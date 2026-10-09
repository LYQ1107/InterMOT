"""Actual bounded FIT-only optimization, INNER calibration, sealed checkpoints."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import itertools
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, update_status
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.learned_authority import TrajectoryAuthorityHead, eligible_prediction

PROTOCOL = OUT / 'protocol/LEARNED_AUTHORITY_V1.json'
CODE = ['scripts/n72r21r1_train_authority.py', 'sam3_intermot/one_click/learned_authority.py']


def supervision(row, reward):
    labels = row['raw_trajectory_labels']
    if reward == 'CURRENT_ONLY': metric = labels['time_zero']; divisor = 1
    else:
        h = int(reward[1:]) if reward in ('H5', 'H20', 'H50') else 100
        metric = labels['future']['H'+str(h)]; divisor = h
        if not metric['complete']: return None
    risk = bool(metric['N10'] or metric['non_target_damage'] or metric['wrong_or_UNKNOWN_writes'])
    value = metric['N01']-metric['N10']
    if reward != 'TARGET_CONTINUITY':
        value -= metric['non_target_damage']+metric['wrong_or_UNKNOWN_writes']
        value += metric['non_target_benefit']
    # Association/fragmentation component retained as explicit proxy, never HOTA.
    if reward == 'TRACKEVAL_ALIGNED_PROXY': value -= .1 * metric['same_target_fragment_displacements']
    benefit = value > 0 and not risk
    return 2 if risk else 1 if benefit else 0, float(np.clip(value/divisor, -3, 3))


def load_records(state, reward):
    manifest = read_json(OUT / 'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json')
    assert sha256(manifest['path']) == manifest['sha256']
    assert sha256(ROOT / 'sam3_intermot/evaluation/joint_trajectory_labels.py') == manifest['label_components_SHA']
    assert sha256(OUT / 'protocol/TRAJECTORY_ORIGIN_DEFINITION.json') == manifest['origin_definition_SHA']
    accepted = []; excluded = Counter()
    for row in read_zstd_jsonl(Path(manifest['path'])):
        if not row['initial_clicked_UID_verified_target']: excluded['unverified_initialization'] += 1; continue
        target = supervision(row, reward)
        if target is None: excluded['incomplete_reward_window'] += 1; continue
        if row['role'] == 'FIT':
            if state == 'BASELINE_ONLY' and row['state_source'] != 'JOINT_BASELINE_SHADOW_P0': continue
            if state == 'TREATMENT_ONLY' and row['state_source'] != 'JOINT_ORIGINAL_TREATMENT_FULL': continue
        if row['role'] not in ('FIT', 'INNER'): continue  # diagnostic0001 never selects
        x = np.array(row['causal_previous_feature_vectors'] + [row['feature_vector']], np.float32)
        assert x.shape == (4, len(FEATURE_NAMES)) and np.isfinite(x).all()
        accepted.append({'x': x, 'class': target[0], 'value': target[1], 'sequence': row['sequence'],
            'role': row['role'], 'group': (row['episode_uid'], row['state_source'], row['frame']),
            'action': row['action'], 'raw': row['raw_trajectory_labels']})
    fit = [r for r in accepted if r['role'] == 'FIT']; inner = [r for r in accepted if r['role'] == 'INNER']
    if not fit or not inner: raise RuntimeError('no usable actual FIT/INNER supervision')
    if not any(r['class'] == 1 for r in fit) or not any(r['class'] == 2 for r in fit):
        raise RuntimeError('insufficient actual benefit/harm diversity; do not fit meaningless success')
    return fit, inner, manifest, dict(excluded)


def tensors(rows, mean, std):
    return (torch.from_numpy((np.stack([r['x'] for r in rows])-mean)/std),
            torch.tensor([r['class'] for r in rows]), torch.tensor([r['value'] for r in rows], dtype=torch.float32))


def loss_components(output, classes, values, weights, *, family):
    classification = F.cross_entropy(output[:, :3], classes, weight=weights, reduction='none')
    value = F.smooth_l1_loss(output[:, 3], values, reduction='none')
    if family == 'GLOBAL_RISK': classification = classification * torch.where(classes == 2, 2., 1.)
    if family == 'ACTION_VALUE': value = value * 3.
    if family == 'PAIRWISE_RANKER':
        # Paired same-prestate loss is added separately, not random batch pairs.
        value = value * .5
    return classification + value


def predictions(output):
    p = output[:, :3].softmax(1).detach().numpy(); value = output[:, 3].detach().numpy()
    return [{'beneficial': float(q[1]), 'harmful': float(q[2]), 'abstain': float(q[0]), 'value': float(v)} for q, v in zip(p, value, strict=True)]


def calibrate(rows, predicted, protocol):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r['action']['family'] != 'KEEP': groups[r['group']].append(i)
    grid = protocol['INNER_calibration_grid']; points = []
    for b, h, v in itertools.product(grid['benefit_min'], grid['harm_max'], grid['value_min']):
        config = {'status': 'CALIBRATED_DIAGNOSTIC', 'benefit_min': b, 'harm_max': h, 'value_min': v}
        selected = []
        for candidates in groups.values():
            eligible = [i for i in candidates if eligible_prediction(predicted[i], config)]
            if eligible:
                selected.append(sorted(eligible, key=lambda i: (-predicted[i]['value'], str(rows[i]['action']['candidate_uid']), rows[i]['action']['family']))[0])
        benefit = sum(rows[i]['class'] == 1 for i in selected); harm = sum(rows[i]['class'] == 2 for i in selected)
        points.append({**config, 'accepted_events': len(selected), 'beneficial_events': benefit, 'harmful_events': harm,
            'harm_rate': harm/len(selected) if selected else None, 'coverage': len(selected)/len(groups) if groups else None,
            'not_independent_sequences': True, 'population_2percent_guarantee': False})
    legal = [p for p in points if p['accepted_events'] >= 5 and p['harm_rate'] <= .02 and p['beneficial_events'] > 0]
    chosen = sorted(legal, key=lambda p: (-p['beneficial_events'], p['harmful_events'], p['accepted_events'], -p['benefit_min'], p['harm_max'], -p['value_min']))[0] if legal else {'status': 'CALIBRATION_ABSTAIN', 'accepted_events': 0, 'nonvacuity_pass': False}
    return chosen, points


def run(family, state, reward, seed):
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    protocol = read_json(PROTOCOL)
    assert family in protocol['families'] and state in protocol['state_contrasts'] and seed in protocol['seeds']
    assert reward == protocol['primary_supervision'] or (family == 'LOGISTIC' and state == 'MIXED' and reward in protocol['LOGISTIC_MIXED_reward_contrasts'])
    uid = '__'.join((family, state, reward, 'seed'+str(seed)))
    record_path = OUT / 'training/authority' / (uid+'.json')
    if record_path.exists():
        record = read_json(record_path)
        assert record['source_code_SHA'] == {p: sha256(ROOT/p) for p in CODE}
        assert record['protocol_SHA'] == sha256(PROTOCOL)
        assert sha256(record['checkpoint_path']) == record['checkpoint_SHA']
        print(json.dumps({'verified_existing_completed_fit': uid}), flush=True); return
    destination = ASSETS / 'training/authority' / uid
    if destination.exists(): raise FileExistsError('retain partial fit; explicit versioned recovery required')
    storage(32 << 20); destination.mkdir(parents=True)
    fit, inner, manifest, excluded = load_records(state, reward)
    scene_counts = Counter(r['sequence'] for r in fit)
    # Every actual FIT scene contributes equally to FIT normalization and loss.
    sample_weights = np.array([1./scene_counts[r['sequence']] for r in fit], np.float32)
    sample_weights /= sample_weights.sum()
    raw = np.stack([r['x'] for r in fit]); mean = (raw[:, -1]*sample_weights[:, None]).sum(axis=0)
    variance = (((raw[:, -1]-mean)**2)*sample_weights[:, None]).sum(axis=0)
    std = np.maximum(np.sqrt(variance), .05).astype(np.float32)
    x, cls, value = tensors(fit, mean, std); ix, icls, ivalue = tensors(inner, mean, std)
    freq = torch.bincount(cls, minlength=3).float(); cw = freq.sum()/freq.clamp_min(1)/3
    sequence_weights = torch.from_numpy(sample_weights*len(fit))
    model = TrajectoryAuthorityHead(family)
    initial = {k: t.detach().clone() for k, t in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=protocol['training']['learning_rate'], weight_decay=protocol['training']['weight_decay'])
    # Same-event pairs only, selected using FIT supervision, not runtime truth.
    groups = defaultdict(list)
    for i, r in enumerate(fit): groups[r['group']].append(i)
    pairs = []
    for ids in groups.values():
        ordered = sorted(ids, key=lambda i: (fit[i]['value'], i))
        if fit[ordered[-1]]['value'] > fit[ordered[0]]['value']:
            pairs.append((ordered[-1], ordered[0]))
    best_loss = float('inf'); best_state = None; best_epoch = None; patience = 0
    steps = 0; nonzero_gradients = 0; max_grad = 0.; began = time.monotonic(); epochs = []
    log_path = destination / 'epochs.jsonl'
    with log_path.open('x') as log:
        for epoch in range(protocol['training']['max_epochs']):
            if time.monotonic()-began > protocol['training']['max_seconds_per_fit']: raise RuntimeError('fit time budget exceeded')
            model.train(); permutation = torch.randperm(len(fit)); losses = []
            for ids in permutation.split(protocol['training']['batch_size']):
                opt.zero_grad(set_to_none=True); output = model(x[ids])
                loss = (loss_components(output, cls[ids], value[ids], cw, family=family)*sequence_weights[ids]).mean()
                loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
                if not np.isfinite(norm): raise RuntimeError('non-finite gradient')
                nonzero_gradients += norm > 0; max_grad = max(max_grad, norm); opt.step(); steps += 1; losses.append(float(loss.detach()))
            pair_loss = 0.
            if family == 'PAIRWISE_RANKER' and pairs:
                opt.zero_grad(set_to_none=True); out = model(x)
                plus, minus = torch.tensor(pairs).T
                pl = F.softplus(-(out[plus, 3]-out[minus, 3])).mean(); pl.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
                nonzero_gradients += norm > 0; max_grad = max(max_grad, norm); opt.step(); steps += 1; pair_loss = float(pl.detach())
            model.eval()
            with torch.inference_mode():
                io = model(ix); inner_loss = float(loss_components(io, icls, ivalue, cw, family=family).mean())
                accuracy = float((io[:, :3].argmax(1) == icls).float().mean())
            epoch_path = destination / ('epoch'+str(epoch+1)+'.pt')
            torch.save({'schema': 'N72R21R1_TRAJECTORY_AUTHORITY_V1', 'family': family, 'model': model.state_dict(),
                'feature_names': list(FEATURE_NAMES), 'FIT_mean': mean.tolist(), 'FIT_std': std.tolist(),
                'selection': {'status': 'UNCALIBRATED_EPOCH_NOT_DEPLOYABLE'}, 'seed': seed, 'epoch': epoch+1,
                'supervision_SHA': manifest['sha256'], 'protocol_SHA': sha256(PROTOCOL)}, epoch_path)
            item = {'epoch': epoch+1, 'optimizer_steps': steps, 'FIT_loss': float(np.mean(losses)),
                'same_prestate_pair_loss': pair_loss, 'INNER_loss': inner_loss, 'INNER_accuracy': accuracy,
                'checkpoint_path': str(epoch_path), 'checkpoint_SHA': sha256(epoch_path)}
            epochs.append(item); log.write(json.dumps(item, sort_keys=True, allow_nan=False)+'\n'); log.flush()
            if inner_loss < best_loss:
                best_loss = inner_loss; best_epoch = epoch+1; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}; patience = 0
            else: patience += 1
            if patience >= protocol['training']['patience']: break
    assert steps and nonzero_gradients and best_state is not None
    model.load_state_dict(best_state, strict=True); model.eval()
    with torch.inference_mode(): pred = predictions(model(ix))
    selection, points = calibrate(inner, pred, protocol)
    checkpoint = destination / 'selected_calibrated.pt'
    torch.save({'schema': 'N72R21R1_TRAJECTORY_AUTHORITY_V1', 'family': family, 'model': best_state,
        'feature_names': list(FEATURE_NAMES), 'FIT_mean': mean.tolist(), 'FIT_std': std.tolist(), 'selection': selection,
        'seed': seed, 'epoch': best_epoch, 'supervision_SHA': manifest['sha256'], 'protocol_SHA': sha256(PROTOCOL)}, checkpoint)
    changed = sum(not torch.equal(t, initial[k]) for k, t in best_state.items())
    assert changed, 'optimizer did not change actual head weights'
    record = {'stage': 'N72R21R1', 'status': 'COMPLETE_ACTUAL_CONTROLLER_OPTIMIZATION', 'fit_uid': uid,
        'family': family, 'state_contrast': state, 'reward': reward, 'seed': seed, 'device': 'cpu',
        'checkpoint_path': str(checkpoint), 'checkpoint_SHA': sha256(checkpoint), 'strict_state_dict_schema': True,
        'FIT_sequences': sorted(scene_counts), 'INNER_sequences': sorted({r['sequence'] for r in inner}),
        'FIT_records': len(fit), 'INNER_records': len(inner), 'FIT_class_counts': dict(Counter(r['class'] for r in fit)),
        'INNER_class_counts': dict(Counter(r['class'] for r in inner)), 'excluded_records': excluded,
        'epochs_completed': len(epochs), 'selected_epoch': best_epoch, 'optimizer_steps': steps,
        'nonzero_gradient_steps': nonzero_gradients, 'max_preclip_gradient_norm': max_grad, 'changed_state_tensors': changed,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'seconds': time.monotonic()-began,
        'same_prestate_FIT_pairs': len(pairs), 'epoch_log_path': str(log_path), 'epoch_log_SHA': sha256(log_path),
        'all_epoch_checkpoints': epochs, 'selection': selection, 'INNER_all18_operating_points': points,
        'supervision_manifest_SHA': sha256(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json'),
        'supervision_SHA': manifest['sha256'], 'protocol_SHA': sha256(PROTOCOL), 'source_code_SHA': {p: sha256(ROOT/p) for p in CODE},
        'FIT_only_normalization': True, 'runtime_GT_feature': False, 'no_VAL_test_confirmation_selection': True,
        'historical_INNER_not_virgin': True, 'not_closed_loop_MOT_evaluation': True, 'not_scientific_success': True}
    write_json('training/authority/'+uid+'.json', record)
    print(json.dumps({'actual_optimization_complete': uid, 'epochs': len(epochs), 'steps': steps,
        'calibration_status': selection['status'], 'INNER_selected_events': selection['accepted_events'],
        'seconds': round(record['seconds'], 2)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--family', required=True); parser.add_argument('--state', required=True)
    parser.add_argument('--reward', default='H100_GLOBAL_RISK'); parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args(); run(args.family, args.state, args.reward, args.seed)
