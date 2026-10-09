"""All registered full-MOT runs sealed first, then common pinned TrackEval/GT."""
from pathlib import Path
from collections import Counter
import json
import subprocess
import time
import numpy as np
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, HISTORY, TRAIN, historical, read_json, write_json, sha256, update_status, storage
from scripts.n72r21r1_fixed_pilot import PROTOCOL, registrations
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.evaluation.safe_intervention_events import assignment_map, analyze_target_events, identity_outcome, contiguous_intervals
from sam3_intermot.evaluation.mot_detection_audit import audit_unchanged_detections
from sam3_intermot.evaluation.one_click_protocol import ratio, open_set_metrics


def verify_all():
    protocol = read_json(PROTOCOL); seals = {}; cases = []
    for seed in protocol['historical_weight_reproduction_seeds']:
        for name, _, _ in registrations(protocol, seed):
            cases.append(name)
            for sequence in protocol['sequences']:
                path = OUT / 'pilot/runtime_seals' / name / (sequence + '.json'); seal = read_json(path)
                assert seal['status'] == 'COMPLETE_ACTUAL_FULL_JOINT_RUNTIME'
                assert seal['protocol_sha256'] == sha256(PROTOCOL) and seal['source_sha256'] == sha256(ROOT / 'scripts/n72r21r1_fixed_pilot.py')
                assert seal['source_code_SHA'] == protocol['source_freeze']
                assert all(sha256(ROOT / p) == digest for p, digest in seal['source_code_SHA'].items())
                assert seal['initialization_sha256'] == sha256(historical('mot_pilot/initialization/' + sequence + '.json'))
                assert seal['global_complete_unique_ownership'] and seal['one_click_only'] and not seal['GT_runtime'] and not seal['future_GT_runtime']
                assert all(sha256(a['path']) == a['sha256'] for a in seal['artifacts'])
                seals[name, sequence] = seal
    assert len(set(cases)) == len(cases) == protocol['actual_case_count_planned']
    assert len(seals) == protocol['sequence_cases_planned']
    return protocol, cases, seals


def artifact(seal, kind):
    return Path(next(a['path'] for a in seal['artifacts'] if a['kind'] == kind))


def offline_statistics(protocol, cases, seals):
    labels_path = historical('development/INITIALIZATION_TRUTH.json')
    labels = {r['episode_uid']: r['target_gt_identity'] for r in read_json(labels_path)['labels']}
    statistics, detection = {}, {}
    for sequence in protocol['sequences']:
        init = read_json(historical('mot_pilot/initialization/' + sequence + '.json')); event = init['event']
        target = labels[event['episode_uid']]; frames, _ = checked_frames(sequence)
        gt = dancetrack_annotations(TRAIN / sequence)
        matched = {int(p['frame']): strict_candidate_matching(rows, gt.get(int(p['frame']), [])) for p, rows in frames}
        baseline = read_zstd_jsonl(artifact(seals['CLICK_C0', sequence], 'trace'))
        baseline_text = artifact(seals['CLICK_C0', sequence], 'trajectory').read_text()
        baseline_public = baseline[event['frame']]['target_public_id']; origin = {}
        for row in baseline:
            for o in row['outputs']:
                identity = matched[row['frame']].get(o['candidate_uid'])
                if identity is not None: origin.setdefault(o['public_id'], identity)
        for case in cases:
            seal = seals[case, sequence]; trace = read_zstd_jsonl(artifact(seal, 'trace'))
            assert [r['frame'] for r in trace] == list(range(len(frames)))
            detection[case + '/' + sequence] = audit_unchanged_detections(baseline_text, artifact(seal, 'trajectory').read_text())
            public = trace[event['frame']]['target_public_id']
            if public is None: public = next(o['public_id'] for o in trace[event['frame']]['outputs'] if o['candidate_uid'] == init['clicked_candidate_uid'])
            observations = []; affected = Counter(); takeover = []; unknown = []; proposed_outcomes = Counter()
            correct = {}; bcorrect = {}; writes = Counter(); probability = []; ranking = []; availability = []
            visible = eligible_correct = available_count = accepted = 0
            for base, actual in zip(baseline, trace, strict=True):
                f = actual['frame']; axis = matched[f]
                assert len({o['candidate_uid'] for o in actual['outputs']}) == len(actual['outputs'])
                assert {o['candidate_uid'] for o in actual['outputs']} == {str(r['candidate_uid']) for r in frames[f][1]}
                bm, am = assignment_map(base), assignment_map(actual)
                if f <= event['frame']:
                    if case != 'NO_HUMAN_C0': assert bm == am
                    continue
                uid = am.get(public); outcome = identity_outcome(uid, axis, target)
                right = axis.get(uid) == target; br = axis.get(bm.get(baseline_public)) == target
                own_keep_uid = actual['authority'].get('own_KEEP_uid', actual['base_assignments'].get(str(public)))
                kr = axis.get(own_keep_uid) == target
                observations.append({'frame': f, 'C0_correct': br, 'actual_correct': right, 'own_KEEP_correct': kr,
                    'effective_action': bool(actual['authority'].get('effective_assignment_change'))})
                correct[f] = right; bcorrect[f] = br; accepted += uid is not None
                target_visible = any(a['identity'] == target for a in gt.get(f, [])); visible += target_visible
                has_target = any(identity == target for identity in axis.values()); available_count += has_target
                eligible_correct += right
                if outcome == 'VERIFIED_OTHER': takeover.append(f)
                if outcome == 'UNKNOWN_UNMATCHED': unknown.append(f)
                for p, identity in origin.items():
                    if p != baseline_public and axis.get(bm.get(p)) == identity and axis.get(am.get(p)) != identity: affected[p] += 1
                decision = actual['identity_decision']
                if decision is not None:
                    proposed = decision['proposed_candidate_uid']; proposed_outcomes[identity_outcome(proposed, axis, target)] += 1
                    probability.append(float(decision['candidate_available_probability']))
                    ranking.append(axis.get(decision['rank1_candidate_uid']) == target)
                    availability.append(has_target)
                if actual['joint_identity_memory_write']:
                    assert actual['joint_memory_write_candidate_uid'] == uid
                    writes[identity_outcome(uid, axis, target)] += 1
            events = analyze_target_events(observations)
            persistence = {}
            for h in (5, 20, 50, 100):
                eligible = [f for f, right in correct.items() if right and not bcorrect[f] and f + h < len(frames)]
                persistent = sum(all(correct[f + k] and not bcorrect[f + k] for k in range(1, h + 1)) for f in eligible)
                persistence['H' + str(h)] = {'eligible_overlapping_correction_frames': len(eligible), 'all_future_strict_benefit': persistent,
                    'rate': ratio(persistent, len(eligible)), 'not_independent_correction_onsets': True}
            unavailable_curve = open_set_metrics(probability, availability, ranking) if probability else None
            statistic = {'strict_correct_UID_frames': eligible_correct, 'visible_frames': visible, 'candidate_available_frames': available_count,
                'strict_UID_recall_visible': ratio(eligible_correct, visible), 'strict_UID_recall_given_candidate': ratio(eligible_correct, available_count),
                'candidate_coverage': ratio(available_count, visible), 'selected_frames': accepted,
                'N01': events['N01_frames'], 'N10': events['N10_frames'], 'N01_minus_N10': events['N01_frames'] - events['N10_frames'],
                'direct_harmful_own_KEEP_decisions': events['direct_harmful_decisions_against_own_KEEP'],
                'direct_beneficial_own_KEEP_decisions': events['direct_beneficial_decisions_against_own_KEEP'],
                'N10_observed_intervals': events['N10_observed_intervals'], 'effective_interventions': seal['effective_interventions'],
                'wrong_person_takeover_frames': len(takeover), 'wrong_person_takeover_runs': contiguous_intervals(takeover),
                'UNKNOWN_selected_frames': len(unknown), 'UNKNOWN_is_not_verified_other': True,
                'non_target_C0_initial_origin_damage_frames': sum(affected.values()), 'distinct_affected_non_target_public_IDs': sorted(affected),
                'damage_frame_counts_by_public_ID': dict(affected), 'correction_persistence': persistence,
                'proposal_outcomes': dict(proposed_outcomes), 'candidate_unavailable_open_set_curve': unavailable_curve,
                'identity_claim_joint_rank_calibration': 'Not available for every new NONE proposal; current trace does not seal rank1 joint probability when NONE wins. Do not infer it from availability.',
                'memory': {'accepted': sum(writes.values()), 'correct': writes['TARGET'], 'verified_other': writes['VERIFIED_OTHER'], 'UNKNOWN': writes['UNKNOWN_UNMATCHED'],
                    'wrong_or_unknown_rate': ratio(writes['VERIFIED_OTHER'] + writes['UNKNOWN_UNMATCHED'], sum(writes.values())),
                    'correct_retention': ratio(writes['TARGET'], available_count),
                    'joint_memory_safety_pass': bool(sum(writes.values()) and (writes['VERIFIED_OTHER'] + writes['UNKNOWN_UNMATCHED']) / sum(writes.values()) <= .02 and available_count and writes['TARGET'] / available_count >= .6)},
                'cached_efficiency': {k: seal[k] for k in ['cached_joint_step_seconds', 'total_cached_replay_seconds', 'cached_joint_FPS_not_pixel_to_output_FPS', 'added_identity_parameters', 'process_cumulative_peak_RSS_bytes_not_per_case_peak', 'association_only_seconds']}}
            statistics[case + '/' + sequence] = statistic
    return statistics, detection


def run_trackeval(cases, sequences):
    pinned = HISTORY / 'third_party/MOTIP/TrackEval'
    commit = subprocess.check_output(['git', '-C', str(pinned), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '12c8791b303e0a0b50f753af204249e622d0281a'
    seqmap = ASSETS / 'fixed_pilot/seqmap.txt'; expected = 'name\n' + '\n'.join(sequences) + '\n'
    if seqmap.exists(): assert seqmap.read_text() == expected
    else:
        seqmap.parent.mkdir(parents=True, exist_ok=True)
        with seqmap.open('x') as handle: handle.write(expected)
    eval_root = ASSETS / 'fixed_pilot/trackeval_v1'
    if eval_root.exists(): raise FileExistsError('preserve evaluator attempt; use versioned explicit resume/repair')
    eval_root.mkdir(parents=True)
    command = _trackeval_command(ASSETS / 'fixed_pilot/trackers', eval_root, cases, seqmap, gt_split='train', gt_folder=TRAIN)
    command[2] = str(pinned / 'scripts/run_mot_challenge.py')
    started = time.monotonic()
    with (eval_root / 'trackeval.log').open('x') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    run = {'command': command, 'returncode': result.returncode, 'seconds': time.monotonic() - started,
        'log_path': str(eval_root / 'trackeval.log'), 'log_sha256': sha256(eval_root / 'trackeval.log'),
        'pinned_TrackEval_commit': commit, 'wrapper_sha256': sha256(ROOT / 'scripts/n72r20r3r2r3_trackeval_entry.py'),
        'same_settings_all_82_trackers': True, 'third_party_checkout_modified': False}
    write_json('pilot/TRACKEVAL_INVOCATION.json', run)
    if result.returncode != 0: raise RuntimeError('real pinned TrackEval failed; retained attempt log')
    metrics = {case: trackeval_summary(parse_trackeval(eval_root, case, sequences)) for case in cases}
    for case, values in metrics.items():
        assert all(values[k] is not None for k in ('HOTA', 'AssA', 'IDF1', 'IDSW', 'DetA', 'LocA', 'MOTA', 'FP', 'FN'))
        assert set(values['per_sequence']) == set(sequences)
    return metrics, run


def sequence_cluster_intervals(metrics, statistics, cases, sequences):
    groups = {}
    for case in cases: groups.setdefault(case.split('_SEED')[0], []).append(case)
    rng = np.random.default_rng(72114); indices = rng.integers(0, len(sequences), (2000, len(sequences)))
    result = {}
    for group, names in groups.items():
        row = {'actual_seeds_per_sequence': len(names), 'actual_sequence_clusters': len(sequences),
               'two_historical_TRAIN_scenes_not_independent_confirmation_or_generalization': True,
               'seeds_and_episodes_averaged_inside_sequence': True, 'uncertainty': {}}
        for key in ('HOTA', 'AssA', 'IDF1', 'IDSW'):
            deltas = np.array([np.mean([metrics[c]['per_sequence'][s][key] for c in names]) - metrics['CLICK_C0']['per_sequence'][s][key] for s in sequences])
            means = deltas[indices].mean(axis=1)
            row['uncertainty'][key] = {'paired_macro_sequence_delta': float(deltas.mean()),
                'sequence_cluster_95pct_CI': np.percentile(means, [2.5, 97.5]).tolist(), 'per_sequence_delta': dict(zip(sequences, deltas.tolist()))}
        for key in ('N01', 'N10', 'effective_interventions', 'direct_harmful_own_KEEP_decisions', 'direct_beneficial_own_KEEP_decisions', 'non_target_C0_initial_origin_damage_frames'):
            row[key + '_mean_seed_sum_sequence'] = sum(float(np.mean([statistics[c + '/' + s][key] for c in names])) for s in sequences)
        row['nonvacuous'] = row['effective_interventions_mean_seed_sum_sequence'] > 0
        row['N01_greater_than_N10'] = row['N01_mean_seed_sum_sequence'] > row['N10_mean_seed_sum_sequence']
        row['not_scientific_PASS_from_safety_alone'] = True
        result[group] = row
    return result


def run():
    storage(128 << 20)
    protocol, cases, seals = verify_all()
    statistics, detection = offline_statistics(protocol, cases, seals)
    metrics, invocation = run_trackeval(cases, protocol['sequences'])
    grouped = sequence_cluster_intervals(metrics, statistics, cases, protocol['sequences'])
    write_json('pilot/FIXED_GATE_RESULT.json', {'stage': 'N72R21R1', 'status': 'COMPLETE_ACTUAL_BOUNDED_FIXED_GATE_PILOT',
        'source_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(PROTOCOL),
        'actual_cases': len(cases), 'actual_full_joint_sequence_cases': len(seals),
        'all_actual_TrackEval_metrics_0to1': metrics, 'offline_target_event_statistics': statistics,
        'exported_unchanged_detection_multiset_audit': detection, 'grouped_paired_sequence_uncertainty': grouped,
        'actual_TrackEval_invocation': invocation, 'scientific_success': None,
        'final_operating_point_selected': False, 'no_VAL_test_new_INNER_confirmation_access': True,
        'scope': 'Historical development pilot only; not Gate3, cross-recording or final generalization.',
        'runtime_seal_SHA': {c + '/' + s: sha256(OUT / 'pilot/runtime_seals' / c / (s + '.json')) for c, s in seals}})
    update_status(phase_B_fixed_pilot_complete=True, completed_new_full_joint_pilot_rollouts=len(seals),
        status='ACTIVE_PHASE_C_D_E_F_MATCHED_STATE_LEARNING_REQUIRED')
    for name, row in grouped.items():
        print(json.dumps({'fixed_group': name, **{k: row[k] for k in ('effective_interventions_mean_seed_sum_sequence', 'N01_mean_seed_sum_sequence', 'N10_mean_seed_sum_sequence', 'nonvacuous', 'N01_greater_than_N10')},
            'delta_HOTA': row['uncertainty']['HOTA']['paired_macro_sequence_delta']}), flush=True)


if __name__ == '__main__': run()
