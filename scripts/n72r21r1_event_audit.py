"""A1/A2: reverified actual full-MOT traces, then TRAIN-only offline truth.

No controller, threshold or heldout access. Observational intervals are not
claimed to identify independent causal onsets; A3 branches remain required.
"""
from collections import Counter
from pathlib import Path
import json
import numpy as np

from scripts.n72r21r1_common import ROOT, OUT, TRAIN, historical, read_json, write_json, sha256, update_status
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21_validation import cases, case_id
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.evaluation.one_click_protocol import iou
from sam3_intermot.evaluation.safe_intervention_events import assignment_map, identity_outcome, analyze_target_events


def verified_seals(version):
    protocol = read_json(historical('protocol/MOT_TRAIN_PILOT.json'))
    seals = {}
    directory = 'reproduction_v2_seals' if version == 2 else 'reproduction_seals'
    source = ROOT / 'scripts' / ('n72r21r1_reproduce_v2.py' if version == 2 else 'n72r21r1_reproduce.py')
    for name, seed, _ in cases(protocol):
        for sequence in protocol['sequences']:
            case = case_id(name, seed)
            path = OUT / 'source_audit' / directory / case / (sequence + '.json')
            seal = read_json(path)
            assert seal['source_sha256'] == sha256(source)
            assert seal['historical_seal_sha256'] == sha256(historical('mot_pilot/runtime_seals/' + case + '/' + sequence + '.json'))
            assert seal['per_frame_committed_states_and_outputs_AA'] and seal['trajectory_byte_identical_to_original']
            assert not seal['GT_parsed_by_runtime']
            assert all(sha256(a['path']) == a['sha256'] for a in seal['artifacts'])
            seals[case, sequence] = seal
    return seals


def seal_trace(seal):
    return read_zstd_jsonl(Path(next(a['path'] for a in seal['artifacts'] if a['kind'] == 'trace')))


def audit():
    # Both versions fully complete; V1 native/gap fields explicitly excluded.
    v1, v2 = verified_seals(1), verified_seals(2)
    receipt = {'stage': 'N72R21R1', 'status': 'COMPLETE_ACTUAL_JOINT_REPRODUCTION',
               'original_full_MOT_case_sequences': 32, 'independent_v1_runs': len(v1), 'independent_v2_runs': len(v2),
               'actual_total_reproduction_runs': len(v1) + len(v2),
               'all_states_outputs_and_trajectory_bytes_equal_original': True,
               'TrackEval_freshly_invoked': False, 'scientific_success': False,
               'native_gap_diagnostic_version': 2,
               'diagnostic_repair_sha256': sha256(OUT / 'source_audit/PRECOMMIT_DIAGNOSTIC_REPAIR.json'),
               'v2_seal_sources': {f'{c}/{s}': sha256(OUT / 'source_audit/reproduction_v2_seals' / c / (s + '.json')) for c, s in v2}}
    write_json('source_audit/BASELINE_REPRODUCTION_RECEIPT.json', receipt)
    labels_path = historical('development/INITIALIZATION_TRUTH.json')
    labels = {r['episode_uid']: r['target_gt_identity'] for r in read_json(labels_path)['labels']}
    old_result = read_json(historical('mot_pilot/RESULT.json'))
    summaries, firsts, propagation, corrections, damage, all_details = {}, {}, {}, {}, {}, {}
    for sequence in ('dancetrack0001', 'dancetrack0002'):
        init = read_json(historical('mot_pilot/initialization/' + sequence + '.json'))
        event = init['event']; target = labels[event['episode_uid']]
        frames, _ = checked_frames(sequence)
        gt = dancetrack_annotations(TRAIN / sequence)
        matched = {int(p['frame']): strict_candidate_matching(rows, gt.get(int(p['frame']), [])) for p, rows in frames}
        baseline = seal_trace(v2['CLICK_C0', sequence]); target_public = baseline[event['frame']]['target_public_id']
        origin = {}
        for row in baseline:
            for o in row['outputs']:
                identity = matched[row['frame']].get(o['candidate_uid'])
                if identity is not None:
                    origin.setdefault(o['public_id'], identity)
        for seed in (72101, 72102, 72103):
            case = 'ACIB_FULL_SEED' + str(seed); key = case + '/' + sequence
            trace = seal_trace(v2[case, sequence])
            assert trace[event['frame']]['target_public_id'] == target_public
            observations = []; nontarget_counter = Counter(); categories = Counter(); damage_direct = Counter()
            first_divergence = first_state_divergence = None
            effective_frames = []; takeover_frames = []; memory_writes = 0
            for (payload, rows), base, actual in zip(frames, baseline, trace, strict=True):
                f = int(payload['frame']); assert f == base['frame'] == actual['frame']
                bm, am = assignment_map(base), assignment_map(actual)
                if f <= event['frame']:
                    assert bm == am and base['state_after'] == actual['state_after']
                    continue
                diag = actual['r21r1_precommit_diagnostic']; labels_now = matched[f]
                actual_uid = am.get(target_public); c0_uid = bm.get(target_public); keep_uid = diag['own_KEEP_target_uid']
                right = labels_now.get(actual_uid) == target; c0_right = labels_now.get(c0_uid) == target
                keep_right = labels_now.get(keep_uid) == target
                effective = bool(diag['own_KEEP_assignment_changed'])
                if effective: effective_frames.append(f)
                if first_divergence is None and bm != am: first_divergence = f
                if first_state_divergence is None and base['state_after'] != actual['state_after']: first_state_divergence = f
                targets = [a for a in gt.get(f, []) if a['identity'] == target]
                max_iou = max((iou(r['box_xyxy'], targets[0]['box']) for r in rows), default=0.) if targets else None
                positive = [u for u, identity in labels_now.items() if identity == target]
                outcome = identity_outcome(actual_uid, labels_now, target)
                if outcome == 'VERIFIED_OTHER': takeover_frames.append(f)
                n10 = c0_right and not right
                if n10:
                    if effective and keep_right: categories['direct_wrong_override_against_own_KEEP'] += 1
                    else: categories['no_current_direct_harmful_override_observational_not_causal'] += 1
                    if actual_uid is None: categories['incorrect_NONE'] += 1
                    if outcome == 'VERIFIED_OTHER': categories['other_person_candidate_takeover'] += 1
                    if diag['displaced_public_ids']: categories['competing_public_ID_displacement'] += 1
                    if not positive: categories['no_strictly_matched_target_candidate'] += 1
                    if targets and max_iou < .5: categories['poor_or_missing_localization'] += 1
                    if outcome == 'UNKNOWN_UNMATCHED': categories['ambiguous_unverified_identity'] += 1
                    if actual['births'] or actual['deaths']: categories['concurrent_birth_or_death_not_proven_cause'] += 1
                    if outcome == 'VERIFIED_OTHER' and diag['precommit_native_tid'] is not None:
                        chosen = next(r for r in rows if r['candidate_uid'] == actual_uid)
                        if int(chosen['native_tid']) == diag['precommit_native_tid']:
                            categories['wrong_identity_native_continuation_not_proven_cause'] += 1
                local_keep = {int(p): uid for p, uid in actual['base_assignments'].items()}
                affected = []
                for public, identity in origin.items():
                    if public == target_public: continue
                    if labels_now.get(bm.get(public)) == identity and labels_now.get(am.get(public)) != identity:
                        nontarget_counter[public] += 1; affected.append(public)
                        if labels_now.get(local_keep.get(public)) == identity:
                            damage_direct[public] += 1
                memory_writes += int(actual['joint_identity_memory_write'])
                decision = actual['identity_decision']
                observations.append({'frame': f, 'C0_correct': c0_right, 'own_KEEP_correct': keep_right, 'actual_correct': right,
                    'effective_action': effective, 'C0_uid': c0_uid, 'actual_uid': actual_uid, 'own_KEEP_uid': keep_uid,
                    'actual_identity_outcome': outcome, 'target_present': bool(targets), 'positive_candidate_uids': positive,
                    'max_candidate_target_IoU': max_iou, 'affected_non_target_public_ids': affected,
                    'births': actual['births'], 'deaths': actual['deaths'], 'selected_action': actual['selected_action'],
                    'proposal_gt_outcome': identity_outcome(diag['proposed_uid'], labels_now, target),
                    'precommit_diagnostic': diag,
                    'rank1_joint_probability': decision['rank1_identity_joint_probability'] if decision else None,
                    'machine_memory_write': actual['joint_identity_memory_write'],
                    'tracker_state_before_sha256': actual['state_before'], 'tracker_state_after_sha256': actual['state_after']})
            events = analyze_target_events(observations)
            expected = old_result['target_statistics'][case][sequence]
            assert events['N01_frames'] == expected['N01'] and events['N10_frames'] == expected['N10']
            assert sum(nontarget_counter.values()) == expected['non_target_initial_origin_correct_frame_damage']
            assert memory_writes == 0  # Actual observed Full, not an assumption for new policies.
            summaries[key] = {'sequence': sequence, 'seed': seed, 'target_gt_identity_offline_only': target,
                'clicked_public_id': target_public, 'N01': events['N01_frames'], 'N10': events['N10_frames'],
                'effective_global_intervention_frames': len(effective_frames),
                'direct_harmful_decisions_against_own_KEEP': events['direct_harmful_decisions_against_own_KEEP'],
                'direct_beneficial_decisions_against_own_KEEP': events['direct_beneficial_decisions_against_own_KEEP'],
                'observed_N10_runs': len(events['N10_observed_intervals']), 'overlapping_N10_categories': dict(categories),
                'verified_other_takeover_frames': len(takeover_frames), 'actual_accepted_machine_bank_writes': memory_writes,
                'accepted_memory_write_is_initial_harm_cause': False,
                'ordinary_tracker_prototype_and_native_feedback_contamination_not_yet_causally_proven': True,
                'distinct_affected_non_target_public_ids': len(nontarget_counter),
                'non_target_initial_origin_correct_frame_damage': sum(nontarget_counter.values())}
            selected = sorted({f for f in (first_divergence, events['first_N10'], events['first_direct_harmful_action'], events['first_N01']) if f is not None})
            by_frame = {r['frame']: r for r in observations}
            firsts[key] = {'first_complete_ownership_divergence': first_divergence, 'first_tracker_state_divergence': first_state_divergence,
                'first_N10': events['first_N10'], 'first_direct_harmful_action': events['first_direct_harmful_action'], 'first_N01': events['first_N01'],
                'A3_predeclared_onsets': selected,
                'windows': {str(f): [by_frame[k] for k in range(max(event['frame'] + 1, f - 2), min(len(frames), f + 3))] for f in selected}}
            propagation[key] = {'observed_N10_intervals': events['N10_observed_intervals'],
                'N10_frames_without_current_direct_harmful_action': events['N10_frames_without_current_direct_harmful_action'],
                'direct_harmful_action_frames': events['direct_harmful_action_frames'],
                'causal_propagation_proven': False, 'requires_same_prestate_A3_branches': True}
            corrections[key] = {'N01_observed_intervals': events['N01_observed_intervals'],
                'direct_beneficial_action_frames': events['direct_beneficial_action_frames'],
                'independent_causal_correction_events_not_proven': True}
            damage[key] = {'initial_C0_origin_proxy_not_global_GT_IDF1': True,
                'frame_damage_by_public_id': dict(nontarget_counter), 'current_action_damage_against_own_KEEP_by_public_id': dict(damage_direct),
                'distinct_affected_public_ids': sorted(nontarget_counter),
                'initial_origin_after_first_C0_verified_match': origin,
                'causal_damage_duration_requires_A3': True}
            all_details[key] = observations
            print(json.dumps({'A1_A2_actual_joint_audit': key, **{k: summaries[key][k] for k in ('N01', 'N10', 'effective_global_intervention_frames', 'direct_harmful_decisions_against_own_KEEP', 'observed_N10_runs')}}, sort_keys=True), flush=True)
    provenance = {'stage': 'N72R21R1', 'runtime_sealed_before_offline_GT': True, 'VAL_or_test_used': False,
        'runtime_GT_inputs': False, 'offline_initialization_truth_SHA': sha256(labels_path),
        'baseline_reproduction_receipt_SHA': sha256(OUT / 'source_audit/BASELINE_REPRODUCTION_RECEIPT.json'),
        'source_sha256': sha256(Path(__file__)),
        'intervals_are_observational_not_independent_causal_origin_counts': True,
        'scientific_success': False, 'A3_complete': False}
    for name, payload in [('N10_ROOT_CAUSE', summaries), ('FIRST_DIVERGENCE_AUDIT', firsts), ('ERROR_PROPAGATION', propagation), ('CORRECTION_EPISODES', corrections), ('NON_TARGET_DAMAGE', damage)]:
        write_json('diagnostics/' + name + '.json', {'provenance': provenance, 'cases': payload})
    # Full frame truth remains local, excluded from code-only delivery.
    write_json('diagnostics/POSTHOC_FRAME_EVENTS.json', {'provenance': provenance, 'cases': all_details})
    update_status(baseline_reproduction_complete=True, completed_new_rollouts=64,
                  phase_A_observational_audit_complete=True, phase_A_root_cause_complete=False,
                  status='ACTIVE_PHASE_A_SAME_PRESTATE_COUNTERFACTUAL')


if __name__ == '__main__':
    audit()
