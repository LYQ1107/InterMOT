"""Posthoc delivery arithmetic only; no runtime, fitting or threshold selection."""
from collections import Counter, defaultdict
import numpy as np


def ratio(n, d):
    return n / d if d else None


def count_metrics(counts):
    c = Counter(counts)
    if any(not isinstance(v, (int, np.integer)) or v < 0 for v in c.values()):
        raise ValueError('actual nonnegative integer counts required')
    if not (c['strict_correct'] <= c['available_frames'] <= c['visible_frames'] <= c['frames']):
        raise ValueError('visible/candidate/strict denominator inconsistency')
    if c['writes'] != c['strict_target_writes'] + c['verified_other_identity_writes'] + c['UNKNOWN_unmatched_writes']:
        raise ValueError('verified target/other/UNKNOWN write partition inconsistent')
    wrong = c['verified_other_identity_writes'] + c['UNKNOWN_unmatched_writes']
    wwr = ratio(wrong, c['writes']); retention = ratio(c['strict_target_writes'], c['available_frames'])
    return {
        'counts': dict(c), 'target_recall_box_IoU0_5': ratio(c['box_correct'], c['visible_frames']),
        'strict_UID_target_recall': ratio(c['strict_correct'], c['visible_frames']),
        'candidate_coverage_given_visible': ratio(c['available_frames'], c['visible_frames']),
        'strict_recognition_recall_given_available': ratio(c['strict_correct'], c['available_frames']),
        'verified_wrong_identity_claim_rate_all_future_frames': ratio(c['verified_wrong_identity_claims'], c['frames']),
        'target_unavailable_FPR_at_frozen_deployment': ratio(c['target_unavailable_accepts'], c['target_unavailable_frames']),
        'visible_GT_gap_FPR_not_physical_absence': ratio(c['GT_gap_accepts'], c['GT_gap_frames']),
        'NONE_false_reject_given_target_candidate': ratio(c['NONE_false_reject_given_available'], c['available_frames']),
        'conservative_non_target_or_unverified_write_rate': wwr,
        'strict_correct_observation_retention': retention,
        'memory_joint_gate': bool(c['writes'] and wwr <= .02 and retention is not None and retention >= .6),
        'zero_write_is_not_safe_memory_PASS': True,
        'UNKNOWN_is_not_verified_wrong_identity_but_is_conservative_write_failure': True,
    }


def sequence_interval(values, bootstrap):
    eligible = {s: float(v) for s, v in values.items() if v is not None}
    if not all(np.isfinite(list(eligible.values()))):
        raise ValueError('nonfinite sequence statistic')
    axis = np.asarray(list(eligible.values()), dtype=float); ci = None
    if len(axis) > 1:
        rng = np.random.default_rng(bootstrap['seed'])
        draw = rng.integers(0, len(axis), (bootstrap['replicates'], len(axis)))
        ci = np.quantile(axis[draw].mean(1), [.025, .975]).tolist()
    return {'sequence_macro_mean': float(axis.mean()) if len(axis) else None,
            'sequence_cluster_95pct_CI': ci, 'eligible_sequences': list(eligible),
            'not_frame_IID': True}


def seed_average_sequence_interval(per_case_sequence, bootstrap):
    if not per_case_sequence:
        raise ValueError('empty compared cases')
    cases = list(per_case_sequence.values()); names = list(cases[0])
    if any(set(c) != set(names) for c in cases):
        raise ValueError('incomplete compared sequence cohort')
    values = {s: float(np.mean([c[s] for c in cases])) if all(c[s] is not None for c in cases) else None for s in names}
    return {**sequence_interval(values, bootstrap),
            'seed_average_inside_sequence_not_independent_seed_clusters': True,
            'registered_sequences': names, 'seeds_or_cases': len(cases)}


def recovery_summary(episodes):
    windows = [w for e in episodes for w in e['strict_UID_reappearance_windows']]
    takeovers = [t for e in episodes for t in e['strict_verified_other_identity_takeover_runs']]
    def summarize(rows):
        delays = [w['strict_delay_seconds'] for w in rows if w['strict_identity_recovered']]
        return {'valid_return_windows': len(rows),
                'strict_identity_recovered_windows': sum(w['strict_identity_recovered'] for w in rows),
                'strict_identity_recovery_recall_all_windows': ratio(sum(w['strict_identity_recovered'] for w in rows), len(rows)),
                'failed_recovery_windows': sum(not w['strict_identity_recovered'] for w in rows),
                'first_accept_strict_correct_windows': sum(w['first_accept_strict_correct'] is True for w in rows),
                'first_accept_verified_wrong_windows': sum(w['first_accept_verified_wrong_identity'] is True for w in rows),
                'no_accept_windows': sum(w['no_accept_in_return_window'] for w in rows),
                'recovered_only_median_strict_delay_seconds': float(np.median(delays)) if delays else None,
                'recovered_only_P95_strict_delay_seconds': float(np.quantile(delays, .95)) if delays else None,
                'failed_windows_not_removed_from_recall_denominator': True}
    by_gap = defaultdict(list)
    for w in windows: by_gap[w['gap_bin']].append(w)
    durations = [t['duration_seconds'] for t in takeovers]
    return {'all_return_windows': summarize(windows), 'returns_by_actual_gap_seconds': {g: summarize(v) for g, v in by_gap.items()},
            'verified_other_identity_takeover_runs': len(takeovers),
            'takeover_total_seconds': float(sum(durations)),
            'takeover_median_run_seconds': float(np.median(durations)) if durations else None,
            'takeover_P95_run_seconds': float(np.quantile(durations, .95)) if durations else None,
            'takeover_runs_without_correct_recovery_before_recording_end': sum(t['no_correct_recovery_before_recording_end'] for t in takeovers),
            'GT_gap_not_proven_physical_absence_or_occlusion': True,
            'within_recording_seconds_not_cross_day_or_session': True}


def require_complete_validation(summary, registered_cases, sequences):
    if set(summary['cases']) != set(registered_cases):
        raise ValueError('all registered VAL cases required')
    for case in registered_cases:
        record = summary['cases'][case]
        if not record['complete_registered_cohort'] or set(record['per_sequence']) != set(sequences):
            raise ValueError('all375 VAL scene-case evaluations required; partial is not final')


def unavailable_cross_recording(scenario):
    return {'scenario': scenario, 'status': 'NOT_EVALUABLE_NO_LAWFUL_LOCAL_MEDIA',
            'target_recall': None, 'ReID_accuracy': None, 'false_takeover': None,
            'valid_episodes': 0, 'metadata_is_not_a_runtime_episode': True,
            'resume_plan': 'outputs/N72R21/datasets/CROSS_RECORDING_RESUME_PLAN.json'}
