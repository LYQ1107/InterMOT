"""Offline event summaries, distinct from runtime or causal attribution.

Intervals are maximal adjacent true frames. They are NOT independent initiating
interventions: baseline becoming correct can start a run without a new action.
"""
from __future__ import annotations


def contiguous_intervals(frames):
    frames = sorted(set(int(f) for f in frames))
    result = []
    for frame in frames:
        if not result or frame != result[-1]['end_frame'] + 1:
            result.append({'start_frame': frame, 'end_frame': frame, 'frames': 1})
        else:
            result[-1]['end_frame'] = frame
            result[-1]['frames'] += 1
    return result


def assignment_map(row):
    return {int(o['public_id']): o['candidate_uid'] for o in row['outputs']}


def identity_outcome(uid, matched, target):
    if uid is None:
        return 'NONE'
    identity = matched.get(uid)
    if identity is None:
        return 'UNKNOWN_UNMATCHED'
    return 'TARGET' if identity == target else 'VERIFIED_OTHER'


def analyze_target_events(rows):
    """Pure posthoc scalars; caller must have sealed runtime before GT access."""
    harm = [r['frame'] for r in rows if r['C0_correct'] and not r['actual_correct']]
    benefit = [r['frame'] for r in rows if not r['C0_correct'] and r['actual_correct']]
    direct_harm = [r['frame'] for r in rows if r['effective_action'] and r['own_KEEP_correct'] and not r['actual_correct']]
    direct_benefit = [r['frame'] for r in rows if r['effective_action'] and not r['own_KEEP_correct'] and r['actual_correct']]
    harmful_actions = set(direct_harm)
    intervals = contiguous_intervals(harm)
    for interval in intervals:
        interval['direct_harmful_action_frames_inside'] = [f for f in direct_harm if interval['start_frame'] <= f <= interval['end_frame']]
        interval['starts_with_direct_harmful_action'] = interval['start_frame'] in harmful_actions
        interval['recovery_frame'] = next((r['frame'] for r in rows if r['frame'] > interval['end_frame'] and r['actual_correct']), None)
        interval['causal_origin_proven'] = False
    return {'N01_frames': len(benefit), 'N10_frames': len(harm),
            'direct_harmful_decisions_against_own_KEEP': len(direct_harm),
            'direct_beneficial_decisions_against_own_KEEP': len(direct_benefit),
            'direct_harmful_action_frames': direct_harm, 'direct_beneficial_action_frames': direct_benefit,
            'N10_observed_intervals': intervals, 'N01_observed_intervals': contiguous_intervals(benefit),
            'intervals_not_independent_causal_events': True,
            'N10_frames_without_current_direct_harmful_action': len(set(harm) - harmful_actions),
            'first_N10': min(harm, default=None), 'first_N01': min(benefit, default=None),
            'first_direct_harmful_action': min(direct_harm, default=None)}
