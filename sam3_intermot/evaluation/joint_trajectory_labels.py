"""Posthoc TRAIN trajectory components. This module is not a runtime scorer."""
from collections import Counter
from .safe_intervention_events import assignment_map, identity_outcome, contiguous_intervals


def non_target_damage(actual, baseline, matched, origins, target_public, target_identity=None):
    am, bm = assignment_map(actual), assignment_map(baseline)
    harmed = [p for p, identity in origins.items() if p != target_public and identity != target_identity and matched.get(bm.get(p)) == identity and matched.get(am.get(p)) != identity]
    helped = [p for p, identity in origins.items() if p != target_public and identity != target_identity and matched.get(bm.get(p)) != identity and matched.get(am.get(p)) == identity]
    fragments = [p for p, identity in origins.items() if p != target_public and identity == target_identity and matched.get(bm.get(p)) == identity and matched.get(am.get(p)) != identity]
    return {'harmed_public_ids': sorted(harmed), 'helped_public_ids': sorted(helped),
            'same_target_fragment_displaced_public_ids': sorted(fragments),
            'UNKNOWN_damage_is_conservative_origin_proxy_not_verified_takeover': True}


def memory_safety(correct, verified_other, unknown, eligible):
    accepted = correct + verified_other + unknown
    risk = (verified_other + unknown) / accepted if accepted else None
    retention = correct / eligible if eligible else None
    return {'accepted': accepted, 'correct': correct, 'verified_other': verified_other, 'UNKNOWN': unknown,
            'wrong_or_unknown_rate': risk, 'correct_retention': retention,
            'safety_and_usefulness_pass': bool(accepted and risk <= .02 and retention is not None and retention >= .6)}


def label_branch(rows, keep, matched, target_identity, *, prestate_origins=None):
    """All raw components retained; current and future windows never conflated."""
    assert len(rows) == len(keep) and rows
    public = keep[0]['target_public_id']; start = keep[0]['frame']
    origins = dict(prestate_origins) if prestate_origins is not None else {}
    if prestate_origins is None:
        for r in keep:
            for o in r['outputs']:
                identity = matched[r['frame']].get(o['candidate_uid'])
                if identity is not None: origins.setdefault(o['public_id'], identity)
    components = []; correct_frames = []; wrong_frames = []; affected = Counter()
    for actual, baseline in zip(rows, keep, strict=True):
        frame = actual['frame']; assert frame == baseline['frame']
        truth = matched[frame]; uid = assignment_map(actual).get(public); base_uid = assignment_map(baseline).get(public)
        right, base_right = truth.get(uid) == target_identity, truth.get(base_uid) == target_identity
        damage = non_target_damage(actual, baseline, truth, origins, public, target_identity)
        outcome = identity_outcome(uid, truth, target_identity)
        base_outcome = identity_outcome(base_uid, truth, target_identity)
        write = bool(actual['memory_write']); write_uid = actual['memory_write_uid']
        if write: assert write_uid == uid, 'write not bound to actually committed UID'
        row = {'offset': frame-start, 'target_correct': int(right), 'N01': int(right and not base_right), 'N10': int(base_right and not right),
            'non_target_damage': len(damage['harmed_public_ids']), 'non_target_benefit': len(damage['helped_public_ids']),
            'harmed_public_ids': damage['harmed_public_ids'], 'verified_other_takeover': int(outcome == 'VERIFIED_OTHER'),
            'same_target_fragment_displacements': len(damage['same_target_fragment_displaced_public_ids']),
            'takeover_increase_vs_KEEP': int(outcome == 'VERIFIED_OTHER') - int(base_outcome == 'VERIFIED_OTHER'),
            'UNKNOWN': int(outcome == 'UNKNOWN_UNMATCHED'), 'target_NONE': int(outcome == 'NONE'),
            'ownership_changed': int(assignment_map(actual) != assignment_map(baseline)),
            'births': len(actual['births']), 'deaths': len(actual['deaths']), 'memory_writes': int(write),
            'wrong_or_UNKNOWN_writes': int(write and truth.get(write_uid) != target_identity),
            'native_diff_vs_KEEP': int(actual['target_native_after'] != baseline['target_native_after']),
            'prototype_drift_vs_KEEP': abs(actual['target_prototype_anchor_cosine_after'] - baseline['target_prototype_anchor_cosine_after'])}
        components.append(row)
        if right: correct_frames.append(frame)
        if row['N10']: wrong_frames.append(frame)
        for p in row['harmed_public_ids']: affected[p] += 1
    windows = {}
    for h in (1, 5, 20, 50, 100):
        future = [r for r in components if 0 < r['offset'] <= h]
        aggregate = {key: sum(r[key] for r in future) for key in components[0] if key not in ('offset', 'harmed_public_ids', 'prototype_drift_vs_KEEP')}
        aggregate.update(actual_future_frames=len(future), complete=len(future) == h,
            distinct_harmed_public_ids=sorted({p for r in future for p in r['harmed_public_ids']}),
            maximum_prototype_drift=max((r['prototype_drift_vs_KEEP'] for r in future), default=0.))
        aggregate['risk_label'] = bool(aggregate['N10'] or aggregate['non_target_damage'] or aggregate['wrong_or_UNKNOWN_writes'])
        aggregate['target_value'] = aggregate['N01'] - aggregate['N10']
        aggregate['global_risk_value'] = aggregate['target_value'] - aggregate['non_target_damage'] + aggregate['non_target_benefit'] - aggregate['wrong_or_UNKNOWN_writes']
        aggregate['benefit_label'] = bool(aggregate['global_risk_value'] > 0 and not aggregate['risk_label'])
        aggregate['proxy_not_actual_HOTA_or_AssA'] = True
        windows['H'+str(h)] = aggregate
    return {'time_zero': components[0], 'future': windows, 'N10_intervals': contiguous_intervals(wrong_frames),
            'first_target_recovery_frame': min((f for f in correct_frames if f > start), default=None),
            'affected_public_frame_counts': dict(affected), 'origin_proxy_not_global_GT_identity_metric': True,
            'origin_reference': 'GENERATING_PRETEST_PREFIX_SEMANTICS' if prestate_origins is not None else 'SHORT_KEEP_WINDOW_CONTINUITY_ONLY',
            'same_target_fragment_displacement_not_other_person_damage': True}
