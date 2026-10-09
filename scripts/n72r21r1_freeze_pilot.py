"""Freeze bounded fixed-control pilot before any new gate trajectory effects."""
from dataclasses import asdict
from scripts.n72r21r1_common import ROOT, OUT, write_json, sha256
from sam3_intermot.one_click.intervention_gate import GatePolicy


def freeze():
    points = [
        {'name': 'P0_BROAD', 'probability_min': .7, 'identity_margin_min': .02, 'base_margin_max': 4., 'quality_min': .2, 'anchor_cosine_min': .4, 'regret_max': 2., 'delay_frames': 1},
        {'name': 'P1_MIDDLE', 'probability_min': .9, 'identity_margin_min': .05, 'base_margin_max': 2., 'quality_min': .5, 'anchor_cosine_min': .6, 'regret_max': 1., 'delay_frames': 2},
        {'name': 'P2_STRICT', 'probability_min': .95, 'identity_margin_min': .1, 'base_margin_max': 1., 'quality_min': .8, 'anchor_cosine_min': .8, 'regret_max': .5, 'delay_frames': 2},
        {'name': 'P3_ABSTAIN', 'probability_min': .99, 'identity_margin_min': .2, 'base_margin_max': 0., 'quality_min': .8, 'anchor_cosine_min': .8, 'regret_max': 0., 'delay_frames': 3},
    ]
    configs = []
    for family in ('confidence', 'recovery', 'global_regret', 'protected', 'delayed', 'abstain'):
        for point in points:
            values = {k: v for k, v in point.items() if k != 'name'}
            configs.append({'name': family.upper() + '_' + point['name'], 'actor': 'ACIB_P0_FROZEN', 'policy': asdict(GatePolicy(family=family, **values))})
    configs += [{'name': 'SHADOW_ACIB_P0', 'actor': 'ACIB_P0_FROZEN', 'policy': asdict(GatePolicy(family='shadow'))},
                {'name': 'ORIGINAL_ACIB_FULL', 'actor': 'ACIB_FULL_FROZEN', 'policy': asdict(GatePolicy(family='original'))}]
    write_json('protocol/FIXED_GATE_PILOT.json', {
        'stage': 'N72R21R1', 'version': 'B_BOUNDED_4POINT_V1', 'frozen_before_new_gate_effects': True,
        'preregistration_sha256': sha256(OUT / 'PREREGISTRATION.json'),
        'purpose': 'Historical-development TRAIN engineering/scientific safety pilot; no best choice from these scenes becomes a final operating point.',
        'sequences': ['dancetrack0001', 'dancetrack0002'], 'historical_weight_reproduction_seeds': [72101, 72102, 72103],
        'new_fit_seeds_unaffected': [72111, 72112, 72113],
        'configurations': configs, 'fixed_unique_configs_per_family': 4, 'registered_per_family_max': 64,
        'common_controls': ['NO_HUMAN_C0', 'CLICK_C0', 'RAW_ANCHOR', 'RAW_MEAN_P1'],
        'actual_case_count_planned': 82, 'sequence_cases_planned': 164,
        'solver_change': False, 'safe_controls_memory': 'P0; write mechanisms evaluated separately, not inferred safe from zero writes.',
        'reject_rule': 'Fixed candidate-proposal gates abstain to KEEP when ACIB proposes NONE; original diagnostic retains original REJECT.',
        'VAL_test_confirmation_or_new_INNER_access': False,
        'all_configs_all_seeds_reported': True, 'zero_intervention_is_not_scientific_PASS': True,
        'progression': 'Continue C/D/E/F/H diagnostics and TRAIN matched-state learning even if fixed gates fail.',
        'source_freeze': {p: sha256(ROOT / p) for p in [
            'sam3_intermot/one_click/intervention_features.py', 'sam3_intermot/one_click/intervention_gate.py',
            'sam3_intermot/one_click/safe_mot_bridge.py', 'sam3_intermot/one_click/joint_intervention_primitives.py']},
    })


if __name__ == '__main__': freeze()
