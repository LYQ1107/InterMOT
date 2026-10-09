"""Aggregate the actual sealed pixel-SOT reference, not a same-candidate gate."""
import json
import numpy as np
from scripts.n72r21_common import OUT, read_json, write_json, sha256


def summarize(episodes):
    visible = sum(e['visible_frames'] for e in episodes)
    correct = sum(e['correct_frames_IoU_0_5'] for e in episodes)
    gaps = [g for e in episodes for g in e['reappearance_episodes']]
    ratio = lambda n,d: n/d if d else None
    return {'episodes': len(episodes), 'frames': sum(e['frames'] for e in episodes),
            'visible_frames': visible, 'correct_frames': correct, 'all_visible_target_recall': ratio(correct,visible),
            'visible_weighted_SOT_success_AUC_21_thresholds': ratio(sum(e['SOT_success_AUC_21_thresholds']*e['visible_frames'] for e in episodes if e['visible_frames']),visible),
            'visible_weighted_SOT_normalized_precision_at_0_2': ratio(sum(e['SOT_normalized_precision_at_0_2']*e['visible_frames'] for e in episodes if e['visible_frames']),visible),
            'verified_wrong_person_takeover_frames': sum(e['verified_wrong_person_takeover_frames'] for e in episodes),
            'visible_GT_gap_false_presence_frames': sum(e['visible_GT_gap_false_presence_frames'] for e in episodes),
            'reappearance_episodes': len(gaps), 'reacquisition_recall': ratio(sum(g['reacquired'] for g in gaps),len(gaps)),
            'reacquisition_recall_by_seconds': {str(s): ratio(sum(g['delay_seconds'] is not None and g['delay_seconds']<=s for g in gaps),len(gaps)) for s in [1,2,5,10]},
            'memory_safety_PASS': False, 'no_NONE_head': True}


def run():
    sequences = read_json(OUT/'protocol/SOT_REFERENCE_PROTOCOL.json')['DanceTrack_train_sequences']
    sources = {}; results = {}; episodes = []
    for sequence in sequences:
        path = OUT/'baselines/sot/evaluations'/f'{sequence}.json'
        data = read_json(path)
        results[sequence] = summarize(data['episodes']); episodes += data['episodes']; sources[sequence] = sha256(path)
    values = np.asarray([r['all_visible_target_recall'] for r in results.values()])
    rng = np.random.default_rng(72104); draw = rng.integers(0,len(values),(2000,len(values)))
    report = {'baseline': 'B6_OFFICIAL_OSTRACK_VITB256_CE', 'status': 'COMPLETE_REGISTERED_TRAIN_REFERENCE_NOT_FINAL_GENERALIZATION',
              'final_goal_file': 'outputs/N72R21/FINAL_GOAL.json', 'pooled': summarize(episodes), 'per_sequence': results,
              'sequence_macro_target_recall': float(values.mean()), 'sequence_cluster_95pct_CI': np.quantile(values[draw].mean(axis=1),[.025,.975]).tolist(),
              'evaluation_sha256': sources, 'same_SAM3_candidate_axis': False, 'original_model_always_emits_box': True,
              'no_target_NONE_or_memory_safety_success_claim': True, 'cross_recording_evaluation': False,
              'new_training': False, 'scientific_success': False, 'next_stage_authorized': False}
    write_json('baselines/SOT_DEVELOPMENT_SUMMARY.json',report)
    print(json.dumps(report['pooled']),flush=True)


if __name__ == '__main__': run()
