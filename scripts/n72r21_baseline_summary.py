"""Descriptive paired sequence-cluster summaries of sealed TRAIN baselines.

Does not select a method/threshold/checkpoint or authorize a next stage.
Incomplete cohorts and historical research exposure are prominently marked.
"""
import json
import numpy as np
from scripts.n72r21_common import OUT,read_json,write_json,sha256
from scripts.n72r21_baselines import CASES


def summarize(episodes):
    ep=[e for e in episodes if 'target' in e]
    visible=sum(e['target']['visible_frames'] for e in ep)
    available=sum(e['target']['candidate_available_frames'] for e in ep)
    frames=sum(e['target']['frames'] for e in ep)
    correct=sum(round(e['target']['target_recall_all_visible']*e['target']['visible_frames']) for e in ep if e['target']['target_recall_all_visible'] is not None)
    writes=sum(e['memory']['accepted_writes'] for e in ep);wrong=sum(e['memory']['wrong_writes'] for e in ep)
    reacq=[r for e in ep for r in e['target']['reacquisition_episodes']]
    accepted=[r for r in reacq if r['first_accept_correct'] is not None]
    def ratio(n,d):return n/d if d else None
    return {'episodes':len(ep),'frames':frames,'visible_frames':visible,'candidate_available_frames':available,
            'correct_frames':correct,'target_recall_all_visible':ratio(correct,visible),
            'verified_wrong_person_takeover_frames':sum(e['target']['verified_wrong_person_takeover_frames'] for e in ep),
            'verified_wrong_person_takeover_episodes':sum(e['target']['verified_wrong_person_takeover_episodes'] for e in ep),
            'visible_GT_gap_false_presence_frames':sum(e['target']['false_presence_frames'] for e in ep),
            'initialization_failures':sum(e['initialization_failure'] for e in ep),
            'reappearance_episodes':len(reacq),'reacquisition_recall':ratio(sum(r['reacquired'] for r in reacq),len(reacq)),
            'reacquisition_first_accept_precision':ratio(sum(r['first_accept_correct'] for r in accepted),len(accepted)),
            'reacquisition_recall_by_seconds':{str(s):ratio(sum(r['recall_by_seconds'][str(s)] for r in reacq),len(reacq)) for s in (1,2,5,10)},
            'writes':writes,'wrong_writes':wrong,'wrong_write_rate':ratio(wrong,writes),
            'correct_observation_retention':ratio(writes-wrong,available),
            'memory_joint_gate':bool(writes and available and wrong/writes<=.02 and (writes-wrong)/available>=.6),
            'zero_write_memory_PASS_allowed':False}


def run():
    protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json');expected=protocol['sequences'];results={}
    for case in CASES:
        records={};sources={}
        for sequence in expected:
            p=OUT/'baselines/evaluations'/case/f'{sequence}.json'
            if p.exists():
                records[sequence]=read_json(p)['episodes'];sources[sequence]=sha256(p)
        all_ep=[e for ep in records.values() for e in ep]
        if not all_ep:continue
        per_sequence={s:summarize(ep) for s,ep in records.items()}
        aggregate=summarize(all_ep)
        results[case]={'pooled':aggregate,'per_sequence':per_sequence,'source_evaluation_sha256':sources,
                      'cohort_complete':set(records)==set(expected),'sequence_macro_target_recall':float(np.mean([r['target_recall_all_visible'] for r in per_sequence.values() if r['target_recall_all_visible'] is not None]))}
    deltas={};base=results.get('B0_RAW_ANCHOR',{}).get('per_sequence',{})
    for case,r in results.items():
        seq=sorted(set(base)&set(r['per_sequence']));values=np.asarray([r['per_sequence'][s]['target_recall_all_visible']-base[s]['target_recall_all_visible'] for s in seq])
        if len(seq)<2:ci=None
        else:
            rng=np.random.default_rng(protocol['bootstrap']['seed']);draw=rng.integers(0,len(seq),(protocol['bootstrap']['replicates'],len(seq)))
            ci=np.quantile(values[draw].mean(axis=1),[.025,.975]).tolist()
        deltas[case]={'paired_sequences':seq,'sequence_macro_target_recall_delta_vs_raw_anchor':float(values.mean()) if len(values) else None,
                      'paired_sequence_cluster_95pct_CI':ci,'not_frame_IID':True,'not_independent_final_validation':True,
                      'complete_preregistered_cohort':len(seq)==len(expected)}
    write_json('baselines/DEVELOPMENT_SUMMARY.json',{'status':'INTERMEDIATE_DESCRIPTIVE_ONLY',
        'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','baseline_results':results,'paired_deltas':deltas,
        'source_scope':'8 historically exposed DanceTrack TRAIN scenes; per-sequence completion explicit',
        'uncalibrated_threshold_comparisons_are_not_a_final_scientific_gate':True,'scientific_success':False,
        'B6_status':'COMPLETE_SEPARATE_PIXEL_SOT_REFERENCE' if (OUT/'baselines/SOT_DEVELOPMENT_SUMMARY.json').exists() else 'PENDING_VERIFIED_PRETRAINED_SOT_ACTUAL_INFERENCE','no_method_selected_from_outer_results':True,
        'cross_recording_test_executed':False,'next_stage_authorized':False})
    print(json.dumps({'cases':len(results),'completed_sequences_per_case':{c:len(r['per_sequence']) for c,r in results.items()}}))


if __name__=='__main__':run()
