"""Post-seal actual P0/P1 in-sample state audit; not T1 learning completion."""
import json
from pathlib import Path
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry


def run():
    from sam3_intermot.one_click.datasets import dancetrack_annotations,dancetrack_truth
    from sam3_intermot.evaluation.one_click_protocol import evaluate_episode
    events={r['episode_uid']:r for r in read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']}
    labels={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    outer='dancetrack0001';sequence='dancetrack0023';seals={}
    for seed in [72101,72102,72103]:
        for policy in ['P0','P1']:
            case=f'{outer}__seed{seed}__{policy}__K8';path=OUT/'training/causal_states/seals'/case/f'{sequence}.json';record=read_json(path)
            for p,h in record['code_sha256'].items():
                if sha256(ROOT/p)!=h:raise ValueError('actual source seal')
            for r in record['artifacts']:
                if sha256(r['path'])!=r['sha256']:raise ValueError('actual state runtime seal')
            seals[case]=record
    gt=dancetrack_annotations(ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence)
    frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
    result={}
    for case,record in seals.items():
        episodes=[]
        for r in record['artifacts']:
            event=events[r['episode_uid']];identity=labels[r['episode_uid']]
            truth=[dancetrack_truth(int(p['frame']),rows,gt.get(int(p['frame']),[]),identity) for p,rows in frames if int(p['frame'])>event['frame']]
            runtime=read_zstd_jsonl(Path(r['path']));metrics=evaluate_episode(runtime,truth,fps=event['fps'],recording_id=sequence)
            metrics.update(episode_uid=r['episode_uid'],initialization_failure=False);episodes.append(metrics)
        result[case]={'episodes':episodes,'source_policy':record['policy'],'source_fit_contains_this_sequence':True,
                      'own_machine_write_not_GT_teacher_forcing':True}
    write_json('training/causal_states/PILOT_POSTHOC_IN_SAMPLE_AUDIT.json',{
        'status':'ACTUAL_CAUSAL_STATE_COLLECTION_PILOT_NOT_T1_MODEL_FIT_OR_GENERALIZATION',
        'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','results':result,
        'model_source':'Completed T0_AMP_R1; empty machine bank during source fit, state mismatch in P1 now explicit',
        'ground_truth_used_only_after_sealed_online_state_collection':True,
        'bank_confidence_semantics':'Candidate availability probability, not calibrated physical presence/identity confidence',
        'physical_absence_truth_or_cross_recording_test':False,'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps({'state_cases_evaluated':len(result),'source_fit_sequence':sequence,'not_independent_validation':True}),flush=True)


if __name__=='__main__':run()
