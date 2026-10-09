"""New learned-policy own-state supervision, separate from frozen round0."""
from pathlib import Path
from collections import Counter
import json
import subprocess
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,TRAIN,read_json,write_json,sha256,update_status,storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.joint_trajectory_labels import label_branch
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES,feature_vector

PROTOCOL=OUT/'protocol/ON_POLICY_ROUND1.json'


def verify_all():
    protocol=read_json(PROTOCOL);seals=[]
    assert all(sha256(ROOT/p)==h for p,h in protocol['source_code_SHA'].items())
    for event in protocol['events']:
        path=OUT/'on_policy/round1/runtime'/event['episode_uid']/'seal.json';seal=read_json(path)
        assert seal['status']=='COMPLETE_ACTUAL_LEARNED_ON_POLICY_JOINT_RUNTIME'
        assert seal['protocol_SHA']==sha256(PROTOCOL) and seal['checkpoint_SHA']==protocol['checkpoint_SHA']
        assert seal['source_code_SHA']==protocol['source_code_SHA'] and seal['inputs_SHA']==protocol['inputs_SHA']
        assert not seal['runtime_GT_input'] and not seal['runtime_future_GT_input'] and not seal['GT_history_replacement']
        assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts'])
        for ref in seal['counterfactual_seals']:
            assert sha256(ref['path'])==ref['sha256'];branch=read_json(ref['path'])
            assert branch['source_code_SHA']==protocol['branch_primitive_code_SHA'] and branch['protocol_sha256']==protocol['branch_method_protocol_SHA']
            assert branch['no_runtime_GT'] and branch['no_GT_best_action']
            assert all(sha256(a['path'])==a['sha256'] for a in branch['artifacts'])
        seals.append((path,seal))
    assert len(seals)==19
    return protocol,seals


def run():
    storage(96<<20);protocol,seals=verify_all()
    identities={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']}
    destination=ASSETS/'on_policy/round1/supervision.jsonl.zst';destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError('preserve existing on-policy supervision, explicit version required')
    records=0;counts=Counter();GT_SHA={}
    with destination.open('xb') as stream:
        compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for sequence in sorted({s['event']['sequence'] for _,s in seals}):
                frames,index_SHA=checked_frames(sequence);gt=dancetrack_annotations(TRAIN/sequence)
                matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
                GT_SHA[sequence]=sha256(TRAIN/sequence/'gt/gt.txt')
                for seal_path,source in [(p,s) for p,s in seals if s['event']['sequence']==sequence]:
                    event=source['event'];target=identities[event['episode_uid']]
                    assert matched[event['frame']].get(event['clicked_candidate_uid'])==target
                    trace=read_zstd_jsonl(Path(next(a['path'] for a in source['artifacts'] if a['kind']=='trace')))
                    assert [r['frame'] for r in trace]==list(range(len(frames)))
                    history={r['frame']:feature_vector(r['authority']['features']).tolist() for r in trace if r['authority']['features'] is not None}
                    origins={};cursor=0
                    for ref in source['counterfactual_seals']:
                        branch=read_json(ref['path']);f=branch['frame']
                        while cursor<f:
                            for observation in trace[cursor]['outputs']:
                                identity=matched[cursor].get(observation['candidate_uid'])
                                if identity is not None:origins.setdefault(observation['public_id'],identity)
                            cursor+=1
                        prestate={p:identity for p,identity in origins.items() if str(p) in trace[f]['base_assignments']}
                        arms=[read_json(a['path']) for a in branch['artifacts']]
                        keep=next(a for a in arms if a['action']['family']=='KEEP')['rows']
                        for arm,artifact in zip(arms,branch['artifacts'],strict=True):
                            assert arm['feature_vector']==feature_vector(arm['features']).tolist()
                            raw=label_branch(arm['rows'],keep,matched,target,prestate_origins=prestate)
                            row={'sequence':sequence,'episode_uid':event['episode_uid'],'role':event['diagnostic_role'],
                                'state_source':protocol['state_source'],'frame':f,'action':arm['action'],
                                'features':arm['features'],'feature_vector':arm['feature_vector'],
                                'causal_previous_feature_vectors':[history.get(k,[0.]*len(FEATURE_NAMES)) for k in range(f-3,f)],
                                'raw_trajectory_labels':raw,'initial_clicked_UID_verified_target':True,
                                'runtime_seal_SHA':sha256(seal_path),'actual_branch_path':artifact['path'],'actual_branch_SHA':artifact['sha256'],
                                'generating_learned_checkpoint_SHA':protocol['checkpoint_SHA'],'candidate_index_SHA':index_SHA,
                                'offline_TRAIN_future_supervision_only':True,'future_GT_is_not_feature':True,
                                'actual_own_prediction_and_global_assignment_history':True,'GT_history_replacement':False}
                            compressor.stdin.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode());records+=1
                            counts[(row['role'],'records')]+=1
                            h=raw['future']['H100']
                            for k in ('complete','benefit_label','risk_label'):counts[(row['role'],'H100_'+k)]+=bool(h['complete'] and h[k])
                print(json.dumps({'on_policy_supervision_scene_complete':sequence,'action_rows':records}),flush=True)
            compressor.stdin.close()
            if compressor.wait():raise RuntimeError('on-policy supervision compressor failed')
        finally:
            if compressor.poll() is None:compressor.terminate();compressor.wait()
    write_json('on_policy/round1/SUPERVISION_MANIFEST.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_LEARNED_ON_POLICY_SUPERVISION',
        'path':str(destination),'sha256':sha256(destination),'records':records,'actual_full_joint_rollouts':19,
        'counts':{'/'.join(k):v for k,v in counts.items()},'protocol_SHA':sha256(PROTOCOL),
        'generating_checkpoint_SHA':protocol['checkpoint_SHA'],'source_evaluator_SHA':sha256(Path(__file__)),
        'label_component_SHA':sha256(ROOT/'sam3_intermot/evaluation/joint_trajectory_labels.py'),'GT_sources_SHA':GT_SHA,
        'origin_definition_SHA':sha256(OUT/'protocol/TRAJECTORY_ORIGIN_DEFINITION.json'),'future_GT_not_runtime_input':True,
        'no_GT_history_replacement':True,'source_state_not_baseline_or_teacher_forced_substitution':True,
        'not_new_correction_fit_yet':True,'not_closed_loop_success':True,'no_VAL_TEST_confirmation_used':True})
    update_status(phase_C5_round1_labels_complete=True,actual_on_policy_round1_rollouts=19)


if __name__=='__main__':run()
