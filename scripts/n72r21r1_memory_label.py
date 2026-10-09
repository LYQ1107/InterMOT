"""Post-seal joint committed-crop labels; future truth never enters features."""
from collections import Counter
from pathlib import Path
import json
import subprocess
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,TRAIN,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_memory_collect import PROTOCOL
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES,feature_vector
from sam3_intermot.evaluation.joint_trajectory_labels import label_branch
from sam3_intermot.evaluation.safe_intervention_events import identity_outcome


def verify_all():
    protocol=read_json(PROTOCOL);sources=[]
    assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    for event in protocol['events']:
        path=OUT/'memory/curriculum/runtime'/event['episode_uid']/'seal.json';s=read_json(path)
        assert s['status']=='COMPLETE_ACTUAL_COMMITTED_MEMORY_CURRICULUM' and s['protocol_SHA']==sha256(PROTOCOL)
        assert s['source_code_SHA']==protocol['source_code_SHA'] and s['inputs_SHA']==protocol['input_SHA']
        assert s['C0_full_trajectory_AA'] and not s['runtime_GT_input'] and not s['runtime_future_GT_input']
        assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts'])
        for ref in s['paired_write_seals']:
            if ref['status']!='COMPLETE_PAIR':assert ref['status']=='SKIPPED_COMMITTED_NONE_NO_REPLACEMENT';continue
            assert sha256(ref['path'])==ref['sha256'];pair=read_json(ref['path'])
            assert pair['protocol_SHA']==sha256(PROTOCOL) and pair['same_current_global_assignment'] and pair['no_runtime_GT']
            assert all(sha256(a['path'])==a['sha256'] for a in pair['artifacts'])
        sources.append((path,s))
    assert len(sources)==22
    return protocol,sources


def run():
    storage(96<<20);protocol,sources=verify_all()
    identities={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']}
    destination=ASSETS/'memory/curriculum/supervision.jsonl.zst'
    if destination.exists():raise FileExistsError('preserve existing committed-memory supervision')
    counts=Counter();paired=[];GT_SHA={};records=0
    with destination.open('xb') as stream:
        compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for sequence in sorted({s['event']['sequence'] for _,s in sources}):
                frames,index_SHA=checked_frames(sequence);gt=dancetrack_annotations(TRAIN/sequence)
                matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
                GT_SHA[sequence]=sha256(TRAIN/sequence/'gt/gt.txt')
                for path,source in [(p,s) for p,s in sources if s['event']['sequence']==sequence]:
                    event=source['event'];target=identities[event['episode_uid']];assert matched[event['frame']].get(event['clicked_candidate_uid'])==target
                    trace=read_zstd_jsonl(Path(next(a['path'] for a in source['artifacts'] if a['kind']=='trace')))
                    assert [r['frame'] for r in trace]==list(range(len(frames)))
                    history={r['frame']:feature_vector(r['committed_observation_features']).tolist() for r in trace if 'committed_observation_features' in r}
                    by_frame={};origins={};cursor=0
                    for ref in source['paired_write_seals']:
                        if ref['status']!='COMPLETE_PAIR':counts['skipped_NONE_pairs']+=1;continue
                        pair=read_json(ref['path']);f=pair['frame']
                        while cursor<=f:
                            for o in trace[cursor]['outputs']:
                                identity=matched[cursor].get(o['candidate_uid'])
                                if identity is not None:origins.setdefault(o['public_id'],identity)
                            cursor+=1
                        arms={a['name']:read_json(a['path']) for a in pair['artifacts']}
                        write=arms['WRITE_CURRENT_COMMITTED']['rows'];deny=arms['DENY_WRITE']['rows']
                        assert write[0]['memory_write_uid']==trace[f]['target_uid'] and not deny[0]['memory_write']
                        current_origins={p:i for p,i in origins.items() if p in {o['public_id'] for o in trace[f]['outputs']}}
                        labels=label_branch(write,deny,matched,target,prestate_origins=current_origins)
                        detail={'sequence':sequence,'episode_uid':event['episode_uid'],'role':event['diagnostic_role'],'frame':f,
                            'pair_seal_SHA':ref['sha256'],'raw_trajectory_labels':labels,'current_write_outcome':identity_outcome(trace[f]['target_uid'],matched[f],target),
                            'future_rank1_correct_WRITE':sum(matched[r['frame']].get(r.get('rank1_candidate_uid'))==target for r in write[1:]),
                            'future_rank1_correct_DENY':sum(matched[r['frame']].get(r.get('rank1_candidate_uid'))==target for r in deny[1:]),
                            'future_frames':len(write)-1,'diagnostic_force_write_not_deployable_policy':True}
                        paired.append(detail);by_frame[f]=detail
                        h=labels['future']['H100'];counts[event['diagnostic_role']+'/complete_H100_pairs']+=h['complete']
                        if h['complete']:
                            counts[event['diagnostic_role']+'/H100_pair_benefit']+=h['benefit_label'];counts[event['diagnostic_role']+'/H100_pair_harm']+=h['risk_label']
                    for actual in trace:
                        f=actual['frame']
                        if f<=event['frame']:continue
                        uid=actual['target_uid'];features=actual['committed_observation_features'];outcome=identity_outcome(uid,matched[f],target)
                        row={'sequence':sequence,'episode_uid':event['episode_uid'],'role':event['diagnostic_role'],'frame':f,
                            'actual_committed_uid':uid,'eligible_current_crop':uid is not None,'current_outcome':outcome,
                            'target_candidate_available':target in matched[f].values(),'features':features,'feature_vector':feature_vector(features).tolist(),
                            'causal_previous_committed_feature_vectors':[history.get(k,[0.]*len(FEATURE_NAMES)) for k in range(f-3,f)],
                            'paired_future':by_frame.get(f),'initial_clicked_UID_verified_target':True,'runtime_seal_SHA':sha256(path),
                            'candidate_index_SHA':index_SHA,'current_or_future_GT_not_feature':True,'GT_history_replacement':False}
                        compressor.stdin.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode());records+=1
                        counts[event['diagnostic_role']+'/records']+=1;counts[event['diagnostic_role']+'/'+outcome]+=1
                print(json.dumps({'committed_memory_labels_scene_complete':sequence,'records':records}),flush=True)
            compressor.stdin.close()
            if compressor.wait():raise RuntimeError('committed memory label compression failed')
        finally:
            if compressor.poll() is None:compressor.terminate();compressor.wait()
    pair_path=write_json('memory/curriculum/PAIRED_FUTURE_LABELS.json',{'pairs':paired,'not_runtime_GT_or_deployed_writer':True})
    write_json('memory/curriculum/SUPERVISION_MANIFEST.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_JOINT_COMMITTED_MEMORY_SUPERVISION',
        'path':str(destination),'sha256':sha256(destination),'records':records,'counts':dict(counts),'actual_full_joint_rollouts':22,
        'actual_paired_write_positions':len(paired),'actual_pair_arms':len(paired)*2,'paired_future_labels_path':str(pair_path),
        'paired_future_labels_SHA':sha256(pair_path),'GT_sources_SHA':GT_SHA,'protocol_SHA':sha256(PROTOCOL),
        'label_component_SHA':sha256(ROOT/'sam3_intermot/evaluation/joint_trajectory_labels.py'),
        'origin_definition_SHA':sha256(OUT/'protocol/TRAJECTORY_ORIGIN_DEFINITION.json'),'source_evaluator_SHA':sha256(Path(__file__)),
        'no_VAL_TEST_confirmation_used':True,'future_GT_not_runtime_input':True,'not_actual_write_head_training_yet':True})
    update_status(phase_F_committed_curriculum_labels_complete=True,actual_committed_memory_curriculum_rollouts=22)


if __name__=='__main__':run()
