"""All80 full memory rollouts sealed before truth, safety and pinned TrackEval."""
from pathlib import Path
import json
import subprocess
import time
import numpy as np
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,TRAIN,HISTORY,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_memory_pilot import PROTOCOL
from scripts.n72r21r1_learned_evaluate import offline,aggregate,artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary


def verify_all():
    protocol=read_json(PROTOCOL);seals={}
    assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    for c in protocol['cases']:
        for sequence in protocol['sequences']:
            s=read_json(OUT/'memory/pilot/runtime_seals'/c['case']/(sequence+'.json'))
            assert s['status']=='COMPLETE_ACTUAL_COMMITTED_MEMORY_FULL_JOINT_RUNTIME'
            assert s['protocol_SHA']==sha256(PROTOCOL) and s['source_code_SHA']==protocol['source_code_SHA'] and s['inputs_SHA']==protocol['input_SHA']
            assert not s['runtime_GT_input'] and not s['runtime_future_GT_input'] and not s['history_rewritten']
            assert s['full_global_exact_unique_candidate_ownership'] and s['extra_clicks']==0
            assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts']);seals[c['case'],sequence]=s
    assert len(seals)==80
    return protocol,seals


def safety_and_drift(protocol,seals,stats):
    for c in protocol['cases']:
        for sequence in protocol['sequences']:
            key=c['case']+'/'+sequence;s=stats[key];counts=s['counts'];safety=s['memory']
            eligible=counts['candidate_available'];good=safety['correct'];accepted=safety['accepted'];risk=safety['wrong_or_unknown_rate']
            safety.update(eligible_available_target_candidates=eligible,
                correct_available_target_retention=good/eligible if eligible else None,
                correct_committed_retention_denominator=counts.get('TARGET',0),
                wrong_or_UNKNOWN_observed_label_upper_bound=risk,
                UNKNOWN_all_counted_as_wrong_for_safety=True,statistical_population_95percent_upper_bound=None,
                correlated_frames_and_only_two_sequence_clusters=True,
                full_safety_and_usefulness_pass=bool(accepted and risk<=.02 and eligible and good/eligible>=.6))
            trace=read_zstd_jsonl(artifact(seals[c['case'],sequence],'trace'))
            similarities=[r['trusted_bank_mean_anchor_cosine_after'] for r in trace if r['trusted_bank_mean_anchor_cosine_after'] is not None]
            s['memory_drift']={'maximum_bank_size':seals[c['case'],sequence]['maximum_bank_size'],
                'mean_trusted_entry_anchor_cosine':float(np.mean(similarities)) if similarities else None,
                'minimum_mean_trusted_entry_anchor_cosine':float(min(similarities)) if similarities else None,
                'rollback_count':trace[-1]['rollback_count_after'],
                'anchor_immutable_but_bank_angle_not_GT_identity_proof':True,'attention_is_frozen_existing_model':True}
            s['writer_control']={'write_policy':c['write_policy'],'mode':c['mode'],
                'unsafe_not_deployable':c['write_policy']['family']=='unsafe','zero_write_not_scientific_PASS':True}


def trackeval(protocol):
    pinned=HISTORY/'third_party/MOTIP/TrackEval';commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()
    assert commit=='12c8791b303e0a0b50f753af204249e622d0281a'
    root=ASSETS/'memory/pilot/trackeval_v1'
    if root.exists():raise FileExistsError('preserve previous memory TrackEval attempt')
    storage(64<<20);root.mkdir(parents=True);seqmap=root/'seqmap.txt'
    with seqmap.open('x') as f:f.write('name\n'+'\n'.join(protocol['sequences'])+'\n')
    cases=[c['case'] for c in protocol['cases']]
    command=_trackeval_command(ASSETS/'memory/pilot/trackers',root,cases,seqmap,gt_split='train',gt_folder=TRAIN)
    command[2]=str(pinned/'scripts/run_mot_challenge.py');log=root/'trackeval.log';began=time.monotonic()
    with log.open('x') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    write_json('memory/pilot/TRACKEVAL_INVOCATION.json',{'command':command,'returncode':result.returncode,'seconds':time.monotonic()-began,
        'log_path':str(log),'log_SHA':sha256(log),'pinned_TrackEval_commit':commit,'same_settings_all40_cases':True,'third_party_modified':False})
    if result.returncode:raise RuntimeError('full memory TrackEval failed, original logs preserved')
    metrics={c:trackeval_summary(parse_trackeval(root,c,protocol['sequences'])) for c in cases}
    assert all(all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')) for m in metrics.values())
    return metrics


def run():
    protocol,seals=verify_all();stats,detection=offline(protocol,seals);safety_and_drift(protocol,seals,stats)
    metrics=trackeval(protocol);groups=aggregate(protocol,stats,metrics)
    coupled=next(c['case'] for c in protocol['cases'] if c['write_policy']['family']=='frozen' and c['mode']=='COUPLED')
    for c in protocol['cases']:
        key='__'.join((c['family'],c['state_contrast'],c['reward'])) if 'family' in c else c['case']
        group=groups[key]
        group['all_cases_nonvacuous_memory_gate']=all(stats[case+'/'+sequence]['memory']['full_safety_and_usefulness_pass']
            for case in group['cases_all_seeds'] for sequence in protocol['sequences'])
        group['macro_delta_vs_same_authority_frozen_bank']={k:float(np.mean([
            metrics[case]['per_sequence'][sequence][k]-metrics[coupled]['per_sequence'][sequence][k]
            for case in group['cases_all_seeds'] for sequence in protocol['sequences']])) for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')} if c['mode']=='COUPLED' else None
        group['not_statistical_independent_cue_confirmation']=True
    write_json('memory/pilot/RESULT.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_COMMITTED_MEMORY_FULL_MOT_EVALUATION',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','actual_sealed_runs':80,'actual_cases':40,'independent_sequences':2,
        'protocol_SHA':sha256(PROTOCOL),'statistics':stats,'TrackEval':metrics,'all_seed_groups':groups,'detection_multiset_audit':detection,
        'frozen_bank_coupled_control':coupled,'all_controls_capacities_aggregations_and_seeds_retained':True,
        'source_evaluator_SHA':sha256(Path(__file__)),'shared_evaluator_SHA':sha256(ROOT/'scripts/n72r21r1_learned_evaluate.py'),
        'historical_development_not_fresh_confirmation':True,'next_stage_authorized':False,'whole_goal_scientific_decision':'PENDING'})
    update_status(phase_F_full_memory_MOT_evaluation_complete=True,actual_full_memory_MOT_runs=80)
    print(json.dumps({'actual_memory_TrackEval_complete':80,'groups':len(groups),'direction_pass':sum(g['direction_pass'] for g in groups.values()),
        'nonvacuous_memory_groups':sum(g['all_cases_nonvacuous_memory_gate'] for g in groups.values()),'Goal_still_active':True}),flush=True)


if __name__=='__main__':run()
