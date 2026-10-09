"""Missing historical/negative controls in genuine complete online joint MOT."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,HISTORY,read_json,write_json,sha256,storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r21r1_collect_joint import source_policy
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r4r1_memory_fit import ReliabilityEnsemble
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.frozen_comparator_controls import FrozenAdapterActor,ShuffledEvidenceMOTBridge,shuffled_evidence
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.association.identity_authority import AdapterEnsemble,AuthorityConfig
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank

PROTOCOL=OUT/'protocol/FROZEN_COMPARATOR_JOINT_PILOT_V1.json'
CASES=('R3R2_ADAPTER','R4R1_NATIVE','SHUFFLED_IDENTITY_ORIGINAL','SHUFFLED_IDENTITY_CONFIDENCE','SHUFFLED_IDENTITY_SHADOW')
ENCODER_SHA='2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154'
GRU_SHA='94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2'


def frozen_sources(sequence):
    order=read_json(HISTORY/'outputs/N72R21/protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences']
    inner=order[(order.index(sequence)+1)%len(order)];fit=sorted(set(order)-{sequence,inner})
    paths=[ROOT.parent/'InterMOT_N72R20R4_assets/models'/f'adapter6__{sequence}__seed{s}.pt' for s in (720321,720322,720323)]
    adapters=[]
    for path in paths:
        record=read_json(path.with_suffix('.json'));assert sha256(path)==record['sha256']
        assert sorted(record['actual_training_sequences'])==fit
        adapters.append({'path':str(path),'sha256':record['sha256'],'sidecar_SHA':sha256(path.with_suffix('.json'))})
    frozen_path=HISTORY/'outputs/N72R20R4R1/authority/frozen_outer'/f'{sequence}.json';frozen=read_json(frozen_path)
    fit_path=HISTORY/'outputs/N72R20R4R1/memory/fit'/f'{sequence}.json'
    native=read_json(fit_path)['models']['NATIVE_RELIABILITY']
    assert all(sha256(r['path'])==r['sha256'] for r in native)
    return {'outer':sequence,'inner':inner,'FIT_six':fit,'adapters':adapters,
        'calibration':frozen['calibration'],'native_spec':frozen['selected']['NATIVE_IDENTITY'],
        'native_models':native,'native_frozen_policy_SHA':sha256(frozen_path),'native_fit_record_SHA':sha256(fit_path)}


def freeze():
    inputs=read_json(OUT/'corpus/RUNTIME_INPUTS.json')
    sources={s:frozen_sources(s) for s in ['dancetrack0001','dancetrack0002']}
    files=['scripts/n72r21r1_frozen_control_pilot.py','sam3_intermot/one_click/frozen_comparator_controls.py',
        'sam3_intermot/one_click/safe_mot_bridge.py','sam3_intermot/one_click/intervention_features.py',
        'sam3_intermot/one_click/joint_intervention_primitives.py','sam3_intermot/association/opportunity_tracker.py',
        'sam3_intermot/association/identity_authority.py','scripts/n72r20r4r1_memory_fit.py']
    write_json('protocol/FROZEN_COMPARATOR_JOINT_PILOT_V1.json',{'stage':'N72R21R1','final_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_control_effects':True,'sequences':list(sources),'cases':[{'case':'CLICK_C0','kind':'REUSE_PREVIOUS_ACTUAL_SEALED_C0'}]+[{'case':c,'kind':c} for c in CASES],
        'sources':sources,'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),'identity_model_source':inputs['frozen_identity_model'],
        'source_code_SHA':{p:sha256(ROOT/p) for p in files},'planned_new_actual_full_joint_runs':10,
        'first_prospective_corpus_event_per_scene_no_GT_outcome_selection':True,
        'shuffle_rule':'Lexically next real current-click-frame UID, no GT identity/best score; disrupt model anchor and all human-anchor gate features, retain original C0 initialization and machine scorer.',
        'shuffle_shadow_requires_C0_full_trajectory_AA':True,
        'native_caveat':'Actual historical frozen native-reliability scorer control, P1-training/P0-deployment distribution shift; NOT a new unchanged-C0-scoring association authority. Label this difference, do not claim it satisfies new pure-controller invariants.',
        'adapter_probability_uncalibrated_not_NONE_posterior':True,
        'max_seconds_per_case':1800,'max_CPU_workers':2,'total_with_teacher_workers_max':4,
        'no_VAL_TEST_confirmation':True,'new_training':False,'next_stage_authorized':False})


def adapter_from(source):
    adapter=AdapterEnsemble([Path(r['path']) for r in source['adapters']],
        expected_shas=[r['sha256'] for r in source['adapters']],forbidden_sequences=[source['outer'],source['inner']])
    assert all(sorted(r['actual_training_sequences'])==source['FIT_six'] for r in adapter.manifest)
    return adapter


def run(sequence):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL);assert sequence in protocol['sequences']
    assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    assert sha256(OUT/'corpus/RUNTIME_INPUTS.json')==protocol['input_SHA'];inputs=read_json(OUT/'corpus/RUNTIME_INPUTS.json')
    event=sorted([e for e in inputs['inputs'] if e['sequence']==sequence],key=lambda e:(e['frame'],e['episode_uid']))[0]
    frames,index_SHA=checked_frames(sequence);assert index_SHA==event['candidate_index_sha256']
    assert sha256(inputs['anchor_path'])==inputs['anchor_sha256'];anchor=np.array(np.load(inputs['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
    click={'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']}
    baseline_seal_path=OUT/'learned_pilot/runtime_seals/CLICK_C0'/(sequence+'.json');baseline=read_json(baseline_seal_path)
    assert baseline['event']==event and all(sha256(a['path'])==a['sha256'] for a in baseline['artifacts'])
    from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
    baseline_trace=read_zstd_jsonl(Path(next(a['path'] for a in baseline['artifacts'] if a['kind']=='trace')))
    source=protocol['sources'][sequence];assert source==frozen_sources(sequence)
    for case in CASES:
        prefix='frozen_controls/runtime_seals/'+case+'/'+sequence+'.json'
        if (OUT/prefix).exists():
            seal=read_json(OUT/prefix);assert seal['protocol_SHA']==sha256(PROTOCOL)
            assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);continue
        storage(32<<20);model=None;actor=None;bridge=None;tracker=None;donor=None;native=None;adapter=None
        if case=='R4R1_NATIVE':
            adapter=adapter_from(source);native=ReliabilityEnsemble(source['native_models'],sequence,'NATIVE_RELIABILITY')
            bank=LearnedIdentityMemoryBank.from_checkpoint(HISTORY/'outputs/N72R18/checkpoints/identity_memory_gru.pt',
                encoder_sha256=ENCODER_SHA,expected_encoder_sha256=ENCODER_SHA,expected_checkpoint_sha256=GRU_SHA)
            tracker=OpportunityTracker(config=AuthorityConfig(**source['calibration']),event=click,adapter=adapter,bank=bank,
                memory_policy=MemoryPolicy(**source['native_spec']['memory']),intervention_policy=InterventionPolicy(**source['native_spec']['policy']),native_predictor=native)
            assert tracker.memory_policy.family=='P0' and tracker.intervention_policy.family=='C0'
        elif case=='R3R2_ADAPTER':
            adapter=adapter_from(source);actor=FrozenAdapterActor(adapter,anchor,event,
                score_min=source['calibration']['write_score'],margin_min=source['calibration']['write_margin'])
            bridge=SafeMOTIdentityBridge(click,actor,policy=GatePolicy(family='original'),frames=len(frames))
        else:
            current_anchor,donor=shuffled_evidence(frames[event['frame']][1],event['clicked_candidate_uid'])
            identity=inputs['frozen_identity_model'];assert sha256(identity['path'])==identity['sha256']
            saved=torch.load(identity['path'],map_location='cpu',weights_only=True);model=DecisionCapture(ACIBMemoryNetwork()).eval()
            model.model.load_state_dict(saved['model'],strict=True)
            actor=TrustedACIBRecognizer(model,current_anchor,event['episode_uid'],policy='P0',capacity=8)
            actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
            gate=GatePolicy(family='original') if case.endswith('ORIGINAL') else GatePolicy(family='shadow') if case.endswith('SHADOW') else source_policy('JOINT_FIXED_BROAD_ON_POLICY_ROUND0')[1]
            bridge=ShuffledEvidenceMOTBridge(click,actor,policy=gate,frames=len(frames))
        if bridge is not None:bridge.configure_fps(event['fps'])
        trace_path=ASSETS/'frozen_controls/traces'/case/(sequence+'.jsonl.zst');trajectory_path=ASSETS/'frozen_controls/trackers'/case/'data'/(sequence+'.txt')
        for path in (trace_path,trajectory_path):
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('preserve partial frozen comparator')
        began=time.monotonic();step_seconds=0.;effective=0
        with trace_path.open('xb') as stream,trajectory_path.open('x') as trajectory:
            compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
            try:
                for payload,rows in frames:
                    if time.monotonic()-began>protocol['max_seconds_per_case']:raise RuntimeError('frozen comparator case cap')
                    f=int(payload['frame']);started=time.perf_counter()
                    if tracker is None:actual=bridge.step(f,rows)
                    else:
                        native_result=tracker.step(rows,f);actual={k:v for k,v in native_result.items() if k not in ('base_matrix','fused_matrix','solver','states_before_commit_axis','proposals')}
                        scores=native_result['identity_scores'];top=int(np.argmax(scores)) if len(scores) and f>event['frame'] else None
                        p=0. if top is None else float(np.clip((scores[top]+1)/2,0,1))
                        uid=native_result['target_uid'];row=next((r for r in rows if str(r['candidate_uid'])==uid),None)
                        actual.update(authority={'family':'HISTORICAL_NATIVE_RELIABILITY_SCORER','approved':False,'effective_assignment_change':False},
                            identity_decision={'selected_candidate_uid':uid,'proposed_candidate_uid':uid,'rank1_candidate_uid':str(rows[top]['candidate_uid']) if top is not None else None,
                                'candidate_available_probability':p,'predicted_box_xyxy':list(row['box_xyxy']) if row else None,
                                'memory_write':False,'memory_write_candidate_uid':None,'presence_probability_semantics':'UNCALIBRATED_HISTORICAL_IDENTITY_SCORE'},
                            joint_identity_memory_write=False,joint_memory_write_candidate_uid=None)
                        assert not native_result['memory']['accepted']
                    step_seconds+=time.perf_counter()-started
                    if f<=event['frame']:assert actual['outputs']==baseline_trace[f]['outputs'] and actual['state_after']==baseline_trace[f]['state_after']
                    if case.endswith('SHADOW'):assert actual['outputs']==baseline_trace[f]['outputs'] and actual['state_after']==baseline_trace[f]['state_after']
                    assert {o['candidate_uid'] for o in actual['outputs']}=={str(r['candidate_uid']) for r in rows}
                    assert len({o['public_id'] for o in actual['outputs']})==len(rows)
                    effective+=bool(actual['authority'].get('effective_assignment_change'))
                    assert not actual['joint_identity_memory_write']
                    decision=actual['identity_decision'];cache=float(model.last['joint_probabilities'][0,:len(rows)].max()) if model is not None and f>event['frame'] and rows else float(decision['candidate_available_probability']) if decision else 0.
                    actual.update(case=case,episode_uid=event['episode_uid'],cached_rank1_joint_probability=cache,
                        runtime_GT_input=False,runtime_future_GT_input=False,negative_control_donor_UID=donor,extra_clicks=0)
                    compressor.stdin.write((json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode());trajectory.write(trajectory_text([actual]))
                compressor.stdin.close()
                if compressor.wait():raise RuntimeError('frozen comparator compression failure')
            finally:
                if compressor.poll() is None:compressor.terminate();compressor.wait()
        if case.endswith('SHADOW'):assert sha256(trajectory_path)==next(a['sha256'] for a in baseline['artifacts'] if a['kind']=='trajectory')
        write_json(prefix,{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_FULL_JOINT_FROZEN_COMPARATOR','case':case,'sequence':sequence,
            'event':event,'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],'inputs_SHA':protocol['input_SHA'],
            'candidate_index_SHA':index_SHA,'baseline_source_seal_SHA':sha256(baseline_seal_path),'frames':len(frames),'effective_interventions':effective,
            'artifacts':[{'kind':kind,'path':str(path),'sha256':sha256(path)} for kind,path in [('trace',trace_path),('trajectory',trajectory_path)]],
            'historical_model_sources':source if not case.startswith('SHUFFLED') else inputs['frozen_identity_model'],
            'negative_control_donor_UID':donor,'native_historical_scorer_differs_from_C0':case=='R4R1_NATIVE',
            'historical_native_training_state':'P1' if case=='R4R1_NATIVE' else None,'native_deployment_state':'P0' if case=='R4R1_NATIVE' else None,
            'no_new_training':True,'runtime_GT_input':False,'runtime_future_GT_input':False,'history_rewritten':False,'extra_clicks':0,
            'full_global_exact_unique_candidate_ownership':True,'cached_joint_step_seconds':step_seconds,'cached_total_seconds':time.monotonic()-began,
            'cached_joint_FPS_not_pixel_FPS':len(frames)/step_seconds,'association_only_seconds':None,
            'controller_parameter_count':sum(m.parameter_count for m in native.models) if native is not None else 0,
            'identity_parameter_count':sum(p.numel() for m in adapter.models for p in m.parameters()) if adapter is not None else sum(p.numel() for p in model.parameters()),
            'cumulative_process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'scientific_success':None,'next_stage_authorized':False})
        print(json.dumps({'frozen_comparator_complete':case,'sequence':sequence,'effective':effective,'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--sequence');a=p.parse_args()
    if a.freeze:freeze()
    else:run(a.sequence)
