"""Separate real-stream transfer for the fixed paired controlled L1 fits.

Reuse the original full-global replay/onset/TrackEval and shared selection.
No edits to original policies or source/model thresholds; no confirmation.
"""
import argparse
from contextlib import contextmanager
from scripts import n72r21r2_main_policy as engine
from scripts.n72r21r2_common import ROOT,OUT,ASSETS,GOAL,read_json,write_json,sha256,preregistration,development_sequence,storage
from scripts.n72r21r2_controlled_l1_source_v1 import PROTOCOL as FIT_PROTOCOL,DATA
from sam3_intermot.evaluation.controlled_unit_objective import CONDITIONS
from sam3_intermot.evaluation.learned_policy_evidence import choose_inner_point
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL=OUT/'mot/CONTROLLED_L1_TRANSFER_PROTOCOL_V1.json'
PREFIX='mot/controlled_l1_transfer_v1'
BASE=ASSETS/'controlled_l1_transfer_v1'
OBJECTIVE='L1_EVENT_CORRECTION'
CODE=tuple(dict.fromkeys((*engine.CODE,'scripts/n72r21r2_controlled_l1_mot_v1.py',
    'scripts/n72r21r2_controlled_l1_mot_driver_v1.py','scripts/n72r21r2_controlled_l1_source_v1.py',
    'scripts/n72r21r2_train_controlled_l1_v1.py','sam3_intermot/evaluation/controlled_unit_objective.py')))


def fit_path(condition,seed):
    return OUT/'objectives/controlled_unit_l1_contrast_v1/fits'/(condition+'__seed'+str(seed)+'.json')


def freeze():
    assert not list((OUT/PREFIX).rglob('*.json')), 'Freeze before transfer results or ownership'
    source=read_json(FIT_PROTOCOL);original=read_json(engine.PROTOCOL)
    assert source['seeds']==original['seeds'] and source['conditions']==list(CONDITIONS)
    assert original['source_sha256']=={n:sha256(ROOT/n) for n in engine.CODE}
    refs={}
    for condition in CONDITIONS:
        for seed in source['seeds']:
            path=fit_path(condition,seed);r=read_json(path)
            assert r['condition']==condition and r['seed']==seed and r['nonzero_gradient_steps']>0
            assert r['protocol_sha256']==sha256(FIT_PROTOCOL) and r['source_dataset_sha256']==sha256(DATA)
            assert sha256(r['checkpoint_path'])==r['checkpoint_sha256']
            refs[str(path)]={'fit_sha256':sha256(path),'checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256']}
    storage(2<<30)
    write_json(str(PROTOCOL.relative_to(OUT)),{'stage':'N72R21R2','goal':GOAL,'frozen':True,
        'source_sha256':{n:sha256(ROOT/n) for n in CODE},'inherited_MAIN_protocol_sha256':sha256(engine.PROTOCOL),
        'fit_protocol_sha256':sha256(FIT_PROTOCOL),'source_dataset_sha256':sha256(DATA),'fit_refs':refs,
        'conditions':list(CONDITIONS),'actual_architecture':'SMALL_MLP','objective':OBJECTIVE,'seeds':source['seeds'],
        'split':original['split'],'points':original['points'],'authority_support_policy':original['authority_support_policy'],
        'selection':'Original all3-seed all INNER8 shared point; only selected FIT16, no best seed/ready subset. Both conditions remain experiments regardless of teacher loss.',
        'engineering_smoke':'Fixed first INNER sequence/first original click, P0 first51 frames, all6 frozen heads; no truth labels or source-target payload access. Not a full-MOT cell or scientific qualification.',
        'runtime':'Original candidate/sole anchor/frozen identity P0 no-write/full-global/current hard-negative/NONE/GT-file-guard behavior. All original frames, actual own-prefix pairs and pinned full-video nine metrics.',
        'SOURCE_teacher_NOT_natural_G1_or_on_policy_proof':True,'original_17_groups_not_replaced_by_these_two_transfer_groups':True,
        'CPU_workers':1,'reuse_slot':'Wait for actual original STATE owner/children to exit and its terminal marker with all192 completed cells/no failures. Never terminate another owner or launch a third full-MOT worker.',
        'reserve_GiB':60,'estimated_extra_assets_GiB_planning':2.,
        'nonzero_action_growth':'Before onset expansion check actual action count and conservative pair byte/time estimate against reserve and2GiB planning; unexpected excess stops with retained evidence, no silent skipping of actions.',
        'confirmation_VAL_TEST_SOT_authorized':False,'next_association_stage_authorized':False,'scientific_goal_complete':False})


def inputs(sequence,condition,objective,seed,point):
    development_sequence(sequence);p=read_json(PROTOCOL)
    assert p['goal']==GOAL and p['source_sha256']=={n:sha256(ROOT/n) for n in CODE}
    if condition not in p['conditions'] or objective!=OBJECTIVE or type(seed) is not int or seed not in p['seeds'] or point not in p['points']:
        raise ValueError('Only the six frozen weights/two original points')
    assert sequence in p['split']['fit']+p['split']['inner']
    assert p['fit_protocol_sha256']==sha256(FIT_PROTOCOL)
    assert p['inherited_MAIN_protocol_sha256']==sha256(OUT/'mot/MAIN_POLICY_PROTOCOL_V1.json')
    path=fit_path(condition,seed);r=read_json(path);ref=p['fit_refs'][str(path)]
    assert sha256(path)==ref['fit_sha256'] and sha256(r['checkpoint_path'])==ref['checkpoint_sha256']==r['checkpoint_sha256']
    assert r['protocol_sha256']==p['fit_protocol_sha256'] and r['source_dataset_sha256']==p['source_dataset_sha256']
    assert r['condition']==condition and r['seed']==seed and r['family']=='SMALL_MLP' and r['objective']==OBJECTIVE
    assert r['nonzero_gradient_steps']>0 and sum(r['changed_weight_elements_by_tensor'].values())>0
    assert not r['actual_model_generated_on_policy'] and not r['next_association_stage_authorized']
    if sequence in p['split']['fit']:
        selected=read_json(OUT/PREFIX/'selections'/(condition+'__'+OBJECTIVE+'.json'))
        assert selected['protocol_sha256']==sha256(PROTOCOL) and selected['selected_point']==point
    uid='CONTROLLED_L1__'+condition+'__'+OBJECTIVE+'__seed'+str(seed)+'__'+point
    return p,uid,r,path


@contextmanager
def scoped_engine():
    names=('PROTOCOL','PREFIX','BASE','CODE','inputs','write_json');previous={n:getattr(engine,n) for n in names}
    def transfer_write(relative,value,**kwargs):
        return write_json(relative,{**value,'actual_architecture_family':'SMALL_MLP',
            'controlled_L1_label_transfer_diagnostic':True,'teacher_source_not_actual_model_generated_on_policy':True,
            'natural_independent_error_roots_not_inferred_from_teacher_units':True},**kwargs)
    try:
        engine.PROTOCOL,engine.PREFIX,engine.BASE,engine.CODE=PROTOCOL,PREFIX,BASE,CODE
        engine.inputs,engine.write_json=inputs,transfer_write
        yield
    finally:
        for name,value in previous.items():setattr(engine,name,value)


def execute(action,sequence,condition,seed,point):
    if action not in ('runtime','onsets','evaluate'):raise ValueError('Only full original replay operations')
    with scoped_engine():
        worker={'runtime':engine.runtime,'onsets':engine.onset_runtime,'evaluate':engine.evaluate}[action]
        if action=='evaluate':worker(sequence,condition,OBJECTIVE,seed,point)
        else:
            with runtime_file_guard():worker(sequence,condition,OBJECTIVE,seed,point)


def select(condition):
    p=read_json(PROTOCOL);assert condition in p['conditions'];cells,refs={},[]
    for point in p['points']:
        cells[point]=[]
        for seed in p['seeds']:
            for seq in p['split']['inner']:
                _,uid,_,_=inputs(seq,condition,OBJECTIVE,seed,point)
                path=OUT/PREFIX/'results'/uid/(seq+'.json');r=read_json(path)
                assert r['protocol_sha256']==sha256(PROTOCOL) and r['controlled_L1_label_transfer_diagnostic']
                cells[point].append(r);refs.append({'path':str(path),'sha256':sha256(path)})
    chosen=choose_inner_point(cells,p['seeds'],p['split']['inner'])
    write_json(PREFIX+'/selections/'+condition+'__'+OBJECTIVE+'.json',{'stage':'N72R21R2','goal':GOAL,
        'protocol_sha256':sha256(PROTOCOL),'condition':condition,'family':'SMALL_MLP','objective':OBJECTIVE,
        'all_INNER_result_refs':refs,'confirmation_authorized':False,**chosen})


def smoke():
    """Fixed first INNER sequence/click/P0, all6 heads, first51 frames only.

    No evaluation truth, source-target JSON or new full-MOT worker. Checks
    common real inputs, actual GT guard and zero-action prefix C0 tensor A/A.
    """
    import numpy as np
    import torch
    from scripts.n72r21r2_baseline import checked_frames
    from scripts.n72r21r2_event_context_v2 import sealed_click
    from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
    from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
    p=read_json(PROTOCOL);seq=p['split']['inner'][0];point=next(iter(p['points']))
    storage(32<<20);torch.set_num_threads(1)
    with runtime_file_guard():
        init=read_json(OUT/'data/initialization'/(seq+'.json'));assert sha256(init['anchor_path'])==init['anchor_sha256']
        event=init['inputs'][0]
        if event['initialization_failure']:raise RuntimeError('Fixed first smoke click failed; no replacement/best click')
        anchor=np.array(np.load(init['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
        frames,index_sha=checked_frames(seq);results=[]
        for condition in p['conditions']:
            for seed in p['seeds']:
                _,_,fit,_=inputs(seq,condition,OBJECTIVE,seed,point)
                with scoped_engine():bridge=engine.build_bridge({**p,'current_point':p['points'][point]},fit,event,anchor)
                c0=MOTIdentityBridge(sealed_click(event,anchor),frames=len(frames));c0.configure_fps(event['fps'])
                actions,aa=0,0
                for payload,rows in frames[:51]:
                    f=payload['frame']
                    if actions==0:assert full_tracker_fingerprint(bridge)==full_tracker_fingerprint(c0)
                    result,base=bridge.step(f,rows),c0.step(f,rows)
                    actions+=bool(result['authority'].get('effective_assignment_change'))
                    if actions==0:
                        assert result['outputs']==base['outputs'] and full_tracker_fingerprint(bridge)==full_tracker_fingerprint(c0)
                        aa+=1
                results.append({'condition':condition,'seed':seed,'actual_frames':min(51,len(frames)),
                                'effective_actions':actions,'zero_action_prefix_tensor_C0_AA_frames':aa})
    write_json(PREFIX+'/FIXED_FIRST51_GT_FREE_SMOKE.json',{'stage':'N72R21R2','goal':GOAL,'protocol_sha256':sha256(PROTOCOL),
        'sequence':seq,'episode_uid':event['episode_uid'],'candidate_index_sha256':index_sha,'point':point,'all6_head_results':results,
        'GT_file_guard_actual':True,'no_source_target_payload_in_runtime':True,'not_full_MOT_or_G1_or_G2':True,'scientific_goal_complete':False})
    print({'actual_fixed_six_head_real_stream_smoke':results,'full_MOT_success_claimed':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','smoke','runtime','onsets','evaluate','select'))
    parser.add_argument('--sequence');parser.add_argument('--condition',choices=CONDITIONS);parser.add_argument('--seed',type=int);parser.add_argument('--point')
    a=parser.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='smoke':smoke()
    elif a.action=='select':select(a.condition)
    else:execute(a.action,a.sequence,a.condition,a.seed,a.point)
