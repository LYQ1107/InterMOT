"""Finite192 transfer cells, reusing the original STATE CPU slot only."""
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT,OUT,ASSETS,PYTHON,GOAL,read_json,write_json,sha256,storage,utcnow
from scripts.n72r21r2_controlled_l1_mot_v1 import PROTOCOL,PREFIX,OBJECTIVE,CODE
from scripts import n72r21r2_state_policy as original_state

TERMINAL='AVAILABLE_STATE_SOURCE_FULL_MOT_FINISHED_SEPARATE_ON_POLICY_AND_GATE_CLOSURE_REQUIRED'


def slot_ready(marker,live_state_process):
    if live_state_process or marker['status']!=TERMINAL or marker.get('active') is not None:
        return False
    if marker['failed_retained'] or len(marker['completed'])!=192 or len(set(marker['completed']))!=192:
        raise RuntimeError('Original STATE terminal failure/incomplete scope requires diagnosis, not bypass')
    if marker['group_status'].get('BASELINE_STATE')!='NOT_RUN_ACTUAL_SUPERVISION_FAILURE_NOT_A_WEIGHT':
        raise RuntimeError('Original baseline measured no-fit disposition must remain explicit')
    return True


def state_processes_live():
    modules=('scripts.n72r21r2_idle_driver_recovery_v2','scripts.n72r21r2_state_policy_driver','scripts.n72r21r2_state_policy')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().decode().split('\0');cwd=(proc/'cwd').resolve()
        except (OSError,UnicodeError):continue
        if cwd==ROOT and any(module in args for module in modules):return True
    return False


def verify_original_state_complete(marker):
    p=read_json(original_state.PROTOCOL);digest=sha256(original_state.PROTOCOL);expected=set()
    assert p['source_sha256']=={n:sha256(ROOT/n) for n in original_state.CODE}
    for mode in ('TREATMENT_STATE','MIXED_STATE'):
        selection=read_json(OUT/original_state.PREFIX/'selections'/(mode+'__'+p['objective']+'.json'))
        assert selection['protocol_sha256']==digest
        pairs=[(point,s) for point in p['points'] for s in p['split']['inner']]
        pairs += [(selection['selected_point'],s) for s in p['split']['fit']]
        for seed in p['seeds']:
            for point,seq in pairs:
                uid='STATE__'+mode+'__'+p['objective']+'__seed'+str(seed)+'__'+point
                expected.add(uid+'/'+seq)
                rpath=OUT/original_state.PREFIX/'runtime'/uid/(seq+'.json')
                opath=OUT/original_state.PREFIX/'onset_runtime'/uid/(seq+'.json')
                result=read_json(OUT/original_state.PREFIX/'results'/uid/(seq+'.json'))
                assert result['protocol_sha256']==digest and result['runtime_seal_sha256']==sha256(rpath) and result['onset_runtime_seal_sha256']==sha256(opath)
    assert expected==set(marker['completed'])


def run_cell(state,condition,seed,point,seq):
    uid='CONTROLLED_L1__'+condition+'__'+OBJECTIVE+'__seed'+str(seed)+'__'+point;key=uid+'/'+seq
    if key in state['completed']:return
    storage(32<<20)
    log=ASSETS/'controlled_l1_transfer_v1/logs'/uid/(seq+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as handle:
        for action,folder in [('runtime','runtime'),('onsets','onset_runtime'),('evaluate','results')]:
            if (OUT/PREFIX/folder/uid/(seq+'.json')).exists():raise FileExistsError('No unowned output overwrite or implicit recovery')
            if action=='onsets':
                seal=read_json(OUT/PREFIX/'runtime'/uid/(seq+'.json'))
                count=sum(e['effective_direct_decisions_NOT_independent_onsets'] for e in seal['episodes'])
                estimate=count*2*101*250000
                if estimate>2<<30:raise RuntimeError('Measured nonzero-action count requires pair-size/time re-audit before expansion; all runtime evidence retained')
                storage(max(32<<20,estimate))
            command=[str(PYTHON),'-u','-m','scripts.n72r21r2_controlled_l1_mot_v1',action,'--condition',condition,'--seed',str(seed),'--point',point,'--sequence',seq]
            process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=handle,stderr=subprocess.STDOUT)
            state['active']={'key':key,'action':action,'pid':process.pid,'log':str(log),'command':command}
            write_json(PREFIX+'/driver.json',state,mutable=True)
            code=process.wait()
            if code:
                state['failed_retained'].append({'key':key,'action':action,'actual_exit':code,'log':str(log)})
                raise RuntimeError('Actual transfer child failed; keep all partial outputs and do not auto-refit/restart')
    state['active']=None;state['completed'].append(key)
    write_json(PREFIX+'/driver.json',state,mutable=True)
    print({'actual_paired_transfer_complete_cells':len(state['completed']),'key':key},flush=True)


def run():
    marker=OUT/PREFIX/'driver.json'
    if marker.exists():raise FileExistsError('Inspect existing exact owner before any new launch')
    p=read_json(PROTOCOL);assert p['source_sha256']=={n:sha256(ROOT/n) for n in CODE}
    state={'stage':'N72R21R2','goal':GOAL,'pid':os.getpid(),'protocol_sha256':sha256(PROTOCOL),
        'status':'WAITING_ORIGINAL_STATE_SLOT_NO_NEW_FULL_MOT_WORKER','CPU_workers':1,'active':None,
        'completed':[],'failed_retained':[],'confirmation_authorized':False,'observed_original_state':None}
    write_json(PREFIX+'/driver.json',state)
    try:
        while True:
            old=read_json(OUT/original_state.PREFIX/'driver.json');live=state_processes_live()
            state['observed_original_state']={'utc':utcnow(),'status':old['status'],'owner_pid':old['pid'],'actual_processes_live':live,'complete_cells':len(old['completed'])}
            write_json(PREFIX+'/driver.json',state,mutable=True)
            if slot_ready(old,live):
                verify_original_state_complete(old);break
            time.sleep(10)
        storage(2<<30);state['status']='ACTIVE_FIXED_TWO_CONDITION_PAIRED_FULL_GLOBAL_TRANSFER';write_json(PREFIX+'/driver.json',state,mutable=True)
        for condition in p['conditions']:
            for point in p['points']:
                for seed in p['seeds']:
                    for seq in p['split']['inner']:run_cell(state,condition,seed,point,seq)
            log=ASSETS/'controlled_l1_transfer_v1/logs'/(condition+'__selection.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as handle:
                process=subprocess.Popen([str(PYTHON),'-u','-m','scripts.n72r21r2_controlled_l1_mot_v1','select','--condition',condition],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},stdout=handle,stderr=subprocess.STDOUT)
                state['active']={'action':'select','condition':condition,'pid':process.pid,'log':str(log)};write_json(PREFIX+'/driver.json',state,mutable=True)
                code=process.wait()
                if code:raise RuntimeError('Actual shared selection failed; no successful subset or best-seed fallback')
            state['active']=None
            selected=read_json(OUT/PREFIX/'selections'/(condition+'__'+OBJECTIVE+'.json'))['selected_point']
            for seed in p['seeds']:
                for seq in p['split']['fit']:run_cell(state,condition,seed,selected,seq)
        assert len(state['completed'])==192
        state['status']='COMPLETE_FIXED192_TRANSFER_CELLS_GATES_AND_WHOLE_GOAL_SEPARATE';write_json(PREFIX+'/driver.json',state,mutable=True)
    except Exception as error:
        state['status']='STOPPED_RETAINED_FAILURE_OR_RESOURCE_NEEDS_DIAGNOSIS';state['error']=repr(error)
        write_json(PREFIX+'/driver.json',state,mutable=True);raise


if __name__=='__main__':run()
