"""Bounded real learned-policy corpus workers; never overwrite partial logs."""
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status


def main():
    protocol=read_json(OUT/'protocol/ON_POLICY_ROUND1.json')
    pending=sorted({e['sequence'] for e in protocol['events']},key=lambda s:(s!='dancetrack0002',s))
    active=[];finished=[];logs=ASSETS/'on_policy/round1/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PATH']=str(ROOT.parent/'InterMOT/.venv/bin')+':'+env.get('PATH','')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(phase_C5_on_policy_running=True)
    while pending or active:
        while pending and len(active)<2:
            sequence=pending.pop(0);events=[e for e in protocol['events'] if e['sequence']==sequence]
            if all((OUT/'on_policy/round1/runtime'/e['episode_uid']/'seal.json').exists() for e in events):
                for e in events:
                    s=read_json(OUT/'on_policy/round1/runtime'/e['episode_uid']/'seal.json')
                    assert s['protocol_SHA']==sha256(OUT/'protocol/ON_POLICY_ROUND1.json')
                    assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts'])
                finished.append({'sequence':sequence,'returncode':0,'verified_existing':True});continue
            log_path=logs/(sequence+'.log')
            if log_path.exists():raise FileExistsError('retain partial on-policy log')
            storage(128<<20);handle=log_path.open('x')
            command=[python,str(ROOT/'scripts/n72r21r1_collect_on_policy.py'),'--sequence',sequence]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,log_path,sequence,command))
        for entry in list(active):
            process,handle,path,sequence,command=entry;result=process.poll()
            if result is None:continue
            handle.close();active.remove(entry)
            finished.append({'sequence':sequence,'returncode':result,'command':command,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'on_policy_worker_terminal':sequence,'returncode':result,'remaining_sequences':len(pending),'live_workers':len(active)}),flush=True)
        if active:time.sleep(.25)
    write_json('on_policy/round1/SCHEDULER_RESULT.json',{'status':'WORKERS_TERMINAL_VERIFY_EPISODE_SEALS','workers':finished,
        'planned_full_joint_episodes':19,'CPU_workers_max':2,'no_GPU_use':True,'not_new_correction_model_training':True})
    update_status(phase_C5_on_policy_running=False)


if __name__=='__main__':main()
