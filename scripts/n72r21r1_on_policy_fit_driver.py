"""All registered correction fits, at most two single-thread CPU workers."""
import itertools
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status


def main():
    protocol=read_json(OUT/'protocol/ON_POLICY_CORRECTION_HEADS_V1.json')
    pending=list(itertools.product(protocol['families'],protocol['data_contrasts'],protocol['seeds']))
    active=[];finished=[];logs=ASSETS/'training/on_policy_correction/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(phase_C5_correction_training_running=True)
    while pending or active:
        while pending and len(active)<2:
            family,contrast,seed=pending.pop(0);uid='__'.join((family,contrast,'seed'+str(seed)))
            record_path=OUT/'training/on_policy_correction'/(uid+'.json')
            if record_path.exists():
                record=read_json(record_path);assert sha256(record['checkpoint_path'])==record['checkpoint_SHA']
                assert record['protocol_SHA']==sha256(OUT/'protocol/ON_POLICY_CORRECTION_HEADS_V1.json')
                assert all(sha256(ROOT/p)==s for p,s in record['source_code_SHA'].items())
                finished.append({'fit_uid':uid,'returncode':0,'verified_existing':True});continue
            path=logs/(uid+'.log')
            if path.exists():raise FileExistsError('preserve partial correction fit logs')
            storage(32<<20);handle=path.open('x')
            command=[python,str(ROOT/'scripts/n72r21r1_train_on_policy.py'),'--family',family,'--contrast',contrast,'--seed',str(seed)]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,path,uid,command))
        for entry in list(active):
            process,handle,path,uid,command=entry;result=process.poll()
            if result is None:continue
            handle.close();active.remove(entry)
            finished.append({'fit_uid':uid,'returncode':result,'command':command,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'correction_fit_terminal':uid,'returncode':result,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('training/on_policy_correction/SCHEDULER_RESULT.json',{'status':'WORKERS_TERMINAL_VERIFY_FITS',
        'workers':finished,'planned_fits':18,'CPU_workers_max':2,'no_GPU_use':True})
    actual=list((OUT/'training/on_policy_correction').glob('*__seed*.json'))
    update_status(phase_C5_correction_training_running=False,actual_on_policy_correction_fits=len(actual))


if __name__=='__main__':main()
