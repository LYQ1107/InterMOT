"""All three actual writer fits, no more than two single-thread workers."""
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status


def main():
    protocol=read_json(OUT/'protocol/JOINT_WRITE_HEADS_V1.json');pending=list(protocol['seeds']);active=[];finished=[]
    logs=ASSETS/'training/joint_write/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(phase_F_joint_write_training_running=True)
    while pending or active:
        while pending and len(active)<2:
            seed=pending.pop(0);uid='JOINT_COMMITTED_CURRENT_AND_FUTURE_RISK_MLP__seed'+str(seed);path=logs/(uid+'.log')
            record_path=OUT/'training/joint_write'/(uid+'.json')
            if record_path.exists():
                r=read_json(record_path);assert sha256(r['checkpoint_path'])==r['checkpoint_SHA']
                assert all(sha256(ROOT/p)==s for p,s in r['source_code_SHA'].items())
                assert r['protocol_SHA']==sha256(OUT/'protocol/JOINT_WRITE_HEADS_V1.json')
                finished.append({'seed':seed,'returncode':0,'verified_existing':True});continue
            if path.exists():raise FileExistsError('preserve partial writer fit log')
            storage(32<<20);handle=path.open('x');command=[python,str(ROOT/'scripts/n72r21r1_train_joint_write.py'),'--seed',str(seed)]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);active.append((process,handle,path,seed))
        for entry in list(active):
            process,handle,path,seed=entry;rc=process.poll()
            if rc is None:continue
            handle.close();active.remove(entry);finished.append({'seed':seed,'returncode':rc,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'joint_write_fit_terminal':seed,'returncode':rc,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('training/joint_write/SCHEDULER_RESULT.json',{'planned_fits':3,'workers':finished,'CPU_workers_max':2,'no_GPU_use':True})
    update_status(phase_F_joint_write_training_running=False,actual_joint_write_fits=len(list((OUT/'training/joint_write').glob('*__seed*.json'))))


if __name__=='__main__':main()
