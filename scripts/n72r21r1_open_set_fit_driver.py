"""Nine registered single-thread CPU fits; preserve every failed attempt."""
import itertools
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_train_open_set import PROTOCOL


def main():
    protocol=read_json(PROTOCOL);pending=list(itertools.product(protocol['families'],protocol['seeds']));active=[];finished=[]
    logs=ASSETS/'training/current_axis_verifier/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(current_axis_verifier_fitting_running=True,git_publication_clean=False)
    while pending or active:
        while pending and len(active)<2:
            family,seed=pending.pop(0);uid=family+'__seed'+str(seed)
            path=OUT/'training/current_axis_verifier'/(uid+'.json')
            if path.exists():
                r=read_json(path);assert r['protocol_SHA']==sha256(PROTOCOL) and sha256(r['checkpoint_path'])==r['checkpoint_SHA']
                assert all(sha256(ROOT/k)==v for k,v in r['source_code_SHA'].items())
                finished.append({'fit_uid':uid,'returncode':0,'verified_existing':True});continue
            path=logs/(uid+'.log')
            if path.exists():raise FileExistsError('preserve partial open-set fit log')
            storage(32<<20);handle=path.open('x')
            command=[python,str(ROOT/'scripts/n72r21r1_train_open_set.py'),'--family',family,'--seed',str(seed)]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,path,uid,command))
        for entry in list(active):
            process,handle,path,uid,command=entry;code=process.poll()
            if code is None:continue
            handle.close();active.remove(entry);finished.append({'fit_uid':uid,'returncode':code,'command':command,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'verifier_fit_terminal':uid,'returncode':code,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('training/current_axis_verifier/SCHEDULER_RESULT.json',{'status':'WORKERS_TERMINAL_REQUIRE_NINE_FIT_VERIFICATION',
        'workers':finished,'planned_fits':9,'max_CPU_workers':2,'no_GPU_use':True,'scheduler_source_SHA':sha256(__file__)})
    update_status(current_axis_verifier_fitting_running=False,actual_current_axis_verifier_fits=len(list((OUT/'training/current_axis_verifier').glob('*__seed*.json'))))
    if any(r['returncode'] for r in finished):raise RuntimeError('at least one verifier fit failed; retain log')


if __name__=='__main__':main()
