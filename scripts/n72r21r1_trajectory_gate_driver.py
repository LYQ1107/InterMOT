"""All12 explicit gate cases on two exposed TRAIN scenes; two CPU workers."""
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_trajectory_gate_pilot import PROTOCOL


def main():
    p=read_json(PROTOCOL);pending=[(c['case'],s) for c in p['cases'] if c['case']!='CLICK_C0' for s in p['sequences']]
    active=[];finished=[];logs=ASSETS/'trajectory_gate/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PATH']=str(ROOT.parent/'InterMOT/.venv/bin')+':'+env.get('PATH','');python=str(ROOT.parent/'InterMOT/.venv/bin/python')
    update_status(explicit_B7_B8_full_joint_running=True)
    while pending or active:
        while pending and len(active)<2:
            case,sequence=pending.pop(0);seal=OUT/'trajectory_gate/runtime_seals'/case/(sequence+'.json')
            if seal.exists():
                r=read_json(seal);assert r['protocol_SHA']==sha256(PROTOCOL) and all(sha256(a['path'])==a['sha256'] for a in r['artifacts'])
                finished.append({'case':case,'sequence':sequence,'returncode':0,'verified_existing':True});continue
            path=logs/(case+'__'+sequence+'.log')
            if path.exists():raise FileExistsError('preserve B7/B8 partial runtime log')
            storage(64<<20);handle=path.open('x');command=[python,str(ROOT/'scripts/n72r21r1_trajectory_gate_pilot.py'),'--sequence',sequence,'--case',case]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,path,case,sequence,command))
        for entry in list(active):
            process,handle,path,case,sequence,command=entry;code=process.poll()
            if code is None:continue
            handle.close();active.remove(entry);finished.append({'case':case,'sequence':sequence,'returncode':code,'command':command,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'B7_B8_joint_terminal':case,'sequence':sequence,'returncode':code,'remaining':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('trajectory_gate/SCHEDULER_RESULT.json',{'workers':finished,'planned_new_actual_runs':24,'max_CPU_workers':2,'driver_source_SHA':sha256(__file__)})
    update_status(explicit_B7_B8_full_joint_running=False,actual_explicit_B7_B8_new_joint_runs=sum(r['returncode']==0 for r in finished))
    if any(r['returncode'] for r in finished):raise RuntimeError('actual B7/B8 case failed; retain log')


if __name__=='__main__':main()
