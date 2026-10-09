"""Bounded full memory comparisons, all80 planned source rollouts retained."""
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_memory_pilot import PROTOCOL


def main():
    protocol=read_json(PROTOCOL);pending=[(c['case'],s) for c in protocol['cases'] for s in protocol['sequences']];active=[];finished=[]
    logs=ASSETS/'memory/pilot/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(phase_F_full_memory_MOT_running=True,planned_full_memory_MOT_runs=80)
    while pending or active:
        while pending and len(active)<3:
            case,sequence=pending.pop(0);seal_path=OUT/'memory/pilot/runtime_seals'/case/(sequence+'.json')
            if seal_path.exists():
                s=read_json(seal_path);assert s['protocol_SHA']==sha256(PROTOCOL)
                assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts']);finished.append({'case':case,'sequence':sequence,'returncode':0,'verified_existing':True});continue
            path=logs/(case+'__'+sequence+'.log')
            if path.exists():raise FileExistsError('preserve partial full memory worker log')
            storage(64<<20);handle=path.open('x');command=[python,str(ROOT/'scripts/n72r21r1_memory_pilot.py'),'--sequence',sequence,'--case',case]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);active.append((process,handle,path,case,sequence))
        for entry in list(active):
            process,handle,path,case,sequence=entry;rc=process.poll()
            if rc is None:continue
            handle.close();active.remove(entry);finished.append({'case':case,'sequence':sequence,'returncode':rc,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'full_memory_worker_terminal':case,'sequence':sequence,'returncode':rc,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('memory/pilot/SCHEDULER_RESULT.json',{'planned_runs':80,'workers':finished,'CPU_workers_max':3,'no_GPU_use':True})
    update_status(phase_F_full_memory_MOT_running=False,actual_full_memory_MOT_runs=sum(r['returncode']==0 for r in finished))


if __name__=='__main__':main()
