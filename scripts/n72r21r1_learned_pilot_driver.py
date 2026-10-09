"""Three bounded CPU workers; every frozen learned comparison retained."""
import os
import subprocess
import time
import json
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status


def main():
    protocol=read_json(OUT/'protocol/LEARNED_FULL_MOT_PILOT_V1.json')
    pending=[(c['case'],s) for c in protocol['cases'] for s in protocol['sequences']]
    active=[]; finished=[]; logs=ASSETS/'learned_pilot/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PATH']=str(ROOT.parent/'InterMOT/.venv/bin')+':'+env.get('PATH','')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python')
    update_status(phase_H_learned_full_MOT_running=True,planned_learned_full_MOT_runs=120)
    while pending or active:
        while pending and len(active)<3:
            case,sequence=pending.pop(0);path=logs/(case+'__'+sequence+'.log')
            seal_path=OUT/'learned_pilot/runtime_seals'/case/(sequence+'.json')
            if seal_path.exists():
                seal=read_json(seal_path);assert seal['protocol_SHA']==sha256(OUT/'protocol/LEARNED_FULL_MOT_PILOT_V1.json')
                assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts'])
                finished.append({'case':case,'sequence':sequence,'returncode':0,'verified_existing':True});continue
            if path.exists():raise FileExistsError('preserve partial learned worker log')
            storage(64<<20);handle=path.open('x')
            command=[python,str(ROOT/'scripts/n72r21r1_learned_pilot.py'),'--sequence',sequence,'--case',case]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,path,case,sequence,command))
        for entry in list(active):
            process,handle,path,case,sequence,command=entry;result=process.poll()
            if result is None:continue
            handle.close();active.remove(entry)
            finished.append({'case':case,'sequence':sequence,'returncode':result,'log_path':str(path),'log_SHA':sha256(path),'command':command})
            print(json.dumps({'learned_worker_terminal':case,'sequence':sequence,'returncode':result,'remaining':len(pending),'live':len(active)}),flush=True)
        if active:time.sleep(.25)
    write_json('learned_pilot/SCHEDULER_RESULT.json',{'status':'WORKERS_TERMINAL_CHECK_SEALS','planned_runs':120,
        'successful_workers':sum(r['returncode']==0 for r in finished),'workers':finished,'CPU_workers_max':3,'no_GPU_use':True})
    update_status(phase_H_learned_full_MOT_running=False,completed_learned_full_MOT_runs=sum(r['returncode']==0 for r in finished))


if __name__=='__main__':main()
