"""Bounded committed-memory TRAIN curriculum, preserve all partial attempts."""
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status


def main():
    protocol_path=OUT/'protocol/COMMITTED_MEMORY_CURRICULUM_V1.json';protocol=read_json(protocol_path)
    pending=sorted({e['sequence'] for e in protocol['events']},key=lambda s:(s!='dancetrack0002',s))
    active=[];finished=[];logs=ASSETS/'memory/curriculum/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(phase_F_committed_curriculum_running=True)
    while pending or active:
        while pending and len(active)<2:
            sequence=pending.pop(0);events=[e for e in protocol['events'] if e['sequence']==sequence]
            if all((OUT/'memory/curriculum/runtime'/e['episode_uid']/'seal.json').exists() for e in events):
                for event in events:
                    s=read_json(OUT/'memory/curriculum/runtime'/event['episode_uid']/'seal.json');assert s['protocol_SHA']==sha256(protocol_path)
                    assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts'])
                finished.append({'sequence':sequence,'returncode':0,'verified_existing':True});continue
            path=logs/(sequence+'.log')
            if path.exists():raise FileExistsError('preserve partial memory curriculum log')
            storage(96<<20);handle=path.open('x');command=[python,str(ROOT/'scripts/n72r21r1_memory_collect.py'),'--sequence',sequence]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);active.append((process,handle,path,sequence))
        for entry in list(active):
            process,handle,path,sequence=entry;rc=process.poll()
            if rc is None:continue
            handle.close();active.remove(entry);finished.append({'sequence':sequence,'returncode':rc,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'memory_curriculum_terminal':sequence,'returncode':rc,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('memory/curriculum/SCHEDULER_RESULT.json',{'planned_full_joint_rollouts':22,'workers':finished,'CPU_workers_max':2,'no_GPU_use':True})
    update_status(phase_F_committed_curriculum_running=False)


if __name__=='__main__':main()
