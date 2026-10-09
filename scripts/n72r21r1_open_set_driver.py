"""Bounded all-axis extraction, no concurrent writers and no GT reads."""
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_open_set_collect import PROTOCOL


def main():
    protocol=read_json(PROTOCOL);pending=sorted({e['sequence'] for e in protocol['events']});active=[];finished=[]
    logs=ASSETS/'availability/current_axis_v1/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(current_axis_open_set_collection_running=True)
    while pending or active:
        while pending and len(active)<2:
            sequence=pending.pop(0);events=[e for e in protocol['events'] if e['sequence']==sequence]
            if all((OUT/'availability/current_axis_v1'/e['episode_uid']/'seal.json').exists() for e in events):
                for event in events:
                    s=read_json(OUT/'availability/current_axis_v1'/event['episode_uid']/'seal.json')
                    assert s['protocol_SHA']==sha256(PROTOCOL) and all(sha256(a['path'])==a['sha256'] for a in s['artifacts'])
                finished.append({'sequence':sequence,'returncode':0,'verified_existing':True});continue
            path=logs/(sequence+'.log')
            if path.exists():raise FileExistsError('preserve previous open-set extraction log')
            storage(32<<20);handle=path.open('x')
            process=subprocess.Popen([python,str(ROOT/'scripts/n72r21r1_open_set_collect.py'),'--sequence',sequence],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,path,sequence))
        for process,handle,path,sequence in list(active):
            code=process.poll()
            if code is None:continue
            handle.close();active.remove((process,handle,path,sequence))
            finished.append({'sequence':sequence,'returncode':code,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'open_set_sequence_terminal':sequence,'returncode':code,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('availability/current_axis_v1/SCHEDULER_RESULT.json',{'workers':finished,'maximum_CPU_workers':2,'new_training':False})
    update_status(current_axis_open_set_collection_running=False)


if __name__=='__main__':main()
