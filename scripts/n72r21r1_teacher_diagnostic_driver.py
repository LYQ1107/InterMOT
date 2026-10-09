"""Two-worker posthoc diagnostic; preserve and verify all completed attempts."""
import json
import os
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status


def main():
    path=OUT/'protocol/TEACHER_STATE_DIAGNOSTIC_V1.json';protocol=read_json(path)
    pending=sorted({e['sequence'] for e in protocol['events']},key=lambda s:(s!='dancetrack0002',s))
    active=[];finished=[];logs=ASSETS/'diagnostics/teacher_state_v1/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python')
    update_status(teacher_state_diagnostic_running=True,git_publication_clean=False)
    while pending or active:
        while pending and len(active)<2:
            sequence=pending.pop(0);events=[e for e in protocol['events'] if e['sequence']==sequence]
            if all((OUT/'diagnostics/teacher_state_v1'/e['episode_uid']/'seal.json').exists() for e in events):
                for event in events:
                    seal=read_json(OUT/'diagnostics/teacher_state_v1'/event['episode_uid']/'seal.json')
                    assert seal['protocol_SHA']==sha256(path)
                    assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts'])
                finished.append({'sequence':sequence,'returncode':0,'verified_existing':True});continue
            log=logs/(sequence+'.log')
            if log.exists():raise FileExistsError('preserve partial teacher diagnostic log')
            storage(64<<20);handle=log.open('x')
            process=subprocess.Popen([python,str(ROOT/'scripts/n72r21r1_teacher_diagnostic.py'),'--sequence',sequence],
                cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,log,sequence))
        for process,handle,log,sequence in list(active):
            code=process.poll()
            if code is None:continue
            handle.close();active.remove((process,handle,log,sequence))
            finished.append({'sequence':sequence,'returncode':code,'log_path':str(log),'log_SHA':sha256(log)})
            print(json.dumps({'teacher_sequence_terminal':sequence,'returncode':code,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('diagnostics/teacher_state_v1/SCHEDULER_RESULT.json',{'workers':finished,'maximum_CPU_workers':2,
        'actual_successful_sequences':sum(r['returncode']==0 for r in finished),'no_new_training':True})
    update_status(teacher_state_diagnostic_running=False)


if __name__=='__main__':main()
