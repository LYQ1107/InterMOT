"""Registered bounded optimizer driver; CPU-only, at most two fit workers."""
from pathlib import Path
import os
import subprocess
import time
import json
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, update_status, storage


def main():
    protocol = read_json(OUT/'protocol/LEARNED_AUTHORITY_V1.json'); jobs = []
    for family in protocol['families']:
        for state in protocol['state_contrasts']:
            for seed in protocol['seeds']: jobs.append((family, state, protocol['primary_supervision'], seed))
    for reward in protocol['LOGISTIC_MIXED_reward_contrasts']:
        for seed in protocol['seeds']: jobs.append(('LOGISTIC', 'MIXED', reward, seed))
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PATH'] = str(ROOT.parent/'InterMOT/.venv/bin')+':'+env.get('PATH','')
    python = str(ROOT.parent/'InterMOT/.venv/bin/python'); active = []; finished = []
    logs = ASSETS/'training/authority_driver_logs'; logs.mkdir(parents=True, exist_ok=True)
    while jobs or active:
        while jobs and len(active) < 2:
            job = jobs.pop(0); uid = '__'.join((job[0], job[1], job[2], 'seed'+str(job[3])))
            record = OUT/'training/authority'/(uid+'.json')
            if record.exists():
                saved = read_json(record); assert sha256(saved['checkpoint_path']) == saved['checkpoint_SHA']
                finished.append({'uid': uid, 'returncode': 0, 'verified_existing': True}); continue
            path = logs/(uid+'.log')
            if path.exists(): raise FileExistsError('do not overwrite partial fit log')
            storage(64<<20); handle = path.open('x')
            command = [python, str(ROOT/'scripts/n72r21r1_train_authority.py'), '--family',job[0], '--state',job[1], '--reward',job[2], '--seed',str(job[3])]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active.append((process, handle, path, uid, command))
        for entry in list(active):
            process, handle, path, uid, command = entry
            result = process.poll()
            if result is None: continue
            handle.close(); active.remove(entry)
            finished.append({'uid':uid,'returncode':result,'log_path':str(path),'log_SHA':sha256(path),'command':command})
            print(json.dumps({'fit_worker_terminal':uid,'returncode':result,'remaining_jobs':len(jobs),'live_workers':len(active)}),flush=True)
        if active: time.sleep(.25)
    successful = sum(r['returncode']==0 for r in finished)
    write_json('training/AUTHORITY_FIT_SCHEDULER_RESULT.json',{'status':'WORKERS_TERMINAL_CHECK_FIT_RECORDS','registered_fits':len(finished),
        'successful_workers':successful,'workers':finished,'CPU_workers_max':2,'no_GPU_use':True,'not_MOT_evaluation':True})
    update_status(completed_new_fits=successful,phase_H_optimizer_workers_running=False)


if __name__=='__main__': main()
