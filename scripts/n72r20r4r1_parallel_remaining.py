"""Move only this task's serial queue to disjoint one-thread CPU workers."""
import argparse
import os
import signal
import subprocess
import sys
import time
from scripts.n72r20r4r1_common import OUT,ROOT,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--serial-pid',type=int,required=True);args=p.parse_args()
    proc=__import__('pathlib').Path(f'/proc/{args.serial_pid}')
    original=(proc/'cmdline').read_bytes();start=(proc/'stat').read_text().split()[21]
    expected=b'scripts/n72r20r4r1_inner.py'
    if expected not in original or b'dancetrack0002' not in original:raise ValueError('not our known serial research worker')
    while not (OUT/'authority/frozen_outer/dancetrack0002.json').exists():time.sleep(5)
    if proc.exists():
        if (proc/'cmdline').read_bytes()!=original or (proc/'stat').read_text().split()[21]!=start:raise ValueError('PID identity changed')
        os.kill(args.serial_pid,signal.SIGTERM)
    queues=[['dancetrack0002','dancetrack0057'],['dancetrack0023','dancetrack0072'],['dancetrack0024'],['dancetrack0039']]
    write_json(OUT/'audit/PARALLEL_CPU_EXECUTION.json',{'own_serial_pid':args.serial_pid,'validated_process_cmdline':original.decode().replace('\0',' '),
        'stopped_only_after_completed_0002_family_freeze':True,'disjoint_queues':queues,'OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1,
        'other_users_processes_stopped':False,'GPU_used':False,'freeze_then_outer_per_fold':True})
    env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    children=[subprocess.Popen([sys.executable,'scripts/n72r20r4r1_fold_pipeline.py','--heldouts',*q],cwd=ROOT,env=env) for q in queues]
    codes=[child.wait() for child in children]
    if any(codes):raise SystemExit(f'worker return codes: {codes}')
