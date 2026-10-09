"""Wait for the sole GPU fit phase; lawful bounded existing-window work."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.n72r21_common import ROOT,OUT,read_json


def other_CPU_runtime_workers():
    names=('scripts/n72r21_t2_replay.py','scripts/n72r21_validation.py','scripts/n72r21_collect_coupled_states.py','scripts/n72r21_t1_replay.py')
    count=0
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            if directory.stat().st_uid!=os.getuid():continue
            command=(directory/'cmdline').read_bytes().split(b'\0')
            if len(command)>1 and any(command[1].decode(errors='replace').endswith(n) for n in names):count+=1
        except (OSError,ProcessLookupError):continue
    return count


def run(gpu):
    started=time.monotonic();announced=False
    while True:
        records=list((OUT/'training/T2_COUPLED_V1').glob('*.json'))
        if len(records)==24 and all(read_json(p).get('completed') for p in records):
            status=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            memory,util=[int(v.strip()) for v in status.stdout.strip().split(',')]
            if memory<=64 and not util:break
        if not announced:print({'LaSOT_waiting_for_all24_T2_fits_and_idle_GPU':True,'GPU_reserved':False},flush=True);announced=True
        if time.monotonic()-started>7200:raise RuntimeError('bounded readiness wait, no resource reserved')
        time.sleep(20)
    windows=read_json(OUT/'protocol/LASOT_FRAME_ONLY_IDENTITY.json')['windows']
    # The generator is the sole GPU process, with one host decode thread;
    # do not add a fifth CPU-only identity replay while four are active.
    for name in windows:
        subprocess.run([sys.executable,'scripts/n72r21_lasot_candidates.py','--name',name,'--gpu',str(gpu)],cwd=ROOT,check=True)
    while other_CPU_runtime_workers()>=4:
        if time.monotonic()-started>7200:raise RuntimeError('bounded CPU slot wait; sealed candidates retained')
        time.sleep(20)
    for name in windows:
        for action in ('replay','evaluate'):
            subprocess.run([sys.executable,'scripts/n72r21_lasot_identity.py',action,'--name',name],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args();run(args.gpu)
