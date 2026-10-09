"""Wait for a genuinely freed CPU producer slot, then finish the frozen pilot."""
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,utcnow


def active_existing_producers():
    names={'n72r21_replay_ready_t2.py','n72r21_validation_ready.py'};active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
            if len(args)>1 and Path(args[1]).name in names:active.append({'pid':int(p.name),'script':Path(args[1]).name})
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return active


def run():
    start=time.monotonic();active=active_existing_producers()
    print({'MOT_pilot_CPU_slot_check':active,'starts_only_after_one_complete_producer_exits':True},flush=True)
    while len(active)>=4:
        if time.monotonic()-start>7200:raise RuntimeError('bounded CPU-slot wait; no existing job restarted or killed')
        time.sleep(20);active=active_existing_producers()
    write_json('mot_pilot/CPU_START_RECEIPT.json',{'utc':utcnow(),'other_live_producers':active,
        'wait_seconds':time.monotonic()-start,'global_CPU_runtime_workers_max':4,'unrelated_processes_stopped':False})
    print({'MOT_pilot_CPU_slot_acquired':True,'other_live_producers':active},flush=True)
    protocol=read_json(OUT/'protocol/MOT_TRAIN_PILOT.json')
    for sequence in protocol['sequences']:
        subprocess.run([sys.executable,'scripts/n72r21_mot_pilot.py','replay','--sequence',sequence],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/n72r21_mot_pilot_evaluate.py'],cwd=ROOT,check=True)


if __name__=='__main__':run()
