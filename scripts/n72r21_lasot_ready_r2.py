"""One bounded R2 run per window, full logs retained, capped CPU concurrency."""
import argparse
import subprocess
import sys
import time
from scripts.n72r21_lasot_ready import other_CPU_runtime_workers
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256


def run(gpu):
    fits=list((OUT/'training/T2_COUPLED_V1').glob('*.json'))
    if len(fits)!=24 or not all(read_json(p).get('completed') for p in fits):raise RuntimeError('all24 fits required')
    windows=read_json(OUT/'protocol/LASOT_FRAME_ONLY_IDENTITY.json')['windows']
    for name in windows:
        log=ASSETS/'diagnostics/LaSOT_GROUNDING_R2_logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        if log.exists():raise FileExistsError('preserve R2 logs, no automatic second attempt')
        with log.open('xb') as stream:
            result=subprocess.run([sys.executable,'scripts/n72r21_lasot_batch_repair_r2.py','--name',name,'--gpu',str(gpu)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        write_json(f'experiments/LaSOT_identity/R2_LOGS/{name}.json',{'path':str(log),'sha256':sha256(log),'bytes':log.stat().st_size,
            'returncode':result.returncode,'repair_protocol_sha256':sha256(OUT/'protocol/LASOT_GROUNDING_BATCH_REPAIR_R2.json')})
        print({'LaSOT_R2_candidate_result':name,'exit_code':result.returncode,'full_log_retained':str(log)},flush=True)
        if result.returncode:raise RuntimeError(f'R2 failed for {name}; preserve evidence, stop this version')
    started=time.monotonic()
    while other_CPU_runtime_workers()>=4:
        if time.monotonic()-started>7200:raise RuntimeError('bounded CPU-slot wait; sealed candidates retained')
        time.sleep(20)
    for name in windows:
        for action in ('replay','evaluate'):
            subprocess.run([sys.executable,'scripts/n72r21_lasot_identity.py',action,'--name',name],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args();run(args.gpu)
