"""Bounded T2 source/fit scheduling without simultaneous GPU fit jobs."""
import argparse
import json
import subprocess
import sys
import time
from scripts.n72r21_common import ROOT,OUT,read_json,storage
from scripts.n72r21_train_t0 import split


def completed(path):return path.exists() and read_json(path).get('completed',False)


def run(action,outers,gpu):
    sequences=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'];seeds=(72101,72102,72103)
    if not set(outers)<=set(sequences):raise ValueError('registered outers')
    pending=list(outers);started=time.monotonic();announced=set()
    while pending:
        progress=False
        # T1 has its own already-running single-GPU driver. T2 fits wait for
        # all T1 fits, not race that driver merely because a GPU looks idle.
        T1_all_done=all(completed(OUT/'training/T1_CAUSAL_V1'/f'{s}__seed{seed}__{condition}.json')
                        for s in sequences for seed in seeds for condition in ('P0','P1','MIXED'))
        for outer in pending.copy():
            if not all(completed(OUT/'training/T1_CAUSAL_V1'/f'{outer}__seed{seed}__MIXED.json') for seed in seeds):continue
            fit_sequences,inner=split(outer,sequences)
            if action=='collect':
                for scope in ('fit','inner'):
                    subprocess.run([sys.executable,'scripts/n72r21_collect_coupled_states.py','--outer',outer,'--scope',scope],cwd=ROOT,check=True)
            else:
                expected=[OUT/'training/coupled_states/seals'/f'{outer}__seed{seed}__T1MIXED_P1_K8'/f'{s}.json'
                          for seed in seeds for s in fit_sequences+[inner]]
                if not T1_all_done or not all(p.exists() for p in expected):continue
                storage(100<<20)
                subprocess.run([sys.executable,'scripts/n72r21_verify_risk_labels.py','--outer',outer,'--sequences',*fit_sequences,inner],cwd=ROOT,check=True)
                subprocess.run([sys.executable,'scripts/n72r21_train_t2.py','--outer',outer,'--gpu',str(gpu)],cwd=ROOT,check=True)
            pending.remove(outer);progress=True
            print(json.dumps({'T2_completed_action':action,'outer':outer}),flush=True)
        state=tuple(pending)
        if state not in announced:
            print(json.dumps({'T2_action':action,'waiting_for_sources':pending,'T1_single_GPU_phase_complete':T1_all_done,
                              'GPU_reserved_while_waiting':False}),flush=True);announced.add(state)
        if pending and not progress:
            if time.monotonic()-started>7200:raise RuntimeError('bounded two-hour T2 driver; artifacts retained')
            time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['collect','fit']);parser.add_argument('--outers',nargs='+',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();run(args.action,args.outers,args.gpu)
