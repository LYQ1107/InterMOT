"""One GPU fit process at a time, only after complete sealed TRAIN/inner states."""
import argparse
import json
import subprocess
import sys
import time
from scripts.n72r21_common import ROOT,OUT,read_json,storage
from scripts.n72r21_train_t0 import split


def run(gpu):
    sequences=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'];seeds=[72101,72102,72103]
    pending=list(sequences);started=time.monotonic();announced=set()
    while pending:
        progressed=False
        for outer in pending.copy():
            tags=[f'{outer}__seed{seed}__{condition}' for seed in seeds for condition in ('P0','P1','MIXED')]
            if all((OUT/'training/T1_CAUSAL_V1'/f'{tag}.json').exists() and read_json(OUT/'training/T1_CAUSAL_V1'/f'{tag}.json').get('completed') for tag in tags):
                pending.remove(outer);print(json.dumps({'all_T1_conditions_complete':outer}),flush=True);progressed=True;continue
            fit_sequences,inner=split(outer,sequences)
            expected=[OUT/'training/causal_states/seals'/f'{outer}__seed{seed}__{policy}__K8'/f'{sequence}.json'
                      for seed in seeds for policy in ('P0','P1') for sequence in fit_sequences+[inner]]
            if not all(p.exists() for p in expected):continue
            storage(100<<20)
            print(json.dumps({'T1_ready_outer':outer,'sealed_state_sources':len(expected)}),flush=True)
            subprocess.run([sys.executable,'scripts/n72r21_train_t1.py','--outer',outer,'--gpu',str(gpu)],cwd=ROOT,check=True)
            pending.remove(outer);progressed=True
        state=tuple(pending)
        if state not in announced:
            print(json.dumps({'T1_waiting_for_sealed_sources':pending,'GPU_reserved_while_waiting':False}),flush=True);announced.add(state)
        if not progressed and pending:
            if time.monotonic()-started>7200:raise RuntimeError('bounded two-hour driver; completed fits and remaining sources retained')
            time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args();run(args.gpu)
