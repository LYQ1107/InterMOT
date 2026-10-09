"""Bounded CPU worker for all preregistered T1 comparisons, after fits seal."""
import argparse
import json
import subprocess
import sys
import time
from scripts.n72r21_common import ROOT,OUT,read_json,storage


def run(sequences):
    registered=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences']
    if not set(sequences)<=set(registered):raise ValueError('registered cohort only')
    pending=list(sequences);announced=set();started=time.monotonic()
    while pending:
        progress=False
        for sequence in pending.copy():
            fits=[OUT/'training/T1_CAUSAL_V1'/f'{sequence}__seed{seed}__{condition}.json'
                  for seed in (72101,72102,72103) for condition in ('P0','P1','MIXED')]
            if not all(path.exists() and read_json(path).get('completed') for path in fits):continue
            storage(100<<20)
            for action in ('replay','evaluate'):
                subprocess.run([sys.executable,'scripts/n72r21_t1_replay.py',action,'--sequences',sequence],cwd=ROOT,check=True)
            print(json.dumps({'all_T1_comparisons_replayed_evaluated':sequence}),flush=True);pending.remove(sequence);progress=True
        state=tuple(pending)
        if state not in announced:
            print(json.dumps({'T1_replay_waiting_for_completed_fits':pending}),flush=True);announced.add(state)
        if pending and not progress:
            if time.monotonic()-started>7200:raise RuntimeError('two-hour CPU replay driver budget; complete/partial outputs retained')
            time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequences',nargs='+',required=True);args=parser.parse_args();run(args.sequences)
