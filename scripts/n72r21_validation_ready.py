"""Single-threaded fixed-sequence worker, no fitting or adaptive scope."""
import argparse
import subprocess
import sys
from scripts.n72r21_common import ROOT,OUT,read_json


def run(sequences):
    registered=read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json')['sequences']
    if not set(sequences)<=set(registered):raise ValueError('registered VAL cohort')
    for sequence in sequences:
        for action in ('replay','evaluate'):
            subprocess.run([sys.executable,'scripts/n72r21_validation.py',action,'--sequences',sequence],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequences',nargs='+',required=True)
    args=parser.parse_args();run(args.sequences)
