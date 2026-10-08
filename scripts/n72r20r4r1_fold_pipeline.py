"""Disjoint-fold local CPU execution; each freeze precedes its outer replay."""
import argparse
from scripts.n72r20r4r1_common import SEQUENCES,torch
from scripts.n72r20r4r1_inner import run_fold as inner
from scripts.n72r20r4r1_joint_inner import run_fold as joint
from scripts.n72r20r4r1_outer import run_fold as outer

if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--heldouts',nargs='+',required=True);args=p.parse_args()
    if not set(args.heldouts)<=set(SEQUENCES):raise ValueError('unregistered fold')
    for sequence in args.heldouts:
        inner(sequence);joint(sequence);outer(sequence)
