"""Actual MAIN feature-support code-only delivery; retain V5-V13 receipts."""
import argparse
from scripts import n72r21r2_connector_publication_v8 as previous
from scripts.n72r21r2_common import ASSETS, write_json

DIRECTORY = ASSETS / "git_connector_publication/checkpoint_MAIN_FEATURE_SUPPORT_V14"
previous.DIRECTORY = DIRECTORY
previous.original.DIRECTORY = DIRECTORY
previous.original.previous.DIRECTORY = DIRECTORY
previous.original.previous.prior.DIRECTORY = DIRECTORY
previous.original.previous.STATIC = {"sam3_intermot/evaluation/current_feature_support.py"}


def verify_git():
    saved = previous.write_json
    def routed(relative, value, **kwargs):
        if relative == "git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json":
            relative = "git_delivery/CHECKPOINT_MAIN_FEATURE_SUPPORT_V14.json"
        return write_json(relative, value, **kwargs)
    try:
        previous.write_json = routed
        previous.verify_git()
    finally:
        previous.write_json = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare")
    group.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    previous.original.previous.prepare(args.prepare) if args.prepare else verify_git()
