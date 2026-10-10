"""Fresh V12 source-only actual-data/task-delivery checkpoint namespace."""
import argparse
from scripts import n72r21r2_connector_publication_v8 as previous
from scripts.n72r21r2_common import ASSETS, write_json

DIRECTORY = ASSETS / "git_connector_publication/checkpoint_ACTUAL_DATA_FULL_TASK_DELIVERY_V12"
previous.DIRECTORY = DIRECTORY
previous.original.DIRECTORY = DIRECTORY
previous.original.previous.DIRECTORY = DIRECTORY
previous.original.previous.prior.DIRECTORY = DIRECTORY
previous.original.previous.STATIC = {"sam3_intermot/evaluation/task_delivery_evidence.py"}


def verify_git():
    saved = previous.write_json
    def routed(relative, value, **kwargs):
        if relative == "git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json":
            relative = "git_delivery/CHECKPOINT_ACTUAL_DATA_FULL_TASK_DELIVERY_V12.json"
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
