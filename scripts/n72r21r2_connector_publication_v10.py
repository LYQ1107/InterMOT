"""Isolated V10 code-only all17-group closure checkpoint, no force ref write."""
import argparse
from scripts import n72r21r2_connector_publication_v8 as previous
from scripts.n72r21r2_common import ASSETS, write_json

DIRECTORY = ASSETS / "git_connector_publication/checkpoint_ALL17_DEVELOPMENT_CLOSURE_V10"
previous.DIRECTORY = DIRECTORY
previous.original.DIRECTORY = DIRECTORY
previous.original.previous.DIRECTORY = DIRECTORY
previous.original.previous.prior.DIRECTORY = DIRECTORY
previous.original.previous.STATIC = {"sam3_intermot/evaluation/development_closure.py",
    "outputs/N72R21R2/mot/DEVELOPMENT_CLOSURE_PROTOCOL_V1.json"}


def verify_git():
    original_write = previous.write_json
    def routed_write(relative, value, **kwargs):
        if relative == "git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json":
            relative = "git_delivery/CHECKPOINT_ALL17_DEVELOPMENT_CLOSURE_V10.json"
        return write_json(relative, value, **kwargs)
    try:
        previous.write_json = routed_write; previous.verify_git()
    finally:
        previous.write_json = original_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare"); group.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    previous.original.previous.prepare(args.prepare) if args.prepare else verify_git()
