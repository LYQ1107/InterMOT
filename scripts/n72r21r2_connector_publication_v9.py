"""Code-only state-source/scene checkpoint, isolated non-force V9 receipts."""
import argparse
from scripts import n72r21r2_connector_publication_v8 as previous
from scripts.n72r21r2_common import ASSETS, write_json

DIRECTORY = ASSETS / "git_connector_publication/checkpoint_STATE_DENSITY_SCENES_V9"
previous.DIRECTORY = DIRECTORY
previous.original.DIRECTORY = DIRECTORY
previous.original.previous.DIRECTORY = DIRECTORY
previous.original.previous.prior.DIRECTORY = DIRECTORY
previous.original.previous.STATIC = {"sam3_intermot/evaluation/whole_video_density.py",
    "sam3_intermot/evaluation/scene_difficulty.py",
    "outputs/N72R21R2/on_policy/STATE_POLICY_PROTOCOL_V1.json",
    "outputs/N72R21R2/data/SCENE_CHARACTERISTICS_PROTOCOL_V1.json"}


def verify_git():
    original_write = previous.write_json
    def routed_write(relative, value, **kwargs):
        if relative == "git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json":
            relative = "git_delivery/CHECKPOINT_STATE_DENSITY_SCENES_V9.json"
        return write_json(relative, value, **kwargs)
    try:
        previous.write_json = routed_write
        previous.verify_git()
    finally:
        previous.write_json = original_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare"); modes.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    previous.original.previous.prepare(args.prepare) if args.prepare else verify_git()
