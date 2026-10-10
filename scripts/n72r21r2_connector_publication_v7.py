"""Separate code-only linear source-shift delivery; no old receipt overwrite."""
import argparse
from scripts import n72r21r2_connector_publication_v6 as original
from scripts.n72r21r2_common import ASSETS, write_json

DIRECTORY = ASSETS / "git_connector_publication/checkpoint_LINEAR_SOURCE_SHIFT_V7"
original.DIRECTORY = DIRECTORY
original.previous.DIRECTORY = DIRECTORY
original.previous.prior.DIRECTORY = DIRECTORY
original.previous.STATIC = {"sam3_intermot/evaluation/linear_source_shift.py",
                            "outputs/N72R21R2/training/LINEAR_SOURCE_SHIFT_PROTOCOL_V1.json"}
BRANCH, save = original.BRANCH, original.save


def verify_api(ref_path, commit_path):
    previous = original.write_json
    def routed_write(relative, value, **kwargs):
        if relative == "git_delivery/CHECKPOINT_PURE_MARGIN_HAND_FILTER_SUPPORT_V6.json":
            relative = "git_delivery/CHECKPOINT_LINEAR_SOURCE_SHIFT_V7.json"
        return write_json(relative, value, **kwargs)
    try:
        original.write_json = routed_write
        original.verify_api(ref_path, commit_path)
    finally:
        original.write_json = previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare")
    group.add_argument("--verify-api-ref")
    parser.add_argument("--canonical-commit")
    args = parser.parse_args()
    if args.prepare:
        original.previous.prepare(args.prepare)
    else:
        if not args.canonical_commit:
            parser.error("Canonical commit readback is required")
        verify_api(args.verify_api_ref, args.canonical_commit)
