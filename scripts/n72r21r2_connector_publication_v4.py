"""Leased code-only matched-state checkpoint, preserving prior publications."""
import argparse
import json
from pathlib import Path
from scripts.n72r21r2_common import ASSETS, read_json, write_json, utcnow
from scripts import n72r21r2_connector_publication_v3 as previous

BRANCH = previous.BRANCH
DIRECTORY = ASSETS / "git_connector_publication/checkpoint_MATCHED_JOINT_STATE_V4"
previous.DIRECTORY = DIRECTORY
previous.prior.DIRECTORY = DIRECTORY
previous.STATIC = {"outputs/N72R21R2/on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json",
                   "outputs/N72R21R2/on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json",
                   "outputs/N72R21R2/availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json"}
git, save = previous.git, previous.save


def adopt(remote):
    prepared = read_json(DIRECTORY / "PREPARED.json")
    assert read_json(DIRECTORY / "IMPORTED.json")["sha"] == remote
    assert git("rev-parse", "HEAD").decode().strip() == prepared["original_local_head"]
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    assert git("rev-parse", remote + "^{tree}").decode().strip() == prepared["target_tree"]
    git("merge-base", "--is-ancestor", prepared["remote_parent"], remote)
    assert read_json(DIRECTORY / "FRESH_REMOTE_READBACK.json")["object"]["sha"] == remote
    backup = "refs/heads/codex/n72r21r2-offline-" + prepared["original_local_head"][:12]
    git("update-ref", backup, prepared["original_local_head"], "0" * 40)
    git("update-ref", "refs/heads/" + BRANCH, remote, prepared["original_local_head"])
    git("update-ref", "refs/remotes/origin/" + BRANCH, remote)
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    receipt = {"utc": utcnow(), "repository": "LYQ1107/InterMOT", "branch": BRANCH, "local_HEAD": remote,
               "fresh_remote_HEAD": remote, "tree": prepared["target_tree"], "clean_worktree": True,
               "preserved_original_local_HEAD": prepared["original_local_head"], "preserved_backup_ref": backup,
               "same_source_tree_no_working_file_rewrite": True, "force_push": False,
               "fresh_remote_parent": prepared["remote_parent"], "code_and_necessary_docs_only": True,
               "scientific_goal_complete": False, "next_stage_authorized": False}
    save("RECEIPT.json", receipt)
    write_json("git_delivery/CHECKPOINT_MATCHED_JOINT_STATE_V4.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare")
    group.add_argument("--emit-chunk", type=int)
    group.add_argument("--import-commit")
    group.add_argument("--adopt")
    args = parser.parse_args()
    if args.prepare:
        previous.prepare(args.prepare)
    elif args.emit_chunk is not None:
        previous.prior.emit_chunk(args.emit_chunk)
    elif args.import_commit:
        previous.prior.import_commit(Path(args.import_commit))
    else:
        adopt(args.adopt)
