"""Code-only current-risk/frontier/recovery checkpoint; new immutable receipts."""
import argparse
import json
from pathlib import Path
import subprocess
from scripts.n72r21r2_common import ASSETS, ROOT, read_json, write_json, utcnow
from scripts import n72r21r2_connector_publication_v3 as previous

BRANCH = previous.BRANCH
DIRECTORY = ASSETS / "git_connector_publication/checkpoint_CURRENT_RISK_TENSOR_RECOVERY_V5"
previous.DIRECTORY = DIRECTORY
previous.prior.DIRECTORY = DIRECTORY
previous.STATIC = {"outputs/N72R21R2/memory/RISK_WRITE_PROTOCOL_V1.json",
                   "outputs/N72R21R2/memory/SAFETY_FRONTIER_PROTOCOL_V1.json",
                   "outputs/N72R21R2/training/MAIN_WAIT_RECOVERY_PROTOCOL_V2.json",
                   "outputs/N72R21R2/on_policy/JOINT_STATE_TENSOR_RECOVERY_PROTOCOL_V2.json",
                   "sam3_intermot/evaluation/memory_safety_frontier.py"}
git, save = previous.git, previous.save


def verify_direct():
    p = read_json(DIRECTORY / "PREPARED.json")
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    head = git("rev-parse", "HEAD").decode().strip()
    assert head == p["original_local_head"]
    command = ["git", "-c", "http.version=HTTP/1.1", "ls-remote", "--heads", "origin", BRANCH]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=25, check=True)
    sha, ref = result.stdout.strip().split()
    assert ref == "refs/heads/" + BRANCH and sha == head
    assert git("rev-parse", head + "^{tree}").decode().strip() == p["target_tree"]
    assert git("rev-parse", head + "^").decode().strip() == p["remote_parent"]
    receipt = {"utc": utcnow(), "repository": "LYQ1107/InterMOT", "branch": BRANCH,
               "local_HEAD": head, "fresh_remote_HEAD": sha, "tree": p["target_tree"], "parent": p["remote_parent"],
               "fresh_read_command": command, "clean_worktree": True, "force_push": False,
               "normal_git_delivery": True, "code_and_necessary_docs_only": True,
               "files": len(p["changes"]), "bytes": sum(c["bytes"] for c in p["changes"]),
               "no_weights_images_GT_embeddings_media_bulk_traces": True,
               "scientific_goal_complete": False, "next_stage_authorized": False}
    save("RECEIPT.json", receipt)
    write_json("git_delivery/CHECKPOINT_CURRENT_RISK_TENSOR_RECOVERY_V5.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare")
    mode.add_argument("--emit-chunk", type=int)
    mode.add_argument("--import-commit")
    mode.add_argument("--verify-direct", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        previous.prepare(args.prepare)
    elif args.emit_chunk is not None:
        previous.prior.emit_chunk(args.emit_chunk)
    elif args.import_commit:
        previous.prior.import_commit(Path(args.import_commit))
    else:
        verify_direct()
