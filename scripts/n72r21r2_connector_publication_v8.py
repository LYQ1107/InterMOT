"""Normal-Git own-onset/main-full-MOT code checkpoint; immutable V8 receipts."""
import argparse
from scripts import n72r21r2_connector_publication_v6 as original
from scripts.n72r21r2_common import ASSETS, read_json, write_json, utcnow

DIRECTORY = ASSETS / "git_connector_publication/checkpoint_OWN_ONSET_MAIN_FULL_MOT_V8"
original.DIRECTORY = DIRECTORY
original.previous.DIRECTORY = DIRECTORY
original.previous.prior.DIRECTORY = DIRECTORY
original.previous.STATIC = {"sam3_intermot/evaluation/learned_policy_evidence.py",
    "outputs/N72R21R2/training/SUPPORT_ONSET_AUDIT_PROTOCOL_V1.json",
    "outputs/N72R21R2/mot/MAIN_POLICY_PROTOCOL_V1.json"}
BRANCH, save = original.BRANCH, original.save


def verify_git():
    """Fresh normal Git ref identity; no fabricated API canonical readback."""
    p = read_json(DIRECTORY / "PREPARED.json")
    push = read_json(DIRECTORY / "NORMAL_GIT_PUSH_ATTEMPT_V1.json")
    assert push["exit_code"] == 0 and not push["force_push"]
    assert not original.git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    head = original.git("rev-parse", "HEAD").decode().strip()
    readback = original.git("ls-remote", "origin", "refs/heads/" + BRANCH).decode().strip()
    remote, ref = readback.split("\t")
    assert ref == "refs/heads/" + BRANCH and head == remote == p["original_local_head"]
    tree = original.git("rev-parse", head + "^{tree}").decode().strip()
    parent = original.git("rev-parse", head + "^").decode().strip()
    assert tree == p["target_tree"] and parent == p["remote_parent"]
    save("FRESH_REMOTE_NORMAL_GIT_READBACK.json", {"output": readback, "remote_sha": remote,
        "same_commit_identity_implies_same_tree_parent": True, "API_canonical_readback_claimed": False})
    receipt = {"utc": utcnow(), "repository": "LYQ1107/InterMOT", "branch": BRANCH,
        "local_HEAD": head, "fresh_remote_HEAD": remote, "tree": tree, "parent": parent,
        "clean_worktree": True, "force_push": False, "normal_git_delivery": True,
        "verified_by_fresh_normal_git_ref_identity": True,
        "API_canonical_readback_claimed": False, "files": len(p["changes"]),
        "bytes": sum(c["bytes"] for c in p["changes"]), "code_and_necessary_docs_only": True,
        "no_weights_images_GT_embeddings_media_bulk_traces": True,
        "scientific_goal_complete": False, "next_stage_authorized": False}
    save("RECEIPT.json", receipt)
    write_json("git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json", receipt)
    print(receipt, flush=True)


def verify_api(ref_path, commit_path):
    previous = original.write_json
    def routed_write(relative, value, **kwargs):
        if relative == "git_delivery/CHECKPOINT_PURE_MARGIN_HAND_FILTER_SUPPORT_V6.json":
            relative = "git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json"
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
    group.add_argument("--verify-git", action="store_true")
    parser.add_argument("--canonical-commit")
    args = parser.parse_args()
    if args.prepare:
        original.previous.prepare(args.prepare)
    elif args.verify_git:
        verify_git()
    else:
        if not args.canonical_commit:
            parser.error("Canonical commit readback required")
        verify_api(args.verify_api_ref, args.canonical_commit)
