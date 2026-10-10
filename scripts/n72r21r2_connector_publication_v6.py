"""Code-only pure-margin/support checkpoint with separate immutable receipts.

Read-only API verification is allowed after a normal Git push; no history or
ref mutation is performed here. Both ref and canonical commit must match.
"""
import argparse
import json
from scripts.n72r21r2_common import ASSETS, ROOT, read_json, write_json, utcnow
from scripts import n72r21r2_connector_publication_v3 as previous

BRANCH = previous.BRANCH
DIRECTORY = ASSETS / "git_connector_publication/checkpoint_PURE_MARGIN_HAND_FILTER_SUPPORT_V6"
previous.DIRECTORY = DIRECTORY
previous.prior.DIRECTORY = DIRECTORY
previous.STATIC = {
    "sam3_intermot/evaluation/authority_support_audit.py",
    "outputs/N72R21R2/availability/PURE_MARGIN_PROTOCOL_V1.json",
    "outputs/N72R21R2/availability/PURE_MARGIN_SELECTED_POINTS_V1.json",
    "outputs/N72R21R2/training/AUTHORITY_SUPPORT_AUDIT_PROTOCOL_V1.json",
    "outputs/N72R21R2/training/SUPPORT_ABLATION_PILOT_PROTOCOL_V1.json",
    "outputs/N72R21R2/events/DIAGNOSTICS_WAIT_RECOVERY_PROTOCOL_V2.json",
}
git, save = previous.git, previous.save


def verify_api(ref_path, commit_path):
    p = read_json(DIRECTORY / "PREPARED.json")
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    head = git("rev-parse", "HEAD").decode().strip()
    ref, commit = read_json(ref_path), read_json(commit_path)
    assert head == p["original_local_head"] == ref["object"]["sha"] == commit["sha"]
    assert ref["ref"] == "refs/heads/" + BRANCH
    assert commit["tree"]["sha"] == p["target_tree"] == git("rev-parse", head + "^{tree}").decode().strip()
    assert [r["sha"] for r in commit["parents"]] == [p["remote_parent"]]
    assert git("rev-parse", head + "^").decode().strip() == p["remote_parent"]
    save("FRESH_REMOTE_API_REF_READBACK.json", ref)
    save("FRESH_REMOTE_API_CANONICAL_COMMIT_READBACK.json", commit)
    receipt = {"utc": utcnow(), "repository": "LYQ1107/InterMOT", "branch": BRANCH,
               "local_HEAD": head, "fresh_remote_HEAD": ref["object"]["sha"], "tree": p["target_tree"],
               "parent": p["remote_parent"], "clean_worktree": True, "force_push": False,
               "normal_git_delivery": True, "verified_by_fresh_API_ref_and_canonical_commit": True,
               "Git_only_readback_not_inferred_from_API_success": True,
               "files": len(p["changes"]), "bytes": sum(c["bytes"] for c in p["changes"]),
               "code_and_necessary_docs_only": True, "no_weights_images_GT_embeddings_media_bulk_traces": True,
               "scientific_goal_complete": False, "next_stage_authorized": False}
    save("RECEIPT.json", receipt)
    write_json("git_delivery/CHECKPOINT_PURE_MARGIN_HAND_FILTER_SUPPORT_V6.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare")
    group.add_argument("--verify-api-ref")
    parser.add_argument("--canonical-commit")
    args = parser.parse_args()
    if args.prepare:
        previous.prepare(args.prepare)
    else:
        if not args.canonical_commit:
            parser.error("Canonical commit is required with API ref verification")
        verify_api(args.verify_api_ref, args.canonical_commit)
