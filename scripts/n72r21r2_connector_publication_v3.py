"""Fresh-head leased code-only memory/open-set checkpoint, no bulk evidence."""
import argparse
import json
from pathlib import Path
from scripts.n72r21r2_common import ROOT, ASSETS, read_json, write_json, utcnow
from scripts import n72r21r2_connector_publication as prior
from scripts.n72r20r4_publish_git import commit_payload

BRANCH = prior.BRANCH
DIRECTORY = ASSETS / "git_connector_publication/checkpoint_MEMORY_OPEN_SET_V3"
prior.DIRECTORY = DIRECTORY
git, save, entries = prior.git, prior.save, prior.entries
STATIC = {"outputs/N72R21R2/memory/ZERO_AUTHORITY_PROTOCOL_V1.json",
          "outputs/N72R21R2/availability/CURRENT_AXIS_PROTOCOL_V1.json",
          "outputs/N72R21R2/availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json",
          "outputs/N72R21R2/training/MAIN_EXECUTION_PROTOCOL_V1.json"}


def prepare(parent):
    assert git("branch", "--show-current").decode().strip() == BRANCH
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    git("merge-base", "--is-ancestor", parent, "HEAD")
    old, new = entries(parent), entries("HEAD")
    assert not old.keys() - new.keys()
    changes = [new[p] for p in sorted(new) if new[p] != old.get(p)]
    for entry in changes:
        path = entry["path"]
        assert path.startswith(("scripts/n72r21r2_", "tests/test_n72r21r2_", "docs/N72R21R2", "sam3_intermot/one_click/")) or path in STATIC, path
        assert entry["type"] == "blob" and Path(path).suffix in (".py", ".md", ".json")
        raw = git("cat-file", "blob", entry["sha"])
        raw.decode("utf-8")
        assert len(raw) < 128 * 1024
        entry["bytes"] = len(raw)
    head = git("rev-parse", "HEAD").decode().strip()
    payload = commit_payload(git("cat-file", "commit", head))
    record = {"repository": "LYQ1107/InterMOT", "branch": BRANCH, "remote_parent": parent,
              "base_tree": git("rev-parse", parent + "^{tree}").decode().strip(), "target_tree": payload["tree"],
              "original_local_head": head, "message": payload["message"], "changes": changes,
              "force": False, "require_fresh_expected_sha_lease": True, "no_bulk_weights_data_media_traces": True,
              "scientific_goal_complete": False}
    save("PREPARED.json", record)
    print(json.dumps({"files": len(changes), "bytes": sum(r["bytes"] for r in changes), "target_tree": payload["tree"], "original_head": head}))


def adopt(remote):
    prepared = read_json(DIRECTORY / "PREPARED.json")
    assert read_json(DIRECTORY / "IMPORTED.json")["sha"] == remote
    assert git("rev-parse", "HEAD").decode().strip() == prepared["original_local_head"]
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    assert git("rev-parse", remote + "^{tree}").decode().strip() == prepared["target_tree"]
    git("merge-base", "--is-ancestor", prepared["remote_parent"], remote)
    fresh = read_json(DIRECTORY / "FRESH_REMOTE_READBACK.json")
    assert fresh["object"]["sha"] == remote
    backup = "refs/heads/codex/n72r21r2-offline-" + prepared["original_local_head"][:12]
    git("update-ref", backup, prepared["original_local_head"], "0" * 40)
    git("update-ref", "refs/heads/" + BRANCH, remote, prepared["original_local_head"])
    git("update-ref", "refs/remotes/origin/" + BRANCH, remote)
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    receipt = {"utc": utcnow(), "repository": "LYQ1107/InterMOT", "branch": BRANCH, "local_HEAD": remote, "fresh_remote_HEAD": remote,
               "tree": prepared["target_tree"], "clean_worktree": True, "preserved_original_local_HEAD": prepared["original_local_head"],
               "preserved_backup_ref": backup, "same_source_tree_no_working_file_rewrite": True, "force_push": False,
               "fresh_remote_parent": prepared["remote_parent"], "code_and_necessary_docs_only": True,
               "scientific_goal_complete": False, "next_stage_authorized": False}
    save("RECEIPT.json", receipt)
    write_json("git_delivery/CHECKPOINT_MEMORY_OPEN_SET_V3.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare")
    mode.add_argument("--emit-chunk", type=int)
    mode.add_argument("--import-commit")
    mode.add_argument("--adopt")
    mode.add_argument("--save-json")
    args = parser.parse_args()
    if args.prepare: prepare(args.prepare)
    elif args.emit_chunk is not None: prior.emit_chunk(args.emit_chunk)
    elif args.import_commit: prior.import_commit(Path(args.import_commit))
    elif args.adopt: adopt(args.adopt)
    else:
        import sys
        print(save(args.save_json, json.loads(sys.stdin.read())))
