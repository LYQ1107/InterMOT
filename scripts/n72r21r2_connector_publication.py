"""R2-scoped identical-tree API publication; preserve all offline ancestry."""
import argparse
import json
from pathlib import Path
import subprocess
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, utcnow
from scripts.n72r20r4_publish_git import commit_payload
from scripts.n72r21r1_connector_publication_round import verified_dates
from scripts.n72r21_connector_publication import remote_commit_bytes

BRANCH = "codex/n72r21r2-event-causal-mot-generalization"
DIRECTORY = ASSETS / "git_connector_publication/checkpoint_M3_M4_V1"


def git(*args, input=None):
    return subprocess.run(["git", *args], cwd=ROOT, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30).stdout


def save(name, value):
    path = DIRECTORY / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) == value:
            return path
        raise FileExistsError("Preserve publication attempts; version new rounds")
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def entries(ref):
    result = {}
    for line in git("ls-tree", "-r", ref).decode().splitlines():
        fields, path = line.split("\t", 1)
        mode, kind, h = fields.split()
        result[path] = {"path": path, "mode": mode, "type": kind, "sha": h}
    return result


def prepare(parent):
    assert git("branch", "--show-current").decode().strip() == BRANCH
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    git("merge-base", "--is-ancestor", parent, "HEAD")
    old, new = entries(parent), entries("HEAD")
    assert not old.keys() - new.keys(), "No deletions in scoped checkpoint delivery"
    changes = [new[p] for p in sorted(new) if new[p] != old.get(p)]
    for entry in changes:
        path = entry["path"]
        allowed = path.startswith(("scripts/n72r21r2_", "tests/test_n72r21r2_", "docs/N72R21R2", "sam3_intermot/one_click/", "sam3_intermot/evaluation/")) or path == "sam3_intermot/backend/sam3_trim_schema_compat.py" or path in (
            "outputs/N72R21R2/FINAL_GOAL.json", "outputs/N72R21R2/PREREGISTRATION.json", "outputs/N72R21R2/events/COUNTERFACTUAL_PROTOCOL_V1.json",
            "outputs/N72R21R2/events/WINDOW_TRACKEVAL_PROTOCOL_V1.json", "outputs/N72R21R2/simple/SIMPLE_PROTOCOL_V1.json")
        assert allowed and entry["type"] == "blob" and Path(path).suffix in (".py", ".md", ".json"), path
        raw = git("cat-file", "blob", entry["sha"])
        raw.decode("utf-8")
        assert len(raw) < 128 * 1024, "No bulk asset in code-only publication"
        entry["bytes"] = len(raw)
    head = git("rev-parse", "HEAD").decode().strip()
    payload = commit_payload(git("cat-file", "commit", head))
    record = {"repository": "LYQ1107/InterMOT", "branch": BRANCH, "remote_parent": parent,
              "base_tree": git("rev-parse", parent + "^{tree}").decode().strip(), "target_tree": payload["tree"],
              "original_local_head": head, "message": payload["message"], "changes": changes,
              "force": False, "no_bulk_weights_data_media_traces": True, "scientific_goal_complete": False}
    save("PREPARED.json", record)
    print(json.dumps({"files": len(changes), "bytes": sum(r["bytes"] for r in changes), "target_tree": payload["tree"], "original_head": head}))


def emit_chunk(chunk):
    selected = read_json(DIRECTORY / "PREPARED.json")["changes"][chunk * 6:(chunk + 1) * 6]
    print(json.dumps([{"path": r["path"], "mode": r["mode"], "type": "blob", "content": git("cat-file", "blob", r["sha"]).decode("utf-8")} for r in selected]))


def import_commit(path):
    record = read_json(path)
    prepared = read_json(DIRECTORY / "PREPARED.json")
    assert record["tree"]["sha"] == prepared["target_tree"]
    assert [p["sha"] for p in record["parents"]] == [prepared["remote_parent"]]
    canonical = verified_dates(record)
    raw = remote_commit_bytes(canonical)
    imported = git("hash-object", "-t", "commit", "-w", "--stdin", input=raw).decode().strip()
    assert imported == record["sha"]
    save("CANONICAL_RAW_METADATA_VERIFIED.json", canonical)
    save("IMPORTED.json", {"sha": imported, "tree": record["tree"]["sha"]})
    print(json.dumps({"imported_exact_canonical_commit": imported, "tree_equal": True}))


def adopt(remote):
    prepared = read_json(DIRECTORY / "PREPARED.json")
    assert read_json(DIRECTORY / "IMPORTED.json")["sha"] == remote
    assert git("rev-parse", "HEAD").decode().strip() == prepared["original_local_head"]
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    assert git("rev-parse", remote + "^{tree}").decode().strip() == prepared["target_tree"]
    backup = "refs/heads/codex/n72r21r2-offline-" + prepared["original_local_head"][:12]
    git("update-ref", backup, prepared["original_local_head"], "0" * 40)
    git("update-ref", "refs/heads/" + BRANCH, remote, prepared["original_local_head"])
    git("update-ref", "refs/remotes/origin/" + BRANCH, remote)
    assert not git("-c", "core.fsmonitor=false", "status", "--porcelain").strip()
    receipt = {"utc": utcnow(), "repository": "LYQ1107/InterMOT", "branch": BRANCH, "local_HEAD": remote, "fresh_remote_HEAD": remote,
               "tree": prepared["target_tree"], "clean_worktree": True, "preserved_original_local_HEAD": prepared["original_local_head"],
               "preserved_backup_ref": backup, "same_source_tree_no_working_file_rewrite": True, "force_push": False,
               "code_and_necessary_docs_only": True, "scientific_goal_complete": False, "next_stage_authorized": False}
    save("RECEIPT.json", receipt)
    write_json("git_delivery/CHECKPOINT_M3_M4_V1.json", receipt)
    print(json.dumps(receipt))


def save_input(path, value):
    return save(path, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare")
    mode.add_argument("--emit-chunk", type=int)
    mode.add_argument("--import-commit")
    mode.add_argument("--adopt")
    mode.add_argument("--save-json")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.emit_chunk is not None:
        emit_chunk(args.emit_chunk)
    elif args.import_commit:
        import_commit(Path(args.import_commit))
    elif args.adopt:
        adopt(args.adopt)
    else:
        import sys
        print(save_input(args.save_json, json.loads(sys.stdin.read())))
