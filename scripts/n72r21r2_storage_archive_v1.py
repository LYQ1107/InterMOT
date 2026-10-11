"""Corrective lossless cold archive of a closed legacy evaluation cache.

User-authorized InterMOT-only storage reclamation, never result erasure.
Each original byte stream has a verified zstd copy and sealed recovery receipt
BEFORE its uncompressed copy is unlinked. No scientific interpretation or GT
evaluation; no weights, manifests, current inputs or other project targets.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from scripts.n72r21r2_common import (
    ROOT, HISTORY, OUT, ASSETS, GOAL, read_json, write_json, sha256,
    utcnow, append_log, update_status,
)

TARGET = HISTORY / "outputs/N72R19R1/metrics/eval_cache"
ARCHIVE = ASSETS / "storage_legacy_n72r19r1_eval_cache_zstd_v1"
PLAN = "audit/storage_legacy_cache_archive_v1/PLAN.json"
CODE = "scripts/n72r21r2_storage_archive_v1.py"


def file_signature(path):
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns, "mode": s.st_mode, "uid": s.st_uid}


def restored_digest(path):
    digest = hashlib.sha256()
    count = 0
    process = subprocess.Popen(["zstd", "-q", "-d", "-c", str(path)],
                               stdout=subprocess.PIPE)
    try:
        for block in iter(lambda: process.stdout.read(1 << 20), b""):
            digest.update(block)
            count += len(block)
    finally:
        process.stdout.close()
    if process.wait() != 0:
        raise ValueError("Failed archive decode: keep original and failed archive")
    return digest.hexdigest(), count


def checked_source(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise ValueError("Only inventoried legacy JSON cache entries")
    path = TARGET / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("No symlink, missing or escaped cache entry")
    if path.stat().st_uid != os.getuid():
        raise ValueError("No other user's files")
    return path


def run():
    assert TARGET.resolve() == TARGET and TARGET.is_dir()
    assert ARCHIVE.resolve().is_relative_to(ASSETS.resolve())
    status_path = HISTORY / "outputs/N72R19R1/stage_status.json"
    status = read_json(status_path)
    assert status["status"] == "COMPLETE_FAIL_CLEAN_PRESERVATION"
    assert status["stop_rule_applied"] is True
    protected = [status_path, HISTORY / "outputs/N72R19R1/FINAL_RESULT.json",
                 HISTORY / "outputs/N72R19R1/protocol.json",
                 HISTORY / "outputs/N72R19R1/corruption_manifest.json",
                 HISTORY / "scripts/n72r19r1_evaluate.py"]
    # Only an existing closed legacy output cache, never dataset image/GT reads.
    if not (OUT / PLAN).exists():
        files = sorted(TARGET.rglob("*.json"))
        if not files:
            raise ValueError("No original cache inventory; do not invent completion")
        rows = []
        for p in files:
            relative = str(p.relative_to(TARGET))
            checked_source(relative)
            rows.append({"relative": relative, "signature": file_signature(p)})
        write_json(PLAN, {"goal": GOAL, "utc": utcnow(), "target": str(TARGET),
            "archive": str(ARCHIVE), "source_sha256": {CODE: sha256(ROOT / CODE)},
            "user_authorization": "清理自己的项目空间",
            "no_matching_legacy_cache_open_file_observed": True,
            "lsof_unrelated_FUSE_warnings_NOT_universal_open_file_guarantee": True,
            "current_R2_outputs_scripts_and_joint_MOT_code_have_no_cache_reference": True,
            "filesystem_free_bytes_before": shutil.disk_usage(TARGET).free,
            "protected_sha256": {str(p): sha256(p) for p in protected},
            "entries": rows, "recoverability": "zstd -d -c ARCHIVE > ORIGINAL_PATH; exact original byte SHA retained",
            "corrective_bounded_storage_cleanup_not_new_experiment": True})
    plan = read_json(OUT / PLAN)
    assert plan["target"] == str(TARGET) and plan["archive"] == str(ARCHIVE)
    assert plan["source_sha256"][CODE] == sha256(ROOT / CODE)
    assert all(sha256(p) == h for p, h in plan["protected_sha256"].items())
    completed = []
    for row in plan["entries"]:
        relative = row["relative"]
        destination = ARCHIVE / (relative + ".zst")
        pending = "audit/storage_legacy_cache_archive_v1/prepared/" + relative
        receipt = "audit/storage_legacy_cache_archive_v1/completed/" + relative
        src = TARGET / relative
        if (OUT / receipt).exists():
            sealed = read_json(OUT / receipt)
            assert not src.exists() and sha256(destination) == sealed["archive_sha256"]
            completed.append(sealed)
            continue
        if (OUT / pending).exists():
            prepared = read_json(OUT / pending)
        else:
            src = checked_source(relative)
            assert file_signature(src) == row["signature"]
            # Storage restoration must work below the floor; temporary output
            # is bounded by one existing file, not another full cache copy.
            if shutil.disk_usage(TARGET).free < src.stat().st_size + (64 << 20):
                raise RuntimeError("Insufficient even for one corrective archive file")
            original_sha = sha256(src)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("Unsealed archive retained: inspect/version, never overwrite")
            subprocess.run(["zstd", "-q", "-3", "-T1", "--check", str(src),
                            "-o", str(destination)], check=True)
            digest, size = restored_digest(destination)
            assert digest == original_sha and size == src.stat().st_size
            assert file_signature(src) == row["signature"] and sha256(src) == original_sha
            prepared = {"goal": GOAL, "utc": utcnow(), "source": str(src),
                "source_signature": row["signature"], "original_sha256": original_sha,
                "archive": str(destination), "archive_sha256": sha256(destination),
                "archive_bytes": destination.stat().st_size,
                "restored_sha256": digest, "restored_bytes": size,
                "lossless_round_trip_verified_before_unlink": True,
                "original_uncompressed_copy_removed": False}
            write_json(pending, prepared)
        assert prepared["source"] == str(src) and prepared["archive"] == str(destination)
        assert sha256(destination) == prepared["archive_sha256"]
        if src.exists():
            checked_source(relative)
            assert file_signature(src) == prepared["source_signature"]
            assert sha256(src) == prepared["original_sha256"]
            # Explicit validated single file, not a recursive or glob deletion.
            subprocess.run(["unlink", str(src)], check=True)
        assert not src.exists()
        sealed = {**prepared, "completed_utc": utcnow(),
                  "original_uncompressed_copy_removed": True}
        write_json(receipt, sealed)
        completed.append(sealed)
        if len(completed) % 10 == 0:
            print({"verified_lossless_archived_files": len(completed),
                   "free_GiB": shutil.disk_usage(TARGET).free / (1 << 30)}, flush=True)
    assert len(completed) == len(plan["entries"])
    assert all(sha256(p) == h for p, h in plan["protected_sha256"].items())
    summary = {"goal": GOAL, "utc": utcnow(), "status": "COMPLETE_LOSSLESS_LEGACY_CACHE_ARCHIVE",
        "plan": {"path": str(OUT / PLAN), "sha256": sha256(OUT / PLAN)},
        "archived_files": len(completed), "original_bytes": sum(c["restored_bytes"] for c in completed),
        "archive_bytes": sum(c["archive_bytes"] for c in completed),
        "current_free_bytes": shutil.disk_usage(TARGET).free,
        "all_original_byte_streams_preserved_with_exact_round_trip_SHA": True,
        "unique_results_weights_manifests_datasets_or_other_projects_deleted": False,
        "scientific_goal_complete": False, "restore_required_for_old_plain_cache_paths": True}
    path = write_json("audit/storage_legacy_cache_archive_v1/COMPLETE.json", summary)
    update_status(latest_project_storage_archive={"path": str(path), "sha256": sha256(path)},
                  project_storage_archive_status=summary["status"],
                  all_current_working_changes_published=False)
    append_log("USER_AUTHORIZED_INTERMOT_ONLY_LOSSLESS_LEGACY_CACHE_ARCHIVE", path=str(path),
               files=len(completed), historical_byte_streams_preserved=True)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    run()
