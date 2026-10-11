"""Independent full byte-stream readback of the user-authorized cold archive.

Never deletes, restores or interprets historical scientific data. Flushes the
existing archives and receipts, then verifies every compressed and decoded SHA.
"""
import argparse
import os
from pathlib import Path
import shutil

from scripts import n72r21r2_storage_archive_v1 as archive
from scripts.n72r21r2_common import (
    ROOT, OUT, GOAL, read_json, write_json, sha256, utcnow, update_status, append_log,
)

SELF = "scripts/n72r21r2_storage_archive_readback_v1.py"
RESULT = "audit/storage_legacy_cache_archive_v1/ACTUAL_READBACK_V1.json"


def checked_archive_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise ValueError("Only original inventoried relative JSON paths")
    path = archive.ARCHIVE / (str(relative) + ".zst")
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("No missing, symbolic or escaped archive")
    if path.stat().st_uid != os.getuid():
        raise ValueError("No other user's archives")
    return path


def flush_existing(path):
    # No rewrite: make already verified data durable before sealing readback.
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def run():
    if (OUT / RESULT).exists():
        raise FileExistsError("Preserve actual readback; version a later audit")
    plan_path = OUT / archive.PLAN
    complete_path = OUT / "audit/storage_legacy_cache_archive_v1/COMPLETE.json"
    plan, complete = read_json(plan_path), read_json(complete_path)
    assert plan["goal"] == complete["goal"] == GOAL
    assert plan["target"] == str(archive.TARGET) and plan["archive"] == str(archive.ARCHIVE)
    assert plan["source_sha256"] == {archive.CODE: sha256(ROOT / archive.CODE)}
    assert complete["plan"] == {"path": str(plan_path), "sha256": sha256(plan_path)}
    assert complete["status"] == "COMPLETE_LOSSLESS_LEGACY_CACHE_ARCHIVE"
    relative_paths = [r["relative"] for r in plan["entries"]]
    assert len(relative_paths) == len(set(relative_paths)) == complete["archived_files"]
    expected = {checked_archive_path(relative) for relative in relative_paths}
    assert set(archive.ARCHIVE.rglob("*.zst")) == expected
    refs, rows = {}, []
    directories = set()
    def attest(path):
        flush_existing(path)
        refs[str(path)] = sha256(path)
        directories.add(Path(path).parent)
    attest(plan_path)
    attest(complete_path)
    for entry in plan["entries"]:
        relative = entry["relative"]
        destination = checked_archive_path(relative)
        prepared_path = OUT / "audit/storage_legacy_cache_archive_v1/prepared" / relative
        receipt_path = OUT / "audit/storage_legacy_cache_archive_v1/completed" / relative
        prepared, receipt = read_json(prepared_path), read_json(receipt_path)
        assert receipt == {**prepared, "completed_utc": receipt["completed_utc"],
                           "original_uncompressed_copy_removed": True}
        assert prepared["original_uncompressed_copy_removed"] is False
        assert prepared["lossless_round_trip_verified_before_unlink"] is True
        assert receipt["source"] == str(archive.TARGET / relative)
        assert receipt["source_signature"] == entry["signature"]
        assert receipt["archive"] == str(destination)
        assert not (archive.TARGET / relative).exists()
        attest(destination)
        assert refs[str(destination)] == receipt["archive_sha256"]
        digest, size = archive.restored_digest(destination)
        assert digest == receipt["original_sha256"] == receipt["restored_sha256"]
        assert size == receipt["restored_bytes"] == entry["signature"]["bytes"]
        assert destination.stat().st_size == receipt["archive_bytes"]
        attest(prepared_path)
        attest(receipt_path)
        rows.append({"relative": relative, "original_sha256": digest,
                     "restored_bytes": size, "archive_bytes": destination.stat().st_size,
                     "archive_allocated_bytes": destination.stat().st_blocks * 512})
        if len(rows) % 30 == 0:
            print({"independently_decoded_and_verified": len(rows)}, flush=True)
    for path, digest in plan["protected_sha256"].items():
        assert sha256(path) == digest
        refs[path] = digest
    for directory in sorted(directories):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    original_bytes = sum(r["restored_bytes"] for r in rows)
    archive_bytes = sum(r["archive_bytes"] for r in rows)
    assert original_bytes == complete["original_bytes"]
    assert archive_bytes == complete["archive_bytes"]
    result = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
        "status": "PASS_ALL_ARCHIVES_INDEPENDENT_EXACT_BYTE_READBACK",
        "source_sha256": {SELF: sha256(ROOT / SELF), archive.CODE: sha256(ROOT / archive.CODE)},
        "archived_files": len(rows), "original_bytes": original_bytes,
        "archive_bytes": archive_bytes, "logical_bytes_reclaimed": original_bytes - archive_bytes,
        "archive_allocated_bytes": sum(r["archive_allocated_bytes"] for r in rows),
        "filesystem_free_bytes_at_readback": shutil.disk_usage(archive.TARGET).free,
        "concurrent_disk_activity_prevents_attributing_df_delta_to_cleanup_alone": True,
        "all_360_archives_and_receipts_fsynced": len(rows) == 360,
        "all_original_byte_streams_independently_decoded_and_SHA_verified": True,
        "datasets_checkpoints_manifests_final_results_other_projects_untouched": True,
        "historical_plain_cache_paths_require_on_demand_restore": True,
        "scientific_goal_complete": False, "next_stage_authorized": False,
        "evidence_sha256": refs, "entries": rows}
    path = write_json(RESULT, result)
    flush_existing(path)
    update_status(latest_project_storage_archive_readback={"path": str(path), "sha256": sha256(path)},
                  all_current_working_changes_published=False)
    append_log("LOSSLESS_COLD_ARCHIVE_INDEPENDENT_READBACK", path=str(path),
               sha256=sha256(path), files=len(rows), logical_bytes_reclaimed=original_bytes - archive_bytes)
    print({k: result[k] for k in ("status", "archived_files", "logical_bytes_reclaimed",
                                 "filesystem_free_bytes_at_readback")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    run()
