"""Lossless cold archive of closed N72R19 JSONL noise-replay records only."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from scripts.n72r21r2_storage_archive_v1 import file_signature, restored_digest
from scripts.n72r21r2_storage_archive_readback_v1 import flush_existing
from scripts.n72r21r2_common import ROOT, HISTORY, OUT, ASSETS, GOAL, read_json, write_json, sha256, utcnow, append_log, update_status

STAGE = HISTORY / "outputs/N72R19"
TARGET = STAGE / "noise_results"
ARCHIVE = ASSETS / "storage_legacy_n72r19_noise_zstd_v2"
PREFIX = "audit/storage_legacy_noise_archive_v2"
CODE = ("scripts/n72r21r2_storage_noise_archive_v2.py",
        "scripts/n72r21r2_storage_archive_v1.py", "scripts/n72r21r2_storage_archive_readback_v1.py")


def checked_path(root, relative, *, archive=False):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".jsonl":
        raise ValueError("Only original inventoried relative JSONL noise records")
    path = root / (str(relative) + ".zst" if archive else str(relative))
    if path.is_symlink() or not path.is_file() or path.resolve() != path or path.stat().st_uid != os.getuid():
        raise ValueError("No missing, symbolic, escaped or other-user record")
    return path


def flush_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def run():
    assert TARGET.resolve() == TARGET and TARGET.is_dir()
    assert ARCHIVE.resolve().is_relative_to(ASSETS.resolve())
    assert read_json(STAGE / "stage_status.json")["status"] == "FINAL_RESULT_WRITTEN"
    assert (STAGE / "FINAL_RESULT.json").is_file()
    plan_path = OUT / PREFIX / "PLAN.json"
    if not plan_path.exists():
        files = sorted(TARGET.rglob("*.jsonl"))
        if not files: raise ValueError("No original inventory: do not invent completion")
        for p in files: checked_path(TARGET, str(p.relative_to(TARGET)))
        protected = sorted(p for p in STAGE.iterdir() if p.is_file())
        write_json(PREFIX + "/PLAN.json", {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
            "user_authorization": "清理自己的项目空间", "target": str(TARGET), "archive": str(ARCHIVE),
            "source_sha256": {name: sha256(ROOT / name) for name in CODE},
            "protected_closed_stage_metadata_sha256": {str(p): sha256(p) for p in protected},
            "current_R2_source_outputs_and_R1_scripts_no_noise_record_reference_observed": True,
            "filesystem_free_bytes_before": shutil.disk_usage(TARGET).free,
            "entries": [{"relative": str(p.relative_to(TARGET)), "signature": file_signature(p)} for p in files],
            "no_training_inference_GT_interpretation_or_scientific_result_change": True})
    plan = read_json(plan_path)
    assert plan["target"] == str(TARGET) and plan["archive"] == str(ARCHIVE)
    assert plan["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    for path, digest in plan["protected_closed_stage_metadata_sha256"].items():
        assert sha256(path) == digest
    rows = []
    for entry in plan["entries"]:
        relative = entry["relative"]
        src, dst = TARGET / relative, ARCHIVE / (relative + ".zst")
        prepared_path = OUT / PREFIX / "prepared" / (relative + ".json")
        receipt_path = OUT / PREFIX / "completed" / (relative + ".json")
        if receipt_path.exists():
            row = read_json(receipt_path)
            assert not src.exists() and sha256(checked_path(ARCHIVE, relative, archive=True)) == row["archive_sha256"]
        else:
            if prepared_path.exists(): row = read_json(prepared_path)
            else:
                checked_path(TARGET, relative)
                assert file_signature(src) == entry["signature"]
                if shutil.disk_usage(TARGET).free < src.stat().st_size + (64 << 20):
                    raise RuntimeError("Not enough for one bounded corrective compressed copy")
                if dst.exists(): raise FileExistsError("Keep any unsealed archive; never overwrite")
                original_sha = sha256(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["zstd", "-q", "-3", "-T1", "--check", str(src), "-o", str(dst)], check=True)
                digest, size = restored_digest(dst)
                assert digest == original_sha and size == entry["signature"]["bytes"]
                assert file_signature(src) == entry["signature"] and sha256(src) == original_sha
                flush_existing(dst); flush_directory(dst.parent)
                row = {"goal": GOAL, "utc": utcnow(), "source": str(src), "source_signature": entry["signature"],
                    "archive": str(dst), "archive_sha256": sha256(dst), "original_sha256": original_sha,
                    "restored_sha256": digest, "restored_bytes": size, "archive_bytes": dst.stat().st_size,
                    "lossless_round_trip_and_archive_fsync_before_unlink": True, "original_uncompressed_copy_removed": False}
                write_json(str(prepared_path.relative_to(OUT)), row)
                flush_existing(prepared_path); flush_directory(prepared_path.parent)
            assert row["source"] == str(src) and row["archive"] == str(dst)
            assert row["source_signature"] == entry["signature"]
            assert sha256(checked_path(ARCHIVE, relative, archive=True)) == row["archive_sha256"]
            if src.exists():
                checked_path(TARGET, relative)
                assert file_signature(src) == entry["signature"] and sha256(src) == row["original_sha256"]
                subprocess.run(["unlink", str(src)], check=True)
                flush_directory(src.parent)
            row = {**row, "original_uncompressed_copy_removed": True, "completed_utc": utcnow()}
            write_json(str(receipt_path.relative_to(OUT)), row)
            flush_existing(receipt_path); flush_directory(receipt_path.parent)
        # Independently decode even an already completed receipt on resume.
        assert restored_digest(dst) == (row["original_sha256"], row["restored_bytes"])
        rows.append(row)
        if len(rows) % 10 == 0: print({"verified_noise_record_archives": len(rows), "free_GiB": shutil.disk_usage(TARGET).free / (1 << 30)}, flush=True)
    for path, digest in plan["protected_closed_stage_metadata_sha256"].items(): assert sha256(path) == digest
    summary = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "status": "COMPLETE_LOSSLESS_CLOSED_NOISE_RECORD_ARCHIVE",
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)}, "archived_files": len(rows),
        "original_bytes": sum(r["restored_bytes"] for r in rows), "archive_bytes": sum(r["archive_bytes"] for r in rows),
        "current_free_bytes": shutil.disk_usage(TARGET).free, "all_original_bytes_round_trip_SHA_verified": True,
        "historical_plain_record_paths_require_on_demand_restore": True,
        "datasets_checkpoints_manifests_final_results_or_other_projects_deleted": False,
        "scientific_goal_complete": False, "next_stage_authorized": False}
    path = write_json(PREFIX + "/COMPLETE.json", summary)
    flush_existing(path); flush_directory(path.parent)
    update_status(latest_project_noise_archive={"path": str(path), "sha256": sha256(path)}, all_current_working_changes_published=False)
    append_log("LOSSLESS_CLOSED_N72R19_NOISE_RECORD_ARCHIVE", path=str(path), files=len(rows), all_bytes_preserved=True)
    print({k: summary[k] for k in ("status", "archived_files", "original_bytes", "archive_bytes", "current_free_bytes")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    run()
