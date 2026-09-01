"""Preserve the post-IoU-repair N70 mapping state before public-absence repair."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N70"
DEST = OUT / "attempts/mapping_cache_attempt2_public_absence"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    names = (
        "diagnosis/mapping_audit.jsonl",
        "diagnosis/mapping_summary.json",
        "diagnosis/mapping_fixture_results.json",
        "cache/candidate_cache_manifest.json",
        "cache/candidate_cache_audit.json",
        "stage_01_status.json",
        "stage_02_status.json",
    )
    copied = []
    for relative in names:
        source = OUT / relative
        record = {"path": str(source), "exists": source.is_file(), "sha256": sha256(source)}
        if source.is_file():
            destination = DEST / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record["preserved_copy"] = str(destination)
            record["bytes"] = source.stat().st_size
        copied.append(record)
    summary = json.loads((OUT / "diagnosis/mapping_summary.json").read_text(encoding="utf-8"))
    audit_rows = [
        json.loads(line)
        for line in (OUT / "diagnosis/mapping_audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    incomplete = [row for row in audit_rows if row.get("candidate_chain_complete") is not True]
    null_public_candidates = sum(
        sum(item.get("public_id") is None for item in row.get("candidate_mappings", []))
        for row in audit_rows
    )
    target_public_null = sum(
        bool(row.get("target_candidate_present") and row.get("target_row") is not None
             and row.get("candidate_mappings", [])[int(row["target_row"])].get("public_id") is None)
        for row in audit_rows
    )
    atomic_json(OUT / "attempts/n70_stage01_mapping_attempt2_public_absence.json", {
        "schema": "N70_MAPPING_ATTEMPT2_PUBLIC_ABSENCE_V1",
        "status": "PASS_REPAIR_WITH_EXPLICIT_PUBLIC_ASSIGNMENT_ABSENCE_PENDING_CONTRACT_FIX",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "N70_STAGE_01_MAPPING_PROVENANCE",
        "root_cause": "After fixing degenerate-box equality, remaining chain-incomplete rows have explicit native/local/global mappings but no N54 public assignment for unmatched candidates; no public ID may be synthesized.",
        "evidence": {
            "mapping_summary_status": summary.get("status"),
            "mapping_error_frames": summary.get("mapping_error_frames"),
            "chain_complete_frames": summary.get("chain_complete_frames"),
            "incomplete_frame_rows": len(incomplete),
            "null_public_candidate_occurrences": null_public_candidates,
            "target_candidate_with_null_public_assignment_frames": target_public_null,
            "target_candidate_absent_frames": summary.get("target_candidate_absent_frames"),
        },
        "preserved_files": copied,
        "repair_scope": "Represent N54 public assignment absence with an explicit status/provenance while retaining public_id=null; distinguish it from native/local/global mapping failure.",
        "forbidden_action": "Do not fill null public IDs from GT, chunk-local IDs, candidate order, or a guessed offset.",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
    })
    print(json.dumps({"status": "PRESERVED", "artifact": str(OUT / "attempts/n70_stage01_mapping_attempt2_public_absence.json"), "incomplete_frame_rows": len(incomplete), "null_public_candidate_occurrences": null_public_candidates, "target_public_null_frames": target_public_null}, sort_keys=True))


if __name__ == "__main__":
    main()
