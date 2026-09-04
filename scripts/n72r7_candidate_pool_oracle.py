#!/usr/bin/env python3
"""Seal the N72R7 B0+target union-pool oracle recall decision."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORENSIC = ROOT / "outputs/N72R7/forensic"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    summary_path = FORENSIC / "candidate_source_summary.json"
    table_path = FORENSIC / "candidate_source_table.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in table_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if summary.get("status") != "PASS_N72R7_CANDIDATE_SOURCE_FORENSICS":
        raise RuntimeError("candidate-source forensic audit is not PASS")
    if len(rows) != int(summary.get("future_frame_count", -1)):
        raise RuntimeError("forensic row count mismatch")
    if any(row.get("runtime_future_gt_used") is not False for row in rows):
        raise RuntimeError("runtime future GT flag in forensic table")
    by_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_sequence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if not row.get("target_visible_posthoc"):
            continue
        action = str(row["action_type"])
        sequence = str(row["sequence"])
        for group in (by_action[action], by_sequence[sequence]):
            group["visible"] += 1
            group["b0_hit"] += int(row["b0_contains_correct_candidate_posthoc"])
            group["target_extra_hit"] += int(row["target_extra_contains_correct_candidate_posthoc"])
            group["union_hit"] += int(row["union_contains_correct_candidate_posthoc"])
    def rates(group: dict[str, int]) -> dict[str, Any]:
        denominator = group["visible"]
        return {
            **dict(group),
            "b0_recall": None if not denominator else group["b0_hit"] / denominator,
            "target_extra_recall": None if not denominator else group["target_extra_hit"] / denominator,
            "union_recall": None if not denominator else group["union_hit"] / denominator,
        }
    oracle = {
        "schema_version": "N72R7_UNION_CANDIDATE_POOL_ORACLE_V1",
        "status": "PASS_UNION_CANDIDATE_POOL_ORACLE_AUDIT",
        "event_count": summary["event_count"],
        "sequence_count": summary["sequence_count"],
        "future_frame_count": summary["future_frame_count"],
        "target_visible_frame_count": summary["target_visible_frame_count"],
        "b0_pool_recall_over_visible": summary["b0_pool_recall_over_visible"],
        "target_extra_recall_over_visible": summary["target_extra_recall_over_visible"],
        "union_pool_recall_over_visible": summary["union_pool_recall_over_visible"],
        "raw_target_row_missing_count": summary["raw_target_row_missing_count"],
        "human_gate_rejected_count": summary["human_gate_rejected_count"],
        "by_action": {key: rates(dict(value)) for key, value in sorted(by_action.items())},
        "by_sequence": {key: rates(dict(value)) for key, value in sorted(by_sequence.items())},
        "decision": "USE_B0_FULL_POOL_FIRST_AND_ADD_TARGET_SESSION_ROWS_WHEN_AVAILABLE",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "inputs": {
            "candidate_source_summary": str(summary_path),
            "candidate_source_summary_sha256": sha256_file(summary_path),
            "candidate_source_table": str(table_path),
            "candidate_source_table_sha256": sha256_file(table_path),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(FORENSIC / "union_pool_oracle.json", oracle)
    stage = {
        "schema_version": "N72R7_STAGE_STATUS_V1",
        "stage": "N72R7-03_UNION_CANDIDATE_POOL_ORACLE_RECALL",
        "status": oracle["status"],
        "event_count": oracle["event_count"],
        "sequence_count": oracle["sequence_count"],
        "future_frame_count": oracle["future_frame_count"],
        "target_visible_frame_count": oracle["target_visible_frame_count"],
        "b0_pool_recall_over_visible": oracle["b0_pool_recall_over_visible"],
        "target_extra_recall_over_visible": oracle["target_extra_recall_over_visible"],
        "union_pool_recall_over_visible": oracle["union_pool_recall_over_visible"],
        "decision": oracle["decision"],
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "next_stage": "N72R7-04_CANDIDATE_POOL_ARCHITECTURE",
        "created_at_utc": oracle["created_at_utc"],
    }
    atomic_json(FORENSIC / "stage_03_status.json", stage)
    atomic_json(ROOT / "outputs/N72R7/stage_03_status.json", stage)
    print(json.dumps({"status": oracle["status"], "b0_recall": oracle["b0_pool_recall_over_visible"], "union_recall": oracle["union_pool_recall_over_visible"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
