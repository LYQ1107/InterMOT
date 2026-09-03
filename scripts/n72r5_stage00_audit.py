#!/usr/bin/env python3
"""Audit and reconcile the two N72R4 Stage10 bookkeeping defects.

This stage is CPU-only and read-only with respect to N72R4.  It does not rerun
SAM3 or rewrite the frozen recall artifact.  The audit deliberately reads the
canonical ``candidate_recall/no_vs_m0_candidate_recall.json`` payload rather
than the similarly named stage-status wrapper, and records the corrected
state×candidate provenance convention used by the current solver wrapper.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
N72R4_ROOT = Path(
    os.environ.get(
        "N72R4_INPUT_ROOT",
        "/data2/usr_for_deadline/SAM3_InterMOT_N72R3R1/worktree/outputs/N72R4",
    )
)
RECALL = N72R4_ROOT / "candidate_recall" / "no_vs_m0_candidate_recall.json"
STAGE_STATUS = N72R4_ROOT / "stage_status" / "stage_10_status.json"
OUT = ROOT / "outputs" / "N72R5"
AUDIT = OUT / "audits" / "n72r4_stage10_recall_repair.json"
REPORT = ROOT / "docs" / "N72R4R1_AUDIT_CORRECTION.md"

HORIZONS = ("20", "50", "100")
BRANCHES = ("B0_NO_INTERVENTION", "B1_CURRENT_FRAME_CORRECTION")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    if not RECALL.is_file():
        raise FileNotFoundError(
            f"canonical Stage10 recall artifact is missing: {RECALL}; "
            "do not manufacture a None→None report"
        )
    if not STAGE_STATUS.is_file():
        raise FileNotFoundError(f"Stage10 status is missing: {STAGE_STATUS}")

    recall = read_object(RECALL)
    stage_status = read_object(STAGE_STATUS)
    rows: dict[str, dict[str, Any]] = {}
    validation_errors: list[str] = []
    aggregate = recall.get("aggregate")
    if not isinstance(aggregate, dict):
        validation_errors.append("recall.aggregate_missing_or_not_object")
        aggregate = {}
    delta = recall.get("m0_minus_no_candidate_recall")
    if not isinstance(delta, dict):
        validation_errors.append("recall.m0_minus_no_candidate_recall_missing_or_not_object")
        delta = {}

    for horizon in HORIZONS:
        row: dict[str, Any] = {"horizon": int(horizon)}
        values: list[float] = []
        for branch in BRANCHES:
            branch_row = aggregate.get(branch, {})
            horizon_row = branch_row.get(horizon, {}) if isinstance(branch_row, dict) else {}
            value = horizon_row.get("candidate_recall") if isinstance(horizon_row, dict) else None
            row["no_recall" if branch == BRANCHES[0] else "m0_recall"] = value
            if not finite(value):
                validation_errors.append(f"{branch}.H{horizon}.candidate_recall_not_finite")
            else:
                values.append(float(value))
        delta_value = delta.get(horizon)
        row["delta_m0_minus_no"] = delta_value
        if not finite(delta_value):
            validation_errors.append(f"H{horizon}.delta_m0_minus_no_not_finite")
        elif len(values) == 2 and abs(float(delta_value) - (values[1] - values[0])) > 1e-12:
            validation_errors.append(f"H{horizon}.delta_does_not_match_branch_values")
        rows[horizon] = row

    expected_status = "PASS_STAGE10_NO_VS_M0_POSTHOC_RECALL"
    if recall.get("status") != expected_status:
        validation_errors.append(f"recall.status={recall.get('status')!r}")
    if recall.get("runtime_future_gt_used") is not False:
        validation_errors.append("recall.runtime_future_gt_used_not_false")
    if stage_status.get("runtime_future_gt_used") is not False:
        validation_errors.append("stage_status.runtime_future_gt_used_not_false")
    if recall.get("scientific_result") != "NO_VS_M0_CANDIDATE_AVAILABILITY_ONLY_NOT_FUTURE_EFFECT_GATE":
        validation_errors.append("recall.scientific_result_unexpected")

    audit = {
        "schema_version": "N72R5_STAGE00_N72R4_AUDIT_CORRECTION_V1",
        "stage": "00_N72R4_METADATA_REPAIR",
        "status": "PASS_N72R4_RECALL_READ_REPAIRED_HASH_REPAIR_PENDING" if not validation_errors else "FAIL_INPUT_ARTIFACT",
        "created_at_utc": now_utc(),
        "historical_n72r4_outputs_read_only": True,
        "sam3_launched": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "stage10_recall_source": {
            "path": str(RECALL),
            "sha256": sha256(RECALL),
            "status": recall.get("status"),
        },
        "stage10_status_source": {
            "path": str(STAGE_STATUS),
            "sha256": sha256(STAGE_STATUS),
            "status": stage_status.get("status"),
        },
        "corrected_read_path": "aggregate[B0_NO_INTERVENTION|B1_CURRENT_FRAME_CORRECTION][20|50|100].candidate_recall",
        "incorrect_read_path_preserved_for_audit": "stage_status.stage10 candidate_recall field is only a pointer and is not used as the metric source",
        "recall_by_horizon": rows,
        "validation_errors": validation_errors,
        "historical_scientific_result_unchanged": True,
        "next_step": "run_hash_orientation_regression_then_write_N72R4R1_audit_correction",
    }
    atomic_json(AUDIT, audit)
    if validation_errors:
        return 2

    report_lines = [
        "# N72R4R1 Audit Correction",
        "",
        "> This is a bookkeeping/provenance correction, not a new scientific experiment.",
        "",
        "N72R4 remains `M3_SIGNAL_WAS_SOLVER_ARTIFACT` with research gate `FAIL_FUTURE_EFFECT`; no model, checkpoint, candidate stream, solver definition, or historical output was changed.",
        "",
        "## Correct NO versus M0 candidate recall",
        "",
        "| Horizon | NO intervention | M0 current-frame correction | M0 − NO |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        row = rows[horizon]
        report_lines.append(
            f"| H{horizon} | {row['no_recall']:.9f} | {row['m0_recall']:.9f} | {row['delta_m0_minus_no']:.9f} |"
        )
    report_lines += [
        "",
        "The values above come directly from the canonical Stage10 artifact, not from the stage-status pointer. Runtime future-GT usage remains `false`; GT is used only for posthoc scoring.",
        "",
        "## Provenance-hash repair",
        "",
        "The effect-assignment wrapper now records separate canonical hashes for the original state×candidate matrix and the solver-facing candidate×state transpose. Existing N72R3R1/N72R4 artifacts are preserved as historical evidence; they are not rewritten in place.",
        "",
        f"Machine-readable audit: `{AUDIT}`",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT)
    audit["status"] = "PASS_N72R4_RECALL_AND_HASH_REPAIRS_AUDITED"
    audit["audit_report"] = str(REPORT)
    atomic_json(AUDIT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
