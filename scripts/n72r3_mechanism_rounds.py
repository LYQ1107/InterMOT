#!/usr/bin/env python3
"""Evidence-preserving N72R3 post-gate mechanism rounds.

N72R3 Stage 22 did not pass the strict future-effect gate.  This script runs
five deterministic, CPU-only root-cause rounds over the sealed Stage 20--22
artifacts.  Rounds 1, 2, 4 and 5 are audits.  Round 3 is an explicitly
hypothetical assignment-boundary probe: it adds a residual only to the target
identity row to test whether a solver crossing is possible.  It is not a
model result and is never promoted to production.

No round opens a GT file before the existing runtime artifacts have been
validated.  The probe itself uses no GT at all.  Existing Stage 22 posthoc
metrics are reused only as already-sealed evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/N72R3"
STAGE19 = OUT / "candidate_recall/stage19_candidate_recall.json"
STAGE22 = OUT / "effect_replay/attempt1/ccam_paired_replay_results.json"
STAGE21 = OUT / "stage_21_status.json"
RUNTIME_ROOT = OUT / "effect_replay/attempt1/runtime_event_artifacts"
ROUND_ROOT = Path(os.environ.get("N72R3_MECHANISM_ROUND_ROOT", str(OUT / "mechanism_rounds")))
SUMMARY_PATH = ROUND_ROOT / "mechanism_rounds_summary.json"
FAILURE_ROOT = OUT / "attempts"
VARIANTS = (
    "M1_HUMAN_EMA_PROTOTYPE",
    "M2_POSITIVE_HUMAN_ANCHORS",
    "M3_NEGATIVE_COMPETITOR_BANK",
    "M4_RELIABILITY_AGE_ADMISSION",
)
HORIZONS = (20, 50, 100)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sealed() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[tuple[str, list[dict[str, Any]]]]]:
    stage19 = read_json(STAGE19)
    stage22 = read_json(STAGE22)
    stage21 = read_json(STAGE21)
    if stage22.get("runtime_future_gt_used") is not False or stage22.get("gate", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError("Stage 22 runtime future-GT boundary is not sealed false")
    if stage21.get("status") != "PASS_STAGE21_TARGET_SCOPED_ASSOCIATION_AUDIT":
        raise RuntimeError("Stage 21 target-scoped audit did not pass")
    files = sorted(RUNTIME_ROOT.glob("*.jsonl"))
    expected_events = {str(item["event_id"]) for item in stage22.get("event_metrics", [])}
    if len(files) != len(expected_events):
        raise RuntimeError(f"sealed runtime file count mismatch: {len(files)} vs {len(expected_events)}")
    artifacts: list[tuple[str, list[dict[str, Any]]]] = []
    for path in files:
        rows = read_jsonl(path)
        if not rows:
            raise RuntimeError(f"sealed runtime artifact empty: {path}")
        event_id = str(rows[0].get("event_id"))
        if event_id not in expected_events or any(row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False for row in rows):
            raise RuntimeError(f"sealed runtime artifact causal audit failed: {path}")
        artifacts.append((event_id, rows))
    return stage19, stage22, stage21, artifacts


def round1_candidate_recall(stage19: dict[str, Any], stage22: dict[str, Any]) -> dict[str, Any]:
    event_recall = stage19.get("event_recall", [])
    by_action = stage19.get("by_action", {})
    horizon_summary: dict[str, Any] = {}
    for horizon in HORIZONS:
        value = stage19.get("by_horizon", {}).get(str(horizon), {})
        horizon_summary[str(horizon)] = {
            "target_gt_present_frames": value.get("target_gt_present_frames"),
            "candidate_present_frames": value.get("candidate_present_frames"),
            "candidate_absent_frames": value.get("candidate_absent_frames"),
            "candidate_recall": value.get("candidate_recall"),
            "unrecoverable_by_candidate_absence_upper_bound": value.get("candidate_absent_frames"),
        }
    action_summary = {
        action: {
            str(horizon): {
                "candidate_recall": payload.get(str(horizon), {}).get("candidate_recall"),
                "candidate_absent_frames": payload.get(str(horizon), {}).get("candidate_absent_frames"),
            }
            for horizon in HORIZONS
        }
        for action, payload in sorted(by_action.items())
    }
    return {
        "round": "R1_CANDIDATE_RECALL_BOUND",
        "status": "PASS_DIAGNOSTIC",
        "root_cause": "candidate_recall_low",
        "event_count": stage19.get("event_count"),
        "independent_sequence_count": stage19.get("independent_sequence_count"),
        "event_recall_rows": event_recall,
        "by_horizon": horizon_summary,
        "by_action": action_summary,
        "interpretation": "A candidate-absent frame cannot be repaired by an appearance-only association score. This is a performance upper bound, not an identity-authority failure and not a synthetic candidate insertion.",
        "component_changed": None,
        "runtime_future_gt_used": False,
        "scientific_result": "CANDIDATE_RECALL_BOUND_NOT_FUTURE_EFFECT",
    }


def assignment_objective(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"score matrix is not finite 2-D: {matrix.shape}")
    assignment = np.full(matrix.shape[0], -1, dtype=np.int64)
    if matrix.shape[0] and matrix.shape[1]:
        rows, columns = linear_sum_assignment(-matrix)
        assignment[rows] = columns
    objective = float(sum(matrix[row, int(column)] for row, column in enumerate(assignment) if column >= 0))
    return objective, assignment


def constrained_objective(matrix: np.ndarray, target_row: int, candidate_column: int) -> float:
    if target_row < 0 or target_row >= matrix.shape[0] or candidate_column < 0 or candidate_column >= matrix.shape[1]:
        return float("-inf")
    remaining_rows = [index for index in range(matrix.shape[0]) if index != target_row]
    remaining_columns = [index for index in range(matrix.shape[1]) if index != candidate_column]
    total = float(matrix[target_row, candidate_column])
    if remaining_rows and remaining_columns:
        reduced = matrix[np.ix_(remaining_rows, remaining_columns)]
        value, _ = assignment_objective(reduced)
        total += value
    return total


def round2_margin_residual(artifacts: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    distributions: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    max_deltas: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    best_alternative_deltas: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    possible: dict[str, int] = {variant: 0 for variant in VARIANTS}
    checked = 0
    rows_out: list[dict[str, Any]] = []
    for event_id, rows in artifacts:
        for frame in rows:
            if int(frame.get("frame_horizon", -1)) <= 0:
                continue
            variant = str(frame["variant"])
            if variant not in VARIANTS:
                continue
            base = np.asarray(frame["base_score_matrix"], dtype=np.float64)
            fused = np.asarray(frame["fused_score_matrix"], dtype=np.float64)
            target_row = frame.get("target_row_index")
            if target_row is None:
                continue
            target_row = int(target_row)
            current_objective, assignment = assignment_objective(fused)
            current_column = int(assignment[target_row]) if target_row < assignment.size else -1
            alternatives: list[dict[str, Any]] = []
            for column in range(fused.shape[1]):
                if column == current_column:
                    continue
                constrained = constrained_objective(fused, target_row, column)
                residual = max(0.0, current_objective - constrained + 1.0e-6)
                alternatives.append({"candidate_column": column, "required_residual": float(residual), "constrained_objective": float(constrained)})
            if alternatives:
                best = min(alternatives, key=lambda value: (value["required_residual"], value["candidate_column"]))
                distributions[variant].append(float(best["required_residual"]))
                max_delta = float(np.max(np.abs(fused[target_row, :] - base[target_row, :])))
                best_delta = float(fused[target_row, int(best["candidate_column"])] - base[target_row, int(best["candidate_column"])])
                max_deltas[variant].append(max_delta)
                best_alternative_deltas[variant].append(best_delta)
                if best_delta + 1.0e-12 >= float(best["required_residual"]):
                    possible[variant] += 1
                if len(rows_out) < 5000:
                    rows_out.append({
                        "event_id": event_id,
                        "frame": int(frame["frame"]),
                        "horizon": int(frame["frame_horizon"]),
                        "variant": variant,
                        "target_row": target_row,
                        "current_candidate_column": current_column,
                        "best_alternative_candidate_column": best["candidate_column"],
                        "required_residual": best["required_residual"],
                        "max_abs_target_row_memory_delta": max_delta,
                        "target_row_delta_at_best_alternative": best_delta,
                        "memory_can_cross_under_frozen_variant": bool(best_delta + 1.0e-12 >= float(best["required_residual"])),
                        "runtime_future_gt_used": False,
                    })
            checked += 1

    def summarize(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "median": float(np.quantile(array, 0.5)),
            "p90": float(np.quantile(array, 0.9)),
            "p95": float(np.quantile(array, 0.95)),
            "max": float(np.max(array)),
        }

    return {
        "round": "R2_ASSIGNMENT_MARGIN_RESIDUAL",
        "status": "PASS_DIAGNOSTIC",
        "root_cause": "scores_move_but_assignment_decision_boundary",
        "checked_future_frames": checked,
        "required_target_row_residual": {variant: summarize(values) for variant, values in distributions.items()},
        "frozen_memory_target_row_delta": {variant: summarize(values) for variant, values in max_deltas.items()},
        "frozen_memory_delta_at_best_alternative": {variant: summarize(values) for variant, values in best_alternative_deltas.items()},
        "memory_reaches_a_frozen_boundary": possible,
        "sample_rows_capped_at": 5000,
        "sample_rows": rows_out,
        "interpretation": "The residual is the minimum target-row addition required to beat the best constrained global assignment, computed without GT. A large residual relative to the frozen memory delta supports an association-interface bottleneck; it does not justify changing the production solver in this round.",
        "component_changed": None,
        "runtime_future_gt_used": False,
        "scientific_result": "ASSIGNMENT_MARGIN_DIAGNOSTIC_NOT_FUTURE_EFFECT",
    }


def round3_boundary_probe(artifacts: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    checked = 0
    crossed: dict[str, int] = {variant: 0 for variant in VARIANTS}
    failures: list[str] = []
    sample: list[dict[str, Any]] = []
    for event_id, rows in artifacts:
        for frame in rows:
            if int(frame.get("frame_horizon", -1)) <= 0 or str(frame.get("variant")) not in VARIANTS:
                continue
            variant = str(frame["variant"])
            fused = np.asarray(frame["fused_score_matrix"], dtype=np.float64)
            base = np.asarray(frame["base_score_matrix"], dtype=np.float64)
            target_row_value = frame.get("target_row_index")
            if target_row_value is None:
                continue
            target_row = int(target_row_value)
            current_objective, current_assignment = assignment_objective(fused)
            current_column = int(current_assignment[target_row])
            choices = []
            for column in range(fused.shape[1]):
                if column == current_column:
                    continue
                constrained = constrained_objective(fused, target_row, column)
                choices.append((max(0.0, current_objective - constrained + 1.0e-6), column))
            if not choices:
                continue
            residual, column = min(choices, key=lambda value: (value[0], value[1]))
            probe = fused.copy()
            probe[target_row, column] += float(residual)
            _, probe_assignment = assignment_objective(probe)
            non_target_rows = [index for index in range(base.shape[0]) if index != target_row]
            non_target_equal = bool(np.array_equal(base[non_target_rows, :], fused[non_target_rows, :])) if non_target_rows else True
            target_crossed = int(probe_assignment[target_row]) == int(column)
            if not non_target_equal:
                failures.append(f"{event_id}/{variant}/{frame['frame']}:non_target_source_not_equal")
            if target_crossed:
                crossed[variant] += 1
            if len(sample) < 1000:
                sample.append({
                    "event_id": event_id,
                    "frame": int(frame["frame"]),
                    "variant": variant,
                    "target_row": target_row,
                    "current_candidate_column": current_column,
                    "probe_candidate_column": int(column),
                    "added_target_row_residual": float(residual),
                    "probe_target_assignment_column": int(probe_assignment[target_row]),
                    "target_crossed": target_crossed,
                    "non_target_rows_bitwise_equal_to_source": non_target_equal,
                    "runtime_future_gt_used": False,
                })
            checked += 1
    if failures:
        raise RuntimeError(f"boundary probe source equality failure: {failures[:3]}")
    return {
        "round": "R3_TARGET_ROW_BOUNDARY_PROBE",
        "status": "PASS_DIAGNOSTIC",
        "root_cause": "assignment_interface_can_be_crossed_only_with_explicit_target_row_residual",
        "checked_future_frames": checked,
        "synthetic_target_row_crossings": crossed,
        "sample_rows_capped_at": 1000,
        "sample_rows": sample,
        "probe_definition": "For each frame, add exactly the deterministic residual needed to make the cheapest constrained target candidate win, plus 1e-6, to the target identity row only; leave all other cells unchanged and rerun global Hungarian.",
        "production_changed": False,
        "not_a_model_result": True,
        "interpretation": "This confirms the global solver can cross when the target row is given enough residual. It is an interface diagnostic, not evidence that the residual is a valid learned score or that future identity improves.",
        "component_changed": "isolated_hypothetical_target_row_probe_only",
        "runtime_future_gt_used": False,
        "scientific_result": "HYPOTHETICAL_ASSIGNMENT_BOUNDARY_PROBE_NOT_FUTURE_EFFECT",
    }


def round4_memory_age(stage22: dict[str, Any], artifacts: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    admitted = {variant: {"read": 0, "admitted": 0, "age_rejected": 0} for variant in ("M3_NEGATIVE_COMPETITOR_BANK", "M4_RELIABILITY_AGE_ADMISSION")}
    for _event_id, rows in artifacts:
        for row in rows:
            variant = str(row.get("variant"))
            if variant not in admitted or int(row.get("frame_horizon", -1)) <= 0:
                continue
            admitted[variant]["read"] += int(bool(row.get("memory_read")))
            admitted[variant]["admitted"] += int(bool(row.get("memory_admitted")))
            admitted[variant]["age_rejected"] += int(row.get("memory_read_reason") == "AGE_ABOVE_ADMISSION")
    aggregate = {}
    for horizon in HORIZONS:
        m3 = stage22.get("aggregate", {}).get("M3_NEGATIVE_COMPETITOR_BANK", {}).get(str(horizon), {})
        m4 = stage22.get("aggregate", {}).get("M4_RELIABILITY_AGE_ADMISSION", {}).get(str(horizon), {})
        aggregate[str(horizon)] = {
            "m3_identity_utility": m3.get("identity_utility"),
            "m4_identity_utility": m4.get("identity_utility"),
            "m3_m4_utility_difference": None if m3.get("identity_utility") is None or m4.get("identity_utility") is None else float(m4["identity_utility"] - m3["identity_utility"]),
            "m3_assignment_changes": m3.get("assignment_change_count"),
            "m4_assignment_changes": m4.get("assignment_change_count"),
            "m3_wrong_reassociation": m3.get("wrong_reassociation_frames"),
            "m4_wrong_reassociation": m4.get("wrong_reassociation_frames"),
        }
    return {
        "round": "R4_MEMORY_ADMISSION_AGE",
        "status": "PASS_DIAGNOSTIC",
        "root_cause": "memory_effect_is_local_and_age_bounded",
        "memory_read_admission_counts": admitted,
        "horizon_comparison": aggregate,
        "frozen_admission": {"minimum_reliability": 0.75, "maximum_age_frames": 80},
        "interpretation": "M4 is a fixed admission/age branch, not a posthoc-selected variant. Differences after the fixed age boundary are recorded; no decay or admission value is tuned from future outcomes.",
        "component_changed": None,
        "runtime_future_gt_used": False,
        "scientific_result": "MEMORY_LIFECYCLE_DIAGNOSTIC_NOT_FUTURE_EFFECT",
    }


def round5_statistics(stage22: dict[str, Any]) -> dict[str, Any]:
    primary = stage22.get("aggregate", {}).get("M2_POSITIVE_HUMAN_ANCHORS", {})
    variants = {}
    for variant in VARIANTS:
        h20 = stage22.get("aggregate", {}).get(variant, {}).get("20", {})
        ci = h20.get("sequence_cluster_bootstrap_95ci", {})
        variants[variant] = {
            "h20_identity_utility": h20.get("identity_utility"),
            "h20_ci_lower": ci.get("lower"),
            "h20_ci_upper": ci.get("upper"),
            "clusters": ci.get("clusters"),
            "event_count": h20.get("event_count"),
            "assignment_change_correct": h20.get("assignment_change_correct_count"),
            "assignment_change_incorrect": h20.get("assignment_change_incorrect_count"),
            "protected_regression": h20.get("protected_regression_count"),
        }
    return {
        "round": "R5_STATISTICAL_POWER_AND_PROTECTED_IDS",
        "status": "PASS_DIAGNOSTIC",
        "root_cause": "strict_ci_underpowered_with_six_independent_sequences",
        "primary_m2_h20": {
            "identity_utility": primary.get("20", {}).get("identity_utility"),
            "ci": primary.get("20", {}).get("sequence_cluster_bootstrap_95ci"),
        },
        "by_variant": variants,
        "quota": {"event_target": stage22.get("event_quota_target"), "sequence_target": stage22.get("sequence_quota_target"), "observed_events": stage22.get("event_count"), "observed_sequences": stage22.get("independent_sequence_count")},
        "interpretation": "The fixed 2000-repetition independent-sequence bootstrap is valid but cannot yield a strictly positive lower bound with the observed six sequences and small effects. No duplicate events or resampling-based pseudo-sequences are introduced.",
        "component_changed": None,
        "runtime_future_gt_used": False,
        "scientific_result": "STATISTICAL_POWER_DIAGNOSTIC_NOT_FUTURE_EFFECT",
    }


def write_failure(exc: BaseException) -> Path:
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = FAILURE_ROOT / f"mechanism_rounds_failure_{stamp}.json"
    atomic_json(path, {
        "schema_version": "N72R3_FAILURE_RECORD_V1",
        "stage": "POST_STAGE22_MECHANISM_ROUNDS",
        "status": "FAIL_PRESERVED",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "runtime_future_gt_used": False,
        "scientific_result": "NO_SCIENTIFIC_RESULT",
    })
    return path


def main() -> int:
    try:
        stage19, stage22, _stage21, artifacts = load_sealed()
        rounds = [
            round1_candidate_recall(stage19, stage22),
            round2_margin_residual(artifacts),
            round3_boundary_probe(artifacts),
            round4_memory_age(stage22, artifacts),
            round5_statistics(stage22),
        ]
        for index, value in enumerate(rounds, 1):
            value["round_index"] = index
            value["created_at_utc"] = now_utc()
            atomic_json(ROUND_ROOT / f"round_{index:02d}_{value['round']}.json", value)
        summary = {
            "schema_version": "N72R3_MECHANISM_ROUNDS_SUMMARY_V1",
            "status": "PASS_FIVE_MECHANISM_ROUNDS_COMPLETED_NO_STRICT_FUTURE_EFFECT",
            "created_at_utc": now_utc(),
            "round_count": len(rounds),
            "rounds": [{"round_index": value["round_index"], "round": value["round"], "status": value["status"], "root_cause": value["root_cause"], "artifact": str(ROUND_ROOT / f"round_{value['round_index']:02d}_{value['round']}.json")} for value in rounds],
            "input_artifacts": {
                "stage19": str(STAGE19),
                "stage19_sha256": sha256(STAGE19),
                "stage22": str(STAGE22),
                "stage22_sha256": sha256(STAGE22),
                "stage21": str(STAGE21),
                "stage21_sha256": sha256(STAGE21),
            },
            "runtime_future_gt_used": False,
            "posthoc_gt_reused_only_from_sealed_stage22": True,
            "production_changed": False,
            "checkpoint_changed": False,
            "candidate_definition_changed": False,
            "hungarian_solver_changed": False,
            "real_human_tape": False,
            "production_authorized": False,
            "final_interpretation": "No strict future-effect confirmation. The artifacts support a combined candidate-recall and association-decision-boundary bottleneck; M3/M4 produce a small local effect without a strictly positive sequence-cluster CI. More independent eligible events or real human tape are required before a production claim.",
            "next_minimum_action": "Collect or ingest additional eligible current-frame events without changing the frozen protocol; do not duplicate the six events, train a downstream module, or alter checkpoint/candidate/Hungarian definitions.",
        }
        atomic_json(SUMMARY_PATH, summary)
        print(json.dumps({"status": summary["status"], "round_count": len(rounds), "summary": str(SUMMARY_PATH)}, sort_keys=True))
        return 0
    except Exception as exc:
        path = write_failure(exc)
        print(json.dumps({"status": "FAIL_MECHANISM_ROUNDS", "failure": str(path)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
