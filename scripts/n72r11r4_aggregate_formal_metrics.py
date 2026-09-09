#!/usr/bin/env python3
"""Audit and aggregate a complete frozen N72R11R4 formal replay.

This is posthoc-only: it never changes a runtime row, event, checkpoint, or
comparison.  Every event must provide a sealed E0/E1A artifact before a
summary is emitted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import sequence_cluster_bootstrap  # noqa: E402

HORIZONS = (20, 50, 100)
DEFAULT_COMPARISON = "E1A_vs_E0"
COUNT_KEYS = (
    "evaluated_frames", "window_frame_count", "target_gt_visible_frames", "target_gt_absent_frames",
    "assignment_change_count", "global_common_assignment_change_count", "target_assignment_change_count",
    "true_correct_crossing_count", "true_incorrect_crossing_count", "directional_improvement_count",
    "directional_regression_count", "neutral_change_count", "id_switch_count", "raw_switch_count",
    "posthoc_correct_switch_count", "posthoc_wrong_switch_count", "posthoc_unassessable_switch_count",
    "recorrection_opportunity_count", "candidate_present_frames", "target_missing_frames",
    "baseline_correct_frames", "baseline_identity_error_frames", "treatment_correct_frames",
    "treatment_identity_error_frames", "protected_compared", "protected_regression_count",
    "protected_improvement_count",
)
SUM_KEYS = ("identity_error_reduction_sum", "delta_iou_sum", "baseline_iou_sum", "treatment_iou_sum")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def finite(value: Any, label: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {label} in {path}: {value!r}")
    return float(value)


def nonnegative_count(value: Any, label: str, path: Path) -> int:
    number = finite(value, label, path)
    result = int(number)
    if number != result or result < 0:
        raise ValueError(f"invalid count {label} in {path}: {value!r}")
    return result


def validate_event(record: Mapping[str, Any], comparison: str) -> tuple[dict[str, Any], Path]:
    event_id = str(record.get("event_id"))
    done_path = resolve(record.get("done"))
    if not done_path.is_file():
        raise FileNotFoundError(done_path)
    done = read_json(done_path)
    if done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT":
        raise ValueError(f"event done is not sealed PASS: {done_path}")
    if done.get("runtime_future_gt_used") is not False or done.get("posthoc_gt_used") is not True:
        raise ValueError(f"event provenance failed: {done_path}")
    seal_path = resolve(done.get("runtime_event_sealed"))
    if not seal_path.is_file():
        raise FileNotFoundError(seal_path)
    seal = read_json(seal_path)
    if seal.get("runtime_future_gt_used") is not False or seal.get("runtime_gt_read") is not False:
        raise ValueError(f"runtime seal has a GT flag != false: {seal_path}")
    posthoc_path = resolve(done.get("posthoc"))
    if not posthoc_path.is_file():
        raise FileNotFoundError(posthoc_path)
    posthoc = read_json(posthoc_path)
    if posthoc.get("status") != "PASS_N72R11_POSTHOC_EVENT" or posthoc.get("posthoc_gt_used") is not True:
        raise ValueError(f"posthoc provenance/status failed: {posthoc_path}")
    if posthoc.get("runtime_future_gt_used") is not False:
        raise ValueError(f"posthoc reports runtime future GT: {posthoc_path}")
    event = posthoc.get("event")
    if not isinstance(event, Mapping):
        raise ValueError(f"posthoc event object missing: {posthoc_path}")
    if str(event.get("event_id")) != event_id or str(event.get("sequence")) != str(record.get("sequence")):
        raise ValueError(f"manifest/event identity mismatch: {posthoc_path}")
    comparisons = event.get("comparisons")
    if not isinstance(comparisons, Mapping) or comparison not in comparisons:
        raise ValueError(f"{comparison} missing: {posthoc_path}")
    horizons = comparisons[comparison]
    if not isinstance(horizons, Mapping):
            raise ValueError(f"malformed {comparison}: {posthoc_path}")
    for horizon in HORIZONS:
        metric = horizons.get(str(horizon))
        if not isinstance(metric, Mapping):
            raise ValueError(f"missing {COMPARISON}/H{horizon}: {posthoc_path}")
        if nonnegative_count(metric.get("evaluated_frames", 0), "evaluated_frames", posthoc_path) <= 0:
            raise ValueError(f"empty H{horizon}: {posthoc_path}")
        for key in COUNT_KEYS:
            nonnegative_count(metric.get(key, 0), key, posthoc_path)
        for key in SUM_KEYS:
            finite(metric.get(key, 0.0), key, posthoc_path)
    return posthoc, posthoc_path


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {key: 0.0 for key in (*COUNT_KEYS, *SUM_KEYS)}
    by_sequence: dict[str, list[float]] = defaultdict(list)
    sequences: list[str] = []
    for item in items:
        metric = item["metric"]
        for key in COUNT_KEYS:
            totals[key] += nonnegative_count(metric.get(key, 0), key, Path(item["posthoc"]))
        for key in SUM_KEYS:
            totals[key] += finite(metric.get(key, 0.0), key, Path(item["posthoc"]))
        sequence = str(item["sequence"])
        sequences.append(sequence)
        by_sequence[sequence].append(float(metric.get("identity_error_reduction", 0.0)))
    evaluated = int(totals["evaluated_frames"])
    denominator = max(evaluated, 1)
    protected = int(totals["protected_compared"])
    result: dict[str, Any] = {key: int(value) if key in COUNT_KEYS else float(value) for key, value in totals.items()}
    result.update(
        {
            "event_count": len(items),
            "independent_sequence_count": len(set(sequences)),
            "baseline_future_identity_error": totals["baseline_identity_error_frames"] / denominator,
            "treatment_future_identity_error": totals["treatment_identity_error_frames"] / denominator,
            "identity_error_reduction": totals["identity_error_reduction_sum"] / denominator,
            "baseline_mean_iou": totals["baseline_iou_sum"] / denominator,
            "treatment_mean_iou": totals["treatment_iou_sum"] / denominator,
            "delta_iou": totals["delta_iou_sum"] / denominator,
            "assignment_change_rate": totals["assignment_change_count"] / denominator,
            "target_assignment_change_rate": totals["target_assignment_change_count"] / denominator,
            "id_switch_rate": totals["id_switch_count"] / denominator,
            "recorrection_rate": totals["recorrection_opportunity_count"] / denominator,
            "candidate_recall": totals["candidate_present_frames"] / denominator,
            "missing_rate": totals["target_missing_frames"] / denominator,
            "protected_regression_rate": None if protected == 0 else totals["protected_regression_count"] / protected,
            "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(
                by_sequence, seed=7211, repetitions=2000
            ) if by_sequence else None,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", choices=("E1A_vs_E0", "E1B_vs_E0", "E2_vs_E1B"), default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError("replay manifest has no records")
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("replay manifest contains a non-object record")
        event_id = str(record.get("event_id"))
        if event_id in seen:
            failures.append({"event_id": event_id, "status": "DUPLICATE_EVENT_ID"})
            continue
        seen.add(event_id)
        if record.get("status") != "PASS":
            failures.append({"event_id": event_id, "status": record.get("status"), "returncode": record.get("returncode"), "log": record.get("log")})
            continue
        try:
            posthoc, posthoc_path = validate_event(record, str(args.comparison))
            metric_by_horizon = posthoc["event"]["comparisons"][str(args.comparison)]
            for horizon in HORIZONS:
                items.append(
                    {
                        "event_id": event_id,
                        "sequence": str(record["sequence"]),
                        "action_type": str(record["action_type"]),
                        "horizon": horizon,
                        "posthoc": str(posthoc_path),
                        "metric": dict(metric_by_horizon[str(horizon)]),
                    }
                )
        except Exception as exc:
            failures.append({"event_id": event_id, "status": "FAIL_AUDIT", "error_type": type(exc).__name__, "error": str(exc), "done": record.get("done")})
    required = int(manifest.get("event_count", 32))
    completeness = {
        "required_events": required,
        "manifest_records": len(records),
        "unique_event_ids": len(seen),
        "audited_pass_events": len(records) - len(failures),
        "failed_or_invalid_events": len(failures),
        "duplicate_event_ids": len(records) - len(seen),
        "missing_event_count": max(required - len(seen), 0),
        "failures": failures,
    }
    by_horizon: dict[str, Any] = {}
    by_action: dict[str, dict[str, Any]] = defaultdict(dict)
    by_sequence: dict[str, dict[str, Any]] = defaultdict(dict)
    actions = sorted({str(record.get("action_type")) for record in records})
    sequences = sorted({str(record.get("sequence")) for record in records})
    for horizon in HORIZONS:
        subset = [item for item in items if item["horizon"] == horizon]
        by_horizon[str(horizon)] = aggregate(subset)
        for action in actions:
            by_action[action][str(horizon)] = aggregate([item for item in subset if item["action_type"] == action])
        for sequence in sequences:
            by_sequence[sequence][str(horizon)] = aggregate([item for item in subset if item["sequence"] == sequence])
    h20 = by_horizon.get("20", {})
    ci = h20.get("sequence_cluster_bootstrap_95ci") or {}
    all_positive = all(float(by_horizon[str(horizon)].get("identity_error_reduction", 0.0)) > 0.0 for horizon in HORIZONS)
    correct = int(h20.get("true_correct_crossing_count", 0))
    incorrect = int(h20.get("true_incorrect_crossing_count", 0))
    complete = not failures and len(records) == required and len(items) == required * len(HORIZONS)
    stage_a_recovered = bool(
        str(args.comparison) == "E1A_vs_E0"
        and
        complete
        and all_positive
        and correct > incorrect
        and all(int(by_horizon[str(horizon)].get("protected_regression_count", 0)) == 0 for horizon in HORIZONS)
    )
    result = {
        "schema_version": "N72R11R4_FORMAL_METRICS_V1",
        "status": "PASS_N72R11R4_STAGE_A_RECOVERED" if stage_a_recovered else (
            "PASS_N72R11R4_STAGE_B_E1B_METRICS" if complete and str(args.comparison) == "E1B_vs_E0" else (
                "PASS_N72R11R4_E2_METRICS" if complete and str(args.comparison) == "E2_vs_E1B" else "FAIL_N72R11R4_FORMAL_METRICS_INCOMPLETE"
            )
        ),
        "created_at_utc": now_utc(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "protocol": manifest.get("protocol"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "model_checkpoint": manifest.get("model_checkpoint"),
        "model_checkpoint_sha256": manifest.get("model_checkpoint_sha256"),
        "variants": manifest.get("variants"),
        "comparison": str(args.comparison),
        "horizons": list(HORIZONS),
        "event_completeness": completeness,
        "complete": complete,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "bootstrap": {"seed": 7211, "repetitions": 2000, "cluster": "independent_sequence"},
        "stage_a_rule": "H20/H50/H100 identity_error_reduction positive, H20 true_correct_crossing_count > true_incorrect_crossing_count, protected regression zero",
        "stage_a_recovered": stage_a_recovered,
        "strict_h20_lower_ci_gt_zero": bool(isinstance(ci.get("lower"), (int, float)) and float(ci["lower"]) > 0.0),
        "h20_correct_crossings": correct,
        "h20_incorrect_crossings": incorrect,
        "by_horizon": by_horizon,
        "by_action": dict(by_action),
        "by_sequence": dict(by_sequence),
    }
    atomic_json(output_path, result)
    print(json.dumps({"status": result["status"], "output": str(output_path), "complete": complete, "failures": len(failures)}, sort_keys=True))
    return 0 if stage_a_recovered or (complete and str(args.comparison) in {"E1B_vs_E0", "E2_vs_E1B"}) else 1


if __name__ == "__main__":
    raise SystemExit(main())
