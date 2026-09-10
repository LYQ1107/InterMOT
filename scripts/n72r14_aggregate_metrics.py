#!/usr/bin/env python3
"""Posthoc causal metrics for the sealed N72R14 B0/P0/T1/G1 replay."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    metric_record,
    sequence_cluster_bootstrap,
)
from scripts import n72r11_on_demand_replay as replay  # noqa: E402
from scripts.n72r13_temporal_oracle import _quality  # noqa: E402


HORIZONS = (20, 50, 100)
VARIANTS = (
    "E0_BASELINE_B0",
    "E1F_TARGET_POOL_ONLY",
    "E1G_TARGET_STATE_ONLY",
    "E1H_PERSISTENT_GLOBAL_STATE",
)
COMPARISONS = (
    ("P0_MINUS_B0", "E0_BASELINE_B0", "E1F_TARGET_POOL_ONLY"),
    ("T1_MINUS_P0", "E1F_TARGET_POOL_ONLY", "E1G_TARGET_STATE_ONLY"),
    ("G1_MINUS_P0", "E1F_TARGET_POOL_ONLY", "E1H_PERSISTENT_GLOBAL_STATE"),
    ("G1_MINUS_T1", "E1G_TARGET_STATE_ONLY", "E1H_PERSISTENT_GLOBAL_STATE"),
    ("G1_MINUS_B0", "E0_BASELINE_B0", "E1H_PERSISTENT_GLOBAL_STATE"),
)
BOOTSTRAP_SEED = 7214
BOOTSTRAP_REPETITIONS = 2000
IOU_THRESHOLD = 0.50
MANIFEST = ROOT / "outputs/N72R14/formal_manifest.json"
OUTPUT = ROOT / "outputs/N72R14/causal_metrics.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _runtime_scan(value: Any, location: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"} and nested is not False:
                errors.append(f"{location}/{key}={nested!r}")
            errors.extend(_runtime_scan(nested, f"{location}/{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_runtime_scan(nested, f"{location}/{index}"))
    return errors


def _load_runtime(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    if manifest.get("status") != "PASS_N72R14_FORMAL_REPLAY" or int(manifest.get("event_count", -1)) != 32:
        raise RuntimeError("N72R14 formal manifest is not the complete PASS artifact")
    events: list[dict[str, Any]] = []
    rows_by_event: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: set[tuple[str, str]] = set()
    for event_record in manifest.get("events", []):
        event_id = str(event_record["event_id"])
        variants = event_record.get("variants", [])
        if event_record.get("status") != "PASS_N72R14_FORMAL_EVENT" or len(variants) != len(VARIANTS):
            raise RuntimeError(f"{event_id}: event manifest incomplete")
        event_rows: dict[str, list[dict[str, Any]]] = {}
        for variant_record in variants:
            variant = str(variant_record["variant"])
            if variant not in VARIANTS or (event_id, variant) in seen:
                raise RuntimeError(f"duplicate/unknown runtime key {event_id}/{variant}")
            seen.add((event_id, variant))
            path = Path(str(variant_record["frames"]))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 101:
                raise RuntimeError(f"{event_id}/{variant}: expected 101 runtime rows")
            event_frame = int(event_record["event_frame"])
            if [int(row.get("frame", -1)) for row in rows] != list(range(event_frame, event_frame + 101)):
                raise RuntimeError(f"{event_id}/{variant}: incomplete frame axis")
            flags = _runtime_scan(rows, f"{event_id}/{variant}")
            if flags:
                raise RuntimeError(f"{event_id}/{variant}: runtime flag violations: {flags[:3]}")
            if rows[0].get("record_kind") != "event_frame_correction" or rows[0].get("memory_read") is not False:
                raise RuntimeError(f"{event_id}/{variant}: event-frame causal contract failed")
            for row in rows[1:]:
                candidates = row.get("candidate_rows")
                if not isinstance(candidates, list) or len({str(item.get("candidate_uid")) for item in candidates}) != len(candidates):
                    raise RuntimeError(f"{event_id}/{variant}:{row.get('frame')}: candidate axis invalid")
                edge = row.get("persistent_state_association", {}).get("edge", {})
                if variant in {"E1G_TARGET_STATE_ONLY", "E1H_PERSISTENT_GLOBAL_STATE"}:
                    if edge.get("state_edge_scale") != 1.0:
                        raise RuntimeError(f"{event_id}/{variant}: state edge scale changed")
            event_rows[variant] = rows
        if set(event_rows) != set(VARIANTS):
            raise RuntimeError(f"{event_id}: variant set mismatch")
        events.append({
            "event_id": event_id,
            "sequence": str(event_record["sequence"]),
            "event_frame": int(event_record["event_frame"]),
            "action_type": str(event_record["action_type"]),
            "dataset_gt_id": int(next(item.get("dataset_gt_id") for item in manifest.get("events", []) if str(item["event_id"]) == event_id) if False else 0),
        })
        rows_by_event[event_id] = event_rows
    # Recover the authoritative frozen event metadata from N72R9 only after
    # runtime validation; this does not open GT.
    protocol = read_json(ROOT / "outputs/N72R9/protocol.json")
    frozen_events = {str(item["event_id"]): dict(item) for item in protocol.get("source_event_selection", {}).get("events", [])}
    if len(frozen_events) != 32 or set(frozen_events) != set(rows_by_event):
        raise RuntimeError("formal runtime event set differs from frozen N72R9 protocol")
    for event in events:
        frozen = frozen_events[event["event_id"]]
        event.update({
            "target_public_id": int(read_json(Path(str(frozen["source_event_manifest"]))).get("target_public_id")),
            "target_dataset_gt_id": int(frozen["dataset_gt_id"]),
            "interaction_source": str(frozen.get("interaction_source", "simulated_from_gt")),
            "not_real_human_evidence": True,
        })
    if len(seen) != 32 * 4:
        raise RuntimeError(f"unique runtime artifact count {len(seen)} != 128")
    return events, rows_by_event


def _assignment_map(row: Mapping[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for candidate in row.get("candidate_rows", []):
        uid = str(candidate.get("candidate_uid"))
        if uid in result:
            raise RuntimeError(f"duplicate candidate UID {uid}")
        public = candidate.get("solver_public_id", candidate.get("public_id"))
        result[uid] = None if public is None else int(public)
    return result


def _target_uid(row: Mapping[str, Any], public_id: int) -> str | None:
    values = [item for item in row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)]
    if len(values) > 1:
        raise RuntimeError(f"duplicate target assignment {public_id}")
    return None if not values else str(values[0]["candidate_uid"])


def _matrix_pairs(row: Mapping[str, Any]) -> dict[tuple[str, int], float]:
    audit = row.get("score_audit", {})
    candidates = row.get("candidate_rows", [])
    uids = [str(item.get("candidate_uid")) for item in candidates]
    publics = [int(value) for value in audit.get("public_id_axis", [])]
    matrix = np.asarray(audit.get("fused_score_matrix", []), dtype=np.float64)
    if matrix.shape != (len(uids), len(publics)) or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid score audit matrix at {row.get('event_id')}:{row.get('frame')}")
    return {(uid, public): float(matrix[i, j]) for i, uid in enumerate(uids) for j, public in enumerate(publics)}


def _score_change(baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> tuple[bool, int]:
    left, right = _matrix_pairs(baseline), _matrix_pairs(treatment)
    common = set(left) & set(right)
    changed = [abs(left[key] - right[key]) > 1.0e-9 for key in common]
    return bool(any(changed)), int(sum(changed))


def _posthoc_pair(
    event: Mapping[str, Any],
    rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_name: str,
    treatment_name: str,
    horizon: int,
    gt: Mapping[int, Mapping[int, Any]],
    protected: Mapping[int, int],
) -> dict[str, Any]:
    baseline_rows = list(rows_by_variant[baseline_name])
    treatment_rows = list(rows_by_variant[treatment_name])
    base_quality = _quality(
        baseline_rows,
        event_frame=int(event["event_frame"]),
        target_public=int(event["target_public_id"]),
        target_gid=int(event["target_dataset_gt_id"]),
        protected=protected,
        gt=gt,
        horizon=int(horizon),
    )
    treatment_quality = _quality(
        treatment_rows,
        event_frame=int(event["event_frame"]),
        target_public=int(event["target_public_id"]),
        target_gid=int(event["target_dataset_gt_id"]),
        protected=protected,
        gt=gt,
        horizon=int(horizon),
    )
    frame_details: list[dict[str, Any]] = []
    score_changed_count = 0
    score_cell_changed_count = 0
    assignment_changed_count = 0
    global_assignment_changed_count = 0
    correct_change = 0
    incorrect_change = 0
    neutral_change = 0
    for offset in range(1, int(horizon) + 1):
        frame = int(event["event_frame"]) + offset
        baseline = baseline_rows[offset]
        treatment = treatment_rows[offset]
        score_changed, cell_count = _score_change(baseline, treatment)
        score_changed_count += int(score_changed)
        score_cell_changed_count += cell_count
        base_uid = _target_uid(baseline, int(event["target_public_id"]))
        treatment_uid = _target_uid(treatment, int(event["target_public_id"]))
        assignment_changed = base_uid != treatment_uid
        assignment_changed_count += int(assignment_changed)
        base_map, treatment_map = _assignment_map(baseline), _assignment_map(treatment)
        common_uids = set(base_map) & set(treatment_map)
        global_changed = sum(int(base_map[uid] != treatment_map[uid]) for uid in common_uids)
        global_assignment_changed_count += int(global_changed > 0)
        gt_item = gt.get(frame, {}).get(int(event["target_dataset_gt_id"]))
        base_iou = treatment_iou = None
        change_kind = "UNASSESSABLE_NO_TARGET_GT"
        if gt_item is not None:
            base_iou, _ = replay.legacy._public_box_for_gt(baseline, int(event["target_public_id"]), gt_item["box"])
            treatment_iou, _ = replay.legacy._public_box_for_gt(treatment, int(event["target_public_id"]), gt_item["box"])
            record = metric_record(
                baseline_iou=float(base_iou),
                treatment_iou=float(treatment_iou),
                baseline_correct=bool(base_iou >= IOU_THRESHOLD),
                treatment_correct=bool(treatment_iou >= IOU_THRESHOLD),
                assignment_changed=bool(assignment_changed),
            )
            change_kind = str(record["assignment_change_type"])
            correct_change += int(record["true_correct_crossing"])
            incorrect_change += int(record["true_incorrect_crossing"])
            neutral_change += int(
                assignment_changed and not record["true_correct_crossing"] and not record["true_incorrect_crossing"]
            )
        frame_details.append({
            "frame": frame,
            "score_changed": score_changed,
            "score_changed_cell_count": cell_count,
            "baseline_target_candidate_uid": base_uid,
            "treatment_target_candidate_uid": treatment_uid,
            "assignment_changed": assignment_changed,
            "global_common_assignment_changed_count": int(global_changed),
            "baseline_target_iou": base_iou,
            "treatment_target_iou": treatment_iou,
            "assignment_change_type": change_kind,
            "runtime_future_gt_used": False,
        })
    def difference(left: str, right: str) -> float | None:
        a, b = base_quality.get(left), treatment_quality.get(right)
        return None if a is None or b is None else float(b - a)
    return {
        "baseline": base_quality,
        "treatment": treatment_quality,
        "delta_target_identity_error_reduction": (
            None
            if base_quality.get("target_identity_error") is None or treatment_quality.get("target_identity_error") is None
            else float(base_quality["target_identity_error"] - treatment_quality["target_identity_error"])
        ),
        "delta_global_identity_error_reduction": (
            None
            if base_quality.get("global_identity_error") is None or treatment_quality.get("global_identity_error") is None
            else float(base_quality["global_identity_error"] - treatment_quality["global_identity_error"])
        ),
        "delta_target_iou": difference("target_iou", "target_iou"),
        "delta_protected_accuracy": difference("protected_accuracy", "protected_accuracy"),
        "id_switch_improvement": int(base_quality.get("id_switch_count", 0) - treatment_quality.get("id_switch_count", 0)),
        "target_missing_reduction": int(base_quality.get("target_missing_frames", 0) - treatment_quality.get("target_missing_frames", 0)),
        "score_changed_frame_count": int(score_changed_count),
        "score_changed_cell_count": int(score_cell_changed_count),
        "assignment_changed_count": int(assignment_changed_count),
        "global_assignment_changed_frame_count": int(global_assignment_changed_count),
        "correct_assignment_change_count": int(correct_change),
        "incorrect_assignment_change_count": int(incorrect_change),
        "neutral_assignment_change_count": int(neutral_change),
        "frame_details": frame_details,
        "runtime_future_gt_used": False,
    }


def _aggregate_comparison(
    event_results: Sequence[Mapping[str, Any]],
    comparison_name: str,
    horizon: int,
) -> dict[str, Any]:
    selected = [item for item in event_results if int(horizon) in item["horizons"]]
    values = [item["horizons"][int(horizon)]["delta_global_identity_error_reduction"] or 0.0 for item in selected]
    by_sequence: dict[str, list[float]] = defaultdict(list)
    by_action_sequence: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item, value in zip(selected, values):
        by_sequence[str(item["sequence"])].append(float(value))
        by_action_sequence[str(item["action_type"])][str(item["sequence"])].append(float(value))
    def mean(name: str) -> float:
        vals = [item["horizons"][int(horizon)].get(name) for item in selected]
        vals = [float(value) for value in vals if value is not None and math.isfinite(float(value))]
        return float(np.mean(vals)) if vals else 0.0
    counts = {
        name: int(sum(int(item["horizons"][int(horizon)].get(name, 0)) for item in selected))
        for name in (
            "score_changed_frame_count",
            "assignment_changed_count",
            "correct_assignment_change_count",
            "incorrect_assignment_change_count",
            "neutral_assignment_change_count",
            "global_assignment_changed_frame_count",
        )
    }
    action_summary = {
        action: {
            "event_count": int(sum(len(vals) for vals in sequence_values.values())),
            "independent_sequence_count": len(sequence_values),
            "mean_global_identity_error_reduction": float(np.mean([value for vals in sequence_values.values() for value in vals])) if sequence_values else 0.0,
            "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(
                sequence_values,
                seed=BOOTSTRAP_SEED + int(horizon) + len(action),
                repetitions=BOOTSTRAP_REPETITIONS,
            ) if sequence_values else sequence_cluster_bootstrap({}, seed=BOOTSTRAP_SEED + int(horizon), repetitions=BOOTSTRAP_REPETITIONS),
        }
        for action, sequence_values in sorted(by_action_sequence.items())
    }
    return {
        "comparison": comparison_name,
        "horizon": int(horizon),
        "event_count": len(selected),
        "independent_sequence_count": len(by_sequence),
        "mean_global_identity_error_reduction": mean("delta_global_identity_error_reduction"),
        "mean_target_identity_error_reduction": mean("delta_target_identity_error_reduction"),
        "mean_target_iou_delta": mean("delta_target_iou"),
        "mean_protected_accuracy_delta": mean("delta_protected_accuracy"),
        "mean_id_switch_improvement": mean("id_switch_improvement"),
        "mean_target_missing_reduction": mean("target_missing_reduction"),
        "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(
            by_sequence,
            seed=BOOTSTRAP_SEED + list(name for name, _, _ in COMPARISONS).index(comparison_name) * 100 + int(horizon),
            repetitions=BOOTSTRAP_REPETITIONS,
        ),
        "counts": counts,
        "action_breakdown": action_summary,
        "runtime_future_gt_used": False,
    }


def aggregate(manifest_path: Path = MANIFEST, output_path: Path = OUTPUT) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    events, rows_by_event = _load_runtime(manifest)
    # All runtime rows have now been read and validated.  Only this posthoc
    # section opens dataset GT.
    event_results: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (str(item["sequence"]), int(item["event_frame"]), str(item["event_id"]))):
        gt = replay.legacy._load_gt(str(event["sequence"]))
        # ``_protected_map`` requires the frozen event-frame C0 row.  The
        # event's source path is loaded without any GT use.
        protocol_events = read_json(ROOT / "outputs/N72R9/protocol.json")["source_event_selection"]["events"]
        frozen_event = next(item for item in protocol_events if str(item["event_id"]) == str(event["event_id"]))
        inputs = replay._load_inputs(frozen_event, horizon=100)
        protected = replay.legacy._protected_map(
            inputs["rows"]["c0_source"][int(event["event_frame"])],
            gt,
            int(event["event_frame"]),
            int(event["target_dataset_gt_id"]),
        )
        comparisons: dict[str, dict[str, Any]] = {}
        for name, baseline, treatment in COMPARISONS:
            comparisons[name] = {}
            for horizon in HORIZONS:
                comparisons[name][str(horizon)] = _posthoc_pair(
                    event,
                    rows_by_event[event["event_id"]],
                    baseline,
                    treatment,
                    horizon,
                    gt,
                    protected,
                )
        event_results.append({
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": int(event["event_frame"]),
            "target_public_id": int(event["target_public_id"]),
            "target_dataset_gt_id": int(event["target_dataset_gt_id"]),
            "protected_public_by_gt_posthoc": protected,
            "comparisons": comparisons,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": str(event.get("interaction_source", "simulated_from_gt")),
            "not_real_human_evidence": True,
            "horizons": {},
        })
        for result in event_results[-1:]:
            for name in comparisons:
                # Main aggregation uses global identity error reduction for
                # each treatment pair.  Keep the detailed pair table above.
                result["horizons"].setdefault(100, {})
        # Keep a compact canonical per-comparison horizon index for the
        # sequence bootstrap helper below.
        event_results[-1]["horizons"] = {
            horizon: comparisons["G1_MINUS_B0"][str(horizon)] for horizon in HORIZONS
        }
    aggregate_summary: dict[str, dict[str, Any]] = {}
    for name, _, _ in COMPARISONS:
        aggregate_summary[name] = {str(horizon): _aggregate_comparison(
            [
                {
                    **item,
                    "horizons": {int(h): item["comparisons"][name][str(h)] for h in HORIZONS},
                }
                for item in event_results
            ],
            name,
            horizon,
        ) for horizon in HORIZONS}
    output = {
        "schema_version": "N72R14_CAUSAL_METRICS_V1",
        "status": "PASS_N72R14_POSTHOC_METRICS",
        "created_at_utc": now_utc(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "event_count": len(event_results),
        "independent_sequence_count": len({item["sequence"] for item in event_results}),
        "horizons": list(HORIZONS),
        "variants": list(VARIANTS),
        "comparisons": aggregate_summary,
        "events": event_results,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS, "unit": "independent_sequence"},
        "gt_usage": "posthoc_only_after_all_runtime_artifacts_validated",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "historical_outputs_modified": False,
    }
    atomic_json(output_path, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = aggregate(args.manifest.resolve(), args.output.resolve())
        print(json.dumps({
            "status": result["status"],
            "event_count": result["event_count"],
            "independent_sequence_count": result["independent_sequence_count"],
            "output": str(args.output.resolve()),
        }, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R14_FAILURE_V1",
            "status": "FAIL_N72R14_POSTHOC_METRICS",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "created_at_utc": now_utc(),
            "runtime_future_gt_used": False,
        }
        atomic_json(args.output.resolve().with_name("causal_metrics_failure.json"), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
