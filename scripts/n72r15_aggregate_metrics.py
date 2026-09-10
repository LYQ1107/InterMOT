#!/usr/bin/env python3
"""Posthoc causal metrics for the sealed N72R15 replay."""

from __future__ import annotations

from collections import defaultdict
import argparse
from datetime import datetime, timezone
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
    "E1I_HUMAN_RELATIVE_STATE",
    "E1J_TRUSTED_GLOBAL_RELATIVE_STATE",
)
COMPARISONS = (
    ("H1_MINUS_B0", "E0_BASELINE_B0", "E1I_HUMAN_RELATIVE_STATE"),
    ("G2_MINUS_B0", "E0_BASELINE_B0", "E1J_TRUSTED_GLOBAL_RELATIVE_STATE"),
    ("G2_MINUS_H1", "E1I_HUMAN_RELATIVE_STATE", "E1J_TRUSTED_GLOBAL_RELATIVE_STATE"),
)
BOOTSTRAP_SEED = 7215
BOOTSTRAP_REPETITIONS = 2000
IOU_THRESHOLD = 0.50
MANIFEST = ROOT / "outputs/N72R15/formal/formal_manifest.json"
OUTPUT = ROOT / "outputs/N72R15/causal_metrics.json"


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
        raise TypeError(f"expected JSON object: {path}")
    return value


def _runtime_scan(value: Any, location: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"} and nested is not False:
                errors.append(f"{location}/{key}={nested!r}")
            errors.extend(_runtime_scan(nested, f"{location}/{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_runtime_scan(nested, f"{location}/{index}"))
    return errors


def _load_runtime(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    if manifest.get("status") != "PASS_N72R15_FORMAL_REPLAY" or int(manifest.get("event_count", -1)) != 32:
        raise RuntimeError("N72R15 formal manifest is not the complete PASS artifact")
    events: list[dict[str, Any]] = []
    rows_by_event: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: set[tuple[str, str]] = set()
    for event_record in manifest.get("events", []):
        event_id = str(event_record["event_id"])
        variants = event_record.get("variants", [])
        if event_record.get("status") != "PASS_N72R15_FORMAL_EVENT" or len(variants) != len(VARIANTS):
            raise RuntimeError(f"{event_id}: event manifest incomplete")
        event_rows: dict[str, list[dict[str, Any]]] = {}
        for variant_record in variants:
            variant = str(variant_record["variant"])
            if variant not in VARIANTS or (event_id, variant) in seen:
                raise RuntimeError(f"duplicate/unknown runtime key {event_id}/{variant}")
            seen.add((event_id, variant))
            path = Path(str(variant_record["frames"]))
            if not path.is_file():
                raise RuntimeError(f"missing runtime frames {path}")
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
                pool = row.get("candidate_pool", {})
                if pool.get("candidate_pool_policy") != "MAIN_B0_ONLY" or pool.get("target_session_candidate_in_solver") is not False:
                    raise RuntimeError(f"{event_id}/{variant}:{row.get('frame')}: candidate policy drift")
                candidates = row.get("candidate_rows", [])
                if not isinstance(candidates, list) or len({str(item.get("candidate_uid")) for item in candidates}) != len(candidates):
                    raise RuntimeError(f"{event_id}/{variant}:{row.get('frame')}: candidate axis invalid")
                if any(item.get("candidate_source") != "MAIN_B0_CANDIDATE" for item in candidates):
                    raise RuntimeError(f"{event_id}/{variant}:{row.get('frame')}: non-MAIN candidate in solver")
                edge = row.get("persistent_state_association", {})
                if variant != "E0_BASELINE_B0" and not bool(edge.get("row_max_preserved")):
                    raise RuntimeError(f"{event_id}/{variant}:{row.get('frame')}: row max not preserved")
            event_rows[variant] = rows
        if set(event_rows) != set(VARIANTS):
            raise RuntimeError(f"{event_id}: variant set mismatch")
        reference = [[str(item.get("candidate_uid")) for item in event_rows[VARIANTS[0]][i].get("candidate_rows", [])] for i in range(101)]
        for variant in VARIANTS[1:]:
            observed = [[str(item.get("candidate_uid")) for item in event_rows[variant][i].get("candidate_rows", [])] for i in range(101)]
            if observed != reference:
                raise RuntimeError(f"{event_id}: candidate axis mismatch {variant}")
        events.append({
            "event_id": event_id,
            "sequence": str(event_record["sequence"]),
            "event_frame": int(event_record["event_frame"]),
            "action_type": str(event_record["action_type"]),
        })
        rows_by_event[event_id] = event_rows
    if len(seen) != 32 * len(VARIANTS):
        raise RuntimeError(f"unique runtime artifact count {len(seen)} != {32 * len(VARIANTS)}")
    protocol = read_json(ROOT / "outputs/N72R9/protocol.json")
    frozen_events = {str(item["event_id"]): dict(item) for item in protocol.get("source_event_selection", {}).get("events", [])}
    if len(frozen_events) != 32 or set(frozen_events) != set(rows_by_event):
        raise RuntimeError("formal runtime event set differs from frozen N72R9 protocol")
    for event in events:
        frozen = frozen_events[event["event_id"]]
        manifest_path = Path(str(frozen["source_event_manifest"]))
        source_manifest = read_json(manifest_path)
        event.update({
            "target_public_id": int(source_manifest["target_public_id"]),
            "target_dataset_gt_id": int(frozen["dataset_gt_id"]),
            "interaction_source": str(frozen.get("interaction_source", "simulated_from_gt")),
            "not_real_human_evidence": True,
        })
    return events, rows_by_event


def _target_uid(row: Mapping[str, Any], public_id: int) -> str | None:
    values = [item for item in row.get("candidate_rows", []) if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)]
    if len(values) > 1:
        raise RuntimeError(f"duplicate target assignment {public_id} at {row.get('event_id')}:{row.get('frame')}")
    return None if not values else str(values[0]["candidate_uid"])


def _assignment_map(row: Mapping[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for candidate in row.get("candidate_rows", []):
        uid = str(candidate.get("candidate_uid"))
        if uid in result:
            raise RuntimeError(f"duplicate candidate UID {uid}")
        public = candidate.get("public_id", candidate.get("solver_public_id"))
        result[uid] = None if public is None else int(public)
    return result


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
    if set(left) != set(right):
        raise RuntimeError("score matrix candidate/public axis mismatch")
    values = [abs(left[key] - right[key]) > 1.0e-9 for key in left]
    return bool(any(values)), int(sum(values))


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
    score_changed_frames = score_changed_cells = assignment_changed = global_changed_frames = 0
    cardinality_changed_frames = correct_change = incorrect_change = neutral_change = 0
    for offset in range(1, int(horizon) + 1):
        frame = int(event["event_frame"]) + offset
        baseline = baseline_rows[offset]
        treatment = treatment_rows[offset]
        score_changed, cell_count = _score_change(baseline, treatment)
        score_changed_frames += int(score_changed)
        score_changed_cells += cell_count
        base_uid = _target_uid(baseline, int(event["target_public_id"]))
        treatment_uid = _target_uid(treatment, int(event["target_public_id"]))
        target_changed = base_uid != treatment_uid
        assignment_changed += int(target_changed)
        base_map, treatment_map = _assignment_map(baseline), _assignment_map(treatment)
        if set(base_map) != set(treatment_map):
            raise RuntimeError(f"candidate axis mismatch at {event['event_id']}:{frame}")
        # Public-axis comparison is read from the explicit solver audits so a
        # candidate's NONE assignment is not confused with a public identity.
        base_public = {
            int(item["public_id"]): item.get("candidate_uid")
            for item in baseline.get("assignment", {}).get("public_assignments", [])
            if item.get("public_id") is not None
        }
        treatment_public = {
            int(item["public_id"]): item.get("candidate_uid")
            for item in treatment.get("assignment", {}).get("public_assignments", [])
            if item.get("public_id") is not None
        }
        changed_public_ids = sorted(public for public in set(base_public) | set(treatment_public) if base_public.get(public) != treatment_public.get(public))
        global_changed = bool(changed_public_ids)
        global_changed_frames += int(global_changed)
        base_count = len([value for value in base_public.values() if value not in (None, "", "None")])
        treatment_count = len([value for value in treatment_public.values() if value not in (None, "", "None")])
        cardinality_changed_frames += int(base_count != treatment_count)
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
                assignment_changed=bool(target_changed),
            )
            change_kind = str(record["assignment_change_type"])
            correct_change += int(record["true_correct_crossing"])
            incorrect_change += int(record["true_incorrect_crossing"])
            neutral_change += int(target_changed and not record["true_correct_crossing"] and not record["true_incorrect_crossing"])
        frame_details.append({
            "frame": frame,
            "score_changed": score_changed,
            "score_changed_cell_count": cell_count,
            "baseline_target_candidate_uid": base_uid,
            "treatment_target_candidate_uid": treatment_uid,
            "assignment_changed": target_changed,
            "changed_public_ids": changed_public_ids,
            "global_assignment_changed": global_changed,
            "assignment_cardinality_changed": base_count != treatment_count,
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
        "delta_target_identity_error_reduction": None if base_quality.get("target_identity_error") is None or treatment_quality.get("target_identity_error") is None else float(base_quality["target_identity_error"] - treatment_quality["target_identity_error"]),
        "delta_global_identity_error_reduction": None if base_quality.get("global_identity_error") is None or treatment_quality.get("global_identity_error") is None else float(base_quality["global_identity_error"] - treatment_quality["global_identity_error"]),
        "delta_target_iou": difference("target_iou", "target_iou"),
        "delta_protected_accuracy": difference("protected_accuracy", "protected_accuracy"),
        "id_switch_improvement": int(base_quality.get("id_switch_count", 0) - treatment_quality.get("id_switch_count", 0)),
        "target_missing_reduction": int(base_quality.get("target_missing_frames", 0) - treatment_quality.get("target_missing_frames", 0)),
        "score_changed_frame_count": int(score_changed_frames),
        "score_changed_cell_count": int(score_changed_cells),
        "assignment_changed_count": int(assignment_changed),
        "global_assignment_changed_frame_count": int(global_changed_frames),
        "assignment_cardinality_changed_frame_count": int(cardinality_changed_frames),
        "correct_assignment_change_count": int(correct_change),
        "incorrect_assignment_change_count": int(incorrect_change),
        "neutral_assignment_change_count": int(neutral_change),
        "frame_details": frame_details,
        "runtime_future_gt_used": False,
    }


def _aggregate_comparison(event_results: Sequence[Mapping[str, Any]], comparison_name: str, horizon: int) -> dict[str, Any]:
    selected = [item for item in event_results if str(horizon) in item["comparisons"][comparison_name]]
    values = [float(item["comparisons"][comparison_name][str(horizon)]["delta_global_identity_error_reduction"] or 0.0) for item in selected]
    by_sequence: dict[str, list[float]] = defaultdict(list)
    by_action_sequence: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item, value in zip(selected, values):
        by_sequence[str(item["sequence"])].append(value)
        by_action_sequence[str(item["action_type"])][str(item["sequence"])].append(value)
    def finite_mean(field: str) -> float:
        values = [item["comparisons"][comparison_name][str(horizon)].get(field) for item in selected]
        values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        return float(np.mean(values)) if values else 0.0
    count_fields = (
        "score_changed_frame_count",
        "score_changed_cell_count",
        "assignment_changed_count",
        "global_assignment_changed_frame_count",
        "assignment_cardinality_changed_frame_count",
        "correct_assignment_change_count",
        "incorrect_assignment_change_count",
        "neutral_assignment_change_count",
    )
    counts = {field: int(sum(int(item["comparisons"][comparison_name][str(horizon)].get(field, 0)) for item in selected)) for field in count_fields}
    action_summary: dict[str, Any] = {}
    for action, sequence_values in sorted(by_action_sequence.items()):
        flattened = [value for values in sequence_values.values() for value in values]
        action_summary[action] = {
            "event_count": len(flattened),
            "independent_sequence_count": len(sequence_values),
            "mean_global_identity_error_reduction": float(np.mean(flattened)) if flattened else 0.0,
            "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(
                sequence_values,
                seed=BOOTSTRAP_SEED,
                repetitions=BOOTSTRAP_REPETITIONS,
            ) if sequence_values else sequence_cluster_bootstrap({}, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPETITIONS),
        }
    return {
        "comparison": comparison_name,
        "horizon": int(horizon),
        "event_count": len(selected),
        "independent_sequence_count": len(by_sequence),
        "mean_global_identity_error_reduction": finite_mean("delta_global_identity_error_reduction"),
        "mean_target_identity_error_reduction": finite_mean("delta_target_identity_error_reduction"),
        "mean_target_iou_delta": finite_mean("delta_target_iou"),
        "mean_protected_accuracy_delta": finite_mean("delta_protected_accuracy"),
        "mean_id_switch_improvement": finite_mean("id_switch_improvement"),
        "mean_target_missing_reduction": finite_mean("target_missing_reduction"),
        "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(
            by_sequence,
            seed=BOOTSTRAP_SEED,
            repetitions=BOOTSTRAP_REPETITIONS,
        ),
        "counts": counts,
        "action_breakdown": action_summary,
        "runtime_future_gt_used": False,
    }


def aggregate(manifest_path: Path = MANIFEST, output_path: Path = OUTPUT) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    events, rows_by_event = _load_runtime(manifest)
    event_results: list[dict[str, Any]] = []
    protocol_events = {str(item["event_id"]): item for item in read_json(ROOT / "outputs/N72R9/protocol.json")["source_event_selection"]["events"]}
    for event in sorted(events, key=lambda item: (str(item["sequence"]), int(item["event_frame"]), str(item["event_id"]))):
        gt = replay.legacy._load_gt(str(event["sequence"]))
        frozen_event = protocol_events[event["event_id"]]
        inputs = replay._load_inputs(frozen_event, horizon=100)
        protected = replay.legacy._protected_map(
            inputs["rows"]["c0_source"][int(event["event_frame"])],
            gt,
            int(event["event_frame"]),
            int(event["target_dataset_gt_id"]),
        )
        comparisons: dict[str, dict[str, Any]] = {}
        for name, baseline, treatment in COMPARISONS:
            comparisons[name] = {
                str(horizon): _posthoc_pair(
                    event,
                    rows_by_event[event["event_id"]],
                    baseline,
                    treatment,
                    horizon,
                    gt,
                    protected,
                )
                for horizon in HORIZONS
            }
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
        })
    aggregate_summary = {
        name: {str(horizon): _aggregate_comparison(event_results, name, horizon) for horizon in HORIZONS}
        for name, _, _ in COMPARISONS
    }
    output = {
        "schema_version": "N72R15_CAUSAL_METRICS_V1",
        "status": "PASS_N72R15_POSTHOC_METRICS",
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
        print(json.dumps({"status": result["status"], "event_count": result["event_count"], "independent_sequence_count": result["independent_sequence_count"], "output": str(args.output.resolve())}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R15_FAILURE_V1",
            "status": "FAIL_N72R15_POSTHOC_METRICS",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "created_at_utc": now_utc(),
            "runtime_future_gt_used": False,
        }
        atomic_json(args.output.resolve().with_name("causal_metrics_failure.json"), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
