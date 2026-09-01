#!/usr/bin/env python3
"""Run the N37 M0--M4 paired future replay in an N37-only output tree.

The replay receives one frozen prefix and one candidate-complete future tape
for all five variants.  The tape event contains only event-time annotation,
human ROI evidence, and correction metadata; dataset GT is deliberately kept
in the offline manifest and loaded only after every variant for that event has
finished.  GT is then used solely by ``event_variant_summary`` for post-hoc
scoring.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.ccam_replay import paired_replay, validate_candidate_tape
from sam3_intermot.datasets.dancetrack import DanceTrackDataset

from scripts.n36_real_eval_common import (
    DATA_ROOT,
    FEATURE_DIM,
    HORIZONS,
    atomic_json,
    build_replay_tape,
    evaluate_trace,
    load_manifest,
    variant_config,
)
from scripts.run_n36_replay import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    VARIANTS,
    cluster_bootstrap,
    event_variant_summary,
    finite,
    protected_regression,
)


OUT = ROOT / "outputs/n37"
EVENT_MANIFEST = OUT / "real_event_manifest.json"
EVENT_DIR = OUT / "replay_event_artifacts"
RESULT = OUT / "ccam_paired_replay_results.json"
STAGE = OUT / "stage_03_status.json"
FULL_LOOP_RESULT = OUT / "full_loop_results.json"
N36_TAPE_MANIFEST = ROOT / "outputs/n36/real_tape/tape_manifest.json"

# The event-time annotation and correction evidence are valid runtime inputs.
# ``dataset_gt_id``, ``other_gt_box`` and all N8 selection fields remain only
# in the offline event manifest and are intentionally not sent to paired_replay.
RUNTIME_EVENT_KEYS = {
    "event_id",
    "event_type",
    "action_type",
    "frame",
    "sequence",
    "public_id",
    "canonical_public_id",
    "current_public_id",
    "other_canonical_public_id",
    "other_auto_tid",
    "gt_box",
    "quality",
    "spatial_corrections",
    "human_embedding",
    "competing_embeddings",
    "future_gt_used_runtime",
    "runtime_future_gt_used",
}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def runtime_event_view(event: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key in RUNTIME_EVENT_KEYS
    }


def runtime_item_view(item: dict[str, Any]) -> dict[str, Any]:
    event = item.get("event")
    if not isinstance(event, dict):
        raise ValueError("event_missing_or_not_mapping")
    required = ("prefix_state", "source_tape", "sequence_frame_count")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"runtime_item_missing:{','.join(missing)}")
    return {
        "prefix_state": copy.deepcopy(item["prefix_state"]),
        "source_tape": str(item["source_tape"]),
        "sequence_frame_count": int(item["sequence_frame_count"]),
        "event": runtime_event_view(event),
    }


def build_runtime_tape(item: dict[str, Any], horizon: int = 100) -> dict[str, Any]:
    tape = build_replay_tape(runtime_item_view(item), horizon=horizon)
    tape["protocol"] = "N37_REAL_CCAM_PAIRED_REPLAY_TAPE_V1"
    tape["event"] = runtime_event_view(item["event"])
    tape["runtime_boundary"] = {
        "future_gt_fields_sent_to_replay": [],
        "offline_gt_fields_stripped": [
            "dataset_gt_id",
            "other_gt_box",
            "other_mask",
            "n8_candidate_id",
            "n8_canonical_public_id",
            "n8_current_public_id",
            "n8_event_type",
            "n8_reported_frame",
        ],
        "human_embedding_is_event_time_roi": True,
        "machine_candidate_embedding_used_as_human_anchor": False,
    }
    return tape


def tape_window_check(tape: dict[str, Any], item: dict[str, Any], horizon: int = 100) -> dict[str, Any]:
    event_frame = int(item["event"]["frame"])
    expected_end = min(int(item["sequence_frame_count"]) - 1, event_frame + int(horizon))
    frames = tape.get("frames", [])
    frame_ids = [int(row.get("frame")) for row in frames]
    expected_count = max(0, expected_end - event_frame)
    issues = []
    if len(frames) != expected_count:
        issues.append(f"future_frame_count:{len(frames)}!={expected_count}")
    if frame_ids != list(range(event_frame + 1, expected_end + 1)):
        issues.append("future_frame_range_or_contiguity_invalid")
    if any("gt" in str(key).lower() or str(key).lower().startswith("future_") for row in frames for candidate in row.get("candidates", []) for key in candidate):
        issues.append("future_gt_key_in_candidate")
    return {
        "valid": not issues,
        "issues": issues,
        "event_frame": event_frame,
        "future_frame_start": event_frame + 1,
        "future_frame_end": expected_end,
        "future_frame_count": len(frames),
        "expected_future_frame_count": expected_count,
        "strictly_future": all(frame > event_frame for frame in frame_ids),
        "contiguous": frame_ids == list(range(event_frame + 1, expected_end + 1)),
    }


def n36_tape_gate() -> dict[str, Any]:
    if not N36_TAPE_MANIFEST.is_file():
        return {"status": "FAIL", "reason": "n36_tape_manifest_missing"}
    payload = json.loads(N36_TAPE_MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "status_pass": payload.get("status") == "PASS",
        "candidate_complete": payload.get("candidate_complete") is True,
        "candidate_set_complete": payload.get("candidate_set_complete") is True,
        "runtime_future_gt_used_false": payload.get("runtime_future_gt_used") is False,
        "runtime_gt_read_false": payload.get("runtime_gt_read") is False,
        "third_party_unmodified": payload.get("third_party_modified") is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "artifact": display_path(N36_TAPE_MANIFEST),
    }


def add_identity_error_aliases(summary: dict[str, Any]) -> None:
    """Expose the frozen target error metric without changing its definition."""
    for metrics_name in ("no_write_metrics", "write_metrics"):
        metrics = summary.get(metrics_name, {})
        for horizon in HORIZONS:
            row = metrics.get("horizons", {}).get(str(horizon), {})
            row["target_identity_error_rate"] = row.get("target_missing_rate")
            row["identity_error_semantics"] = (
                "posthoc target public-id absent or below IoU threshold on visible GT frames"
            )


def compact_candidate_audit(audit: Any) -> dict[str, Any]:
    """Keep the per-frame audit contract without serializing large arrays."""
    if not isinstance(audit, dict):
        return {"present": False}
    deltas = np.asarray(audit.get("appearance_score_deltas", []), dtype=float)
    fused_scores = np.asarray(audit.get("fused_scores", []), dtype=float)
    appearance_scores = np.asarray(
        audit.get("appearance_memory_scores", []), dtype=float
    )
    assignment = audit.get("assignment_after_scope", audit.get("assignment", []))
    return {
        "present": True,
        "candidate_public_id_mapping_complete": bool(
            audit.get("candidate_public_id_mapping_complete", False)
        ),
        "candidate_count": len(audit.get("candidates", [])),
        "assignment_count": len(assignment) if isinstance(assignment, list) else 0,
        "assignment_after_scope": copy.deepcopy(assignment),
        "appearance_memory_enabled": bool(
            audit.get("appearance_memory_enabled", False)
        ),
        "appearance_score_delta_max_abs": (
            float(np.max(np.abs(deltas))) if deltas.size else 0.0
        ),
        "fused_score_count": int(fused_scores.size),
        "appearance_memory_score_shape": [int(value) for value in appearance_scores.shape],
        "frame": audit.get("frame"),
        "runtime_future_gt_used": audit.get("runtime_future_gt_used", False),
    }


def compact_future_trace(trace: Any) -> list[dict[str, Any]]:
    """Serialize future rows and a bounded machine-readable audit per frame."""
    if not isinstance(trace, list):
        return []
    output = []
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        rows = []
        for row in entry.get("rows", []):
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                rows.append([int(row[0]), np.asarray(row[1], dtype=float).tolist()])
        output.append(
            {
                "frame": int(entry["frame"]),
                "rows": rows,
                "candidate_audit": compact_candidate_audit(
                    entry.get("candidate_audit")
                ),
            }
        )
    return output


def run(
    manifest_path: Path = EVENT_MANIFEST,
    *,
    max_events: int | None = None,
    event_ids: set[str] | None = None,
    event_dir: Path = EVENT_DIR,
    result_path: Path = RESULT,
    stage_path: Path = STAGE,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    full_loop = json.loads(FULL_LOOP_RESULT.read_text(encoding="utf-8")) if FULL_LOOP_RESULT.is_file() else {}
    upstream_checks = {
        "manifest_pass": manifest.get("status") == "PASS",
        "full_loop_pass": full_loop.get("status") == "PASS",
        "full_loop_all_events_pass": full_loop.get("event_pass_count") == full_loop.get("event_count") == manifest.get("event_count"),
        "full_loop_runtime_future_gt_false": full_loop.get("runtime_future_gt_used") is False,
        "n36_real_tape_pass": n36_tape_gate().get("status") == "PASS",
    }
    events = list(manifest.get("events", []))
    if event_ids is not None:
        events = [item for item in events if str(item["event"]["event_id"]) in event_ids]
    if max_events is not None:
        events = events[: int(max_events)]

    event_rows: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    validation: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    event_dir.mkdir(parents=True, exist_ok=True)

    if not all(upstream_checks.values()):
        failure = {
            "status": "FAIL",
            "error": "n37_replay_upstream_gate_failed",
            "checks": upstream_checks,
        }
        errors.append(failure)
        payload = {
            "protocol": "N37_REAL_CCAM_M0_M4_PAIRED_REPLAY_V1",
            "status": "NOT_RUN_UPSTREAM_BLOCKED",
            "real_data_status": "NOT_RUN_UPSTREAM_BLOCKED",
            "event_count": 0,
            "successful_event_count": 0,
            "independent_sequence_count": 0,
            "runtime_future_gt_used": False,
            "gt_used_only_posthoc_scoring": True,
            "upstream_checks": upstream_checks,
            "errors": errors,
        }
        atomic_json(result_path, payload)
        atomic_json(
            stage_path,
            {
                "stage": "N37-03",
                "status": payload["status"],
                "real_data_status": payload["real_data_status"],
                "artifacts": [display_path(result_path)],
                "upstream_checks": upstream_checks,
                "errors": errors,
                "downstream_authorized": False,
            },
        )
        return payload

    for item in events:
        event = item["event"]
        event_id = str(event["event_id"])
        sequence = str(event["sequence"])
        action_row: dict[str, Any] = {
            "event_id": event_id,
            "sequence": sequence,
            "event_frame": int(event["frame"]),
            "action_type": event["action_type"],
            "interaction_source": "simulated_from_gt",
            "synthetic": False,
            "variants": {},
        }
        replays: dict[str, dict[str, Any]] = {}
        tape = None
        tape_check = None
        try:
            tape = build_runtime_tape(item, horizon=100)
            tape_check = tape_window_check(tape, item, horizon=100)
            tape_validation = validate_candidate_tape(tape, feat_dim=FEATURE_DIM)
            validation[f"{event_id}:tape"] = {
                "tape_window": tape_check,
                "candidate_tape": tape_validation,
            }
            if not tape_check["valid"]:
                raise RuntimeError(f"candidate tape window check failed: {tape_check}")
            if not tape_validation["valid"] or not tape_validation["candidate_complete"]:
                raise RuntimeError(f"candidate tape validation failed: {tape_validation}")
        except Exception as exc:
            failure = {
                "event_id": event_id,
                "sequence": sequence,
                "event_frame": int(event["frame"]),
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            errors.append(failure)
            for name in VARIANTS:
                action_row["variants"][name] = {
                    **failure,
                    "variant": name,
                }
                atomic_json(event_dir / event_id / f"{name}.json", action_row["variants"][name])
            action_row["status"] = "FAIL"
            event_rows.append(action_row)
            print(json.dumps({"event_id": event_id, "status": "FAIL"}, sort_keys=True), flush=True)
            continue

        # All five branches consume the same in-memory serialized tape.  GT
        # is intentionally not loaded until this loop has completed.
        for name in VARIANTS:
            try:
                config, _description = variant_config(name)
                replay = paired_replay(
                    tape,
                    config=config,
                    feat_dim=FEATURE_DIM,
                    write_branch_uses_appearance_memory=(name != "M0"),
                )
                if replay.get("status") != "PASS":
                    raise RuntimeError(f"paired_replay returned {replay.get('status')}: {replay.get('validation')}")
                replays[name] = replay
            except Exception as exc:
                failure = {
                    "event_id": event_id,
                    "sequence": sequence,
                    "event_frame": int(event["frame"]),
                    "variant": name,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                errors.append(failure)
                action_row["variants"][name] = failure

        # This is the first point at which offline GT may enter the process.
        # It is used only for post-hoc scoring of already completed replays.
        dataset = DanceTrackDataset(str(DATA_ROOT), sequences=[sequence], split="train")
        gt_frames = dataset.load_gt(sequence)
        for name in VARIANTS:
            if name in action_row["variants"]:
                continue
            replay = replays[name]
            artifact_path = event_dir / event_id / f"{name}.json"
            try:
                summary = event_variant_summary(
                    str(event["action_type"]), event, name, replay, gt_frames
                )
                add_identity_error_aliases(summary)
                summary["validation"] = validation[f"{event_id}:tape"]
                summary["runtime_event_keys"] = sorted(runtime_event_view(event))
                summary["posthoc_gt_loaded_after_all_variants"] = True
                summary["future_trace"] = {
                    "memory_write_false": compact_future_trace(
                        replay["branches"]["memory_write=False"]["future_trace"]
                    ),
                    "memory_write_true": compact_future_trace(
                        replay["branches"]["memory_write=True"]["future_trace"]
                    ),
                }
                summary["trace_semantics"] = {
                    "rows_are_future_frames_only": True,
                    "contains_candidate_audit": True,
                    "gt_not_in_runtime_trace": True,
                }
                atomic_json(artifact_path, summary)
                # Per-frame traces live in the event artifact.  Keep the
                # aggregate result compact so it remains a practical audit
                # index rather than duplicating every frame's audit arrays.
                aggregate_summary = copy.deepcopy(summary)
                aggregate_summary.pop("future_trace", None)
                aggregate_summary["future_trace_artifact"] = display_path(artifact_path)
                action_row["variants"][name] = aggregate_summary
                variant_rows[name].append(aggregate_summary)
            except Exception as exc:
                failure = {
                    "event_id": event_id,
                    "sequence": sequence,
                    "event_frame": int(event["frame"]),
                    "variant": name,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                errors.append(failure)
                action_row["variants"][name] = failure
                atomic_json(artifact_path, failure)
        action_row["status"] = "PASS" if all(
            row.get("status") == "PASS" for row in action_row["variants"].values()
        ) else "FAIL"
        event_rows.append(action_row)
        print(json.dumps({"event_id": event_id, "status": action_row["status"]}, sort_keys=True), flush=True)
        del dataset, gt_frames, replays, tape
        gc.collect()

    bootstrap: dict[str, Any] = {}
    gate_checks: dict[str, Any] = {}
    for name in VARIANTS:
        rows = variant_rows[name]
        bootstrap[name] = {str(horizon): cluster_bootstrap(rows, horizon) for horizon in HORIZONS}
        regression_rows = []
        for action_row in event_rows:
            summary = action_row.get("variants", {}).get(name, {})
            if summary.get("status") != "PASS":
                continue
            event = next(
                item["event"] for item in events if item["event"]["event_id"] == action_row["event_id"]
            )
            regression_rows.append(
                protected_regression(
                    summary["no_write_metrics"],
                    summary["write_metrics"],
                    event,
                    horizon=20,
                )
            )
        gate_checks[name] = {
            "sequence_cluster_h20_lower_ci": bootstrap[name]["20"].get("lower"),
            "sequence_cluster_h50_lower_ci": bootstrap[name]["50"].get("lower"),
            "sequence_cluster_h100_lower_ci": bootstrap[name]["100"].get("lower"),
            "protected_regression": regression_rows,
            "protected_no_obvious_regression": bool(regression_rows)
            and all(row["no_obvious_regression"] for row in regression_rows),
        }

    successful_events = sum(row.get("status") == "PASS" for row in event_rows)
    independent_sequences = len({row.get("sequence") for row in event_rows if row.get("sequence")})
    complete_run = (
        max_events is None
        and event_ids is None
        and len(events) == manifest.get("event_count") == 24
    )
    execution_status = (
        "PASS"
        if complete_run and not errors and successful_events == len(events)
        else ("PARTIAL" if successful_events else "FAIL")
    )
    all_variant_rows = all(
        len(row.get("variants", {})) == len(VARIANTS)
        and all(summary.get("status") == "PASS" for summary in row.get("variants", {}).values())
        for row in event_rows
    )
    authorization_checks = {
        "real_tape_complete": upstream_checks["n36_real_tape_pass"],
        "real_full_loop_pass": upstream_checks["full_loop_pass"],
        "complete_24_event_replay": complete_run and successful_events == 24,
        "all_events_have_five_pass_variants": all_variant_rows,
        "at_least_twelve_independent_event_sequences": independent_sequences >= 12,
        "paired_replay_post_treatment_leakage_free": execution_status == "PASS"
        and all(
            all(
                bool(summary.get("causal_boundary", {}).get("only_branch_difference_memory_write", False))
                and bool(summary.get("causal_boundary", {}).get("event_frame_excluded_from_future_tape", False))
                and summary.get("posthoc_gt_loaded_after_all_variants") is True
                for summary in row.get("variants", {}).values()
            )
            for row in event_rows
        ),
    }
    for name in ("M2", "M3", "M4"):
        authorization_checks[f"{name}_h20_sequence_cluster_lower_ci_gt_zero"] = bool(
            gate_checks.get(name, {}).get("sequence_cluster_h20_lower_ci") is not None
            and gate_checks[name]["sequence_cluster_h20_lower_ci"] > 0.0
        )
        authorization_checks[f"{name}_protected_no_obvious_regression"] = bool(
            gate_checks.get(name, {}).get("protected_no_obvious_regression", False)
        )
    future_effect_gate_pass = all(authorization_checks.values())
    effects = [gate_checks.get(name, {}).get("sequence_cluster_h20_lower_ci") for name in ("M2", "M3", "M4")]
    finite_effects = [float(value) for value in effects if finite(value)]
    if not finite_effects:
        effect_status = "NOT_COMPUTABLE"
    elif all(value > 0 for value in finite_effects):
        effect_status = "POSITIVE"
    elif all(value <= 0 for value in finite_effects):
        effect_status = "NEGATIVE" if any(value < 0 for value in finite_effects) else "NULL"
    else:
        effect_status = "NULL"

    payload = {
        "protocol": "N37_REAL_CCAM_M0_M4_PAIRED_REPLAY_V1",
        "status": execution_status,
        "real_data_status": execution_status,
        "synthetic": False,
        "split": "train/train_fold",
        "event_manifest": display_path(manifest_path),
        "event_count": len(events),
        "successful_event_count": successful_events,
        "independent_sequence_count": independent_sequences,
        "runtime_future_gt_used": False,
        "gt_used_only_posthoc_scoring": True,
        "variants": VARIANTS,
        "events": event_rows,
        "validation": validation,
        "upstream_checks": upstream_checks,
        "sequence_cluster_bootstrap": bootstrap,
        "future_effect_gate": {
            "status": "PASS" if future_effect_gate_pass else "NOT_AUTHORIZED",
            "checks": authorization_checks,
            "horizon_primary": 20,
            "metric": "identity_utility_delta = mean(target IoU delta, missing-rate reduction)",
            "cluster_unit": "independent sequence",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "strict_lower_ci_requirement": "M2/M3/M4 H20 lower CI strictly > 0",
        },
        "ccam_future_effect": effect_status,
        "calibration_head": "AUTHORIZED" if future_effect_gate_pass else "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "AUTHORIZED_PILOT_ONLY" if future_effect_gate_pass else "NOT_AUTHORIZED",
        "errors": errors,
        "artifacts": {
            "event_manifest": display_path(manifest_path),
            "event_artifacts": display_path(event_dir),
            "result": display_path(result_path),
            "full_loop": display_path(FULL_LOOP_RESULT),
        },
        "metric_notes": {
            "identity_error": "target public-id posthoc missing/below IoU threshold; no GT enters runtime",
            "idf1_hota_assa": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
            "re_correction_count": "posthoc contiguous identity-error opportunity proxy; not observed clicks",
            "protected_identity_regression": "compares write/no-write posthoc GT metrics for non-event identities",
        },
    }
    atomic_json(result_path, payload)
    stage = {
        "stage": "N37-03",
        "status": execution_status,
        "real_data_status": execution_status,
        "artifacts": [display_path(result_path), display_path(event_dir)],
        "event_count": len(events),
        "successful_event_count": successful_events,
        "independent_sequence_count": independent_sequences,
        "runtime_future_gt_used": False,
        "gt_used_only_posthoc_scoring": True,
        "upstream_checks": upstream_checks,
        "future_effect_gate": payload["future_effect_gate"],
        "ccam_future_effect": effect_status,
        "calibration_head": payload["calibration_head"],
        "selector": payload["selector"],
        "decoder_lora": payload["decoder_lora"],
        "errors": errors,
        "downstream_authorized": bool(execution_status == "PASS"),
        "next_action": (
            "Write N37 Stage D gate/report; do not train before it."
            if execution_status == "PASS"
            else "Preserve replay failures and do not authorize learning."
        ),
    }
    atomic_json(stage_path, stage)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EVENT_MANIFEST)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--event-id", action="append", default=None)
    parser.add_argument("--event-dir", type=Path, default=EVENT_DIR)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--stage", type=Path, default=STAGE)
    args = parser.parse_args()
    payload = run(
        args.manifest,
        max_events=args.max_events,
        event_ids=None if args.event_id is None else set(args.event_id),
        event_dir=args.event_dir,
        result_path=args.result,
        stage_path=args.stage,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "event_count": payload["event_count"],
                "successful_event_count": payload["successful_event_count"],
                "ccam_future_effect": payload.get("ccam_future_effect"),
                "future_effect_gate": payload.get("future_effect_gate", {}).get("status"),
                "output": display_path(args.result),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
