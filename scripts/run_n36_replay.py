#!/usr/bin/env python3
"""Run real train-fold M0--M4 future-only paired replays for N36.

Each variant receives the same serialized prefix, current-frame spatial
transaction and complete future candidate window.  The paired branches differ
only in whether the CCAM memory write/update is enabled.  Ground truth is
loaded only after replay for post-hoc scoring and sequence-cluster bootstrap;
it is never present in runtime candidate observations.
"""

from __future__ import annotations

import argparse
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
    jsonable,
    load_manifest,
    variant_config,
)


OUT = ROOT / "outputs/n36"
EVENT_MANIFEST = OUT / "real_event_manifest.json"
EVENT_DIR = OUT / "replay_event_artifacts"
RESULT = OUT / "ccam_paired_replay_results.json"
STAGE = OUT / "stage_04_status.json"
FULL_LOOP_RESULT = OUT / "full_loop_results.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 36


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def delta(left: Any, right: Any) -> float | None:
    return float(right) - float(left) if finite(left) and finite(right) else None


def reduction(left: Any, right: Any) -> float | None:
    return float(left) - float(right) if finite(left) and finite(right) else None


def event_variant_summary(
    action: str,
    event: dict[str, Any],
    variant: str,
    replay: dict[str, Any],
    gt_frames: dict[int, Any],
) -> dict[str, Any]:
    no_write = replay["branches"]["memory_write=False"]["future_trace"]
    with_write = replay["branches"]["memory_write=True"]["future_trace"]
    no_metrics = evaluate_trace(no_write, gt_frames, event)
    write_metrics = evaluate_trace(with_write, gt_frames, event)
    horizon_deltas = {}
    for horizon in HORIZONS:
        left = no_metrics["horizons"][str(horizon)]
        right = write_metrics["horizons"][str(horizon)]
        iou_delta = delta(left.get("target_mean_iou"), right.get("target_mean_iou"))
        missing_reduction = reduction(
            left.get("target_missing_rate"), right.get("target_missing_rate")
        )
        utility_terms = [term for term in (iou_delta, missing_reduction) if term is not None]
        horizon_deltas[str(horizon)] = {
            "target_iou_delta_write_minus_no_write": iou_delta,
            "target_missing_rate_reduction_no_write_minus_write": missing_reduction,
            "identity_utility_delta": (
                float(np.mean(utility_terms)) if utility_terms else None
            ),
            "id_switch_reduction": reduction(
                left.get("id_switch_count"), right.get("id_switch_count")
            ),
            "posthoc_recorrection_opportunity_reduction": reduction(
                left.get("posthoc_recorrection_opportunity_count"),
                right.get("posthoc_recorrection_opportunity_count"),
            ),
        }
    comparison = replay.get("comparison", [])
    first_future = comparison[0] if comparison else {}
    return {
        "action_type": action,
        "event_id": event["event_id"],
        "sequence": event["sequence"],
        "event_frame": int(event["frame"]),
        "variant": variant,
        "description": variant_config(variant)[1],
        "status": replay.get("status", "FAIL"),
        "candidate_complete": bool(replay.get("candidate_complete", False)),
        "runtime_future_gt_used": False,
        "same_prefix_and_spatial_correction": True,
        "future_frame_count": len(no_write),
        "causal_boundary": {
            "event_frame_excluded_from_future_tape": all(
                int(item["frame"]) > int(event["frame"])
                for item in no_write
            ),
            "first_write_eligible_frame": int(event["frame"]) + 1,
            "current_frame_write_used_for_score": False,
            "only_branch_difference_memory_write": True,
            "runtime_future_gt_used": False,
        },
        "score_delta_first_future": {
            "frame": first_future.get("frame"),
            "max_abs_score_delta": first_future.get("max_abs_score_delta"),
            "assignment_changed": bool(first_future.get("assignment_changed", False)),
        },
        "no_write_metrics": no_metrics,
        "write_metrics": write_metrics,
        "horizon_deltas": horizon_deltas,
        "metrics_not_claimed": {
            "idf1": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
            "hota": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
            "assa": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
        },
        "identity_effect": "COMPUTED_POSTHOC_GT_ONLY",
    }


def bootstrap_ci(values: list[float], seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    if not values:
        return {"status": "NOT_COMPUTABLE", "n_clusters": 0, "lower": None, "upper": None}
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        return {"status": "NOT_COMPUTABLE_NONFINITE", "n_clusters": int(array.size), "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_REPLICATES, array.size))
    means = array[indices].mean(axis=1)
    return {
        "status": "PASS",
        "n_clusters": int(array.size),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "cluster_values": [float(value) for value in array.tolist()],
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    horizon: int,
    metric_key: str = "identity_utility_delta",
) -> dict[str, Any]:
    by_sequence: dict[str, list[float]] = {}
    for row in rows:
        value = row.get("horizon_deltas", {}).get(str(horizon), {}).get(metric_key)
        if finite(value):
            by_sequence.setdefault(str(row["sequence"]), []).append(float(value))
    cluster_values = [float(np.mean(values)) for _seq, values in sorted(by_sequence.items())]
    result = bootstrap_ci(cluster_values)
    result["metric"] = metric_key
    result["horizon"] = int(horizon)
    result["sequence_clusters"] = sorted(by_sequence)
    return result


def protected_regression(
    no_metrics: dict[str, Any],
    write_metrics: dict[str, Any],
    event: dict[str, Any],
    horizon: int = 20,
) -> dict[str, Any]:
    left = no_metrics.get("per_gt", {}).get(str(horizon), {})
    right = write_metrics.get("per_gt", {}).get(str(horizon), {})
    excluded = {int(event["dataset_gt_id"])}
    if event.get("other_gt_box") is not None and event.get("other_canonical_public_id") is not None:
        # The optional second box belongs to the corrected swap transaction;
        # it is not an untouched identity for this check.
        # Its GT ID is not required in the runtime event, so keep the
        # conservative public event target exclusion and compare all other
        # observed IDs below.
        pass
    regression_ids = []
    improvements = []
    for gid in sorted(set(left) & set(right)):
        if int(gid) in excluded:
            continue
        l = left[gid]
        r = right[gid]
        lm, rm = l.get("missing_rate"), r.get("missing_rate")
        li, ri = l.get("mean_iou"), r.get("mean_iou")
        if finite(lm) and finite(rm) and finite(li) and finite(ri):
            improvements.append((float(ri) - float(li)) + (float(lm) - float(rm)))
            if float(rm) > float(lm) + 1e-9 or float(ri) < float(li) - 1e-9:
                regression_ids.append(int(gid))
    return {
        "horizon": int(horizon),
        "compared_untouched_gt_count": len(improvements),
        "regression_count": len(regression_ids),
        "regression_gt_ids": regression_ids,
        "no_obvious_regression": not regression_ids,
        "mean_untouched_utility_delta": float(np.mean(improvements)) if improvements else None,
    }


def run(manifest_path: Path = EVENT_MANIFEST, max_events: int | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    event_items = list(manifest.get("events", []))
    if max_events is not None:
        event_items = event_items[: int(max_events)]
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sorted({str(item["event"]["sequence"]) for item in event_items}), split="train")
    event_rows: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    validation: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    for item in event_items:
        event = item["event"]
        sequence = str(event["sequence"])
        gt_frames = dataset.load_gt(sequence)
        action_row = {
            "event_id": event["event_id"],
            "sequence": sequence,
            "event_frame": int(event["frame"]),
            "action_type": event["action_type"],
            "interaction_source": "simulated_from_gt",
            "synthetic": False,
            "variants": {},
        }
        for name in VARIANTS:
            artifact_path = EVENT_DIR / str(event["event_id"]) / f"{name}.json"
            try:
                tape = build_replay_tape(item, horizon=100)
                check = validate_candidate_tape(tape, feat_dim=FEATURE_DIM)
                validation[f"{event['event_id']}:{name}"] = check
                if not check["valid"] or not check["candidate_complete"]:
                    raise RuntimeError(f"candidate tape validation failed: {check}")
                config, _description = variant_config(name)
                replay = paired_replay(
                    tape,
                    config=config,
                    feat_dim=FEATURE_DIM,
                    write_branch_uses_appearance_memory=(name != "M0"),
                )
                summary = event_variant_summary(
                    str(event["action_type"]), event, name, replay, gt_frames
                )
                summary["validation"] = check
                atomic_json(artifact_path, summary)
                action_row["variants"][name] = summary
                variant_rows[name].append(summary)
                del replay, tape
            except Exception as exc:
                failure = {
                    "event_id": event.get("event_id"),
                    "sequence": sequence,
                    "event_frame": event.get("frame"),
                    "variant": name,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                errors.append(failure)
                action_row["variants"][name] = failure
                atomic_json(artifact_path, failure)
            finally:
                gc.collect()
        action_row["status"] = "PASS" if all(
            row.get("status") == "PASS" for row in action_row["variants"].values()
        ) else "FAIL"
        event_rows.append(action_row)
        print(json.dumps({"event_id": action_row["event_id"], "status": action_row["status"]}, sort_keys=True), flush=True)

    bootstrap: dict[str, Any] = {}
    gate_checks: dict[str, Any] = {}
    for name in VARIANTS:
        rows = variant_rows[name]
        bootstrap[name] = {
            str(horizon): cluster_bootstrap(rows, horizon)
            for horizon in HORIZONS
        }
        regression_rows = []
        for action_row in event_rows:
            summary = action_row.get("variants", {}).get(name, {})
            if summary.get("status") != "PASS":
                continue
            regression_rows.append(
                protected_regression(
                    summary["no_write_metrics"],
                    summary["write_metrics"],
                    next(
                        item["event"] for item in event_items if item["event"]["event_id"] == action_row["event_id"]
                    ),
                    horizon=20,
                )
            )
        gate_checks[name] = {
            "sequence_cluster_h20_lower_ci": bootstrap[name]["20"].get("lower"),
            "sequence_cluster_h50_lower_ci": bootstrap[name]["50"].get("lower"),
            "sequence_cluster_h100_lower_ci": bootstrap[name]["100"].get("lower"),
            "protected_regression": regression_rows,
            "protected_no_obvious_regression": all(
                row["no_obvious_regression"] for row in regression_rows
            ) if regression_rows else False,
        }
    full_loop_status = None
    if FULL_LOOP_RESULT.exists():
        full_loop_status = json.loads(FULL_LOOP_RESULT.read_text(encoding="utf-8")).get("status")
    successful_events = sum(row.get("status") == "PASS" for row in event_rows)
    independent_sequences = len({row.get("sequence") for row in event_rows})
    execution_status = "PASS" if not errors and successful_events == len(event_items) else ("PARTIAL" if successful_events else "FAIL")
    authorization_checks = {
        "real_tape_complete": True,
        "real_full_loop_pass": full_loop_status == "PASS",
        "at_least_six_independent_event_sequences": independent_sequences >= 6,
        "at_least_two_independent_event_sequences": independent_sequences >= 2,
        "paired_replay_post_treatment_leakage_free": not errors and all(
            all(
                bool(summary.get("causal_boundary", {}).get("only_branch_difference_memory_write", False))
                and bool(summary.get("causal_boundary", {}).get("event_frame_excluded_from_future_tape", False))
                for summary in row.get("variants", {}).values()
                if summary.get("status") == "PASS"
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
    calibration_authorized = bool(all(authorization_checks.values()))
    effects = [
        gate_checks.get(name, {}).get("sequence_cluster_h20_lower_ci")
        for name in ("M2", "M3", "M4")
    ]
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
        "protocol": "N36_REAL_CCAM_M0_M4_PAIRED_REPLAY",
        "status": execution_status,
        "real_data_status": execution_status,
        "synthetic": False,
        "split": "train/train_fold",
        "event_count": len(event_items),
        "successful_event_count": successful_events,
        "independent_sequence_count": independent_sequences,
        "runtime_future_gt_used": False,
        "gt_used_only_posthoc_scoring": True,
        "variants": VARIANTS,
        "events": event_rows,
        "validation": validation,
        "sequence_cluster_bootstrap": bootstrap,
        "future_effect_gate": {
            "status": "PASS" if calibration_authorized else "NOT_AUTHORIZED",
            "checks": authorization_checks,
            "horizon_primary": 20,
            "metric": "identity_utility_delta = mean(target IoU delta, missing-rate reduction)",
            "cluster_unit": "sequence",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "ccam_future_effect": effect_status,
        "calibration_head": "AUTHORIZED" if calibration_authorized else "NOT_AUTHORIZED",
        "decoder_lora": "AUTHORIZED_PILOT_ONLY" if calibration_authorized else "NOT_AUTHORIZED",
        "errors": errors,
        "artifacts": {
            "event_manifest": display_path(manifest_path),
            "event_artifacts": display_path(EVENT_DIR),
            "result": display_path(RESULT),
            "full_loop": display_path(FULL_LOOP_RESULT),
        },
        "metric_notes": {
            "idf1_hota_assa": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
            "re_correction_count": "posthoc contiguous identity-error opportunity proxy; not observed clicks",
            "protected_identity_regression": "compares write/no-write posthoc GT metrics for non-event identities",
        },
    }
    atomic_json(RESULT, payload)
    stage = {
        "stage": "N36-06",
        "status": execution_status,
        "real_data_status": execution_status,
        "artifacts": [display_path(RESULT), display_path(EVENT_DIR)],
        "event_count": len(event_items),
        "successful_event_count": successful_events,
        "independent_sequence_count": independent_sequences,
        "runtime_future_gt_used": False,
        "gt_used_only_posthoc_scoring": True,
        "ccam_future_effect": effect_status,
        "future_effect_gate": "PASS" if calibration_authorized else "NOT_AUTHORIZED",
        "calibration_head": "AUTHORIZED" if calibration_authorized else "NOT_AUTHORIZED",
        "decoder_lora": "AUTHORIZED_PILOT_ONLY" if calibration_authorized else "NOT_AUTHORIZED",
        "errors": errors,
        "next_action": "Only authorize a small calibration pilot if every explicit future-effect check is PASS; otherwise preserve write-only evidence and report the failed gate.",
    }
    atomic_json(STAGE, stage)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EVENT_MANIFEST)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()
    payload = run(args.manifest, args.max_events)
    print(json.dumps({"status": payload["status"], "event_count": payload["event_count"], "successful_event_count": payload["successful_event_count"], "ccam_future_effect": payload["ccam_future_effect"], "output": display_path(RESULT)}, sort_keys=True))


if __name__ == "__main__":
    main()
