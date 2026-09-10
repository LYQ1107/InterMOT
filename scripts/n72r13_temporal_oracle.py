#!/usr/bin/env python3
"""Sequential N72R13 temporal-intervention oracle.

The oracle is intentionally post-hoc and is never presented as a runtime
method.  Runtime candidate construction and both branches are sealed before
GT is opened for the current opportunity.  GT then chooses APPLY only when
both target and global 20-frame values are strictly positive.
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
import re
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import sequence_cluster_bootstrap  # noqa: E402
from sam3_intermot.reacquisition.temporal_intervention_value import (  # noqa: E402
    PRIMARY_VALUE_HORIZON,
    VALUE_HORIZONS,
    build_pctis_proposal,
    clone_temporal_state,
    rollout_intervention_pair,
    serialize_intervention_pair,
    temporal_state_digest,
)
from sam3_intermot.reacquisition.temporal_state_policy import initialize_temporal_state  # noqa: E402
from scripts import n72r11_on_demand_replay as replay  # noqa: E402


HORIZON = 100
IOU_THRESHOLD = 0.50
BOOTSTRAP_SEED = 7213
BOOTSTRAP_REPETITIONS = 2000
DEFAULT_PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_PCTIS = ROOT / "outputs/N72R11R4/pctis_onpolicy_finetune/pctis_onpolicy_finetuned.pt"
DEFAULT_OUTPUT = ROOT / "outputs/N72R13/temporal_oracle"
OPPORTUNITY_AUDIT = ROOT / "outputs/N72R13/intervention_opportunity_audit.json"


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


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _initial_state(inputs: Mapping[str, Any]):
    event_frame = int(inputs["event_frame"])
    target_rows = list(inputs["rows"]["target_stream_source"][event_frame].get("candidate_rows", []))
    raw = target_rows[0].get("official_raw_sam_id") if target_rows else None
    scope = target_rows[0].get("native_scope", target_rows[0].get("native_tid_scope")) if target_rows else None
    return initialize_temporal_state(
        anchor_feature=inputs["anchor"],
        anchor_box=inputs["anchor_box"],
        previous_raw_sam_id=None if raw is None else int(raw),
        previous_native_scope=None if scope is None else str(scope),
    )


def _event_frame_row(inputs: Mapping[str, Any], variant: str) -> dict[str, Any]:
    row = replay._event_frame_row(inputs, variant)
    row["action_type"] = str(inputs.get("action_type", "UNKNOWN"))
    row["oracle_upper_bound_only"] = True
    row["runtime_eligible"] = False
    return row


def _scan_runtime(value: Any, location: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"} and nested is not False:
                errors.append(f"{location}/{key_text}={nested!r}")
            errors.extend(_scan_runtime(nested, f"{location}/{key_text}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_scan_runtime(nested, f"{location}/{index}"))
    return errors


def _quality(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_frame: int,
    target_public: int,
    target_gid: int,
    protected: Mapping[int, int],
    gt: Mapping[int, Mapping[int, Any]],
    horizon: int,
) -> dict[str, Any]:
    by_frame = {int(row["frame"]): row for row in rows}
    target_visible = 0
    target_correct = 0
    target_iou_sum = 0.0
    target_missing = 0
    global_visible = 0
    global_correct = 0
    protected_visible = 0
    protected_correct = 0
    identity_error_frames = 0
    id_switches = 0
    recorrection = 0
    previous_uid: str | None = None
    previous_error = False
    missing_frames: list[int] = []
    for frame in range(int(event_frame) + 1, int(event_frame) + int(horizon) + 1):
        row = by_frame.get(frame)
        if row is None:
            raise RuntimeError(f"trajectory is missing frame {frame}")
        gt_frame = gt.get(frame, {})
        target = gt_frame.get(int(target_gid))
        target_uid, _target_candidate = replay.legacy._target_binding(row, int(target_public))
        if target is not None:
            target_visible += 1
            target_iou, _ = replay.legacy._public_box_for_gt(row, int(target_public), target["box"])
            target_iou_sum += float(target_iou)
            correct = bool(target_iou >= IOU_THRESHOLD)
            target_correct += int(correct)
            identity_error_frames += int(not correct)
            target_missing += int(target_uid is None)
            missing_frames.append(frame) if target_uid is None else None
            current_error = not correct
            recorrection += int(current_error and not previous_error)
            previous_error = current_error
        if previous_uid is not None and target_uid is not None and str(previous_uid) != str(target_uid):
            id_switches += 1
        if target_uid is not None:
            previous_uid = str(target_uid)
        identities = {int(target_gid): int(target_public), **{int(gid): int(pid) for gid, pid in protected.items()}}
        for gid, pid in identities.items():
            item = gt_frame.get(int(gid))
            if item is None:
                continue
            global_visible += 1
            iou, _ = replay.legacy._public_box_for_gt(row, int(pid), item["box"])
            global_correct += int(iou >= IOU_THRESHOLD)
            if int(gid) != int(target_gid):
                protected_visible += 1
                protected_correct += int(iou >= IOU_THRESHOLD)
    return {
        "horizon": int(horizon),
        "target_visible_frames": int(target_visible),
        "target_correct_frames": int(target_correct),
        "target_accuracy": None if target_visible == 0 else float(target_correct / target_visible),
        "target_iou": None if target_visible == 0 else float(target_iou_sum / target_visible),
        "target_identity_error_frames": int(identity_error_frames),
        "target_identity_error": None if target_visible == 0 else float(identity_error_frames / target_visible),
        "target_missing_frames": int(target_missing),
        "global_visible_identity_frames": int(global_visible),
        "global_correct_identity_frames": int(global_correct),
        "global_pid_accuracy": None if global_visible == 0 else float(global_correct / global_visible),
        "global_identity_error": None if global_visible == 0 else float(1.0 - global_correct / global_visible),
        "protected_visible_identity_frames": int(protected_visible),
        "protected_correct_identity_frames": int(protected_correct),
        "protected_accuracy": None if protected_visible == 0 else float(protected_correct / protected_visible),
        "id_switch_count": int(id_switches),
        "recorrection_opportunity_count": int(recorrection),
        "missing_frame_ids": missing_frames,
    }


def _pair_posthoc(
    pair: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    event: Mapping[str, Any],
    gt: Mapping[int, Mapping[int, Any]],
    protected: Mapping[int, int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    keep_rows = pair["keep"]["passive"].runtime_rows
    apply_rows = pair["apply"]["passive"].runtime_rows
    for horizon in VALUE_HORIZONS:
        if int(horizon) > len(keep_rows) or int(horizon) > len(apply_rows):
            continue
        keep = _quality(
            keep_rows,
            event_frame=int(pair["intervention_frame"]),
            target_public=int(inputs["target_public_id"]),
            target_gid=int(event["dataset_gt_id"]),
            protected=protected,
            gt=gt,
            horizon=int(horizon),
        )
        applied = _quality(
            apply_rows,
            event_frame=int(pair["intervention_frame"]),
            target_public=int(inputs["target_public_id"]),
            target_gid=int(event["dataset_gt_id"]),
            protected=protected,
            gt=gt,
            horizon=int(horizon),
        )
        output[str(horizon)] = {
            "keep": keep,
            "apply": applied,
            "delta_target_accuracy": None if keep["target_accuracy"] is None or applied["target_accuracy"] is None else float(applied["target_accuracy"] - keep["target_accuracy"]),
            "delta_global_pid_accuracy": None if keep["global_pid_accuracy"] is None or applied["global_pid_accuracy"] is None else float(applied["global_pid_accuracy"] - keep["global_pid_accuracy"]),
            "delta_target_iou": None if keep["target_iou"] is None or applied["target_iou"] is None else float(applied["target_iou"] - keep["target_iou"]),
            "delta_protected_accuracy": None if keep["protected_accuracy"] is None or applied["protected_accuracy"] is None else float(applied["protected_accuracy"] - keep["protected_accuracy"]),
            "delta_id_switch_improvement": int(keep["id_switch_count"] - applied["id_switch_count"]),
        }
    return output


def _pair_decision(posthoc: Mapping[str, Any]) -> bool:
    primary = posthoc.get(str(PRIMARY_VALUE_HORIZON), {})
    target = primary.get("delta_target_accuracy")
    global_value = primary.get("delta_global_pid_accuracy")
    return bool(target is not None and global_value is not None and float(target) > 0.0 and float(global_value) > 0.0)


def _proposal_context(inputs: Mapping[str, Any], frame: int, state: Any, model: Any, device: torch.device) -> dict[str, Any]:
    local = dict(inputs)
    local["require_positive_geometry"] = True
    return build_pctis_proposal(inputs=local, frame=int(frame), state=state, pctis_model=model, device=device)


def _run_smoke(
    events: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    """Find one deterministic RECOVER opportunity and run exactly one pair."""

    audit = read_json(OPPORTUNITY_AUDIT)
    candidate_keys = {
        (str(item["event_id"]).split(":secondary:", 1)[0], int(item["frame"]))
        for item in audit.get("opportunities", [])
        if str(item.get("action_type")) == "RECOVER_IDENTITY"
    }
    ordered = sorted(
        [dict(event) for event in events if str(event.get("action_type")) == "RECOVER_IDENTITY"],
        key=lambda item: (str(item["sequence"]), int(item["event_frame"]), str(item["event_id"])),
    )
    failures: list[dict[str, Any]] = []
    for event in ordered:
        if not any(key[0] == str(event["event_id"]) for key in candidate_keys):
            continue
        try:
            inputs = replay._load_inputs(event, horizon=HORIZON)
            inputs = dict(inputs)
            inputs["action_type"] = str(event["action_type"])
            inputs["require_positive_geometry"] = True
            state = _initial_state(inputs)
            for frame in range(int(event["event_frame"]) + 1, int(event["event_frame"]) + 81):
                proposal = _proposal_context(inputs, frame, state, model, device)
                if not proposal["effective_opportunity"]:
                    base = {"pool": proposal["pool"], "pool_audit": proposal["pool_audit"], "scored": proposal["base"]}
                    replay_row = __import__("sam3_intermot.reacquisition.temporal_intervention_value", fromlist=["_commit_base_frame"])._commit_base_frame
                    replay_row(inputs=inputs, frame=frame, state=state, base=base, branch="PASSIVE_BASE")
                    continue
                pair = rollout_intervention_pair(
                    inputs=inputs,
                    frame=frame,
                    state_before=state,
                    pctis_proposal=proposal,
                    max_horizon=PRIMARY_VALUE_HORIZON,
                    require_positive_geometry=True,
                )
                checks = {
                    "initial_state_equal_for_branches": pair.get("initial_state_equal_for_branches") is True,
                    "runtime_future_gt_flags_false": pair.get("runtime_future_gt_used") is False,
                    "runtime_gt_read_flags_false": pair.get("runtime_gt_read") is False,
                    "frame_axis_keep": [int(row["frame"]) for row in pair["keep"]["passive"].runtime_rows] == list(range(frame + 1, frame + PRIMARY_VALUE_HORIZON + 1)),
                    "frame_axis_apply": [int(row["frame"]) for row in pair["apply"]["passive"].runtime_rows] == list(range(frame + 1, frame + PRIMARY_VALUE_HORIZON + 1)),
                    "current_assignment_differs": pair.get("current_assignment_differs") is True,
                    "keep_apply_state_or_assignment_differs": pair["keep"]["state_after_current_sha256"] != pair["apply"]["state_after_current_sha256"] or pair.get("current_assignment_differs") is True,
                    "serialized_runtime_flags": not _scan_runtime(serialize_intervention_pair(pair)),
                    "actual_effective_intervention_opportunity": proposal.get("effective_opportunity") is True,
                }
                artifact = serialize_intervention_pair(pair)
                artifact["smoke_checks"] = checks
                artifact["smoke"] = True
                artifact["oracle_upper_bound_only"] = False
                artifact_path = output_root / "smoke" / f"{_slug(str(event['event_id']))}_{frame}_runtime_sealed.json"
                atomic_json(artifact_path, artifact)
                if not all(checks.values()):
                    raise RuntimeError(f"N72R13 paired smoke checks failed: {checks}")
                status = {
                    "schema_version": "N72R13_STAGE_STATUS_V1",
                    "stage": "N72R13-03-RECOVER-PAIRED-SMOKE",
                    "status": "PASS_N72R13_RECOVER_PAIRED_SMOKE",
                    "event_id": str(event["event_id"]),
                    "sequence": str(event["sequence"]),
                    "action_type": str(event["action_type"]),
                    "intervention_frame": int(frame),
                    "runtime_artifact": str(artifact_path),
                    "runtime_artifact_sha256": sha256_file(artifact_path),
                    "checks": checks,
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "posthoc_gt_used": False,
                    "interaction_source": "simulated_from_gt",
                    "real_human_evidence": False,
                    "not_real_human_evidence": True,
                    "created_at_utc": now_utc(),
                    "historical_outputs_modified": False,
                }
                atomic_json(output_root.parent / "stage_03_status.json", status)
                return status
        except Exception as exc:
            failure = {
                "event_id": str(event.get("event_id")),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            atomic_json(output_root / "smoke" / f"{_slug(str(event.get('event_id')))}_failure.json", failure)
    status = {
        "schema_version": "N72R13_STAGE_STATUS_V1",
        "stage": "N72R13-03-RECOVER-PAIRED-SMOKE",
        "status": "FAIL_N72R13_RECOVER_PAIRED_SMOKE",
        "failures": failures,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root.parent / "stage_03_status.json", status)
    raise RuntimeError(f"N72R13 paired smoke failed: {failures}")


def _fresh_baseline_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    # E0 is the frozen N72R9 BASE reference.  The new Oracle trajectory is
    # freshly generated below and is never read from an old E1B/E1C result.
    rows = replay._strip_runtime_rows(inputs["baseline_rows"], HORIZON)
    errors = _scan_runtime(rows)
    if errors:
        raise RuntimeError(f"frozen E0 has runtime boundary errors: {errors[:5]}")
    return rows


def _run_event(
    event: Mapping[str, Any],
    *,
    model: Any,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    inputs = replay._load_inputs(event, horizon=HORIZON)
    inputs = dict(inputs)
    inputs["action_type"] = str(event["action_type"])
    inputs["require_positive_geometry"] = True
    event_dir = output_root / "events" / _slug(str(event["event_id"]))
    event_dir.mkdir(parents=True, exist_ok=True)
    initial = _initial_state(inputs)
    e0_rows = _fresh_baseline_rows(inputs)
    oracle_state = clone_temporal_state(initial)
    oracle_rows = [_event_frame_row(inputs, "ORACLE_TIV")]
    decisions: list[dict[str, Any]] = []
    runtime_opportunities: list[dict[str, Any]] = []
    gt: Mapping[int, Mapping[int, Any]] | None = None
    protected: Mapping[int, int] | None = None
    end_frame = int(inputs["event_frame"]) + HORIZON
    skipped_boundary = 0
    for frame in range(int(inputs["event_frame"]) + 1, end_frame + 1):
        proposal = _proposal_context(inputs, frame, oracle_state, model, device)
        if proposal["effective_opportunity"] and int(frame) + 50 <= end_frame:
            pair = rollout_intervention_pair(
                inputs=inputs,
                frame=frame,
                state_before=oracle_state,
                pctis_proposal=proposal,
                max_horizon=50,
                require_positive_geometry=True,
            )
            runtime_artifact = serialize_intervention_pair(pair)
            runtime_artifact["oracle_upper_bound_only"] = True
            runtime_artifact["runtime_eligible"] = False
            runtime_path = event_dir / "opportunities" / f"frame_{int(frame):06d}_runtime_sealed.json"
            atomic_json(runtime_path, runtime_artifact)
            if _scan_runtime(runtime_artifact):
                raise RuntimeError(f"runtime paired artifact has GT/authority flags: {frame}")
            if gt is None:
                gt = replay.legacy._load_gt(str(inputs["sequence"]))
                protected = replay.legacy._protected_map(
                    inputs["rows"]["c0_source"][int(inputs["event_frame"])],
                    gt,
                    int(inputs["event_frame"]),
                    int(event["dataset_gt_id"]),
                )
            assert protected is not None
            posthoc = _pair_posthoc(pair, inputs=inputs, event=event, gt=gt, protected=protected)
            apply_tiv = _pair_decision(posthoc)
            selected_branch = pair["apply"] if apply_tiv else pair["keep"]
            # The live state object is used only to continue this sequential
            # oracle trajectory; serialize_intervention_pair deliberately
            # removes it from the artifact.
            oracle_state = clone_temporal_state(selected_branch["_state_object"])
            selected_row = selected_branch["current_commit_row"]
            oracle_rows.append(selected_row)
            decision = {
                "frame": int(frame),
                "selection_accepted": True,
                "proposal_target_changed": True,
                "delta_target_accuracy5": posthoc.get("5", {}).get("delta_target_accuracy"),
                "delta_target_accuracy20": posthoc.get("20", {}).get("delta_target_accuracy"),
                "delta_target_accuracy50": posthoc.get("50", {}).get("delta_target_accuracy"),
                "delta_global_pid_accuracy5": posthoc.get("5", {}).get("delta_global_pid_accuracy"),
                "delta_global_pid_accuracy20": posthoc.get("20", {}).get("delta_global_pid_accuracy"),
                "delta_global_pid_accuracy50": posthoc.get("50", {}).get("delta_global_pid_accuracy"),
                "oracle_condition": "APPLY_IF_DELTA_TARGET_ACCURACY20_GT_0_AND_DELTA_GLOBAL_PID_ACCURACY20_GT_0",
                "selected_variant": "APPLY_TIV" if apply_tiv else "KEEP_BASE",
                "gt_used_for_decision": True,
                "runtime_eligible": False,
                "oracle_upper_bound_only": True,
                "runtime_pair_artifact": str(runtime_path),
                "runtime_pair_artifact_sha256": sha256_file(runtime_path),
                "posthoc": posthoc,
            }
            decisions.append(decision)
            runtime_opportunities.append({
                "frame": int(frame),
                "runtime_artifact": str(runtime_path),
                "runtime_artifact_sha256": sha256_file(runtime_path),
                "oracle_selected_variant": decision["selected_variant"],
            })
        else:
            if proposal["effective_opportunity"]:
                skipped_boundary += 1
            base = {"pool": proposal["pool"], "pool_audit": proposal["pool_audit"], "scored": proposal["base"]}
            from sam3_intermot.reacquisition.temporal_intervention_value import _commit_base_frame

            oracle_rows.append(_commit_base_frame(inputs=inputs, frame=frame, state=oracle_state, base=base, branch="PASSIVE_BASE"))
    # No post-hoc information is put into this runtime seal.  It is written
    # before the event-level post-hoc summary is constructed.
    runtime_seal = {
        "schema_version": "N72R13_TIV_ORACLE_RUNTIME_EVENT_SEALED_V1",
        "status": "PASS_N72R13_ORACLE_RUNTIME_SEALED",
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "event_frame": int(event["event_frame"]),
        "e0_rows": e0_rows,
        "oracle_rows": oracle_rows,
        "opportunity_count": len(runtime_opportunities),
        "skipped_boundary_opportunity_count": int(skipped_boundary),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "oracle_upper_bound_only": True,
        "runtime_eligible": False,
    }
    runtime_seal_path = event_dir / "runtime_event_sealed.json"
    atomic_json(runtime_seal_path, runtime_seal)
    runtime_errors = _scan_runtime(runtime_seal)
    if runtime_errors:
        raise RuntimeError(f"oracle runtime seal contains forbidden flags: {runtime_errors[:5]}")
    if gt is None:
        gt = replay.legacy._load_gt(str(inputs["sequence"]))
        protected = replay.legacy._protected_map(
            inputs["rows"]["c0_source"][int(inputs["event_frame"])],
            gt,
            int(inputs["event_frame"]),
            int(event["dataset_gt_id"]),
        )
    assert protected is not None
    score_event = {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "event_frame": int(event["event_frame"]),
        "target_public_id": int(inputs["target_public_id"]),
        "target_dataset_gt_id": int(event["dataset_gt_id"]),
        "rows": {
            "E0_BASELINE_B0": {int(row["frame"]): row for row in e0_rows},
            "ORACLE_TIV": {int(row["frame"]): row for row in oracle_rows},
        },
    }
    comparisons: dict[str, Any] = {}
    quality_by_horizon: dict[str, Any] = {}
    for horizon in VALUE_HORIZONS:
        metric = replay.legacy._score_pair(score_event, "E0_BASELINE_B0", "ORACLE_TIV", int(horizon), gt, protected)
        comparisons[str(horizon)] = metric
        e0_quality = _quality(e0_rows, event_frame=int(event["event_frame"]), target_public=int(inputs["target_public_id"]), target_gid=int(event["dataset_gt_id"]), protected=protected, gt=gt, horizon=int(horizon))
        oracle_quality = _quality(oracle_rows, event_frame=int(event["event_frame"]), target_public=int(inputs["target_public_id"]), target_gid=int(event["dataset_gt_id"]), protected=protected, gt=gt, horizon=int(horizon))
        quality_by_horizon[str(horizon)] = {
            "E0": e0_quality,
            "ORACLE_TIV": oracle_quality,
            "delta_target_accuracy": None if e0_quality["target_accuracy"] is None or oracle_quality["target_accuracy"] is None else float(oracle_quality["target_accuracy"] - e0_quality["target_accuracy"]),
            "delta_global_pid_accuracy": None if e0_quality["global_pid_accuracy"] is None or oracle_quality["global_pid_accuracy"] is None else float(oracle_quality["global_pid_accuracy"] - e0_quality["global_pid_accuracy"]),
            "delta_id_switch_improvement": int(e0_quality["id_switch_count"] - oracle_quality["id_switch_count"]),
        }
    payload = {
        "schema_version": "N72R13_TIV_ORACLE_EVENT_V1",
        "status": "PASS_N72R13_ORACLE_EVENT",
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": int(event["event_frame"]),
        "target_public_id": int(inputs["target_public_id"]),
        "target_dataset_gt_id": int(event["dataset_gt_id"]),
        "runtime_event_sealed": str(runtime_seal_path),
        "runtime_event_sealed_sha256": sha256_file(runtime_seal_path),
        "decisions": decisions,
        "runtime_opportunities": runtime_opportunities,
        "comparisons": comparisons,
        "quality_by_horizon": quality_by_horizon,
        "protected_map_size": len(protected),
        "gt_used_for_decision": True,
        "runtime_eligible": False,
        "oracle_upper_bound_only": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    artifact_path = event_dir / "event.json"
    atomic_json(artifact_path, payload)
    return {**payload, "artifact_path": str(artifact_path)}


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for horizon in VALUE_HORIZONS:
        h = str(horizon)
        metric_keys = ["evaluated_frames", "baseline_identity_error_frames", "treatment_identity_error_frames", "identity_error_reduction_sum", "target_missing_frames", "protected_regression_count", "protected_compared", "assignment_change_count", "true_correct_crossing_count", "true_incorrect_crossing_count", "id_switch_count", "recorrection_opportunity_count"]
        totals = {key: 0.0 for key in metric_keys}
        seq_values: dict[str, list[float]] = defaultdict(list)
        target_deltas: list[float] = []
        global_deltas: list[float] = []
        idsw_improvements: list[float] = []
        for record in records:
            metric = record["comparisons"][h]
            for key in metric_keys:
                totals[key] += float(metric.get(key, 0.0) or 0.0)
            seq_values[str(record["sequence"])].append(float(metric.get("identity_error_reduction") or 0.0))
            quality = record["quality_by_horizon"][h]
            if quality.get("delta_target_accuracy") is not None:
                target_deltas.append(float(quality["delta_target_accuracy"]))
            if quality.get("delta_global_pid_accuracy") is not None:
                global_deltas.append(float(quality["delta_global_pid_accuracy"]))
            idsw_improvements.append(float(quality["delta_id_switch_improvement"]))
        evaluated = totals["evaluated_frames"]
        aggregate[h] = {
            "horizon": int(horizon),
            "event_count": len(records),
            "independent_sequence_count": len(seq_values),
            **{key: int(value) if key.endswith(("_frames", "_count", "_compared")) else float(value) for key, value in totals.items()},
            "identity_error_reduction": None if evaluated == 0 else float(totals["identity_error_reduction_sum"] / evaluated),
            "baseline_future_identity_error": None if evaluated == 0 else float(totals["baseline_identity_error_frames"] / evaluated),
            "oracle_future_identity_error": None if evaluated == 0 else float(totals["treatment_identity_error_frames"] / evaluated),
            "target_accuracy_delta_event_mean": None if not target_deltas else float(np.mean(target_deltas)),
            "global_pid_accuracy_delta_event_mean": None if not global_deltas else float(np.mean(global_deltas)),
            "id_switch_improvement_event_mean": None if not idsw_improvements else float(np.mean(idsw_improvements)),
            "sequence_cluster_bootstrap_95ci": sequence_cluster_bootstrap(seq_values, seed=BOOTSTRAP_SEED + int(horizon), repetitions=BOOTSTRAP_REPETITIONS),
        }
    return aggregate


def _events_from_protocol(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    events = payload.get("source_event_selection", {}).get("events")
    if not isinstance(events, list) or len(events) != 32:
        raise RuntimeError(f"frozen protocol must contain 32 events, found {len(events) if isinstance(events, list) else None}")
    ids = [str(event.get("event_id")) for event in events]
    if len(ids) != len(set(ids)):
        raise RuntimeError("frozen protocol contains duplicate event IDs")
    return [dict(event) for event in events]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--pctis-checkpoint", type=Path, default=DEFAULT_PCTIS)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--event-id", default=None)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    protocol_path = args.protocol.resolve()
    checkpoint = args.pctis_checkpoint.resolve()
    try:
        if int(args.horizon) != HORIZON:
            raise RuntimeError(f"N72R13 formal Oracle is frozen at H{HORIZON}, got H{args.horizon}")
        events = _events_from_protocol(protocol_path)
        if args.event_id is not None:
            events = [event for event in events if str(event["event_id"]) == str(args.event_id)]
            if len(events) != 1:
                raise RuntimeError(f"requested event is not unique in frozen protocol: {args.event_id}")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        device = torch.device(str(args.device))
        model = replay._load_scorer_adapter(checkpoint, device, scorer_kind="pctis")
        if args.smoke:
            status = _run_smoke(events, model=model, device=device, output_root=output_root)
            print(json.dumps({"status": status["status"], "event_id": status.get("event_id"), "frame": status.get("intervention_frame"), "output": status.get("runtime_artifact")}, sort_keys=True))
            return 0
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for event in sorted(events, key=lambda item: str(item["event_id"])):
            try:
                records.append(_run_event(event, model=model, device=device, output_root=output_root))
            except Exception as exc:
                failure = {
                    "event_id": str(event.get("event_id")),
                    "sequence": str(event.get("sequence")),
                    "action_type": str(event.get("action_type")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                atomic_json(output_root / "events" / f"{_slug(str(event.get('event_id')))}.failure.json", failure)
        if failures:
            status = "FAIL_N72R13_ORACLE_RUNTIME"
        elif len(records) != 32 or len({str(row["event_id"]) for row in records}) != 32:
            status = "FAIL_N72R13_ORACLE_EVENT_COMPLETENESS"
        else:
            status = "PASS_N72R13_ORACLE_RUNTIME_AND_POSTHOC"
        metrics = {
            "schema_version": "N72R13_TIV_ORACLE_METRICS_V1",
            "status": status,
            "event_count": len(records),
            "expected_event_count": 32,
            "independent_sequence_count": len({str(row["sequence"]) for row in records}),
            "horizons": list(VALUE_HORIZONS),
            "aggregate": _aggregate(records) if records else {},
            "records": [
                {
                    "event_id": str(row["event_id"]),
                    "sequence": str(row["sequence"]),
                    "action_type": str(row["action_type"]),
                    "artifact_path": str(row["artifact_path"]),
                    "decision_count": len(row["decisions"]),
                    "apply_count": sum(int(item["selected_variant"] == "APPLY_TIV") for item in row["decisions"]),
                    "runtime_opportunity_count": len(row["runtime_opportunities"]),
                }
                for row in records
            ],
            "failures": failures,
            "gt_used_for_decision": True,
            "runtime_eligible": False,
            "oracle_upper_bound_only": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "pctis_checkpoint": str(checkpoint),
            "pctis_checkpoint_sha256": sha256_file(checkpoint),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS, "unit": "independent_sequence"},
            "created_at_utc": now_utc(),
        }
        atomic_json(output_root / "temporal_oracle_metrics.json", metrics)
        stage = {
            "schema_version": "N72R13_STAGE_STATUS_V1",
            "stage": "N72R13-04-TEMPORAL-ORACLE",
            "status": status,
            "metrics": str(output_root / "temporal_oracle_metrics.json"),
            "event_count": len(records),
            "expected_event_count": 32,
            "failure_count": len(failures),
            "gt_used_for_decision": True,
            "runtime_eligible": False,
            "oracle_upper_bound_only": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(output_root.parent / "stage_04_status.json", stage)
        print(json.dumps({"status": status, "events": len(records), "failures": len(failures), "output": str(output_root / "temporal_oracle_metrics.json")}, sort_keys=True))
        return 0 if status == "PASS_N72R13_ORACLE_RUNTIME_AND_POSTHOC" else 1
    except Exception as exc:
        failure = {
            "schema_version": "N72R13_TIV_ORACLE_FAILURE_V1",
            "status": "FAIL_N72R13_ORACLE_RUNTIME",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(output_root / "oracle_failure.json", failure)
        print(json.dumps({"status": failure["status"], "error": str(exc), "failure": str(output_root / "oracle_failure.json")}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
