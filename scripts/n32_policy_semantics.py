#!/usr/bin/env python3
"""Shared, CPU-only semantic checks for N32 policy artifacts.

The N32 protocol keeps undefined metrics as JSON ``null``.  In particular,
visible-window metrics have no denominator when H20 contains no visible GT,
and drift metrics have no denominator when their corresponding sample count is
zero.  This module is deliberately dependency-free so manifest builders,
artifact writers, reconcilers, and merge gates use the same rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


VISIBLE_UNDEFINED = "LEGITIMATELY_UNDEFINED_NO_VISIBLE_GT"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _box_area(box: Any) -> float | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area if math.isfinite(area) else None


def box_proxy_sample_count(h20: Mapping[str, Any]) -> int:
    """Infer defined box-proxy samples from persisted H20 observations."""
    rows = h20.get("rows")
    if not isinstance(rows, list):
        return 0
    count = 0
    for row in rows:
        if not isinstance(row, Mapping) or row.get("prediction_present") is not True:
            continue
        area = _box_area(row.get("predicted_box", row.get("box")))
        if area is not None and area > 0.0:
            count += 1
    return count


def drift_status(h20: Mapping[str, Any]) -> dict[str, Any]:
    """Return finite/legally-undefined status without changing metric values."""
    mask_count = h20.get("mask_area_sample_count")
    mask_count_valid = finite(mask_count) and float(mask_count) >= 0 and float(mask_count).is_integer()
    mask_count_int = int(mask_count) if mask_count_valid else None
    mask_drift = h20.get("mask_area_drift")
    if finite(mask_drift):
        mask_status = "FINITE"
    elif mask_count_int == 0 and mask_drift is None:
        mask_status = "LEGITIMATELY_UNDEFINED_NO_MASK_SAMPLES"
    elif mask_count_int is not None and mask_count_int > 0:
        mask_status = "NONFINITE_WITH_MASK_SAMPLES"
    else:
        mask_status = "MASK_SAMPLE_COUNT_UNDEFINED"

    box_count = box_proxy_sample_count(h20)
    box_drift = h20.get("box_area_drift_proxy")
    if finite(box_drift):
        box_status = "FINITE"
    elif box_count == 0 and box_drift is None:
        box_status = "LEGITIMATELY_UNDEFINED_NO_BOX_SAMPLES"
    else:
        box_status = "NONFINITE_WITH_BOX_SAMPLES"
    return {
        "mask_area_sample_count": mask_count_int,
        "mask_area_drift_status": mask_status,
        "box_area_proxy_sample_count": box_count,
        "box_area_drift_status": box_status,
    }


def visible_h20_status(h20: Mapping[str, Any], *, require_explicit_undefined: bool = False) -> dict[str, Any]:
    """Validate H20 visible metrics and preserve the zero-visible semantics."""
    reasons: list[str] = []
    evaluated = h20.get("evaluated_frame_count")
    visible = h20.get("visible_frame_count")
    absent = h20.get("absent_gt_frame_count")
    rows = h20.get("rows")
    row_count = len(rows) if isinstance(rows, list) else None
    if evaluated != 20:
        reasons.append("h20_evaluated_frame_count_not_20")
    visible_int = (
        int(visible)
        if finite(visible) and float(visible) >= 0 and float(visible).is_integer()
        else None
    )
    if not finite(absent) or float(absent) < 0 or not float(absent).is_integer():
        absent_int = None
    else:
        absent_int = int(absent)

    if visible_int == 0:
        all_target_absent = isinstance(rows, list) and len(rows) == 20 and all(
            isinstance(row, Mapping) and row.get("target_present") is False for row in rows
        )
        if absent_int != 20:
            reasons.append("h20_zero_visible_absent_count_not_20")
        if row_count != 20:
            reasons.append("h20_zero_visible_row_count_not_20")
        if not all_target_absent:
            reasons.append("h20_zero_visible_rows_not_all_target_absent")
        if h20.get("mean_box_iou_visible") is not None:
            reasons.append("h20_zero_visible_iou_must_be_null")
        if h20.get("missing_prediction_rate_visible") is not None:
            reasons.append("h20_zero_visible_missing_must_be_null")
        if require_explicit_undefined and h20.get("visible_metric_status") != VISIBLE_UNDEFINED:
            reasons.append("h20_zero_visible_status_missing")
        return {
            "status": VISIBLE_UNDEFINED,
            "valid": not reasons,
            "reasons": reasons,
            "evaluated_frame_count": evaluated,
            "visible_frame_count": visible_int,
            "absent_gt_frame_count": absent_int,
            "row_count": row_count,
            "all_target_absent": all_target_absent,
        }

    if visible_int is None or visible_int <= 0:
        reasons.append("h20_visible_frame_count_invalid")
    if not finite(h20.get("mean_box_iou_visible")):
        reasons.append("h20_mean_box_iou_visible_nonfinite")
    if not finite(h20.get("missing_prediction_rate_visible")):
        reasons.append("h20_missing_prediction_rate_visible_nonfinite")
    return {
        "status": "DEFINED_VISIBLE_GT",
        "valid": not reasons,
        "reasons": reasons,
        "evaluated_frame_count": evaluated,
        "visible_frame_count": visible_int,
        "absent_gt_frame_count": absent_int,
        "row_count": row_count,
        "all_target_absent": False,
    }


def policy_metric_issues(policy_row: Mapping[str, Any], *, require_explicit_visible_status: bool = False) -> list[str]:
    """Return strict policy/H20 issues using the N32 denominator rules."""
    issues: list[str] = []
    if policy_row.get("status") != "PASS":
        issues.append("status_not_pass")
    if policy_row.get("available") is not True:
        issues.append("available_not_true")
    if not finite(policy_row.get("reward")):
        issues.append("reward_nonfinite")
    if policy_row.get("future_frame_count") != 20:
        issues.append("future_frame_count_not_20")
    metrics = policy_row.get("metrics")
    h20 = metrics.get("20") if isinstance(metrics, Mapping) else None
    if not isinstance(h20, Mapping):
        return issues + ["h20_missing"]
    visible = visible_h20_status(h20, require_explicit_undefined=require_explicit_visible_status)
    issues.extend(visible["reasons"])
    drift = drift_status(h20)
    if drift["mask_area_drift_status"] == "NONFINITE_WITH_MASK_SAMPLES":
        issues.append("h20_mask_area_drift_nonfinite_with_samples")
    elif drift["mask_area_drift_status"] == "MASK_SAMPLE_COUNT_UNDEFINED":
        issues.append("h20_mask_area_sample_count_undefined")
    if drift["box_area_drift_status"] == "NONFINITE_WITH_BOX_SAMPLES":
        issues.append("h20_box_area_drift_proxy_nonfinite_with_samples")
    return issues


def zero_visible_completion_class(policy_row: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Classify a zero-visible policy as protocol B1/B2, or return reasons."""
    issues = policy_metric_issues(policy_row)
    metrics = policy_row.get("metrics")
    h20 = metrics.get("20") if isinstance(metrics, Mapping) else None
    visible = visible_h20_status(h20) if isinstance(h20, Mapping) else {"status": None, "valid": False, "reasons": []}
    if visible.get("status") != VISIBLE_UNDEFINED:
        issues.append("not_zero_visible_h20")
    if issues:
        return None, issues
    failure = policy_row.get("failure")
    action_trace = policy_row.get("action_trace")
    if not isinstance(action_trace, Mapping):
        return None, ["action_trace_missing"]
    action_failure = action_trace.get("failure")
    if failure is None and action_trace.get("status") == "PASS" and action_failure is None and action_trace.get("rollback_used") is not True:
        return "B1", []
    if (
        isinstance(failure, str)
        and bool(failure)
        and failure == action_failure
        and action_trace.get("status") == "ROLLBACK"
        and action_trace.get("rollback_used") is True
        and action_trace.get("prompt_attempted") is True
        and action_trace.get("prompt_returned_target") is False
        and action_trace.get("mapping_valid") is True
        and action_trace.get("target_state_present") is True
        and policy_row.get("future_frame_count") == 20
        and "future_" not in failure.lower()
        and "future_" not in str(action_failure).lower()
    ):
        return "B2", []
    return None, ["zero_visible_policy_outcome_not_B1_or_B2"]
