"""Compare exported MOT detections independently of identity-dependent matching.

HOTA global-ID alignment and CLEAR continuity can change matches even when the
geometric detections are unchanged. Audit the actual exported multiset instead
of assuming equality of the resulting DetA/LocA/FP/FN metrics.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json


def _finite_decimal(value: str) -> Decimal:
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("invalid MOT numeric field") from exc
    if not number.is_finite():
        raise ValueError("nonfinite MOT numeric field")
    return number


def _positive_integer(value: str) -> int:
    number = _finite_decimal(value)
    if number <= 0 or number != number.to_integral_value():
        raise ValueError("MOT frame and public ID must be positive integers")
    return int(number)


def detection_multiset(text: str) -> tuple:
    """Exact numeric (frame,x,y,w,h,confidence) rows, sorted with duplicates.

    Ignore only public ID and row order. Empty frames are verified separately
    by runtime trace/frame-axis checks; they cannot be inferred from MOT rows.
    """
    rows = []
    ownership = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 10:
            raise ValueError("expected ten-field MOT trajectory row")
        frame = _positive_integer(fields[0])
        public = _positive_integer(fields[1])
        if (frame, public) in ownership:
            raise ValueError("duplicate public ID in the same frame")
        ownership.add((frame, public))
        values = tuple(_finite_decimal(v) for v in fields[2:7])
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError("MOT width and height must be positive")
        rows.append((frame, *values))
    return tuple(sorted(rows))


def audit_unchanged_detections(reference: str, candidate: str) -> dict:
    expected = detection_multiset(reference)
    actual = detection_multiset(candidate)
    if actual != expected:
        raise ValueError("exported per-frame detection multiset changed")
    canonical = [
        [row[0], *(format(v.normalize(), "f") if v else "0" for v in row[1:])]
        for row in actual
    ]
    payload = json.dumps(canonical, separators=(",", ":"), allow_nan=False).encode()
    return {
        "exact_numeric_detection_multiset_equal": True,
        "canonical_multiset_sha256": hashlib.sha256(payload).hexdigest(),
        "trajectory_rows": len(actual),
        "frames_with_detections": len(Counter(row[0] for row in actual)),
        "empty_frame_axis_checked_separately_in_runtime_trace": True,
        "comparison_fields": ["frame", "x", "y", "width", "height", "confidence"],
        "ignored_fields": ["public_id", "row_order", "unused_MOT_padding"],
        "derived_TrackEval_detection_metrics_required_equal": False,
    }
