#!/usr/bin/env python3
"""Run the N72R3 Stage 12 simulated-observer audit regression.

The valid path uses only current-frame GT after the prediction is frozen and
is not a scientific tracking result.  A rejecting accessor is also used as a
negative control to prove that a future-read signal cannot be overwritten by
the public audit serializer.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.interaction.continuous_observer import GTFrameAccessor
from sam3_intermot.interaction.n72r2_simulated_observer import N72R2SimulatedHumanObserver
from sam3_intermot.interaction.simulator import GTFrame


OUT = ROOT / "outputs" / "N72R3"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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


class RejectingAccessor:
    def __init__(self) -> None:
        self.current: int | None = None

    def begin_prediction(self, frame: int) -> None:
        self.current = int(frame)

    def mark_prediction_done(self) -> None:
        return None

    def observe(self, frame: int):
        raise RuntimeError(f"blocked future/current probe at {frame}")


def valid_current_frame_probe() -> dict:
    frame = 5
    accessor = GTFrameAccessor({frame: GTFrame(boxes=[np.asarray([1, 1, 5, 5])], gt_ids=[2])})
    observer = N72R2SimulatedHumanObserver(accessor, "toy", "stage12-valid")
    observer.begin_prediction(frame)
    observer.freeze_prediction({"pre": True})
    current = observer.read_current_gt_for_simulation()
    observer.simulate_action("AUTHORITATIVE_CORRECT", public_id=17, current_gt_input=current)
    observer.freeze_post({"post": True})
    observer.write_memory(17, embedding=np.ones(4, dtype=np.float32), source="current_frame_authoritative_roi")
    future_memory_visible = observer.read_memory(frame + 1, 17) is not None
    audit = observer.audit_dict()
    return {
        "status": "PASS"
        if audit["gt_read_before_prediction"] == 0
        and audit["gt_read_future"] == 0
        and audit["runtime_future_gt_used"] is False
        and audit["event_frame_read_hidden"] is True
        and audit["first_memory_read_offset"] == 1
        and future_memory_visible
        else "FAIL",
        "audit": audit,
        "future_memory_visible_at_t_plus_one": future_memory_visible,
    }


def future_read_negative_control() -> dict:
    observer = N72R2SimulatedHumanObserver(RejectingAccessor(), "toy", "stage12-negative-control")
    observer.begin_prediction(5)
    observer.freeze_prediction({"pre": True})
    try:
        observer.read_current_gt_for_simulation()
    except RuntimeError:
        pass
    else:
        raise AssertionError("negative-control accessor unexpectedly returned a frame")
    audit = observer.audit_dict()
    return {
        "detected": audit["gt_read_future"] == 1 and audit["runtime_future_gt_used"] is True,
        "audit": audit,
        "is_negative_control": True,
    }


def main() -> int:
    valid = valid_current_frame_probe()
    negative = future_read_negative_control()
    payload = {
        "schema_version": "N72R3_STAGE12_GT_AUDIT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "12_SIMULATED_OBSERVER_AUDIT_FIX",
        "status": "PASS_STAGE12_GT_AUDIT_CORRECTED" if valid["status"] == "PASS" and negative["detected"] else "FAIL_STAGE12_GT_AUDIT",
        "interaction_source": "simulated_from_gt",
        "valid_current_frame_probe": valid,
        "future_read_negative_control": negative,
        "runtime_future_gt_used": False,
        "negative_control_future_read_detected": bool(negative["detected"]),
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
        "root_cause_fixed": "audit_dict no longer overwrites gt_read_future-derived runtime_future_gt_used",
    }
    atomic_json(OUT / "audits" / "stage12_gt_audit.json", payload)
    atomic_json(
        OUT / "stage_12_status.json",
        {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "stage": "12_SIMULATED_OBSERVER_AUDIT_FIX",
            "status": payload["status"],
            "created_at_utc": payload["created_at_utc"],
            "artifact": str(OUT / "audits" / "stage12_gt_audit.json"),
            "runtime_future_gt_used": False,
            "negative_control_detected_future_gt": bool(negative["detected"]),
            "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
        },
    )
    return 0 if payload["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

