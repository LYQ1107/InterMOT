"""Toy tests for the N72R3 simulated-observer audit correction."""

import numpy as np

from sam3_intermot.interaction.n72r2_simulated_observer import N72R2SimulatedHumanObserver
from sam3_intermot.interaction.simulator import GTFrame
from sam3_intermot.interaction.continuous_observer import GTFrameAccessor


class RejectingAccessor:
    def __init__(self) -> None:
        self.current = None

    def begin_prediction(self, frame: int) -> None:
        self.current = int(frame)

    def mark_prediction_done(self) -> None:
        return None

    def observe(self, frame: int):
        raise RuntimeError(f"blocked future/current probe at {frame}")


def test_future_gt_bit_is_not_overwritten_by_audit_dict() -> None:
    accessor = RejectingAccessor()
    observer = N72R2SimulatedHumanObserver(accessor, "toy", "future-probe")
    observer.begin_prediction(5)
    observer.freeze_prediction({"pre": True})
    try:
        observer.read_current_gt_for_simulation()
    except RuntimeError:
        pass
    else:
        raise AssertionError("rejecting accessor unexpectedly returned GT")
    audit = observer.audit_dict()
    assert audit["gt_read_future"] == 1
    assert audit["runtime_future_gt_used"] is True


def test_valid_current_gt_and_t_plus_one_memory_remain_gt_free() -> None:
    accessor = GTFrameAccessor({5: GTFrame(boxes=[np.asarray([1, 1, 5, 5])], gt_ids=[2])})
    observer = N72R2SimulatedHumanObserver(accessor, "toy", "valid-probe")
    observer.begin_prediction(5)
    observer.freeze_prediction({"pre": True})
    current = observer.read_current_gt_for_simulation()
    observer.simulate_action("AUTHORITATIVE_CORRECT", public_id=17, current_gt_input=current)
    observer.freeze_post({"post": True})
    observer.write_memory(17, embedding=np.ones(4, dtype=np.float32), source="current_frame_authoritative_roi")
    assert observer.read_memory(6, 17) is not None
    audit = observer.audit_dict()
    assert audit["event_frame_read_hidden"] is True
    assert audit["first_memory_read_offset"] == 1
    assert audit["runtime_future_gt_used"] is False
