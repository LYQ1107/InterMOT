import numpy as np
import pytest
import torch
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.zero_authority_memory import FIXED_MEMORY_CASES, fixed_memory_policy, memory_actor, ZeroAuthorityMemoryBridge
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.one_click.intervention_gate import GatePolicy


def vectors():
    a, b = np.zeros(512, np.float32), np.zeros(512, np.float32)
    a[0], b[1] = 1., 1.
    return a, b


def rows(frame):
    a, b = vectors()
    return [{"candidate_uid": str(frame) + "A", "feature": a, "box_xyxy": [frame, 0, frame + 20, 80], "native_tid": 1, "native_scope": "s", "conf": .99},
            {"candidate_uid": str(frame) + "B", "feature": b, "box_xyxy": [100, 0, 120, 80], "native_tid": 2, "native_scope": "s", "conf": .99}]


def original():
    torch.manual_seed(730101)
    a, _ = vectors()
    actor = TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()).eval(), a, "sole", policy="P0")
    actor.start_recording("sequence", fps=20, width=1920, height=1080, initial_frame=0, initial_box=[0, 0, 20, 80])
    return actor


def bridge(case):
    actor = memory_actor(original(), case)
    event = {"event_frame": 0, "human_anchor": actor.anchor, "target_candidate_uid": "0A", "target_box_xyxy": [0, 0, 20, 80]}
    result = ZeroAuthorityMemoryBridge(event, actor, frames=12)
    result.configure_fps(20)
    return result


def test_actual_nonzero_bank_changes_leave_full_tensor_MOT_state_identical():
    torch.set_num_threads(1)
    frozen, mean = bridge("FROZEN"), bridge("MEAN")
    writes = 0
    for f in range(12):
        assert full_tracker_fingerprint(frozen) == full_tracker_fingerprint(mean)
        a, b = frozen.step(f, rows(f)), mean.step(f, rows(f))
        assert a["outputs"] == b["outputs"] and full_tracker_fingerprint(frozen) == full_tracker_fingerprint(mean)
        assert not b["authority"]["approved"]
        writes += b["joint_identity_memory_write"]
        if b["joint_identity_memory_write"]:
            assert b["joint_memory_write_candidate_uid"] == b["target_uid"]
    assert writes == 11 and not frozen.identity.bank and len(mean.identity.bank) == 8


def test_zero_authority_rejects_gate_mutation():
    b = bridge("MEAN")
    b.policy = GatePolicy(family="off")
    with pytest.raises(ValueError):
        b.step(0, rows(0))


def test_risk_writer_not_silently_substituted_with_historical_or_unfit_head():
    assert len(FIXED_MEMORY_CASES) == 6
    assert fixed_memory_policy("PENDING_TRUSTED").family == "diverse"
    with pytest.raises(ValueError):
        fixed_memory_policy("RISK_LEARNED")
