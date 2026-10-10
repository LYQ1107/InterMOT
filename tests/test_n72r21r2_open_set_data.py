import numpy as np
import torch
from tests.test_n72r21_mot_bridge import inputs
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r21r2_open_set_data import current_axis_snapshot, label_uid
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge


def test_all_current_candidates_plus_NONE_without_advancing_actual_tracker_or_actor():
    torch.manual_seed(730105)
    rows, event = inputs()
    actor = TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()).eval(), event["human_anchor"], "unit", policy="P0")
    actor.start_recording("unit", fps=20, width=100, height=100, initial_frame=0, initial_box=event["target_box_xyxy"])
    bridge = SafeMOTIdentityBridge(event, actor, policy=GatePolicy(family="shadow"))
    baseline = MOTIdentityBridge(event)
    bridge.configure_fps(20)
    for f in range(4):
        if f:
            axis = current_axis_snapshot(bridge, f, rows)
            assert [r["candidate_uid"] for r in axis["axis"]] == ["p", "q", None]
            assert bridge.tracker.frame == f - 1 and bridge.identity.last_frame == f - 1
            assert all(np.isfinite(r["feature_vector"]).all() for r in axis["axis"])
        a, b = bridge.step(f, rows), baseline.step(f, rows)
        assert a["outputs"] == b["outputs"] and a["state_after"] == b["state_after"]
        assert not bridge.identity.bank


def test_UNKNOWN_is_never_verified_other_and_NONE_is_candidate_availability_not_visibility():
    truth = {"p": 7, "q": 8, "unmatched": None}
    assert label_uid("unmatched", truth, 7, True) == "UNKNOWN_UNMATCHED"
    assert label_uid("missing_uid", truth, 7, True) == "UNKNOWN_UNMATCHED"
    assert label_uid("q", truth, 7, True) == "VERIFIED_OTHER"
    assert label_uid("p", truth, 7, True) == "TARGET"
    assert label_uid(None, truth, 7, True) == "INCORRECT_NONE"
    # Even a physically visible person can lack a valid current candidate.
    assert label_uid(None, {"q": 8}, 7, False) == "CORRECT_NONE"
