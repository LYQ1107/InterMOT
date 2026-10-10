from copy import deepcopy
import numpy as np
import pytest
from sam3_intermot.one_click.learned_onset_branches import complete_learned_state_fingerprint, chronological_nonoverlapping_candidates, onset_pair
from sam3_intermot.one_click.authority_support_ablation import SupportAblationBridge, POLICIES
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from scripts.n72r21r2_simple_onset_audit import compact_full
import torch
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from scripts.n72r21_t2_replay import DecisionCapture


def setup():
    vector = np.zeros(512, np.float32)
    vector[0] = 1.
    event = dict(event_frame=0, human_anchor=vector, target_candidate_uid="0",
                 target_box_xyxy=[0, 0, 20, 80])
    torch.set_num_threads(1)
    torch.manual_seed(7)
    actor = TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()), vector, "unit", policy="P0")
    actor.start_recording("unit", fps=20., width=200, height=100,
                          initial_frame=0, initial_box=[0, 0, 20, 80])
    return event, actor, vector


class SelectRaw:
    def predict(self, runtime, branch, keep_runtime):
        return dict(beneficial=.999, harmful=.001, value=1. if branch == "RAW_IDENTITY_TOP" else 0.)


def test_full_controller_clone_and_real_full_global_current_then_keep_pair():
    event, actor, vector = setup()
    other = np.zeros(512, np.float32); other[1] = 1.
    def current(frame):
        return [dict(candidate_uid="0" if frame == 0 else "a" + str(frame), feature=vector if frame == 0 else other,
                     box_xyxy=[0, 0, 20, 80], native_tid=1, conf=.99),
                dict(candidate_uid="b" + str(frame), feature=other if frame == 0 else vector,
                     box_xyxy=[60, 0, 80, 80], native_tid=2, conf=.99)]
    frames = [(dict(frame=f), current(f)) for f in range(5)]
    point = dict(claim_min=.8, risk_max=.02, global_regret_max=.2, anchor_advantage_min=.1, confirmation_delay=1)
    bridge = SupportAblationBridge(event, actor, predictor=SelectRaw(), point=point, frames=5, support_policy=POLICIES[-1])
    bridge.configure_fps(20.)
    bridge.step(0, current(0))
    before = complete_learned_state_fingerprint(bridge)
    deployed = bridge.clone()
    result = deployed.step(1, current(1))
    assert result["authority"]["effective_assignment_change"]
    result.update(full_tracker_state_before_sha256=full_tracker_fingerprint(bridge),
                  full_tracker_state_after_sha256=full_tracker_fingerprint(deployed),
                  full_learned_state_before_sha256=before,
                  full_learned_state_after_sha256=complete_learned_state_fingerprint(deployed))
    arms = onset_pair(bridge, frames, 1, result, compact_full, horizon=3)
    assert arms["OWN_KEEP"][0]["target_uid"] != arms["ACTUAL_CURRENT_THEN_KEEP"][0]["target_uid"]
    assert [r["frame"] for r in arms["OWN_KEEP"]] == [1, 2, 3, 4]
    assert all(r["selected_action"]["family"] == "KEEP" for r in arms["ACTUAL_CURRENT_THEN_KEEP"][1:])
    assert complete_learned_state_fingerprint(bridge) == before
    changed = bridge.clone(); changed.event_feature_history[0] = [1.] * 32
    assert complete_learned_state_fingerprint(changed) != before
    with pytest.raises(ValueError):
        onset_pair(bridge, frames, 2, result, compact_full)


def test_spacing_is_only_correlation_control():
    assert chronological_nonoverlapping_candidates([203, 1, 2, 101, 102, 102]) == [1, 102, 203]
