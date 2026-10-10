from copy import deepcopy
import numpy as np
import pytest
import torch
from scripts.n72r21_t2_replay import DecisionCapture
from tests.test_n72r21_mot_bridge import inputs
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_runtime import Evidence
from sam3_intermot.one_click.fresh_memory_risk import INPUT_NAMES, MemoryRiskHead, FreshMemoryRiskPredictor, FreshRiskCommittedMemory, write_input
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.zero_authority_memory import ZeroAuthorityMemoryBridge
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge


def saved(tmp_path, *, status="INNER_SELECTED_CURRENT_WRITE_ONLY"):
    model = MemoryRiskHead("LOGISTIC")
    with torch.no_grad():
        model.net.weight.zero_()
        model.net.bias.copy_(torch.tensor([10., -10., -10.]))
    path = tmp_path / "writer.pt"
    torch.save(dict(schema="N72R21R2_COMMITTED_WRITE_CURRENT_RISK_V1", family="LOGISTIC", model=model.state_dict(),
                    input_names=list(INPUT_NAMES), FIT_mean=[0.] * len(INPUT_NAMES), FIT_scale=[1.] * len(INPUT_NAMES),
                    association_authority=False, selection=dict(status=status, probability_min=.98, unknown_max=.02, temperature=1.)), path)
    return path


def test_current_future_bank_entries_and_GT_never_writer_features():
    rows, event = inputs()
    features = dict.fromkeys(FEATURE_NAMES, 0.)
    vector, values = write_input(features, event["human_anchor"], [], None, 1, rows[0])
    assert len(vector) == 42 and values["own_pending_confirmations"] == 1
    with pytest.raises(ValueError):
        write_input({**features, "future_label": 1}, event["human_anchor"], [], None, 1, rows[0])
    with pytest.raises(ValueError):
        write_input(features, event["human_anchor"], [], None, 1, {**rows[0], "gt_id": 1})
    evidence = Evidence(rows[0]["feature"], "unit", 1, None, .05, 1., "unit", 1., 1., "p")
    with pytest.raises(ValueError):
        write_input(features, event["human_anchor"], [evidence], None, 1, rows[0])


def test_strict_schema_and_uncalibrated_or_abstain_are_not_safe_writes(tmp_path):
    with pytest.raises(ValueError):
        FreshMemoryRiskPredictor(saved(tmp_path, status="UNCALIBRATED_NOT_DEPLOYABLE"))
    predictor = FreshMemoryRiskPredictor(saved(tmp_path, status="CALIBRATION_ABSTAIN"))
    rows, event = inputs()
    actor = FreshRiskCommittedMemory(DecisionCapture(ACIBMemoryNetwork()), event["human_anchor"], "unit", predictor=predictor)
    actor.start_recording("unit", fps=20., width=100, height=100, initial_frame=0, initial_box=event["target_box_xyxy"])
    bridge = ZeroAuthorityMemoryBridge(event, actor, frames=3)
    bridge.configure_fps(20.)
    bridge.step(0, rows)
    result = bridge.step(1, rows)
    assert not result["joint_identity_memory_write"] and not bridge.identity.bank


def test_nonzero_committed_writes_preserve_full_global_C0_and_clone_pending(tmp_path):
    torch.set_num_threads(1)
    rows, event = inputs()
    predictor = FreshMemoryRiskPredictor(saved(tmp_path))
    actor = FreshRiskCommittedMemory(DecisionCapture(ACIBMemoryNetwork()), event["human_anchor"], "unit", predictor=predictor)
    actor.start_recording("unit", fps=20., width=100, height=100, initial_frame=0, initial_box=event["target_box_xyxy"])
    bridge = ZeroAuthorityMemoryBridge(event, actor, frames=12)
    baseline = MOTIdentityBridge(event)
    bridge.configure_fps(20.)
    for frame in range(10):
        result, reference = bridge.step(frame, rows), baseline.step(frame, rows)
        assert result["outputs"] == reference["outputs"]
        if frame:
            assert result["joint_identity_memory_write"] and result["joint_memory_write_candidate_uid"] == result["target_uid"] == "p"
            assert len(bridge.identity.bank) == min(frame, 8)
            assert result["committed_memory_diagnostic"]["confirmations"] == frame
    clone = bridge.clone()
    clone.identity.write_pending["feature"][0] = .2
    assert bridge.identity.write_pending["feature"][0] == 1.
    assert not bridge.identity.bank[-1].embedding.flags.writeable
