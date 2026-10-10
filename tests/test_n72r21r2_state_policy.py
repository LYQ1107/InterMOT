import builtins
from copy import deepcopy
import numpy as np
import pytest
import torch
from scripts import n72r21r2_state_policy as adapter
from scripts import n72r21r2_main_policy as engine
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from scripts.n72r21_t2_replay import DecisionCapture


def test_process_local_adapter_restores_every_main_symbol_on_error(monkeypatch):
    names = ("PROTOCOL", "PREFIX", "BASE", "CODE", "inputs", "write_json")
    previous = {name: getattr(engine, name) for name in names}
    with pytest.raises(RuntimeError):
        with adapter.scoped_engine():
            assert engine.PROTOCOL == adapter.PROTOCOL and engine.PREFIX == adapter.PREFIX
            assert engine.inputs is adapter.inputs
            assert engine.CODE == adapter.CODE and engine.BASE == adapter.BASE
            raise RuntimeError("preserved failed attempt")
    assert all(getattr(engine, name) == value for name, value in previous.items())


def test_runtime_adapter_enforces_gt_boundary_and_restores_guard(monkeypatch):
    original_open = builtins.open
    original_protocol = engine.PROTOCOL
    def cannot_read_gt(*args):
        assert engine.PREFIX == adapter.PREFIX
        builtins.open("/unread/gt/gt.txt")
    monkeypatch.setattr(engine, "runtime", cannot_read_gt)
    with pytest.raises(ValueError, match="GT/offline-label"):
        adapter.execute("runtime", "dancetrack0074", "MIXED_STATE", 730101, "P0_CLAIM80")
    assert engine.PROTOCOL == original_protocol and builtins.open is original_open


def test_routed_receipts_preserve_actual_architecture_and_do_not_claim_on_policy(monkeypatch):
    records = []
    monkeypatch.setattr(adapter, "write_json", lambda path, data, **kw: records.append((path, data)))
    with adapter.scoped_engine():
        engine.write_json(adapter.PREFIX + "/unit.json", dict(family="TREATMENT_STATE", effective=2))
    path, value = records[0]
    assert path == adapter.PREFIX + "/unit.json" and value["family"] == "TREATMENT_STATE"
    assert value["actual_architecture_family"] == "SMALL_MLP"
    assert not value["actual_model_generated_on_policy_training"]


def test_actual_small_mlp_loader_and_causal_full_global_core_are_reused(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    vector, other = np.zeros(512, np.float32), np.zeros(512, np.float32)
    vector[0], other[1] = 1., 1.
    actor = TrustedACIBRecognizer(DecisionCapture(ACIBMemoryNetwork()), vector, "unit", policy="P0")
    actor.start_recording("unit", fps=20., width=200, height=100, initial_frame=0, initial_box=[0, 0, 20, 80])
    model = EventAuthorityHead("SMALL_MLP", hidden=32)
    checkpoint = tmp_path / "untrained_unit_only.pt"
    torch.save(dict(schema="N72R21R2_EVENT_AUTHORITY_V1", family="SMALL_MLP", hidden=32,
        model=model.state_dict(), FIT_mean=[0.] * 32, FIT_scale=[1.] * 32,
        authority_status="TRAINED_UNCALIBRATED_NOT_DEPLOYABLE"), checkpoint)
    # This fixture tests mechanics ONLY; no training/gradient/effect claim.
    monkeypatch.setattr(engine, "make_actor", lambda event, anchor: (deepcopy(actor), None))
    event = dict(sequence="unit", episode_uid="unit", frame=0, frames=6, fps=20.,
        clicked_candidate_uid="a0", box_xyxy=[0, 0, 20, 80])
    policy = dict(current_point=dict(claim_min=.98, risk_max=0., confirmation_delay=3,
        global_regret_max=.2, anchor_advantage_min=.1), authority_support_policy="NO_COST_DISPLACEMENT_OR_ANCHOR_ADVANTAGE_CEILING")
    with adapter.scoped_engine():
        bridge = engine.build_bridge(policy, dict(checkpoint_path=str(checkpoint)), event, vector)
        baseline = MOTIdentityBridge(dict(event_frame=0, human_anchor=vector,
            target_candidate_uid="a0", target_box_xyxy=[0, 0, 20, 80]), frames=6)
        baseline.configure_fps(20.)
        for f in range(6):
            rows = [dict(candidate_uid="a" + str(f), feature=vector, box_xyxy=[0, 0, 20, 80], native_tid=1, conf=.99),
                    dict(candidate_uid="b" + str(f), feature=other, box_xyxy=[60, 0, 80, 80], native_tid=2, conf=.99)]
            assert full_tracker_fingerprint(bridge) == full_tracker_fingerprint(baseline)
            result, base = bridge.step(f, rows), baseline.step(f, rows)
            assert result["outputs"] == base["outputs"]
            assert len({o["public_id"] for o in result["outputs"]}) == 2
            assert len({o["candidate_uid"] for o in result["outputs"]}) == 2
            assert full_tracker_fingerprint(bridge) == full_tracker_fingerprint(baseline)
        assert not bridge.identity.bank
