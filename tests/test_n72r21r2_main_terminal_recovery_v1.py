from pathlib import Path
import pytest
from scripts import n72r21r2_main_terminal_recovery_v1 as recovery


def test_dead_active_rejects_live_reused_bad_pid_and_failure(monkeypatch):
    marker = {"pid": 123, "active": {"pid": 456}, "failed_retained": []}
    monkeypatch.setattr(Path, "exists", lambda path: False)
    recovery.dead_active(marker)
    for value in ({"pid": True}, {"active": None}, {"failed_retained": [1]}):
        with pytest.raises(RuntimeError): recovery.dead_active({**marker, **value})
    monkeypatch.setattr(Path, "exists", lambda path: True)
    with pytest.raises(RuntimeError): recovery.dead_active(marker)


def test_original_driver_state_is_mutated_and_completion_not_aliased():
    state = {"completed": [], "group_status": {}, "failed_retained": [], "pid": 789, "active": None}
    old = {"completed": ["sealed"], "group_status": {"SCALAR": "finished"}, "pid": 123, "active": {"pid": 456}}
    assert recovery.restored_state(state, old) is state
    assert state["completed"] == ["sealed"] and state["group_status"] == old["group_status"]
    assert state["pid"] == 789 and state["active"] is None
    state["completed"].append("next"); assert old["completed"] == ["sealed"]
    with pytest.raises(RuntimeError): recovery.restored_state(state, old)


def test_BASE_only_is_scoped_and_restored_on_failure():
    original_base, original_out, original_protocol = recovery.policy.BASE, recovery.policy.OUT, recovery.policy.PROTOCOL
    with pytest.raises(RuntimeError):
        with recovery.recovery_paths():
            assert recovery.policy.BASE == recovery.RECOVERY_BASE and original_base != recovery.RECOVERY_BASE
            assert recovery.policy.OUT == original_out and recovery.policy.PROTOCOL == original_protocol
            raise RuntimeError("intentional")
    assert recovery.policy.BASE == original_base


def test_noninterrupted_operation_cannot_route_new_scientific_cell(monkeypatch):
    monkeypatch.setattr(recovery, "frozen", lambda: {})
    with pytest.raises(ValueError): recovery.operation("runtime", "dancetrack0052", "LOGISTIC_RISK", "L3_HARM_AWARE", 730103, "P0_CLAIM80")


def test_mutable_marker_only_routed():
    routed = recovery.routed_output(recovery.MARKER, recovery.FRESH)
    assert routed / recovery.MARKER == recovery.OUT / recovery.FRESH
    for path in ("PREREGISTRATION.json", "mot/main_policy_v1/results/a.json", "training/event_authority"):
        assert routed / path == recovery.OUT / path
