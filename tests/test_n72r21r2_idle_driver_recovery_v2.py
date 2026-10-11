from pathlib import Path
import pytest
from scripts import n72r21r2_idle_driver_recovery_v2 as recovery


def test_only_exact_mutable_marker_existence_check_is_routed():
    original = "availability/exact_cf_v1/driver.json"
    fresh = "audit/idle_driver_terminal_recovery_v2/owners/EXACT_CF.json"
    routed = recovery.routed_output(original, fresh)
    assert routed / original == recovery.OUT / fresh
    for scientific in ("availability/exact_cf_v1/results/dancetrack0074.json", "PREREGISTRATION.json", "on_policy/joint_state_v1/supervision"):
        assert routed / scientific == recovery.OUT / scientific
    assert recovery.OUT / original != routed / original


def test_active_partial_failure_and_live_or_reused_pid_are_never_idle_restart_authority(monkeypatch):
    base = {"pid": 123, "active": None, "failed_retained": []}
    monkeypatch.setattr(recovery.legacy, "process_exists", lambda pid: False)
    recovery.legacy.check_idle_dead(base)
    for changed in ({"active": {"pid": 456}}, {"failed_retained": [{"returncode": 1}]}, {"pid": True}):
        with pytest.raises(RuntimeError): recovery.legacy.check_idle_dead({**base, **changed})
    monkeypatch.setattr(recovery.legacy, "process_exists", lambda pid: True)
    with pytest.raises(RuntimeError): recovery.legacy.check_idle_dead(base)


def test_exact_four_idle_kinds_no_active_MAIN_memory_or_source_retry():
    assert set(recovery.KINDS) == {"EXACT_CF", "STATE_POLICY", "PURE_MARGIN", "STATE_FIT"}
    assert all(Path(value[1]).name == "driver.json" for value in recovery.KINDS.values())
