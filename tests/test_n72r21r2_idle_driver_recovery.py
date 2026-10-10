import os
import pytest
from scripts import n72r21r2_idle_driver_recovery_v1 as recovery
from scripts.n72r21r2_idle_driver_recovery_v1 import check_idle_dead, process_exists


def test_live_or_reused_foreign_pid_is_never_restart_authority():
    assert process_exists(os.getpid())
    with pytest.raises(RuntimeError):
        check_idle_dead({"pid": os.getpid(), "active": None, "failed_retained": []})


def test_dead_idle_marker_is_recoverable_not_a_scientific_failure(monkeypatch):
    assert not process_exists(-1)
    monkeypatch.setattr(recovery, "process_exists", lambda pid: False)
    check_idle_dead({"pid": 123, "active": None, "failed_retained": []})


@pytest.mark.parametrize("extra", [{"active": {"pid": 123}}, {"failed_retained": [{"returncode": 1}]}])
def test_partial_child_or_failed_attempt_requires_separate_recovery(extra, monkeypatch):
    monkeypatch.setattr(recovery, "process_exists", lambda pid: False)
    with pytest.raises(RuntimeError):
        check_idle_dead({"pid": 123, "active": None, "failed_retained": [], **extra})


@pytest.mark.parametrize("pid", [-1, 0, True, "123", None])
def test_invalid_pid_is_not_dead_owner_evidence(pid):
    with pytest.raises(RuntimeError):
        check_idle_dead({"pid": pid, "active": None, "failed_retained": []})
