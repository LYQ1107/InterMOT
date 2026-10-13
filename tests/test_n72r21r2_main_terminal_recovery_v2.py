from pathlib import Path
import pytest
from scripts import n72r21r2_main_terminal_recovery_v2 as recovery


def test_fixed_interruption_and_no_prior_module_mutation():
    assert recovery.INTERRUPTED.endswith("seed730103__P0_CLAIM80/dancetrack0052")
    assert "LOGISTIC_RISK" in recovery.old.INTERRUPTED
    assert recovery.old.PROTOCOL != recovery.PROTOCOL
    assert recovery.MODULE in recovery.MODULES
    assert "scripts.n72r21r2_main_storage_resume_v3" in recovery.MODULES


def test_only_interrupted_BASE_is_routed_and_restored():
    original = recovery.policy.BASE
    with pytest.raises(RuntimeError):
        with recovery.recovery_paths():
            assert recovery.policy.BASE == recovery.RECOVERY_BASE
            raise RuntimeError("intentional")
    assert recovery.policy.BASE == original


def test_unregistered_cell_cannot_run(monkeypatch):
    monkeypatch.setattr(recovery, "frozen", lambda: {})
    with pytest.raises(ValueError):
        recovery.operation("runtime", "dancetrack0052", "DUAL_IDENTITY_STATE",
                           "L3_HARM_AWARE", 730102, "P0_CLAIM80")


def test_registered_runtime_has_guard_and_restores_BASE(monkeypatch):
    from contextlib import contextmanager
    monkeypatch.setattr(recovery, "frozen", lambda: {})
    observations = []
    @contextmanager
    def guard():
        observations.append("guard_enter")
        yield
        observations.append("guard_exit")
    monkeypatch.setattr(recovery.old, "runtime_file_guard", guard)
    monkeypatch.setattr(recovery.policy, "runtime",
                        lambda *args: observations.append((recovery.policy.BASE, args)))
    original = recovery.policy.BASE
    recovery.operation("runtime", "dancetrack0052", "DUAL_IDENTITY_STATE",
                       "L3_HARM_AWARE", 730103, "P0_CLAIM80")
    assert observations[0] == "guard_enter" and observations[-1] == "guard_exit"
    assert observations[1][0] == recovery.RECOVERY_BASE
    assert recovery.policy.BASE == original


def test_scoped_live_worker_prevents_evidence(monkeypatch):
    def reject(modules):
        assert recovery.MODULE in modules
        raise RuntimeError("live worker")
    monkeypatch.setattr(recovery.old, "no_live_scoped_workers", reject)
    with pytest.raises(RuntimeError, match="live worker"):
        recovery.evidence()


def test_freeze_never_overwrites_existing_protocol(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda _: True)
    with pytest.raises(FileExistsError):
        recovery.freeze()


def test_marker_routing_does_not_touch_scientific_outputs():
    routed = recovery.old.routed_output(recovery.MARKER, recovery.FRESH)
    assert routed / recovery.MARKER == recovery.OUT / recovery.FRESH
    for name in ("PREREGISTRATION.json", "mot/main_policy_v1/runtime/a.json",
                 "mot/main_policy_v1/results/a.json", "training/event_authority"):
        assert routed / name == recovery.OUT / name


@pytest.mark.parametrize("fail_after_hydration", [False, True])
def test_original_driver_hydrated_once_and_restored(tmp_path, monkeypatch, fail_after_hydration):
    import json
    marker = {"pid": 123, "completed": ["sealed"], "group_status": {"finished": "old"},
              "failed_retained": [], "active": {"pid": 456}}
    row = {"old_marker": marker, "old_marker_sha256": "hash"}
    protocol = {**row, "original_marker_archive": "archive"}
    monkeypatch.setattr(recovery, "OUT", tmp_path)
    monkeypatch.setitem(recovery.old.routed_output.__globals__, "OUT", tmp_path)
    monkeypatch.setattr(recovery, "frozen", lambda: protocol)
    monkeypatch.setattr(recovery, "evidence", lambda: row)
    monkeypatch.setattr(recovery, "exclusive", lambda: None)
    monkeypatch.setattr(recovery.old, "dead_active", lambda _: None)
    monkeypatch.setattr(recovery, "sha256", lambda _: "hash")
    monkeypatch.setattr(recovery, "read_json", lambda _: marker)
    reservations = []
    monkeypatch.setattr(recovery, "storage", reservations.append)
    def write(relative, value, **kwargs):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    monkeypatch.setattr(recovery, "write_json", write)
    driver = recovery.driver
    original = driver.OUT, driver.write_json, driver.run_cell, driver.ASSETS
    def original_run():
        state = {"pid": 999, "completed": [], "group_status": {}, "failed_retained": [], "active": None}
        driver.write_json(recovery.MARKER, state)
        assert state["pid"] == 999 and state["active"] is None
        assert state["completed"] == ["sealed"] and state["group_status"] == {"finished": "old"}
        state["completed"].append("new")
        driver.write_json(recovery.MARKER, state, mutable=True)
        assert marker["completed"] == ["sealed"]
        if fail_after_hydration:
            raise RuntimeError("test restoration")
    monkeypatch.setattr(driver, "run", original_run)
    if fail_after_hydration:
        with pytest.raises(RuntimeError, match="test restoration"):
            recovery.run()
    else:
        recovery.run()
    assert (driver.OUT, driver.write_json, driver.run_cell, driver.ASSETS) == original
    assert reservations == [6 << 30]
    assert json.loads((tmp_path / recovery.FRESH).read_text())["completed"] == ["sealed", "new"]
