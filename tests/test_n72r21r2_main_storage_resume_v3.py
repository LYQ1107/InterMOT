from pathlib import Path
import json
import pytest
from scripts import n72r21r2_main_storage_resume_v3 as recovery


def test_new_owner_only_routes_mutable_marker_not_scientific_outputs():
    routed = recovery.prior.previous.routed_output(recovery.MARKER, recovery.FRESH)
    assert routed / recovery.MARKER == recovery.OUT / recovery.FRESH
    for path in ("mot/main_policy_v1/runtime/a.json", "mot/MAIN_POLICY_PROTOCOL_V1.json", "training/event_authority"):
        assert routed / path == recovery.OUT / path
    assert "main_storage_resume_v3" not in str(recovery.prior.policy.BASE)
    assert recovery.OLD_OWNER != recovery.OUT / recovery.FRESH


@pytest.mark.parametrize("raise_after_first_write", [False, True])
def test_actual_original_driver_state_hydrated_once_and_globals_restored(tmp_path, monkeypatch, raise_after_first_write):
    old = {"pid": 123, "completed": ["sealed"], "group_status": {"finished": "old"}, "failed_retained": [], "active": None}
    p = {"goal": recovery.GOAL, "frozen": True, "source_sha256": {recovery.SELF: "hash"},
         "original_policy_protocol_sha256": "hash"}
    for key in ("actual192_readback", "old_marker_archive", "old_owner"):
        p[key] = {"path": key, "sha256": "hash"}
    monkeypatch.setattr(recovery, "OUT", tmp_path)
    monkeypatch.setitem(recovery.prior.previous.routed_output.__globals__, "OUT", tmp_path)
    monkeypatch.setattr(recovery, "sha256", lambda path: "hash")
    monkeypatch.setattr(recovery, "read_json", lambda path: old if Path(path) == tmp_path / recovery.MARKER else p)
    monkeypatch.setattr(recovery, "checked_readback", lambda: {"evidence": {"old_marker": old, "old_marker_sha256": "hash"}})
    monkeypatch.setattr(recovery, "dead_idle_storage_owner", lambda *args: {})
    reservations = []
    monkeypatch.setattr(recovery, "storage", reservations.append)
    def write(relative, value, **kwargs):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    monkeypatch.setattr(recovery, "write_json", write)
    driver = recovery.prior.driver
    original_globals = driver.OUT, driver.write_json, driver.ASSETS
    def original_run():
        assert driver.OUT / recovery.MARKER == tmp_path / recovery.FRESH
        value = {"pid": 999, "completed": [], "group_status": {}, "failed_retained": [], "active": None}
        driver.write_json(recovery.MARKER, value)
        assert value["pid"] == 999 and value["completed"] == ["sealed"]
        assert value["group_status"] == {"finished": "old"}
        value["completed"].append("new")
        driver.write_json(recovery.MARKER, value, mutable=True)
        assert old["completed"] == ["sealed"]
        if raise_after_first_write:
            raise RuntimeError("test restoration")
    monkeypatch.setattr(driver, "run", original_run)
    if raise_after_first_write:
        with pytest.raises(RuntimeError, match="test restoration"):
            recovery.run()
    else:
        recovery.run()
    assert reservations == [6 << 30]
    assert (driver.OUT, driver.write_json, driver.ASSETS) == original_globals
    assert json.loads((tmp_path / recovery.FRESH).read_text())["completed"] == ["sealed", "new"]
