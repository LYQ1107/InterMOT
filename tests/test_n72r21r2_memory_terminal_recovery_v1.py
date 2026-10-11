from pathlib import Path
import pytest
from scripts import n72r21r2_memory_terminal_recovery_v1 as recovery


def test_exact_click_case_grid_and_duplicates_are_not_exposures():
    events = [{"episode_uid": "a"}, {"episode_uid": "b"}]
    assert recovery.expected_grid(events, ["p", "q"]) == {"p/a", "p/b", "q/a", "q/b"}
    for e, c in ((events + events, ["p"]), (events, ["p", "p"])):
        with pytest.raises(ValueError): recovery.expected_grid(e, c)


def test_only_interrupted_sequence_routes_ASSETS_and_restores_exception():
    original_assets, original_out, original_protocol = recovery.policy.ASSETS, recovery.policy.OUT, recovery.policy.PROTOCOL
    with pytest.raises(RuntimeError):
        with recovery.recovery_paths():
            assert recovery.policy.ASSETS == recovery.RECOVERY_ASSETS
            assert recovery.policy.OUT == original_out and recovery.policy.PROTOCOL == original_protocol
            raise RuntimeError("intentional")
    assert recovery.policy.ASSETS == original_assets


def test_other_sequence_operation_is_refused(monkeypatch):
    monkeypatch.setattr(recovery, "frozen", lambda: {})
    with pytest.raises(ValueError): recovery.operation("runtime", "dancetrack0052")


def test_tracker_view_exclusive_verified_symlink_no_copy_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ASSETS", tmp_path)
    monkeypatch.setattr(recovery, "RECOVERY_ASSETS", tmp_path / "recovery")
    target = Path(__file__)
    # Target must be inside the actual scoped assets root, never arbitrary files.
    row = {"path": str(tmp_path / "recovery/memory_risk_v1/trackers/a/data/x.txt"), "target": str(target), "sha256": recovery.sha256(target)}
    with pytest.raises(AssertionError): recovery.tracker_view([row], create=True)
    monkeypatch.setattr(recovery, "ASSETS", target.resolve().parent)
    # Target/root check is isolated; new tracker path remains independently scoped.
    recovery.tracker_view([row], create=True)
    path = Path(row["path"])
    assert path.is_symlink() and path.resolve() == target.resolve()
    recovery.tracker_view([row], create=False)
    with pytest.raises(FileExistsError): recovery.tracker_view([row], create=True)
    with pytest.raises(AssertionError): recovery.tracker_view([{**row, "sha256": "0" * 64}], create=False)
