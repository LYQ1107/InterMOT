from pathlib import Path

import pytest

from scripts import n72r21r2_storage_guard_recovery_v1 as guard
from scripts import n72r21r2_main_storage_resume_v2 as main
from scripts import n72r21r2_joint_state_storage_resume_v5 as state


def test_actual_storage_traceback_required_and_parent_exit_not_inferred(tmp_path, monkeypatch):
    log = tmp_path / "owner.log"
    log.write_text("Traceback (most recent call last):\n" + guard.STORAGE_ERROR + "\n")
    monkeypatch.setattr(Path, "exists", lambda path: False if str(path).startswith("/proc/") else path.is_file())
    result = guard.dead_idle_storage_owner({"pid": 123, "active": None}, log)
    assert result["actual_storage_guard_traceback_observed"]
    assert not result["actual_old_parent_exit_status_observed"]
    for text in ("stale ACTIVE", "RuntimeError: other cause", guard.STORAGE_ERROR, "Traceback (most recent call last):\n" + guard.STORAGE_ERROR + "\nnew unrelated tail"):
        log.write_text(text)
        with pytest.raises(RuntimeError): guard.dead_idle_storage_owner({"pid": 123, "active": None}, log)


@pytest.mark.parametrize("marker", [{"pid": True}, {"pid": -1}, {"pid": None}, {"pid": 123, "active": {"pid": 456}}])
def test_invalid_or_unsealed_owner_never_recovered(marker, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda path: False)
    with pytest.raises(RuntimeError): guard.dead_idle_storage_owner(marker, tmp_path / "never-read.log")


def test_reused_or_live_pid_is_not_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda path: True)
    with pytest.raises(RuntimeError): guard.dead_idle_storage_owner({"pid": 123, "active": None}, tmp_path / "never-read.log")


def test_main_existing_completion_routing_preserves_scientific_paths():
    routed = main.previous.routed_output(main.MARKER, main.FRESH)
    assert routed / main.MARKER == main.OUT / main.FRESH
    for relative in ("mot/main_policy_v1/results/uid/video.json", "mot/MAIN_POLICY_PROTOCOL_V1.json", "training/event_authority"):
        assert routed / relative == main.OUT / relative
    assert main.CODE[0].endswith("main_storage_resume_v2.py")
    assert "main_storage_resume_v2" not in str(main.policy.BASE)


def test_state_lifecycle_context_restores_all_original_globals_on_failure(monkeypatch):
    saved = state.original.PREFIX, state.original.ATTEMPT_ASSETS, state.original.write_json
    with pytest.raises(RuntimeError):
        with state.routed():
            assert state.original.PREFIX == state.PREFIX
            assert state.original.ATTEMPT_ASSETS == state.ASSETS / "joint_state_storage_resume_v5"
            raise RuntimeError("intentional")
    assert (state.original.PREFIX, state.original.ATTEMPT_ASSETS, state.original.write_json) == saved


@pytest.mark.parametrize("sequence,action", [("dancetrack0037", "runtime"), ("dancetrack0086", "train"), ("val_video", "label")])
def test_state_only_original_remaining_operations_allowed(sequence, action, monkeypatch):
    monkeypatch.setattr(state, "validate", lambda: {"remaining_sequences_original_order": ["dancetrack0052", "dancetrack0086"]})
    monkeypatch.setattr(state, "development_sequence", lambda value: None)
    monkeypatch.setattr(state.original, "operation", lambda *args: pytest.fail("invalid operation executed"))
    with pytest.raises(ValueError): state.operation(sequence, action)


def test_state_original_nineteen_and_order_not_replaced():
    registered = ["sequence" + str(i) for i in range(19)]
    assert state.previous.partition(registered, registered[:-2]) == registered[-2:]
    for complete in (registered + ["outsider"], [registered[0]] * 2):
        with pytest.raises(ValueError): state.previous.partition(registered, complete)


def test_ambiguous_receipts_rejected_before_any_canonical_adoption(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "OUT", tmp_path)
    with pytest.raises(ValueError): state.completed_evidence("sequence")
    for prefix in ("on_policy/joint_state_terminal_resume_v3", state.previous.PREFIX):
        folder = tmp_path / prefix / "recovered_sequences"; folder.mkdir(parents=True)
        (folder / "sequence.json").write_text("{}")
    with pytest.raises(ValueError): state.completed_evidence("sequence")
