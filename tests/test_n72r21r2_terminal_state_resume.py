from pathlib import Path
import pytest
from scripts import n72r21r2_joint_state_terminal_resume_v3 as resume


def test_all_nineteen_in_original_order_not_ready_subset():
    seq = ["s" + str(i) for i in range(19)]
    split = {"fit": seq[:12], "inner": seq[12:]}
    original = {"status": "TERMINAL", "failed_retained": [{"sequence": s} for s in reversed(seq)]}
    old = {"status": "ACTIVE_TERMINAL_FAILURES_ONLY_RECOVERY", "completed": [], "failed_retained": [],
           "sequences": seq, "active": {"sequence": seq[0]}}
    assert resume.selected_sequences(original, old, split) == seq
    old["sequences"] = seq[:-1]
    with pytest.raises(ValueError, match="nineteen"):
        resume.selected_sequences(original, old, split)


def test_completed_or_original_live_attempt_cannot_be_silently_adopted():
    original = {"status": "ACTIVE"}
    with pytest.raises(ValueError, match="first-attempt"):
        resume.selected_sequences(original, {"completed": [], "failed_retained": []}, {})
    original["status"] = "TERMINAL"
    with pytest.raises(ValueError, match="first-attempt"):
        resume.selected_sequences(original, {"completed": ["s"], "failed_retained": []}, {})


def test_exact_cwd_module_and_non_zombie_required(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(resume, "ROOT", project)
    proc = tmp_path / "proc"
    path = proc / "42"
    path.mkdir(parents=True)
    path.joinpath("cmdline").write_bytes(b"python\0-m\0" + resume.MODULES[0].encode() + b"\0")
    path.joinpath("cwd").symlink_to(project)
    path.joinpath("status").write_text("State:\tS (sleeping)\n")
    assert resume.process(42, proc)["live_scoped_state_process"]
    path.joinpath("status").write_text("State:\tZ (zombie)\n")
    assert not resume.process(42, proc)["live_scoped_state_process"]
    path.joinpath("status").write_text("State:\tS (sleeping)\n")
    path.joinpath("cmdline").write_bytes(b"python\0-m\0unrelated\0")
    assert not resume.process(42, proc)["live_scoped_state_process"]
    assert not resume.process(43, proc)["present"]
    assert not resume.process(True, proc)["present"]


def test_foreign_cwd_is_not_a_scoped_owner(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(resume, "ROOT", project)
    path = tmp_path / "proc/42"
    path.mkdir(parents=True)
    path.joinpath("cmdline").write_bytes(resume.MODULES[0].encode() + b"\0")
    path.joinpath("status").write_text("State:\tR (running)\n")
    path.joinpath("cwd").symlink_to(tmp_path)
    assert not resume.process(42, tmp_path / "proc")["live_scoped_state_process"]


def test_routes_only_new_evidence_and_restores_originals(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "read_json", lambda path: {"sequences": ["s"], "source_sha256": {"test": "sha"}})
    monkeypatch.setattr(resume, "validate_frozen_sources", lambda p: None)
    monkeypatch.setattr(resume, "sha256", lambda path: "protocol-sha")
    writes = []
    monkeypatch.setattr(resume, "write_json", lambda relative, value, **kwargs: writes.append((relative, value)))
    old = resume.tensor.PREFIX, resume.tensor.ASSETS, resume.tensor.write_json
    def fake_run(sequence, name):
        assert (sequence, name) == ("s", "runtime")
        assert resume.tensor.PREFIX == resume.PREFIX
        assert resume.tensor.ASSETS == resume.ATTEMPT_ASSETS
        resume.tensor.write_json(resume.PREFIX + "/runtime/s.json", {"source_sha256": "original"})
        resume.tensor.write_json("on_policy/joint_state_v1/plans/s.json", {"plan": "unchanged"})
    monkeypatch.setattr(resume.tensor, "run", fake_run)
    resume.operation("s", "runtime")
    assert writes[0][1]["terminal_resume_protocol_sha256"] == "protocol-sha"
    assert writes[0][1]["source_sha256"] == "original"
    assert writes[1][1] == {"plan": "unchanged"}
    assert (resume.tensor.PREFIX, resume.tensor.ASSETS, resume.tensor.write_json) == old
    with pytest.raises(ValueError, match="registered"):
        resume.operation("not-registered", "runtime")


def test_routes_restored_even_when_actual_operation_fails(monkeypatch):
    monkeypatch.setattr(resume, "read_json", lambda path: {"sequences": ["s"], "source_sha256": {}})
    monkeypatch.setattr(resume, "validate_frozen_sources", lambda p: None)
    old = resume.tensor.PREFIX, resume.tensor.ASSETS, resume.tensor.write_json
    def fail(sequence, name):
        raise RuntimeError("retain partial")
    monkeypatch.setattr(resume.tensor, "run", fail)
    with pytest.raises(RuntimeError, match="retain partial"):
        resume.operation("s", "label")
    assert (resume.tensor.PREFIX, resume.tensor.ASSETS, resume.tensor.write_json) == old
