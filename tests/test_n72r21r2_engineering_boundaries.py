"""Actual frozen encoder plus bounded storage/publication negative controls.

Mock publication prepares never write refs or access the network. Temporary
fixtures are not DanceTrack observations or scientific checkpoint fits.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import n72r21r2_common as common
from scripts import n72r21r2_candidates as candidates
from scripts import n72r21r2_connector_publication_v3 as publication
from sam3_intermot.identity_probe.encoders import OSNetEncoder


def test_actual_frozen_OSNet_parameters_buffers_and_checkpoint_unchanged_after_encode():
    torch.set_num_threads(1)
    weight = common.read_json(common.OUT / "audit/CHECKPOINT_LINEAGE.json")["weights"]["osnet"]
    assert common.sha256(weight["path"]) == weight["sha256"]
    encoder = OSNetEncoder(weight["path"], "cpu")
    before = {name: tensor.detach().clone() for name, tensor in encoder.model.state_dict().items()}
    assert not encoder.model.training and not any(p.requires_grad for p in encoder.model.parameters())
    batch = torch.ones(2, 3, 256, 128, requires_grad=True)
    with torch.enable_grad():
        first, second = encoder.encode(batch), encoder.encode(batch)
    assert first.shape == (2, 512) and torch.isfinite(first).all()
    assert torch.equal(first, second) and not first.requires_grad and first.grad_fn is None
    assert torch.allclose(first.norm(dim=1), torch.ones(2), atol=1e-6)
    assert all(torch.equal(before[name], tensor) for name, tensor in encoder.model.state_dict().items())
    assert not any(p.grad is not None for p in encoder.model.parameters())
    assert common.sha256(weight["path"]) == weight["sha256"]


@pytest.mark.parametrize("encoder", ["sam3", "osnet"])
def test_candidate_integrity_rejects_changed_frozen_encoder_SHA_before_tape_or_seal(monkeypatch, encoder):
    protocol = {"candidate_generation": {"sam3_checkpoint_sha256": "a" * 64, "osnet_checkpoint_sha256": "b" * 64}}
    index = {"sam3_checkpoint_sha256": "a" * 64, "osnet_checkpoint_sha256": "b" * 64}
    index[encoder + "_checkpoint_sha256"] = "c" * 64
    monkeypatch.setattr(candidates, "development_sequence", lambda sequence: protocol)
    monkeypatch.setattr(candidates, "read_json", lambda path: index if Path(path).name == "index.json" else
                        {"status": "PASS_N72R20R2_CANDIDATE_STREAM_SEQUENCE"})
    def forbidden(*args, **kwargs):
        pytest.fail("Changed encoder must reject before loading tape or writing a seal")
    monkeypatch.setattr(candidates, "load_candidate_frames", forbidden)
    monkeypatch.setattr(candidates, "write_json", forbidden)
    with pytest.raises(AssertionError): candidates.integrity("unit")


@pytest.mark.parametrize("reserve", [0, 8 << 20, 512 << 20, 8 << 30])
def test_actual_storage_floor_includes_reserve_and_exact_boundary(monkeypatch, tmp_path, reserve):
    monkeypatch.setattr(common, "ASSETS", tmp_path)
    floor = 60 << 30
    monkeypatch.setattr(common.shutil, "disk_usage", lambda path: SimpleNamespace(free=floor + reserve - 1))
    with pytest.raises(RuntimeError, match="60GiB"): common.storage(reserve)
    monkeypatch.setattr(common.shutil, "disk_usage", lambda path: SimpleNamespace(free=floor + reserve))
    result = common.storage(reserve)
    assert result["filesystem_free_bytes"] == floor + reserve and result["floor_GiB"] == 60
    assert result["personal_limit_not_inferred_from_filesystem_free_space"]


def publication_fixture(monkeypatch, *, path="scripts/n72r21r2_unit.py", payload=b"# unit code\n",
                        kind="blob", dirty=False, delete=False):
    parent, head, tree = "a" * 40, "b" * 40, "c" * 40
    commands, saved = [], {}
    entry = {"path": path, "mode": "100644", "type": kind, "sha": "d" * 40}
    def git(*args):
        commands.append(args)
        if args == ("branch", "--show-current"): return publication.BRANCH.encode()
        if args == ("-c", "core.fsmonitor=false", "status", "--porcelain"): return b" M unit" if dirty else b""
        if args == ("merge-base", "--is-ancestor", parent, "HEAD"): return b""
        if args == ("cat-file", "blob", "d" * 40): return payload
        if args == ("rev-parse", "HEAD"): return head.encode()
        if args == ("rev-parse", parent + "^{tree}"): return tree.encode()
        if args == ("cat-file", "commit", head):
            return (f"tree {tree}\nparent {parent}\nauthor Unit <unit@example.org> 0 +0000\n"
                    "committer Unit <unit@example.org> 0 +0000\n\nunit message\n").encode()
        pytest.fail("Unexpected Git action: " + repr(args))
    monkeypatch.setattr(publication, "git", git)
    monkeypatch.setattr(publication, "entries", lambda ref: {path: entry} if ref == "HEAD" and not delete else
                        {path: entry} if ref == parent and delete else {})
    monkeypatch.setattr(publication, "save", lambda name, value: saved.update({name: value}))
    monkeypatch.setattr(publication, "STATIC", {"sam3_intermot/evaluation/complete_mot_aa_evidence.py"})
    return parent, commands, saved


@pytest.mark.parametrize("path", ["scripts/n72r21r2_unit.py", "tests/test_n72r21r2_unit.py",
                                   "docs/N72R21R2_UNIT.md", "sam3_intermot/evaluation/complete_mot_aa_evidence.py"])
def test_code_only_prepare_allows_exact_source_docs_and_records_nonforce_lease(monkeypatch, path):
    parent, commands, saved = publication_fixture(monkeypatch, path=path)
    publication.prepare(parent)
    record = saved["PREPARED.json"]
    assert record["changes"][0]["path"] == path
    assert not record["force"] and record["require_fresh_expected_sha_lease"] and record["no_bulk_weights_data_media_traces"]
    assert all(command[0] not in ("push", "update-ref", "reset", "checkout") for command in commands)


@pytest.mark.parametrize("path", ["outputs/N72R21R2/mot/results.json", "dataset/train/unit/gt/gt.txt",
    "weights/model.pt", "sam3_intermot/one_click/weights.pt", "docs/N72R21R2_preview.png",
    "third_party/TrackEval/edit.py", "scripts/n72r21r1_change.py"])
def test_code_only_prepare_rejects_outputs_data_weights_media_and_unrelated_sources(monkeypatch, path):
    parent, _, saved = publication_fixture(monkeypatch, path=path)
    with pytest.raises(AssertionError): publication.prepare(parent)
    assert not saved


@pytest.mark.parametrize("case", ["large", "binary", "submodule", "dirty", "delete"])
def test_code_only_prepare_never_silently_rewrites_binary_bulk_dirty_or_deleted_tree(monkeypatch, case):
    options = {"large": {"payload": b"x" * (128 << 10)}, "binary": {"payload": b"\xff\x00"},
               "submodule": {"kind": "commit"}, "dirty": {"dirty": True}, "delete": {"delete": True}}
    parent, _, saved = publication_fixture(monkeypatch, **options[case])
    with pytest.raises((AssertionError, UnicodeDecodeError)): publication.prepare(parent)
    assert not saved
