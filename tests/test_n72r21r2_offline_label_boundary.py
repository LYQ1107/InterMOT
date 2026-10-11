"""Offline labeler verification/GT-open order using only temporary unit truth."""
import json
import pytest
from scripts import n72r21r2_label_counterfactual as labeler
from scripts.n72r21r2_common import sha256
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard


class OfflineGTReached(Exception):
    pass


def label_fixture(monkeypatch, tmp_path):
    out, train = tmp_path / "stage", tmp_path / "train"
    gt = train / "unit/gt/gt.txt"
    gt.parent.mkdir(parents=True)
    gt.write_text("1,42,10,10,20,40,1,1,1\n")
    init = out / "data/initialization/unit.json"
    init.parent.mkdir(parents=True)
    init.write_text(json.dumps({"inputs": [{"episode_uid": "unit0", "frame": 0, "frames": 1, "initialization_failure": False}]}))
    truth = out / "data/initialization_truth/unit.json"
    truth.parent.mkdir(parents=True)
    truth.write_text(json.dumps({"GT_sha256": sha256(gt), "labels": [{"episode_uid": "unit0", "target_gt_identity": 42}]}))
    observation = {"verified": False, "raw_GT_parser_calls": 0, "GT_id": None}
    def verified(sequence):
        observation["verified"] = True
        return [], {"episodes": [{"episode_uid": "unit0", "frames": [0]}]}
    real_annotations = labeler.dancetrack_annotations
    def annotations(path):
        assert observation["verified"]
        observation["raw_GT_parser_calls"] += 1
        actual = real_annotations(path)
        observation["GT_id"] = actual[0][0]["identity"]
        raise OfflineGTReached("Unit stops before corpus or scientific labels are produced")
    monkeypatch.setattr(labeler, "OUT", out)
    monkeypatch.setattr(labeler, "ASSETS", tmp_path / "assets")
    monkeypatch.setattr(labeler, "TRAIN", train)
    monkeypatch.setattr(labeler, "development_sequence", lambda sequence: {"split": {"fit": ["unit"]}})
    monkeypatch.setattr(labeler, "verified", verified)
    monkeypatch.setattr(labeler, "storage", lambda *args: {})
    monkeypatch.setattr(labeler, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(labeler, "sha256", lambda path: sha256(path) if path == gt else "protocol-unit")
    monkeypatch.setattr(labeler, "checked_frames", lambda sequence: ([(dict(frame=0), [])], "unit-index"))
    monkeypatch.setattr(labeler, "dancetrack_annotations", annotations)
    return observation


def test_offline_GT_labels_only_after_source_verification_and_never_runtime(monkeypatch, tmp_path):
    observation = label_fixture(monkeypatch, tmp_path)
    with runtime_file_guard(), pytest.raises(ValueError, match="forbidden in runtime"):
        labeler.run("unit")
    assert observation["verified"] and observation["raw_GT_parser_calls"] == 0
    with pytest.raises(OfflineGTReached): labeler.run("unit")
    assert observation["raw_GT_parser_calls"] == 1 and observation["GT_id"] == 42
    assert not list((tmp_path / "assets").rglob("*.jsonl.zst"))


def test_unverified_source_cannot_open_raw_GT_or_produce_offline_labels(monkeypatch, tmp_path):
    observation = label_fixture(monkeypatch, tmp_path)
    def fail(sequence):
        raise ValueError("unsealed unit source")
    monkeypatch.setattr(labeler, "verified", fail)
    with pytest.raises(ValueError, match="unsealed unit source"): labeler.run("unit")
    assert observation["raw_GT_parser_calls"] == 0 and observation["GT_id"] is None
