import hashlib
import numpy as np
import pytest
from scripts.n72r21r2_memory_risk_data import reconstruct_bank_input
from scripts.n72r21r2_memory_risk_driver import all_sources_ready
from scripts.n72r21r2_train_memory_risk import teacher_weights, assess
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.acib_runtime import unit, Evidence
from sam3_intermot.one_click.fresh_memory_risk import write_input
from tests.test_n72r21_mot_bridge import inputs


def test_exact_past_bank_pending_reconstruction_matches_own_runtime_encoder():
    rows, event = inputs()
    vector = unit(rows[0]["feature"])
    digest = hashlib.sha256(vector.tobytes()).hexdigest()
    anchor = event["human_anchor"]
    snapshot = {"anchor_sha256": hashlib.sha256(anchor.tobytes()).hexdigest(), "bank": [{
        "frame": 1, "candidate_uid": "p", "embedding_sha256": digest, "recording_id": "unit", "camera": None,
        "timestamp_seconds": .05, "confidence": .9, "source": "actual", "quality": 1., "anchor_consistency": 1.}],
        "pending_only_not_trusted": {"frame": 1, "count": 1, "embedding_SHA": digest}}
    features = dict.fromkeys(FEATURE_NAMES, 0.)
    restored, values = reconstruct_bank_input(anchor, snapshot, {"frame": 1, "target_uid": "p"}, {1: {"p": rows[0]}}, 2, rows[0], features)
    bank = [Evidence(vector, "unit", 1, None, .05, .9, "actual", 1., 1., "p")]
    pending = dict(frame=1, count=1, feature=vector, box=rows[0]["box_xyxy"], native=(None, rows[0]["native_tid"]))
    actual, expected = write_input(features, anchor, bank, pending, 2, rows[0])
    assert np.array_equal(restored, actual) and values == expected
    snapshot["bank"][0]["frame"] = 2
    with pytest.raises(ValueError):
        reconstruct_bank_input(anchor, snapshot, {"frame": 1, "target_uid": "p"}, {1: {"p": rows[0]}}, 2, rows[0], features)


def test_teacher_cases_are_correlated_and_UNKNOWN_counts_as_write_risk():
    rows = [{"sequence": "one", "episode_uid": "a", "source_case": "frozen", "class": 0},
            {"sequence": "one", "episode_uid": "a", "source_case": "mean", "class": 2},
            {"sequence": "two", "episode_uid": "b", "source_case": "frozen", "class": 1}]
    weights = teacher_weights(rows)
    assert np.allclose(weights, [.25, .25, .5])
    result = assess(rows, np.array([[.9, .05, .05]] * 3),
                    dict(status="DIAGNOSTIC", probability_min=.5, unknown_max=.5), weights)
    assert result["weighted_wrong_plus_UNKNOWN_risk"] == .75
    assert result["sampled_UNKNOWN_accepted"] == 1


def test_all24_memory_sources_and_actual_M_A_results_required(tmp_path, monkeypatch):
    from scripts import n72r21r2_memory_risk_driver as driver
    monkeypatch.setattr(driver, "OUT", tmp_path)
    sequences = ["s" + str(i) for i in range(24)]
    for folder in ("memory/risk_v1/supervision", "memory/M_A/results"):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        for sequence in sequences[:-1]:
            (directory / (sequence + ".json")).write_text("{}")
    assert not driver.all_sources_ready(sequences)
    for folder in ("memory/risk_v1/supervision", "memory/M_A/results"):
        (tmp_path / folder / (sequences[-1] + ".json")).write_text("{}")
    assert driver.all_sources_ready(sequences)
