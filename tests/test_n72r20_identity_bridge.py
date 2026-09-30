from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from scripts.n72r20_identity_bridge_eval import (
    _method_record,
    evaluate_anchor,
    normalize,
    read_zstd_jsonl,
)
from sam3_intermot.identity_probe.dataset import GTBox


def _candidate(box: list[float], feature: np.ndarray, offset: int) -> dict[str, object]:
    return {
        "box_xyxy": box,
        "embedding_offset": offset,
        "embedding_dim": 512,
        "_feature": normalize(feature),
    }


def test_method_record_keeps_hard_negative_separate_from_target():
    state = np.zeros(512, dtype=np.float32)
    state[0] = 1.0
    target = np.zeros(512, dtype=np.float32)
    target[0] = 0.9
    target[1] = 0.4358899
    negative = np.zeros(512, dtype=np.float32)
    negative[0] = 0.8
    negative[1] = 0.6
    rows = [
        _candidate([0.0, 0.0, 10.0, 10.0], target, 0),
        _candidate([20.0, 0.0, 30.0, 10.0], negative, 1),
    ]

    record, next_state = _method_record(
        method="B0_HUMAN_ANCHOR_ONLY",
        sequence="synthetic",
        track_id=1,
        anchor_frame=1,
        future_frame=2,
        rows=rows,
        state=state,
        target_box=(0.0, 0.0, 10.0, 10.0),
        update_allowed=False,
        competitor_boxes=((20.0, 0.0, 30.0, 10.0),),
    )

    assert record["target_candidate_available"] is True
    assert record["hard_negative_available"] is True
    assert record["identity_evaluable"] is True
    assert record["win"] is True
    assert record["rank"] == 1
    assert record["update_applied"] is False
    np.testing.assert_allclose(next_state, state)


def test_b2_record_is_scored_before_caller_applies_frozen_gru_update():
    state = np.zeros(512, dtype=np.float32)
    state[0] = 1.0
    observation = np.zeros(512, dtype=np.float32)
    observation[0] = 1.0
    row, next_state = _method_record(
        method="B2_FROZEN_N72R18_GRU",
        sequence="synthetic",
        track_id=1,
        anchor_frame=1,
        future_frame=2,
        rows=[_candidate([0.0, 0.0, 10.0, 10.0], observation, 0)],
        state=state,
        target_box=(0.0, 0.0, 10.0, 10.0),
        update_allowed=True,
    )

    assert row["update_applied"] is True
    assert row["selected_candidate_is_target_posthoc"] is True
    np.testing.assert_allclose(next_state, state)


def test_runtime_replay_updates_on_target_not_visible_frames():
    class Sequence:
        seq_length = 3

        def image_path(self, frame):
            return frame

        def boxes(self, frame):
            target = GTBox("synthetic", frame, 1, 0.0, 0.0, 10.0, 10.0, 1.0, 1, 1.0)
            other = GTBox("synthetic", frame, 2, 20.0, 0.0, 10.0, 10.0, 1.0, 1, 1.0)
            return (target, other) if frame == 2 else (other,)

    class Encoder:
        def encode(self, _image, _boxes):
            vector = np.zeros((1, 512), dtype=np.float32)
            vector[0, 0] = 1.0
            return vector

    class IdentityGRU(torch.nn.Module):
        def forward(self, previous, observation):
            return observation, torch.ones((previous.shape[0], 1)), observation

    target = np.zeros(512, dtype=np.float32)
    target[0] = 1.0
    competitor = np.zeros(512, dtype=np.float32)
    competitor[0] = 0.8
    competitor[1] = 0.6
    candidates = {
        1: [
            _candidate([0.0, 0.0, 10.0, 10.0], target, 0),
            _candidate([20.0, 0.0, 30.0, 10.0], competitor, 1),
        ],
        2: [
            _candidate([20.0, 0.0, 30.0, 10.0], competitor, 2),
        ],
    }
    anchor_record = {
        "anchor_frame": 1,
        "anchor": {"track_id": 1, "tlwh": [0.0, 0.0, 10.0, 10.0]},
        "future": [],
    }

    rows = evaluate_anchor(
        sequence_name="synthetic",
        anchor_record=anchor_record,
        sequence=Sequence(),
        candidates=candidates,
        encoder=Encoder(),
        gru=IdentityGRU(),
        update_policy="immediate",
        confirmation_margin_threshold=None,
    )

    b2 = [row for row in rows if row["method"] == "B2_FROZEN_N72R18_GRU"]
    assert len(b2) == 2
    assert b2[0]["target_visible"] is True
    assert b2[1]["target_visible"] is False
    assert b2[1]["update_applied"] is True


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required by the candidate cache contract")
def test_zstd_metadata_reader_reports_decompression_errors_and_rows(tmp_path):
    source = tmp_path / "metadata.jsonl"
    compressed = tmp_path / "metadata.jsonl.zst"
    source.write_text(json.dumps({"frame": 0, "candidates": []}) + "\n", encoding="utf-8")
    with compressed.open("wb") as handle:
        subprocess.run(["zstd", "-q", "-c", str(source)], stdout=handle, check=True)

    assert read_zstd_jsonl(compressed) == {0: []}
