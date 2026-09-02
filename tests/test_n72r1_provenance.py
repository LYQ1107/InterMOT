"""N72R1 provenance/UID toy contract tests; no scientific data is used."""

from __future__ import annotations

import numpy as np
import pytest

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.provenance.candidate_v2 import build_candidate_v2_row, validate_candidate_v2_row
from sam3_intermot.provenance.mapping import canonical_candidate_uid_v2


def _metadata() -> dict[str, str]:
    return {
        "source_run_id": "toy-run",
        "sequence": "toy-seq",
        "video_id": "toy-video",
        "checkpoint_sha256": "a" * 64,
        "runtime_config_sha256": "b" * 64,
        "session_id": "toy-session",
        "segment_id": "segment-0",
        "window_id": "window-0",
        "chunk_id": "chunk-0",
    }


def _observation(raw: int = 17) -> PromptObjectObservation:
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:8, 3:10] = True
    return PromptObjectObservation(
        frame_idx=4,
        sam_object_id=9001,
        raw_sam_object_id=raw,
        mask=mask,
        box_xyxy=np.asarray([3, 2, 10, 8], dtype=float),
        confidence=0.9,
        presence_score=0.8,
    )


def test_uid_v2_is_sensitive_to_all_identity_axes() -> None:
    base = dict(
        source_run_id="run",
        sequence="seq",
        session_id="session",
        segment_id="segment",
        window_id="window",
        chunk_id="chunk",
        frame_idx=4,
        candidate_index=0,
        official_raw_sam_id=17,
        adapter_external_id=9001,
        box_digest="a" * 64,
        mask_sha256="b" * 64,
    )
    first = canonical_candidate_uid_v2(**base)
    assert first == canonical_candidate_uid_v2(**base)
    for key, value in (("session_id", "other"), ("chunk_id", "other"), ("candidate_index", 1), ("mask_sha256", "c" * 64)):
        changed = dict(base)
        changed[key] = value
        assert canonical_candidate_uid_v2(**changed) != first


def test_candidate_v2_requires_explicit_local_global_and_never_emits_public_id() -> None:
    metadata = _metadata()
    row = build_candidate_v2_row(
        _observation(),
        metadata=metadata,
        candidate_index=0,
        segment_local_id="local-token",
        sequence_global_id="global-token",
        feature=np.ones(512, dtype=np.float32),
    )
    assert row["segment_local_id"] == "local-token"
    assert row["sequence_global_id"] == "global-token"
    assert "public_id" not in row
    assert validate_candidate_v2_row(row) == []
    with pytest.raises(ValueError, match="axes"):
        build_candidate_v2_row(
            _observation(),
            metadata=metadata,
            candidate_index=0,
            segment_local_id="",
            sequence_global_id="global-token",
        )


def test_machine_exporter_rejects_human_feature_source() -> None:
    with pytest.raises(ValueError, match="human evidence"):
        build_candidate_v2_row(
            _observation(),
            metadata=_metadata(),
            candidate_index=0,
            segment_local_id="local-token",
            sequence_global_id="global-token",
            feature=np.ones(512, dtype=np.float32),
            feature_source="human_confirmed_roi",
        )
