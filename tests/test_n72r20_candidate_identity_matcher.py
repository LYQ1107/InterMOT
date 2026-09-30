from __future__ import annotations

import numpy as np
import pytest
import torch

from sam3_intermot.identity_memory.candidate_identity_matcher import CandidateIdentityMatcher


def _candidate(index: int, embedding: np.ndarray | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_id": 7,
        "bbox": [float(index), 0.0, float(index + 10), 20.0],
        "mask": None,
        "sam_score": 0.9,
        "decoder_token": None,
    }
    if embedding is not None:
        row["embedding"] = embedding
    return row


def test_extract_score_and_rank_precomputed_embeddings() -> None:
    matcher = CandidateIdentityMatcher()
    first = np.zeros(512, dtype=np.float32)
    second = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    second[1] = 1.0
    candidates = [_candidate(0, first), _candidate(1, second)]
    features = matcher.extract_candidate_features(candidates)
    state = np.zeros(512, dtype=np.float32)
    state[0] = 1.0
    scores = matcher.score_candidates(state, features)

    assert features.shape == (2, 512)
    assert torch.allclose(scores, torch.tensor([1.0, 0.0]), atol=1.0e-6)
    assert matcher.rank_candidates(scores) == [0, 1]


def test_feature_extractor_callback_is_used_without_persisting_candidate_payload() -> None:
    seen: list[int] = []

    def extractor(candidates: tuple[object, ...]) -> np.ndarray:
        seen.append(len(candidates))
        return np.tile(np.arange(1.0, 513.0, dtype=np.float32), (len(candidates), 1))

    matcher = CandidateIdentityMatcher(feature_extractor=extractor)
    features = matcher.extract_candidate_features([_candidate(0), _candidate(1)])

    assert seen == [2]
    assert features.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(features, dim=-1), torch.ones(2))


def test_runtime_gt_fields_are_rejected() -> None:
    candidate = _candidate(0, np.ones(512, dtype=np.float32))
    candidate["gt_id"] = 12

    with pytest.raises(ValueError, match="forbidden GT/identity"):
        CandidateIdentityMatcher().extract_candidate_features([candidate])


def test_rank_is_stable_for_ties_and_empty_input() -> None:
    matcher = CandidateIdentityMatcher()

    assert matcher.rank_candidates(torch.tensor([0.4, 0.4, 0.2])) == [0, 1, 2]
    assert matcher.rank_candidates(torch.empty(0)) == []
