"""Unit contracts for N72R19 deterministic noisy-memory replay."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sam3_intermot.identity_memory.noise import (
    NoiseSpec,
    corrupt_observation,
    training_noise_spec,
)


def _vectors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.tensor([1.0, 0.0, 0.0, 0.0])
    clean = torch.tensor([1.0, 0.0, 0.0, 0.0])
    competitors = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    mask = torch.tensor([True, True])
    return state, clean, competitors, mask


def test_noise_a_is_deterministic_and_replaces_visible_identity():
    state, clean, competitors, mask = _vectors()
    spec = NoiseSpec("wrong_identity_injection", 1.0)
    first = corrupt_observation(state, clean, competitors, mask, spec, "episode", 3)
    second = corrupt_observation(state, clean, competitors, mask, spec, "episode", 3)
    assert first.applied is True
    assert first.present is True
    assert first.replacement_index in {0, 1}
    assert first.replacement_index == second.replacement_index
    assert torch.equal(first.observation, second.observation)


def test_noise_b_holds_the_previous_state_by_marking_observation_missing():
    state, clean, competitors, mask = _vectors()
    result = corrupt_observation(
        state,
        clean,
        competitors,
        mask,
        NoiseSpec("missing_observation", 1.0),
        "episode",
        3,
    )
    assert result.applied is True
    assert result.present is False
    assert result.source == "missing"


def test_noise_c_selects_the_current_state_hard_negative():
    state = torch.tensor([0.0, 0.99, 0.1, 0.0])
    clean = torch.tensor([1.0, 0.0, 0.0, 0.0])
    competitors = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    mask = torch.tensor([True, True])
    result = corrupt_observation(
        state,
        clean,
        competitors,
        mask,
        NoiseSpec("hard_negative_replacement", 1.0),
        "episode",
        3,
    )
    assert result.replacement_index == 1
    assert torch.equal(result.observation, competitors[1])


def test_training_noise_spec_is_reproducible():
    first = [training_noise_spec("episode", step) for step in range(20)]
    second = [training_noise_spec("episode", step) for step in range(20)]
    assert first == second


def test_n72r19_goal_is_frozen():
    path = Path("outputs/N72R19/FINAL_GOAL.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["stage"] == "N72R19"
    assert document["goal"] == "Robust Human Identity Memory Learning"
    assert document["goal_family"] == "Human Identity Memory Learning"
    assert document["goal_frozen"] is True
    assert document["sam3_required"] is False
    assert document["mot_required"] is False
