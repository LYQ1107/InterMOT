import json
from pathlib import Path

import torch

from sam3_intermot.identity_memory.encoder import FEATURE_DIMENSION
from sam3_intermot.identity_memory.losses import hard_negative_info_nce
from sam3_intermot.identity_memory.updater import VARIANT_GRU, VARIANT_GRU_GATE, build_updater


def test_n72r18_goal_is_frozen_to_memory_learning():
    goal = json.loads(Path("outputs/N72R18/FINAL_GOAL.json").read_text(encoding="utf-8"))
    assert goal["stage"] == "N72R18"
    assert goal["goal"] == "Human Identity Memory Learning"
    assert goal["primary_endpoint"] == "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE"
    assert goal["goal_frozen"] is True
    assert goal["sam3_required"] is False
    assert goal["mot_evaluation_required"] is False


def test_gru_and_gate_return_normalized_state_and_reliability():
    previous = torch.nn.functional.normalize(torch.randn(3, FEATURE_DIMENSION), dim=-1)
    observation = torch.nn.functional.normalize(torch.randn(3, FEATURE_DIMENSION), dim=-1)
    for variant in (VARIANT_GRU, VARIANT_GRU_GATE):
        model = build_updater(variant)
        state, reliability, candidate = model(previous, observation)
        assert state.shape == previous.shape
        assert candidate.shape == previous.shape
        assert reliability.shape == (3,)
        assert torch.allclose(state.norm(dim=-1), torch.ones(3), atol=1e-5)
        assert torch.all((reliability >= 0) & (reliability <= 1))
        if variant == VARIANT_GRU:
            assert torch.allclose(reliability, torch.ones_like(reliability))


def test_hard_negative_infonce_uses_all_visible_competitors():
    state = torch.tensor([[1.0, 0.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0, 0.0]])
    negatives = torch.tensor([[[0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]])
    mask = torch.tensor([[True, True]])
    loss = hard_negative_info_nce(state, positive, negatives, mask, temperature=0.1)
    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
    assert float(loss.item()) > 0
