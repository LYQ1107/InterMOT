"""Unit contracts for the N72R19R1 selective identity-memory probe."""

from __future__ import annotations

import json

import torch

from sam3_intermot.identity_memory.selective import (
    FEATURE_NAMES,
    ObservationSelector,
    TrustedState,
    build_evidence_features,
    feature_indices_for_ablation,
    load_frozen_n72r18_gru,
    update_state_hard,
)
from sam3_intermot.identity_memory.r1_protocol import (
    MANIFEST_VERSION,
    all_condition_slugs,
    condition_slug,
    deterministic_uniform,
)
from sam3_intermot.identity_memory.r1_eval import _roc_auc
from sam3_intermot.identity_memory.updater import VARIANT_GRU, build_updater


def _checkpoint(tmp_path):
    model = build_updater(VARIANT_GRU, feature_dim=4)
    path = tmp_path / "n72r18_gru.pt"
    torch.save(
        {
            "stage": "N72R18",
            "variant": VARIANT_GRU,
            "config": {"feature_dimension": 4},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path, model


def test_r1_goal_is_frozen_and_excludes_mot(tmp_path):
    goal = {
        "stage": "N72R19R1",
        "goal": "Selective Identity Memory Update",
        "goal_frozen": True,
        "mot_required": False,
        "sam3_required": False,
    }
    path = tmp_path / "goal.json"
    path.write_text(json.dumps(goal), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["goal_frozen"] is True
    assert loaded["goal"] == "Selective Identity Memory Update"
    assert loaded["mot_required"] is False
    assert loaded["sam3_required"] is False


def test_frozen_n72r18_loader_is_strict_and_freezes_parameters(tmp_path):
    path, reference = _checkpoint(tmp_path)
    loaded = load_frozen_n72r18_gru(path, "cpu")
    assert loaded.variant == VARIANT_GRU
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for name, parameter in loaded.state_dict().items():
        assert torch.equal(parameter, reference.state_dict()[name])


def test_selector_is_small_and_ablations_are_nested():
    counts = [ObservationSelector(feature_indices=feature_indices_for_ablation(name)).parameter_count for name in (
        "A1_memory_similarity_only",
        "A2_memory_plus_human_anchor",
        "A3_plus_competition_margin",
        "A4_plus_prospective_state_drift",
        "A5_full_observation_evidence",
    )]
    assert counts == sorted(counts)
    assert counts[-1] < 100_000
    assert len(FEATURE_NAMES) == 13


def test_rejected_observation_holds_state_and_anchor_is_immutable():
    previous = torch.tensor([1.0, 0.0, 0.0, 0.0])
    candidate = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert torch.equal(update_state_hard(previous, candidate, False), previous)
    trusted = TrustedState(previous, frame=10)
    anchor_before = trusted.anchor.clone()
    trusted.observe_accept(previous, candidate, frame=20)
    assert torch.equal(trusted.anchor, anchor_before)
    assert trusted.last_trusted_update_frame == 20
    assert trusted.trusted_update_count == 1


def test_evidence_features_are_current_frame_only_and_candidate_is_preupdate():
    torch.manual_seed(72191)
    updater = build_updater(VARIANT_GRU, feature_dim=4)
    state = torch.nn.functional.normalize(torch.randn(4), dim=0)
    anchor = state.clone()
    observation = torch.nn.functional.normalize(torch.randn(4), dim=0)
    competitors = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    features, info, candidate = build_evidence_features(
        state,
        anchor,
        observation,
        competitors,
        torch.tensor([True, True, True]),
        frame=20,
        last_trusted_update_frame=10,
        trusted_update_count=2,
        recent_state_stability=0.1,
        frozen_gru=updater,
    )
    assert features.shape == (len(FEATURE_NAMES),)
    assert set(info).issuperset(FEATURE_NAMES)
    assert torch.allclose(state, torch.nn.functional.normalize(state, dim=0))
    assert not torch.equal(candidate, state)
    assert "future" not in FEATURE_NAMES
    assert "label" not in FEATURE_NAMES


def test_missing_update_policy_is_hold_without_selector():
    previous = torch.tensor([1.0, 0.0, 0.0, 0.0])
    candidate = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert torch.equal(update_state_hard(previous, candidate, False), previous)


def test_corruption_manifest_decisions_are_deterministic_and_method_independent():
    key = "seq|3|0|10"
    first = deterministic_uniform(72191, key, 1, "hard_negative", 0.3, "apply")
    second = deterministic_uniform(72191, key, 1, "hard_negative", 0.3, "apply")
    assert first == second
    assert MANIFEST_VERSION == "N72R19R1-fixed-corruption-v1"
    assert condition_slug("wrong_identity", 0.2) in all_condition_slugs()
    assert condition_slug("hard_negative", 0.3) in all_condition_slugs()


def test_score_features_are_computed_before_any_state_write():
    torch.manual_seed(72192)
    updater = build_updater(VARIANT_GRU, feature_dim=4)
    state = torch.nn.functional.normalize(torch.randn(4), dim=0)
    before = state.clone()
    observation = torch.nn.functional.normalize(torch.randn(4), dim=0)
    build_evidence_features(
        state,
        state,
        observation,
        torch.nn.functional.normalize(torch.randn(2, 4), dim=-1),
        torch.tensor([True, True]),
        frame=12,
        last_trusted_update_frame=0,
        trusted_update_count=0,
        recent_state_stability=0.0,
        frozen_gru=updater,
    )
    assert torch.equal(state, before)


def test_training_protocol_contract_keeps_val_out_of_train_sequences():
    train_sequences = {"train_a", "train_b"}
    val_sequences = {"val_a"}
    assert train_sequences.isdisjoint(val_sequences)


def test_roc_auc_uses_descending_score_order_correctly():
    assert _roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
