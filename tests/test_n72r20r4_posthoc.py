import pytest

from scripts.n72r20r4_posthoc import residual_bound
from sam3_intermot.association.identity_authority import AuthorityConfig


def decision(gap, scores, uid="a", authority=1):
    return {"target_public_id": 7, "base_assignments": {"7": uid},
            "authority": authority, "identity_scores": scores,
            "authority_features": [0, 0, gap]}


def test_global_gap_certificate_distinguishes_score_win_from_authority():
    rows = [{"candidate_uid": "a"}, {"candidate_uid": "b"}]
    config = AuthorityConfig(mode="fixed", strength=1, identity_mean=0,
                             identity_std=1, base_scale=0.2).to_dict()
    result = residual_bound(decision(3.5, [0, 1]), rows, config)
    assert result["max_possible_target_residual_advantage"] == pytest.approx(0.2)
    assert result["certified_blocked"]
    assert not residual_bound(decision(0.1, [0, 1]), rows, config)["certified_blocked"]
    assert not residual_bound(decision(-1, [0, 1], uid=None), rows, config)["baseline_target_assigned"]
    assert residual_bound(decision(0.1, [0, 1], authority=0), rows, config)["certified_blocked"]


def test_residual_certificate_includes_NONE_alternative():
    config = AuthorityConfig(mode="fixed", strength=1, identity_mean=0,
                             identity_std=1, base_scale=1).to_dict()
    result = residual_bound(decision(0.5, [-1]), [{"candidate_uid": "a"}], config)
    assert result["max_possible_target_residual_advantage"] == 1
    assert not result["certified_blocked"]
