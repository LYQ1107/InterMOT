import numpy as np
import pytest
import torch
from sam3_intermot.association.causal_state_commit import memory_metrics, commit_memory
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank


class Updater(torch.nn.Module):
    def forward(self, old, observation):
        return torch.nn.functional.normalize(old+observation, dim=-1), torch.ones(old.shape[0]), observation


def test_zero_write_is_not_memory_safety():
    m = memory_metrics([{"frame": 1, "eligible": True, "correct": True, "accepted": False}])
    assert m["wrong_write_rate"] is None
    assert m["correct_write_retention"] == 0
    assert m["safety_pass"] is False


def test_real_consensus_write_after_scoring_and_anchor_immutable():
    anchor = np.zeros(512, dtype=np.float32); anchor[0] = 1
    bank = LearnedIdentityMemoryBank.from_updater(Updater())
    bank.initialize_public_identity(public_id=1001, association_state_id=1, frame=0, human_anchor=anchor, interaction_source="test")
    old_hash = bank.records[1001].state_hash()
    rows = [{"candidate_uid": "a", "feature": anchor, "conf": 1}]
    solver = {"public_assignments": [{"public_id": 1001, "candidate_uid": "a"}]}
    result = commit_memory(bank, AuthorityConfig(memory="P1"), frame=1, rows=rows, base_solver=solver, solver=solver, public_id=1001, scores=np.array([0.8]), score_state_hash=old_hash)
    assert result["accepted"]
    assert bank.records[1001].machine_update_count == 1
    np.testing.assert_array_equal(bank.records[1001].human_anchor, anchor)
    with pytest.raises(RuntimeError, match="score-before-update"):
        commit_memory(bank, AuthorityConfig(memory="P1"), frame=1, rows=rows, base_solver=solver, solver=solver, public_id=1001, scores=np.array([0.8]), score_state_hash=old_hash)


def test_explicit_denominators():
    rows = [{"frame": 1, "eligible": True, "accepted": True, "correct": True}, {"frame": 2, "eligible": True, "accepted": True, "correct": False}, {"frame": 3, "eligible": True, "accepted": False, "correct": True}]
    m = memory_metrics(rows)
    assert m["wrong_write_rate"] == 0.5
    assert m["correct_write_retention"] == 0.5
