from __future__ import annotations

from scripts.n72r20_aggregate_identity_bridge import (
    _summary,
    decide,
    paired_delta,
    paired_policy_rate_delta,
)


def _row(sequence: str, method: str, frame: int, *, win: bool, rank: int = 1) -> dict[str, object]:
    return {
        "sequence": sequence,
        "method": method,
        "track_id": 1,
        "anchor_frame": 1,
        "frame": frame,
        "gap": frame - 1,
        "candidate_available": True,
        "target_candidate_available": True,
        "identity_evaluable": True,
        "win": win,
        "rank": rank,
        "margin": 0.1 if win else -0.1,
        "update_applied": method != "B0_HUMAN_ANCHOR_ONLY",
        "wrong_memory_write_posthoc": False,
    }


def test_aggregation_keeps_candidate_coverage_separate_from_identity_win_rate():
    rows = [
        _row("seq1", "B0_HUMAN_ANCHOR_ONLY", 2, win=True),
        {
            **_row("seq1", "B0_HUMAN_ANCHOR_ONLY", 3, win=False, rank=2),
            "target_candidate_available": False,
            "identity_evaluable": False,
            "win": None,
            "rank": None,
            "margin": None,
        },
    ]
    summary = _summary(rows, "H20")
    assert summary["candidate_coverage"] == 0.5
    assert summary["identity_evaluable_frames"] == 1
    assert summary["hard_negative_identity_win_rate"] == 1.0


def test_paired_delta_uses_same_anchor_frame_keys():
    rows = [
        _row("seq1", "B0_HUMAN_ANCHOR_ONLY", 2, win=False, rank=2),
        _row("seq1", "B2_FROZEN_N72R18_GRU", 2, win=True),
        _row("seq2", "B0_HUMAN_ANCHOR_ONLY", 2, win=True),
        _row("seq2", "B2_FROZEN_N72R18_GRU", 2, win=False, rank=2),
        _row("seq3", "B0_HUMAN_ANCHOR_ONLY", 2, win=True),
    ]
    result = paired_delta(
        rows,
        "B2_FROZEN_N72R18_GRU",
        "B0_HUMAN_ANCHOR_ONLY",
        horizon=100,
        field="win",
    )
    assert result["paired_frames"] == 2
    assert result["paired_sequence_count"] == 2
    assert result["delta_left_minus_right"] == 0.0


def test_decision_requires_coverage_gain_and_contamination_gates():
    def method_block(rank: float, coverage: float, wrong_rate: float | None) -> dict[str, object]:
        horizon = {
            "candidate_coverage": coverage,
            "rank_1_accuracy": rank,
            "wrong_memory_write_rate": wrong_rate,
        }
        return {"horizons": {"H20": dict(horizon), "H50": dict(horizon), "H100": dict(horizon)}}

    immediate = {
        "B0_HUMAN_ANCHOR_ONLY": method_block(0.80, 0.90, None),
        "B2_FROZEN_N72R18_GRU": method_block(0.90, 0.90, 0.40),
    }
    confirmed = {
        "B2_FROZEN_N72R18_GRU": method_block(0.88, 0.90, 0.10),
    }
    paired = {"delta_left_minus_right": 0.10, "sequence_cluster_ci95": {"lower": 0.05}}
    contamination = {"delta_immediate_minus_2frame": 0.30, "sequence_cluster_ci95": {"lower": 0.10}}

    decision, reason = decide(
        complete=True,
        immediate=immediate,
        confirmed=confirmed,
        paired=paired,
        contamination_delta=contamination,
    )

    assert decision == "PASS_REAL_CANDIDATE_IDENTITY_MEMORY"
    assert "gates passed" in reason


def test_policy_rate_delta_is_sequence_clustered():
    immediate = [
        {"sequence": "seq1", "method": "B2_FROZEN_N72R18_GRU", "gap": 1, "target_visible": True, "update_applied": True, "wrong_memory_write_posthoc": True},
        {"sequence": "seq2", "method": "B2_FROZEN_N72R18_GRU", "gap": 1, "target_visible": True, "update_applied": True, "wrong_memory_write_posthoc": True},
    ]
    confirmed = [
        {"sequence": "seq1", "method": "B2_FROZEN_N72R18_GRU", "gap": 1, "target_visible": True, "update_applied": True, "wrong_memory_write_posthoc": False},
        {"sequence": "seq2", "method": "B2_FROZEN_N72R18_GRU", "gap": 1, "target_visible": True, "update_applied": True, "wrong_memory_write_posthoc": False},
    ]
    result = paired_policy_rate_delta(
        immediate,
        confirmed,
        method="B2_FROZEN_N72R18_GRU",
        horizon=100,
        numerator_field="wrong_memory_write_posthoc",
    )
    assert result["delta_immediate_minus_2frame"] == 1.0
    assert result["sequence_cluster_ci95"]["lower"] == 1.0
