from __future__ import annotations

from scripts.n72r20_aggregate_identity_bridge import (
    _summary,
    paired_delta,
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
