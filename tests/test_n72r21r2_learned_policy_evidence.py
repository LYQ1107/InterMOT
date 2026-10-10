from copy import deepcopy
import pytest
from sam3_intermot.evaluation.learned_policy_evidence import (METRICS, compact_policy_result,
    assert_decision_preserved, density_from_counts, video_macro, choose_inner_point)


def result(changed=False):
    keep = [dict(public_id=1, candidate_uid="a"), dict(public_id=2, candidate_uid="b")]
    actual = [dict(public_id=1, candidate_uid="b"), dict(public_id=2, candidate_uid="a")] if changed else keep
    return dict(frame=1, outputs=actual, target_uid=actual[0]["candidate_uid"], target_public_id=1,
        selected_action=dict(family="GLOBAL_SWAP" if changed else "KEEP"), state_before="s0", state_after="s1", births=[], deaths=[],
        full_tracker_state_before_sha256="tensor0", full_tracker_state_after_sha256="tensor1",
        joint_identity_memory_write=False, joint_memory_write_candidate_uid=None,
        huge_solver_matrix=[0.] * 10000, authority=dict(effective_assignment_change=changed,
        chosen_branch="RAW_IDENTITY_TOP" if changed else "KEEP", prediction=dict(beneficial=.9, harmful=.01, value=.1),
        features=dict(current=1), all_current_choices=[dict(branch="KEEP", features=dict(current=1))],
        own_unmodified_KEEP_outputs=keep))


def test_compaction_preserves_global_decision_and_current_own_keep_no_matrices():
    for changed in (False, True):
        original = result(changed)
        compact = compact_policy_result(original)
        assert_decision_preserved(original, compact)
        assert "huge_solver_matrix" not in compact
        assert compact["authority"]["own_KEEP_target_uid"] == "a"
        assert (compact["authority"]["own_KEEP_outputs"] is not None) == changed
        assert ("all_current_choices" in compact["authority"]) == changed
        bad = deepcopy(compact); bad["outputs"][0]["candidate_uid"] = "forged"
        with pytest.raises(AssertionError):
            assert_decision_preserved(original, bad)


def test_preregistered_density_full_frame_axis_no_gt():
    assert density_from_counts([1] * 8 + [12] * 2)["whole_video_density"] == "SPARSE"
    assert density_from_counts([5] * 8 + [0] * 2)["whole_video_density"] == "MEDIUM"
    assert density_from_counts([1, 6, 9])["whole_video_density"] == "MIXED"
    with pytest.raises(ValueError):
        density_from_counts([])


def cell(sequence, seed, delta=0., actions=0, n01=0, n10=0):
    return dict(sequence=sequence, seed=seed, valid_initializations=1,
        paired_deltas_vs_C0={"click0": {k: delta for k in METRICS}},
        effective_direct_decisions_NOT_independent_onsets=actions,
        per_episode_target_components={"click0": dict(N01_frames=n01, N10_frames=n10)},
        own_onset_summary=dict(risky_decisions=0, severe_non_target_harm_decisions=0,
                              beneficial_complete_H100_decisions=actions, incomplete_H100_decisions=0))


def test_macro_seed_and_clicks_are_not_independent_video_clusters():
    rows = [cell(s, seed, delta=seed) for s in ("a", "b") for seed in (1, 2, 3)]
    rows[0]["paired_deltas_vs_C0"]["click1"] = {k: 5. for k in METRICS}
    output = video_macro(rows, [1, 2, 3], ["a", "b"])
    assert output["per_video"]["a"]["HOTA"] == pytest.approx(8 / 3)
    assert output["macro"]["HOTA"] == pytest.approx(7 / 3)
    with pytest.raises(ValueError):
        video_macro(rows[:-1], [1, 2, 3], ["a", "b"])


def test_zero_action_and_positive_hota_never_qualify_for_confirmation():
    rows = [cell(s, seed, delta=.1) for s in ("a", "b", "c") for seed in (1, 2, 3)]
    selected = choose_inner_point({"P0_CLAIM80": rows, "P1_CLAIM50": rows}, [1, 2, 3], ["a", "b", "c"])
    assert selected["selected_point"] == "P0_CLAIM80"
    assert selected["status"].startswith("NO_NONVACUOUS")
    assert not selected["confirmation_authorized"]


def test_harm_and_incomplete_windows_precede_macro_reward_without_root_claim():
    sequences, seeds = ["a", "b", "c"], [1, 2, 3]
    safe = [cell(s, seed, .01, 1, 1) for s in sequences for seed in seeds]
    unsafe = deepcopy(safe)
    for c in unsafe:
        c["paired_deltas_vs_C0"]["click0"]["HOTA"] = .2
        c["own_onset_summary"]["incomplete_H100_decisions"] = 1
    selected = choose_inner_point({"P0_CLAIM80": safe, "P1_CLAIM50": unsafe}, seeds, sequences)
    assert selected["selected_point"] == "P0_CLAIM80"
    assert not selected["independent_G1_roots_proven"] and not selected["confirmation_authorized"]
