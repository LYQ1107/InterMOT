import numpy as np
import pytest
from sam3_intermot.evaluation.scene_difficulty import describe_video, describe_episode, max_box_overlap, transition_reappearance


def gt(identity, box=(0., 0., 2., 2.)):
    return dict(identity=identity, box=box, visibility=1.)


def test_gt_people_and_constant_visibility_are_not_candidate_density_or_occlusion_truth():
    result = describe_video({0: [gt(1), gt(2), gt(3)], 1: [gt(1)]}, 3)
    assert result["GT_people_per_original_frame"]["mean"] == pytest.approx(4 / 3)
    assert result["GT_video_local_identities_NOT_cross_scene_people"] == 3
    assert result["GT_visibility_constant_NOT_informative_occlusion_truth"]
    assert not result["occlusion_ground_truth_available"] and not result["density_reclassification"]
    assert max_box_overlap([gt(1), gt(2, (1., 0., 3., 2.))]) == {1: .5, 2: .5}


def test_return_after_absence_not_first_arrival_or_independent_root():
    assert transition_reappearance([False, True, False, True], initially_present=False)["observational_returns_NOT_independent_events"] == 1
    result = transition_reappearance([True, False, False, True], initially_present=True)
    assert result["preceding_absence_frames"]["mean"] == 2.


def fixture():
    rows = [dict(candidate_uid="p", feature=np.array([1., 0.])),
            dict(candidate_uid="q", feature=np.array([0., 1.])),
            dict(candidate_uid="u", feature=np.array([1., 0.]))]
    frames = [(dict(frame=f), rows) for f in range(4)]
    trace = [dict(frame=f, target_public_id=7, target_uid="p", outputs=[]) for f in range(4)]
    trace[2]["outputs"] = [dict(candidate_uid="p", public_id=9)]
    trace[3]["outputs"] = [dict(candidate_uid="p", public_id=7)]
    matches = {f: dict(p=1, q=2, u=None) for f in range(4)}
    annotations = {f: [gt(1), gt(2)] for f in range(4)}
    return frames, trace, matches, annotations, [1., 0.], dict(frame=0, episode_uid="one"), 1


def test_unassigned_is_not_competitor_owned_unknown_is_not_verified_negative():
    result = describe_episode(*fixture())
    counts = result["counts_NOT_independent_events"]
    assert counts["positive_ownership_UNASSIGNED"] == counts["positive_ownership_OWN_OTHER_PUBLIC_ID"] == counts["positive_ownership_OWN_TARGET_PUBLIC_ID"] == 1
    assert counts["strict_positive_wins"] == 3
    assert result["distributions"]["positive_minus_hardest_verified_OTHER_margin"]["mean"] == 1.
    assert result["distributions"]["hardest_nonpositive_including_UNKNOWN_anchor_cosine"]["mean"] == 1.
    assert not result["independent_causal_event_roots_proven"]


def test_axes_duplicate_annotations_and_noncurrent_outputs_are_rejected():
    with pytest.raises(ValueError, match="frame axis"):
        describe_video({3: [gt(1)]}, 3)
    with pytest.raises(ValueError, match="unique"):
        describe_video({0: [gt(1), gt(1)]}, 1)
    args = fixture(); args[1][1]["outputs"] = [dict(candidate_uid="not_current", public_id=9)]
    with pytest.raises(ValueError, match="ownership"):
        describe_episode(*args)


def test_offline_descriptors_cannot_legalize_gt_assisted_runtime():
    args = fixture(); args[1][1]["runtime_future_gt_used"] = True
    with pytest.raises(ValueError, match="GT-aided"):
        describe_episode(*args)
