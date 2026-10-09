import pytest

from sam3_intermot.evaluation.one_click_protocol import FrameTruth, evaluate_episode, iou, open_set_metrics, runs


def row(frame, uid=None, box=None, p=.1, write=False):
    return {"frame": frame, "target_present_probability": p, "candidate_available_probability": p,
            "selected_candidate_uid": uid, "rank1_candidate_uid": uid, "predicted_box_xyxy": box,
            "memory_write": write, "runtime_future_gt_used": False}


def test_one_click_visibility_absence_reappearance_and_wrong_person_are_distinct():
    a, b = (0,0,10,10), (20,20,30,30)
    truth = [FrameTruth(0,True,a,frozenset({'a'}), (b,)),
             FrameTruth(1,False,None,frozenset(), (b,)),
             FrameTruth(2,True,a,frozenset({'a'}), (b,)),
             FrameTruth(3,True,a,frozenset(), (b,))]
    runtime = [row(0,'a',a,.9,True), row(1,'b',b,.8,True), row(2,'a',a,.9,True), row(3,None,None,.1)]
    result = evaluate_episode(runtime,truth,fps=2,recording_id='session-a')
    assert result['target']['target_recall_all_visible'] == 2/3
    assert result['target']['target_recall_given_candidate'] == 1
    assert result['target']['verified_wrong_person_takeover_frames'] == 1
    assert result['target']['P0_true_absence_fpr'] == 1
    assert result['target']['P1_present_unavailable_fpr'] == 0
    assert result['target']['reacquisition_recall'] == 1
    assert result['memory']['wrong_write_rate'] == 1/3


def test_zero_writes_and_no_negative_data_do_not_pass_safety_or_open_set():
    a = (0,0,10,10)
    r = evaluate_episode([row(0,'a',a,.9)], [FrameTruth(0,True,a,frozenset({'a'}))],fps=30,recording_id='a')
    assert r['memory']['wrong_write_rate'] is None
    assert not r['memory']['joint_safety_pass']
    assert r['open_set']['recall_at_fpr_2pct'] is None


def test_tied_scores_are_not_optimistically_broken_at_fpr_limit():
    result = open_set_metrics([.8,.8],[True,False],[True,False])
    assert result['recall_at_fpr_2pct'] == 0
    assert result['PR_AUC_average_precision'] == .5


@pytest.mark.parametrize('bad',[-.1,1.1,float('nan')])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError): open_set_metrics([bad],[True],[True])


def test_sparse_annotation_and_noncontiguous_frames_not_true_absence():
    with pytest.raises(ValueError):
        evaluate_episode([row(0),row(2)],[FrameTruth(0,False,None,frozenset()),FrameTruth(2,False,None,frozenset())],fps=30,recording_id='a')
    with pytest.raises(ValueError):
        evaluate_episode([row(0)],[FrameTruth(0,False,None,frozenset(),annotation_complete=False)],fps=30,recording_id='a')


def test_incomplete_person_gt_does_not_claim_verified_takeovers():
    r = evaluate_episode([row(0,'b',(20,20,30,30))],[FrameTruth(0,True,(0,0,10,10),frozenset(),complete_person_labels=False)],fps=30,recording_id='a')
    assert r['target']['verified_wrong_person_takeover_frames'] is None


def test_duplicate_decisions_and_future_gt_rejected():
    t = FrameTruth(0,False,None,frozenset())
    with pytest.raises(ValueError): evaluate_episode([row(0),row(0)],[t,t],fps=30,recording_id='a')
    r=row(0);r['runtime_future_gt_used']=True
    with pytest.raises(ValueError): evaluate_episode([r],[t],fps=30,recording_id='a')


def test_geometry_and_runs():
    assert iou((0,0,10,10),(0,0,10,10)) == 1
    assert iou(None,(0,0,10,10)) == 0
    assert runs([False,True,True,False,True]) == [(1,3),(4,5)]
