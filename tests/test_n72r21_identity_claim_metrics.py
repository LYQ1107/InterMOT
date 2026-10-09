from sam3_intermot.evaluation.one_click_protocol import FrameTruth
from sam3_intermot.evaluation.one_click_identity import strict_identity_claim_metrics
import pytest


def test_high_availability_wrong_rank_is_not_identity_PR_positive():
    truth=[FrameTruth(1,True,(0,0,10,20),frozenset({'target'})),FrameTruth(2,True,(0,0,10,20),frozenset({'target'}))]
    trace=[{'frame':1,'rank1_candidate_uid':'other','selected_candidate_uid':'other','rank1_identity_joint_probability':.9,'runtime_future_gt_used':False},
           {'frame':2,'rank1_candidate_uid':'target','selected_candidate_uid':'target','rank1_identity_joint_probability':.8,'runtime_future_gt_used':False}]
    result=strict_identity_claim_metrics(trace,truth,verified_rank1_labels=[False,True])
    assert result['identity_claim_PR_AUC']==.5
    assert result['strict_identity_recall_given_candidate']==.5
    assert result['rank1_correct_claim_recall_at_wrong_rank_FPR2pct']==0.


def test_no_candidate_is_not_positive_and_cannot_invent_truth():
    truth=[FrameTruth(1,False,None,frozenset())]
    trace=[{'frame':1,'rank1_candidate_uid':None,'selected_candidate_uid':None,'rank1_identity_joint_probability':0.,'runtime_future_gt_used':False}]
    result=strict_identity_claim_metrics(trace,truth)
    assert result['identity_claim_PR_AUC'] is None and result['identity_claim_ECE']==0
    assert result['strict_identity_precision_selected'] is None


def test_unmatched_candidate_is_UNKNOWN_not_verified_negative():
    truth=[FrameTruth(1,True,(0,0,10,20),frozenset({'target'}))]
    trace=[{'frame':1,'rank1_candidate_uid':'unmatched','selected_candidate_uid':'unmatched','rank1_identity_joint_probability':.9,'runtime_future_gt_used':False}]
    result=strict_identity_claim_metrics(trace,truth)
    assert result['calibration_verified_frames']==0
    assert result['calibration_UNKNOWN_excluded_frames']==1
    assert result['identity_claim_PR_AUC'] is None and result['identity_claim_ECE'] is None
    assert result['strict_identity_recall_all_visible']==0.
    assert result['strict_identity_precision_selected']==0.


def test_incomplete_annotations_or_inconsistent_positive_are_rejected():
    trace=[{'frame':1,'rank1_candidate_uid':'other','selected_candidate_uid':'other','rank1_identity_joint_probability':.9,'runtime_future_gt_used':False}]
    incomplete=[FrameTruth(1,True,(0,0,10,20),frozenset({'target'}),annotation_complete=False)]
    with pytest.raises(ValueError,match='incomplete'):strict_identity_claim_metrics(trace,incomplete)
    complete=[FrameTruth(1,True,(0,0,10,20),frozenset({'target'}))]
    with pytest.raises(ValueError,match='positive'):strict_identity_claim_metrics(trace,complete,verified_rank1_labels=[True])
