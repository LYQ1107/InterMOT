from sam3_intermot.evaluation.one_click_protocol import FrameTruth
from sam3_intermot.evaluation.one_click_identity import strict_identity_claim_metrics


def test_high_availability_wrong_rank_is_not_identity_PR_positive():
    truth=[FrameTruth(1,True,(0,0,10,20),frozenset({'target'})),FrameTruth(2,True,(0,0,10,20),frozenset({'target'}))]
    trace=[{'frame':1,'rank1_candidate_uid':'other','selected_candidate_uid':'other','rank1_identity_joint_probability':.9,'runtime_future_gt_used':False},
           {'frame':2,'rank1_candidate_uid':'target','selected_candidate_uid':'target','rank1_identity_joint_probability':.8,'runtime_future_gt_used':False}]
    result=strict_identity_claim_metrics(trace,truth)
    assert result['identity_claim_PR_AUC']==.5
    assert result['strict_identity_recall_given_candidate']==.5
    assert result['rank1_correct_claim_recall_at_wrong_rank_FPR2pct']==0.


def test_no_candidate_is_not_positive_and_cannot_invent_truth():
    truth=[FrameTruth(1,False,None,frozenset())]
    trace=[{'frame':1,'rank1_candidate_uid':None,'selected_candidate_uid':None,'rank1_identity_joint_probability':0.,'runtime_future_gt_used':False}]
    result=strict_identity_claim_metrics(trace,truth)
    assert result['identity_claim_PR_AUC'] is None and result['identity_claim_ECE']==0
    assert result['strict_identity_precision_selected'] is None
