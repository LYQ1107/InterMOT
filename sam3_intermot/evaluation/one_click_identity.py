"""Explicit secondary identity-claim calibration, not availability PR-AUC."""
import numpy as np
from sam3_intermot.evaluation.one_click_protocol import open_set_metrics,ratio


def strict_identity_claim_metrics(runtime,truth):
    if len(runtime)!=len(truth) or not runtime:raise ValueError('full aligned frame axes')
    if [r['frame'] for r in runtime]!=[t.frame for t in truth]:raise ValueError('frame axis')
    if any(r['runtime_future_gt_used'] is not False for r in runtime):raise ValueError('GT boundary')
    y=np.asarray([r.get('rank1_candidate_uid') is not None and r.get('rank1_candidate_uid') in t.valid_target_candidate_uids for r,t in zip(runtime,truth)])
    probability=[r['rank1_identity_joint_probability'] for r in runtime]
    accepted=np.asarray([r['selected_candidate_uid'] is not None for r in runtime])
    strict_correct=np.asarray([r['selected_candidate_uid'] is not None and r['selected_candidate_uid'] in t.valid_target_candidate_uids for r,t in zip(runtime,truth)])
    available=np.asarray([bool(t.valid_target_candidate_uids) for t in truth]);visible=np.asarray([t.target_visible for t in truth])
    curve=open_set_metrics(probability,y,y)
    return {'actual_strict_correct_selected_frames':int(strict_correct.sum()),
        'strict_identity_recall_all_visible':ratio(int(strict_correct.sum()),int(visible.sum())),
        'strict_identity_recall_given_candidate':ratio(int(strict_correct.sum()),int(available.sum())),
        'strict_identity_precision_selected':ratio(int(strict_correct.sum()),int(accepted.sum())),
        'rank1_correct_frames':int(y.sum()),'rank1_wrong_or_unavailable_frames':int((~y).sum()),
        'identity_claim_PR_AUC':curve['PR_AUC_average_precision'],'identity_claim_ECE':curve['ECE'],
        'identity_claim_calibration_bins':curve['calibration_bins'],
        'rank1_correct_claim_recall_at_wrong_rank_FPR2pct':curve['recall_at_fpr_2pct'],
        'score':'Learned joint probability that rank1 current candidate is target; zero when candidate set empty',
        'secondary_FPR_denominator':'All wrong-rank or unavailable frames, including wrong-rank frames with a valid target candidate',
        'does_not_replace_primary_target_unavailable_FPR2pct':True,
        'original_open_set_PR_AUC_ECE_are_candidate_availability_not_identity_claim':True,
        'not_physical_absence_truth':True}
