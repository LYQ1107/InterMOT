"""Explicit secondary identity-claim calibration, not availability PR-AUC."""
import numpy as np
from sam3_intermot.evaluation.one_click_protocol import open_set_metrics,ratio


def strict_identity_claim_metrics(runtime,truth,*,verified_rank1_labels=None):
    if len(runtime)!=len(truth) or not runtime:raise ValueError('full aligned frame axes')
    if [r['frame'] for r in runtime]!=[t.frame for t in truth]:raise ValueError('frame axis')
    if any(r['runtime_future_gt_used'] is not False for r in runtime):raise ValueError('GT boundary')
    if any(not t.annotation_complete for t in truth):raise ValueError('incomplete annotations cannot invent identity truth')
    rank_correct=[r.get('rank1_candidate_uid') is not None and r.get('rank1_candidate_uid') in t.valid_target_candidate_uids for r,t in zip(runtime,truth)]
    if verified_rank1_labels is None:
        verified_rank1_labels=[True if correct else False if r.get('rank1_candidate_uid') is None else None for r,correct in zip(runtime,rank_correct)]
    if len(verified_rank1_labels)!=len(runtime):raise ValueError('verified identity label axis')
    for r,correct,label in zip(runtime,rank_correct,verified_rank1_labels):
        if label is not None and type(label) is not bool:raise ValueError('identity labels must be True, False, or UNKNOWN None')
        if (label is True)!=bool(correct):raise ValueError('verified positive must strictly match the target UID')
        if r.get('rank1_candidate_uid') is None and label is not False:raise ValueError('empty candidate set must be a structural no-claim negative')
    known=np.asarray([v is not None for v in verified_rank1_labels],dtype=bool)
    y=np.asarray([v is True for v in verified_rank1_labels],dtype=bool)
    probability=np.asarray([r['rank1_identity_joint_probability'] for r in runtime],dtype=float)
    if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise ValueError('invalid identity probability, including UNKNOWN frames')
    accepted=np.asarray([r['selected_candidate_uid'] is not None for r in runtime])
    strict_correct=np.asarray([r['selected_candidate_uid'] is not None and r['selected_candidate_uid'] in t.valid_target_candidate_uids for r,t in zip(runtime,truth)])
    available=np.asarray([bool(t.valid_target_candidate_uids) for t in truth]);visible=np.asarray([t.target_visible for t in truth])
    curve=open_set_metrics(probability[known],y[known],y[known])
    return {'actual_strict_correct_selected_frames':int(strict_correct.sum()),
        'strict_identity_recall_all_visible':ratio(int(strict_correct.sum()),int(visible.sum())),
        'strict_identity_recall_given_candidate':ratio(int(strict_correct.sum()),int(available.sum())),
        'strict_identity_precision_selected':ratio(int(strict_correct.sum()),int(accepted.sum())),
        'rank1_correct_frames':int(y.sum()),'rank1_verified_negative_frames':int((known&~y).sum()),
        'rank1_unknown_identity_frames':int((~known).sum()),
        'calibration_verified_frames':int(known.sum()),'calibration_UNKNOWN_excluded_frames':int((~known).sum()),
        'rank1_structural_empty_candidate_frames':sum(r.get('rank1_candidate_uid') is None for r in runtime),
        'unverified_selected_candidates_not_counted_as_verified_correct':True,
        'identity_claim_PR_AUC':curve['PR_AUC_average_precision'],'identity_claim_ECE':curve['ECE'],
        'identity_claim_calibration_bins':curve['calibration_bins'],
        'rank1_correct_claim_recall_at_wrong_rank_FPR2pct':curve['recall_at_fpr_2pct'],
        'score':'Learned joint probability that rank1 current candidate is target; zero when candidate set empty',
        'secondary_FPR_denominator':'Verified other-identity rank1 candidates plus structural empty-candidate no-claim frames; UNKNOWN unmatched candidates excluded',
        'does_not_replace_primary_target_unavailable_FPR2pct':True,
        'original_open_set_PR_AUC_ECE_are_candidate_availability_not_identity_claim':True,
        'not_physical_absence_truth':True}
