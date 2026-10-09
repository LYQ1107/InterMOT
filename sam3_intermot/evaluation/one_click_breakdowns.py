"""Post-seal descriptive counts; never imported by runtime or training."""
from collections import Counter
import numpy as np
from sam3_intermot.evaluation.one_click_protocol import iou,ratio,runs
from sam3_intermot.evaluation.one_click_curves import open_set_metrics_fast


def density_bin(n):
    if not isinstance(n,(int,np.integer)) or n<0:raise ValueError('real nonnegative candidate count')
    return 'sparse' if n<=4 else 'medium' if n<=8 else 'crowded'


def gap_bin(seconds):
    if not np.isfinite(seconds) or seconds<0:raise ValueError('actual nonnegative seconds')
    return 'short_le5' if seconds<=5 else 'medium_gt5_le20' if seconds<=20 else 'long_gt20_within_recording'


def frame_facts(candidates,annotations,matched,identity,anchor):
    target=[a for a in annotations if a['identity']==identity]
    if len(target)>1:raise ValueError('duplicate target annotation')
    if set(matched)!={c['candidate_uid'] for c in candidates}:raise ValueError('candidate matching axis')
    positive=[c for c in candidates if matched[c['candidate_uid']]==identity]
    negative=[c for c in candidates if matched[c['candidate_uid']] is not None and matched[c['candidate_uid']]!=identity]
    if len(positive)>1:raise ValueError('one-to-one target matching required')
    def score(c):
        v=np.asarray(c['feature'],np.float32);norm=np.linalg.norm(v)
        if norm<=1e-8 or not np.isfinite(v).all():raise ValueError('real nonzero candidate vector')
        return float((v/norm)@anchor)
    margin=score(positive[0])-max(score(c) for c in negative) if positive and negative else None
    overlap=max((iou(target[0]['box'],a['box']) for a in annotations if a['identity']!=identity),default=0.) if target else None
    return {'visible':bool(target),'target_box':target[0]['box'] if target else None,'available':bool(positive),
        'matched':matched,'identity':identity,'density':density_bin(len(candidates)),'candidate_count':len(candidates),
        'raw_anchor_hard_negative_margin':margin,'GT_geometric_overlap_proxy':overlap}


class Counts:
    def __init__(self):
        self.count=Counter();self.prob=[];self.available=[];self.rank_correct=[];self.claim_prob=[];self.claim_label=[]

    def add(self,row,fact):
        c=self.count;axis=fact['matched'];identity=fact['identity'];selected=row['selected_candidate_uid'];rank=row.get('rank1_candidate_uid');write=row.get('memory_write_candidate_uid')
        if any(uid is not None and uid not in axis for uid in (selected,rank,write)):raise ValueError('UID outside actual candidate axis')
        if bool(row['memory_write'])!=(write is not None):raise ValueError('memory write UID/flag mismatch')
        accepted=selected is not None;strict=accepted and axis[selected]==identity
        verified_wrong=accepted and axis[selected] is not None and axis[selected]!=identity
        c['frames']+=1;c['visible_frames']+=fact['visible'];c['available_frames']+=fact['available'];c['accepted']+=accepted
        c['strict_correct']+=strict;c['box_correct']+=bool(accepted and fact['visible'] and iou(row['predicted_box_xyxy'],fact['target_box'])>=.5)
        c['verified_wrong_identity_claims']+=verified_wrong;c['UNKNOWN_unmatched_identity_claims']+=accepted and axis[selected] is None
        c['candidate_missing_visible_frames']+=fact['visible'] and not fact['available'];c['target_unavailable_frames']+=not fact['available']
        c['target_unavailable_accepts']+=accepted and not fact['available'];c['GT_gap_frames']+=not fact['visible'];c['GT_gap_accepts']+=accepted and not fact['visible']
        c['NONE_false_reject_given_available']+=not accepted and fact['available'];c['available_but_incorrect_or_NONE']+=fact['available'] and not strict
        c['writes']+=write is not None
        if write is not None:
            c['strict_target_writes']+=axis[write]==identity;c['verified_other_identity_writes']+=axis[write] is not None and axis[write]!=identity;c['UNKNOWN_unmatched_writes']+=axis[write] is None
        label=False if rank is None else None if axis[rank] is None else axis[rank]==identity
        self.prob.append(row['candidate_available_probability']);self.available.append(fact['available']);self.rank_correct.append(label is True)
        if label is None:c['UNKNOWN_rank1_calibration_frames']+=1
        else:self.claim_prob.append(row['rank1_identity_joint_probability']);self.claim_label.append(label)

    def result(self):
        c=self.count;wrong=c['writes']-c['strict_target_writes']
        return {'counts':dict(c),'target_recall_box_IoU0_5':ratio(c['box_correct'],c['visible_frames']),
            'strict_UID_target_recall':ratio(c['strict_correct'],c['visible_frames']),
            'candidate_coverage_given_visible':ratio(c['available_frames'],c['visible_frames']),
            'strict_recognition_recall_given_available':ratio(c['strict_correct'],c['available_frames']),
            'verified_wrong_identity_claim_rate_all_future_frames':ratio(c['verified_wrong_identity_claims'],c['frames']),
            'target_unavailable_FPR_at_frozen_deployment':ratio(c['target_unavailable_accepts'],c['target_unavailable_frames']),
            'visible_GT_gap_FPR_not_physical_absence':ratio(c['GT_gap_accepts'],c['GT_gap_frames']),
            'NONE_false_reject_given_target_candidate':ratio(c['NONE_false_reject_given_available'],c['available_frames']),
            'conservative_non_target_or_unverified_write_rate':ratio(wrong,c['writes']),
            'strict_correct_observation_retention':ratio(c['strict_target_writes'],c['available_frames']),
            'zero_write_is_not_safe_memory_PASS':True,
            'candidate_availability_calibration':open_set_metrics_fast(self.prob,self.available,self.rank_correct),
            'verified_identity_claim_calibration':open_set_metrics_fast(self.claim_prob,self.claim_label,self.claim_label)}


def recovery_events(trace,facts,*,fps):
    if not np.isfinite(fps) or fps<=0:raise ValueError('actual FPS required')
    if len(trace)!=len(facts) or not trace:raise ValueError('complete nonempty recovery axis')
    visible=[f['visible'] for f in facts];strict=[r['selected_candidate_uid'] is not None and f['matched'].get(r['selected_candidate_uid'])==f['identity'] for r,f in zip(trace,facts)]
    wrong=[r['selected_candidate_uid'] is not None and f['matched'].get(r['selected_candidate_uid']) is not None and f['matched'][r['selected_candidate_uid']]!=f['identity'] for r,f in zip(trace,facts)]
    returns=[]
    for start,end in runs(np.logical_not(visible)):
        # Match the existing primary eligibility; do not replace its metrics.
        if start==0 or end==len(trace):continue
        stop=next((i for i in range(end,len(trace)) if not visible[i]),len(trace));success=next((i for i in range(end,stop) if strict[i]),None)
        accepted=next((i for i in range(end,stop) if trace[i]['selected_candidate_uid'] is not None),None)
        returns.append({'return_frame':trace[end]['frame'],'gap_frames':end-start,'gap_seconds':(end-start)/fps,'gap_bin':gap_bin((end-start)/fps),
            'visible_return_window_frames':stop-end,'strict_identity_recovered':success is not None,'strict_delay_frames':success-end if success is not None else None,
            'strict_delay_seconds':(success-end)/fps if success is not None else None,'first_accept_strict_correct':bool(strict[accepted]) if accepted is not None else None,
            'first_accept_verified_wrong_identity':bool(wrong[accepted]) if accepted is not None else None,
            'verified_wrong_claims_in_window':sum(wrong[end:stop]),'no_accept_in_return_window':accepted is None})
    takeovers=[]
    for start,end in runs(wrong):
        recovery=next((i for i in range(end,len(trace)) if strict[i]),None)
        takeovers.append({'start_frame':trace[start]['frame'],'end_frame_inclusive':trace[end-1]['frame'],'duration_seconds':(end-start)/fps,
            'correct_identity_recovery_delay_seconds_after_takeover':(recovery-end)/fps if recovery is not None else None,
            'no_correct_recovery_before_recording_end':recovery is None})
    return {'strict_UID_reappearance_windows':returns,'strict_verified_other_identity_takeover_runs':takeovers,
        'GT_gap_not_physical_absence_or_occlusion':True,'failed_recoveries_retained':True}
