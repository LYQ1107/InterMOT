"""Frozen historical/negative controls, not a new scorer or trained authority."""
from copy import copy
import hashlib
import numpy as np
from .acib_runtime import unit,overlap
from .intervention_features import validate_runtime_rows,opportunity_features
from .intervention_gate import decide_authority
from .safe_mot_bridge import SafeMOTIdentityBridge
from .joint_intervention_primitives import prepare_proposal,check_action,commit_action
from sam3_intermot.association.opportunity_solver import AssociationAction


class FrozenAdapterActor:
    """Actual adapter ensemble current-axis scores and historic INNER cutoff."""
    def __init__(self,model,anchor,event,*,score_min,margin_min):
        self.model=model;self.anchor=unit(anchor);self.anchor.setflags(write=False)
        self.anchor_sha=hashlib.sha256(self.anchor.tobytes()).hexdigest()
        self.bank=[];self.pending=None;self.last_frame=event['frame'];self.last_accept=event['frame']
        self.last_box=list(event['box_xyxy']);self.recording=event['sequence']
        self.score_min=float(score_min);self.margin_min=float(margin_min)

    def step(self,frame,rows):
        validate_runtime_rows(rows)
        if frame<=self.last_frame:raise ValueError('causal one-frame adapter decision')
        features=np.stack([unit(r['feature']) for r in rows]) if rows else np.empty((0,len(self.anchor)),np.float32)
        scores=self.model.scores(self.anchor,features)
        order=sorted(range(len(rows)),key=lambda i:(-float(scores[i]),str(rows[i]['candidate_uid'])))
        best=order[0] if order else None
        score=None if best is None else float(scores[best])
        margin=None if best is None else score-(float(scores[order[1]]) if len(order)>1 else -1.)
        selected=best if best is not None and score>=self.score_min and margin>=self.margin_min else None
        row=rows[selected] if selected is not None else None
        if row is not None:self.last_box=list(row['box_xyxy']);self.last_accept=frame
        self.last_frame=frame
        probability=0. if score is None else float(np.clip((score+1)/2,0,1))
        return {'frame':frame,'recording_id':self.recording,'selected_candidate_uid':str(row['candidate_uid']) if row else None,
            'rank1_candidate_uid':str(rows[best]['candidate_uid']) if best is not None else None,
            'predicted_box_xyxy':list(row['box_xyxy']) if row else None,'identity_score':score,'identity_margin':margin if margin is not None else 0.,
            'candidate_available_probability':probability,'presence_probability_semantics':'UNCALIBRATED_ADAPTER_COSINE_NOT_AVAILABILITY',
            'memory_write':False,'memory_write_candidate_uid':None,'machine_bank_size':0,'anchor_sha256':self.anchor_sha,
            'runtime_gt_used':False,'runtime_future_gt_used':False,'extra_clicks':0}


def shuffled_evidence(rows,clicked_uid):
    """Predeclared lexical current-click-frame UID donor; no GT/best selection."""
    axis=sorted(rows,key=lambda r:str(r['candidate_uid']))
    index=next(i for i,r in enumerate(axis) if str(r['candidate_uid'])==clicked_uid)
    if len(axis)<2:raise ValueError('negative control needs another real current candidate')
    donor=axis[(index+1)%len(axis)]
    return unit(donor['feature']),str(donor['candidate_uid'])


def separated_evidence_features(bridge,frame,rows,prepared,check,**kwargs):
    # The feature-only view never reaches scoring/solve/commit. The original
    # human initialization and all native scorer machine state remain intact.
    proxy=copy(bridge);proxy.tracker=copy(bridge.tracker)
    proxy.tracker.event={**bridge.tracker.event,'human_anchor':bridge.identity.anchor}
    return opportunity_features(proxy,frame,rows,prepared,check,**kwargs)


class ShuffledEvidenceMOTBridge(SafeMOTIdentityBridge):
    """Disrupt actor and human-evidence gate cues, not C0 machine evidence."""
    def step(self,frame,rows):
        validate_runtime_rows(rows)
        if frame!=self.tracker.frame+1:raise ValueError('one original frame one global decision')
        if self.identity is None or frame<=self.event_frame or self.policy.family=='off':return super().step(frame,rows)
        prepared=prepare_proposal(self,frame,rows);action=prepared['action']
        check=check_action(self,frame,rows,prepared['preview'],action)
        uid=prepared['proposal']['selected_candidate_uid'];candidate=next((r for r in rows if str(r['candidate_uid'])==uid),None)
        pending=self.authority_pending
        agreement=0. if candidate is None or pending is None else float(np.dot(candidate['feature'],pending['feature']))
        confirmed=bool(candidate is not None and pending is not None and pending['frame']==frame-1 and agreement>=.9 and overlap(candidate['box_xyxy'],pending['box'])>=.3)
        count=pending['count']+1 if confirmed else 1 if candidate is not None else 0
        self.authority_pending=None if candidate is None else {'frame':frame,'count':count,'feature':np.array(candidate['feature'],copy=True),'box':list(candidate['box_xyxy'])}
        features=separated_evidence_features(self,frame,rows,prepared,check,confirmations=count,
            previous_intervention=self.last_intervention,previous_agreement=agreement)
        approved,reasons,prediction=decide_authority(self.policy,features,feasible=check['feasible'],
            proposed_change=bool(check.get('assignment_changed')),predictor=self.predictor)
        actual=action if approved else AssociationAction('KEEP',self.tracker.target_public)
        result=commit_action(self,frame,rows,prepared,actual)
        if approved:self.last_intervention=frame
        result['authority']={'family':self.policy.family,'approved':approved,'proposed_action':action.to_dict(),
            'effective_assignment_change':bool(approved and check['assignment_changed']),'features':features,'reasons':reasons,
            'prediction':prediction,'own_KEEP_uid':prepared['preview']['target_uid'],'future_GT_used':False,
            'negative_control_all_human_identity_cues_shuffled':True,'C0_human_initialization_not_modified':True}
        return result
