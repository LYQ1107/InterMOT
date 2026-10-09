"""Frozen identity proposals into the existing joint MOT solver.

Own current/past state only. This is an explicitly untrained MOT-state
transfer diagnostic; it does not redesign the matcher or use oracle labels.
"""
from copy import deepcopy
import numpy as np
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_solver import AssociationAction,action_for_candidate,solve_counterfactual_global_assignment
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.identity_authority import AuthorityConfig


def copy_identity(identity):
    if identity is None:return None
    model=getattr(identity,'model',None)
    memo={} if model is None else {id(model):model}
    result=deepcopy(identity,memo)
    if hasattr(result,'anchor_sha'):
        result.anchor.setflags(write=False)
        for entry in result.bank:entry.embedding.setflags(write=False)
    return result


def committed_feedback(previous,provisional,proposal,rows,frame,actual_uid):
    """Veto unmatched proposals without writing them; feedback actual motion."""
    same=proposal['selected_candidate_uid']==actual_uid
    actual=provisional if same else previous
    if same:return actual,{**proposal,'proposed_candidate_uid':proposal['selected_candidate_uid'],'proposal_memory_write':proposal['memory_write'],
        'joint_memory_write_vetoed':False,'joint_assignment_feedback_applied':True}
    row=next((r for r in rows if str(r['candidate_uid'])==actual_uid),None)
    if actual_uid is not None and row is None:raise ValueError('committed UID outside current candidate axis')
    actual.last_frame=int(frame)
    if hasattr(actual,'anchor_sha'):
        actual.pending=None
        if row is not None:actual.last_box=list(row['box_xyxy']);actual.last_accept=int(frame)
        bank_size=len(actual.bank)
    else:
        actual._pending_embedding=None;actual._pending_count=0
        actual.last_box=tuple(row['box_xyxy']) if row is not None else None
        actual.last_native_tid=str(row['native_tid']) if row is not None else None
        bank_size=len(actual.bank)
    decision={**proposal,'proposed_candidate_uid':proposal['selected_candidate_uid'],'proposal_memory_write':proposal['memory_write'],
        'selected_candidate_uid':actual_uid,'predicted_box_xyxy':list(row['box_xyxy']) if row is not None else None,
        'memory_write':False,'memory_write_candidate_uid':None,'machine_bank_size':bank_size,
        'joint_memory_write_vetoed':bool(proposal['memory_write']),'joint_assignment_feedback_applied':True}
    return actual,decision


class MOTIdentityBridge:
    def __init__(self,event,identity=None,*,no_human=False,frames=None):
        if no_human and (identity is not None or not isinstance(frames,int) or frames<=0):raise ValueError('no-click control cannot use an identity actor')
        self.event_frame=int(event['event_frame']);self.identity=identity;self.no_human=no_human
        if no_human:event={**event,'event_frame':frames}
        self.tracker=OpportunityTracker(config=AuthorityConfig(mode='off',source='raw',memory='P0',lifecycle='dynamic'),
            event=event,bank=None,intervention_policy=InterventionPolicy(family='C0',source='raw'),memory_policy=MemoryPolicy('P0'))

    def step(self,frame,rows):
        if frame!=self.tracker.frame+1:raise ValueError('one original frame one joint decision')
        if self.identity is None or frame<=self.event_frame:
            d=self.tracker.step(rows,frame);proposal=None;feasible=None;reason=None
        else:
            previous=copy_identity(self.identity);provisional=copy_identity(self.identity)
            if hasattr(provisional,'anchor_sha'):proposal=provisional.step(frame,rows)
            else:
                from sam3_intermot.one_click.runtime import Candidate
                candidates=[Candidate(str(r['candidate_uid']),r['feature'],tuple(r['box_xyxy']),float(np.clip(r.get('conf',0.),0,1)),str(r['native_tid'])) for r in rows]
                proposal=provisional.step(frame,candidates,fps=self.fps)
            # Current-frame KEEP solve uses this treatment's own previously
            # committed joint state, never another run's future state.
            preview=self.tracker.clone().step(rows,frame)
            public=self.tracker.target_public;uid=proposal['selected_candidate_uid']
            if public is None:raise RuntimeError('sole click never bound a public identity')
            if uid==preview['target_uid']:action=AssociationAction('KEEP',public)
            elif uid is None:action=AssociationAction('REJECT_TARGET',public)
            else:action=action_for_candidate(public,uid,preview['solver'])
            publics=preview['states_before_commit_axis'];states=[self.tracker.states[p].pid for p in publics]
            check=solve_counterfactual_global_assignment(rows,preview['base_matrix'],states,publics,action,frame=frame)
            feasible=check['feasible'];reason=check.get('reason')
            d=self.tracker.step(rows,frame,forced_action=action if feasible else None)
            if feasible and d['target_uid']!=uid:raise RuntimeError('forced proposal did not become actual joint target ownership')
            self.identity,proposal=committed_feedback(previous,provisional,proposal,rows,frame,d['target_uid'])
        outputs=d['outputs'];uids=[o['candidate_uid'] for o in outputs];publics=[o['public_id'] for o in outputs]
        if len(uids)!=len(set(uids)) or len(publics)!=len(set(publics)) or set(uids)!={str(r['candidate_uid']) for r in rows}:
            raise RuntimeError('complete one-to-one multi-object candidate ownership required')
        if proposal is not None and proposal['memory_write'] and proposal['memory_write_candidate_uid']!=d['target_uid']:
            raise RuntimeError('uncommitted identity proposal cannot write persistent memory')
        return {k:v for k,v in d.items() if k not in ('base_matrix','solver','states_before_commit_axis','proposals')}|{
            'identity_decision':proposal,'proposal_feasible':feasible,'proposal_failure_reason':reason,
            'joint_identity_memory_write':bool(proposal and proposal['memory_write']),
            'joint_memory_write_candidate_uid':proposal['memory_write_candidate_uid'] if proposal else None,
            'extra_clicks':0,'runtime_gt_read':False,'runtime_future_gt_used':False,
            'full_multi_object_trajectory':True,'frozen_weight_MOT_state_distribution_shift':self.identity is not None}

    def configure_fps(self,fps):
        if not np.isfinite(fps) or fps<=0:raise ValueError('actual FPS required')
        self.fps=float(fps)
