"""Identity bank updates after exact current global ownership, not proposals."""
import numpy as np
from .safe_mot_bridge import SafeMOTIdentityBridge
from .intervention_features import validate_runtime_rows,opportunity_features
from .joint_intervention_primitives import prepare_proposal,check_action,commit_action
from .intervention_gate import decide_authority
from sam3_intermot.association.opportunity_solver import AssociationAction


class CommittedMemoryMOTBridge(SafeMOTIdentityBridge):
    def step(self,frame,rows):
        validate_runtime_rows(rows)
        if frame!=self.tracker.frame+1:raise ValueError('one original frame one complete global decision')
        if self.identity is None or frame<=self.event_frame:return super().step(frame,rows)
        prepared=prepare_proposal(self,frame,rows);proposal_check=check_action(self,frame,rows,prepared['preview'],prepared['action'])
        proposal_features=opportunity_features(self,frame,rows,prepared,proposal_check,previous_intervention=self.last_intervention)
        approved,reasons,prediction=decide_authority(self.policy,proposal_features,feasible=proposal_check['feasible'],
            proposed_change=bool(proposal_check.get('assignment_changed')),predictor=self.predictor)
        action=prepared['action'] if approved else AssociationAction('KEEP',self.tracker.target_public)
        check=check_action(self,frame,rows,prepared['preview'],action)
        committed_uid=prepared['preview']['target_uid'] if action.family=='KEEP' else action.candidate_uid
        specific={**prepared,'proposal':{**prepared['proposal'],'selected_candidate_uid':committed_uid}}
        features=opportunity_features(self,frame,rows,specific,check,previous_intervention=self.last_intervention)
        actual=commit_action(self,frame,rows,prepared,action)
        assert actual['target_uid']==committed_uid
        writer=self.identity.observe_committed(frame,rows,actual['target_uid'],features)
        if writer['accepted'] and writer['candidate_uid']!=actual['target_uid']:raise RuntimeError('uncommitted crop write')
        if approved:self.last_intervention=frame
        actual.update(joint_identity_memory_write=writer['accepted'],joint_memory_write_candidate_uid=writer['candidate_uid'],
            committed_memory_diagnostic=writer,committed_observation_features=features,
            memory_state_after_actual_commit=self.identity.snapshot())
        actual['identity_decision'].update(memory_write=writer['accepted'],memory_write_candidate_uid=writer['candidate_uid'],
            machine_bank_size=len(self.identity.bank),bank_source_recordings=[e.recording_id for e in self.identity.bank])
        actual['authority']={'family':self.policy.family,'approved':approved,'effective_assignment_change':bool(approved and check['assignment_changed']),
            'features':proposal_features,'own_KEEP_uid':prepared['preview']['target_uid'],'reasons':reasons,'prediction':prediction}
        return actual
