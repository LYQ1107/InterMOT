"""Causal committed-crop writer, distinct from identity proposal and authority."""
from dataclasses import dataclass,asdict
import hashlib
import numpy as np
from .acib_runtime import Evidence,unit,overlap
from .acib_trusted_runtime import TrustedACIBRecognizer
from .intervention_features import feature_vector,validate_runtime_rows


@dataclass(frozen=True)
class MemoryCommitPolicy:
    family:str='frozen'
    capacity:int=8
    aggregation:str='attention'
    delay_frames:int=2
    probability_min:float=.7
    anchor_min:float=.6
    quality_min:float=.5
    motion_min:float=.2
    diversity_gap:int=5
    novelty_cosine_max:float=.98
    risk_max:float=.02
    rollback_confirmations:int=2

    def __post_init__(self):
        if self.family not in ('frozen','unsafe','consensus','delayed','diverse','risk','rollback'):raise ValueError('unregistered writer')
        if self.capacity not in (1,4,8) or self.aggregation not in ('attention','mean','confidence'):raise ValueError('invalid bank configuration')
        if self.delay_frames<1 or self.diversity_gap<1 or self.rollback_confirmations<2:raise ValueError('invalid causal evidence window')


class CommittedIdentityMemory(TrustedACIBRecognizer):
    def __init__(self,model,anchor,token,*,write_policy=None,write_predictor=None):
        self.write_policy=write_policy or MemoryCommitPolicy();self.write_predictor=write_predictor
        if self.write_policy.family=='risk' and write_predictor is None:raise ValueError('actual fitted joint-state writer required before runtime')
        super().__init__(model,anchor,token,capacity=self.write_policy.capacity,policy='P0')
        self.write_pending=None;self.last_commit_observation=None;self.write_snapshots=[];self.rollback_count=0;self.disagreement_count=0

    def step(self,frame,rows):
        validate_runtime_rows(rows);trusted=self.bank;aggregation=self.write_policy.aggregation
        if trusted and aggregation!='attention':
            weights=np.ones(len(trusted)) if aggregation=='mean' else np.maximum([e.confidence for e in trusted],1e-6)
            weights=weights/weights.sum();vector=unit(sum(w*e.embedding for w,e in zip(weights,trusted,strict=True)))
            vector.setflags(write=False);latest=trusted[-1]
            self.bank=[Evidence(vector,latest.recording_id,latest.frame,latest.camera,latest.timestamp_seconds,
                float(sum(w*e.confidence for w,e in zip(weights,trusted,strict=True))),
                'DERIVED_'+aggregation.upper()+'_NOT_NEW_CROP_WRITE',float(sum(w*e.quality for w,e in zip(weights,trusted,strict=True))),float(vector@self.anchor))]
        view_size=len(self.bank)
        try:result=super().step(frame,rows)
        finally:self.bank=trusted
        result.update(machine_bank_size=len(self.bank),memory_write=False,memory_write_candidate_uid=None,
            write_policy=asdict(self.write_policy),pending_used_for_identity_score=False,
            only_actual_global_commit_may_write=True,model_bank_view_entries=view_size,
            bank_source_recordings=[e.recording_id for e in self.bank])
        return result

    def observe_committed(self,frame,rows,committed_uid,features):
        validate_runtime_rows(rows)
        if frame!=self.last_frame or (self.last_commit_observation is not None and frame<=self.last_commit_observation):raise ValueError('observe each actual current commit once')
        candidate=next((r for r in rows if str(r['candidate_uid'])==committed_uid),None)
        if committed_uid is not None and candidate is None:raise ValueError('write UID outside actual current axis')
        self.last_commit_observation=frame;policy=self.write_policy;reasons=[];prediction=None;rollback=False
        if candidate is None:self.write_pending=None;return {'accepted':False,'candidate_uid':None,'reasons':['COMMITTED_NONE'],'rollback':False}
        vector=unit(candidate['feature']);anchor_cos=float(vector@self.anchor);pending=self.write_pending
        consecutive=bool(pending and pending['frame']==frame-1 and float(vector@pending['feature'])>=.9 and overlap(candidate['box_xyxy'],pending['box'])>=.3)
        confirmations=pending['count']+1 if consecutive else 1
        self.write_pending={'frame':frame,'count':confirmations,'feature':vector.copy(),'box':list(candidate['box_xyxy']),
            'native':(candidate.get('native_scope'),candidate['native_tid'])}
        if policy.family=='frozen':reasons.append('FROZEN_ANCHOR_NO_WRITE')
        if policy.family not in ('frozen','unsafe'):
            if features['proposed_probability']<policy.probability_min:reasons.append('LOW_COMMITTED_IDENTITY_PROBABILITY')
            if anchor_cos<policy.anchor_min:reasons.append('ANCHOR_DISAGREEMENT')
            if features['quality']<policy.quality_min:reasons.append('LOW_QUALITY')
            if features['motion_iou']<policy.motion_min:reasons.append('LOW_MOTION_SUPPORT')
            if not features['native_same']:reasons.append('NATIVE_CONTINUITY_DISAGREEMENT')
        if policy.family in ('delayed','diverse','risk','rollback') and confirmations<policy.delay_frames:reasons.append('PENDING_CAUSAL_CONFIRMATION')
        if policy.family=='diverse' and self.bank:
            latest=self.bank[-1];same_native=bool(pending and pending['native']==self.write_pending['native'])
            if frame-latest.frame<policy.diversity_gap or (same_native and float(vector@latest.embedding)>policy.novelty_cosine_max):
                reasons.append('REDUNDANT_TRACKLET_EVIDENCE')
        if policy.family=='risk':
            if self.write_predictor is None:raise ValueError('actual joint-state fitted writer required')
            prediction=self.write_predictor.predict(feature_vector(features))
            if prediction['harmful']>policy.risk_max or prediction['beneficial']<1-policy.risk_max:reasons.append('PREDICTED_WRITE_RISK')
        if policy.family=='rollback' and self.bank:
            bank_agreement=max(float(vector@e.embedding) for e in self.bank)
            self.disagreement_count=self.disagreement_count+1 if anchor_cos<.2 and bank_agreement>.8 else 0
            if self.disagreement_count>=policy.rollback_confirmations and self.write_snapshots:
                self.bank=list(self.write_snapshots.pop());self.rollback_count+=1;self.disagreement_count=0
                self.write_pending=None;rollback=True;reasons.append('CAUSAL_ANCHOR_BANK_CONTRADICTION_ROLLBACK')
        accepted=not reasons
        if accepted:
            self.write_snapshots.append(tuple(self.bank));self.write_snapshots=self.write_snapshots[-4:]
            vector.setflags(write=False)
            self.bank.append(Evidence(vector,self.recording,int(frame),self.camera,frame/self.fps,
                float(features['proposed_probability']),'ACTUAL_GLOBAL_COMMIT_'+policy.family.upper(),
                float(features['quality']),anchor_cos,str(committed_uid)))
            self.bank=self.bank[-policy.capacity:]
        if hashlib.sha256(self.anchor.tobytes()).hexdigest()!=self.anchor_sha:raise RuntimeError('immutable human anchor changed')
        return {'accepted':accepted,'candidate_uid':str(committed_uid) if accepted else None,
            'reasons':reasons or ['CURRENT_COMMITTED_CROP_CONFIRMED'],'confirmations':confirmations,
            'prediction':prediction,'rollback':rollback,'trusted_size':len(self.bank),
            'pending_used_for_identity_score':False,'proposed_UID_not_write_authority':True,
            'causal_evidence_agreement_not_statistical_independence_proof':True}

    def snapshot(self):
        return super().snapshot()|{'write_policy':asdict(self.write_policy),'pending_only_not_trusted':None if self.write_pending is None else {
            'frame':self.write_pending['frame'],'count':self.write_pending['count'],
            'embedding_SHA':hashlib.sha256(self.write_pending['feature'].tobytes()).hexdigest()},
            'rollback_count':self.rollback_count,'bounded_prewrite_snapshot_count':len(self.write_snapshots)}
