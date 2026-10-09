"""Unknown-FPS domain diagnostic: time inputs disabled, never invent seconds.

Not the ordinary trained deployment condition or a time-valid success test.
Current/past real crops and frame-count confirmation remain causal.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from sam3_intermot.one_click.acib_runtime import unit,overlap


@dataclass(frozen=True)
class FrameEvidence:
    embedding: np.ndarray
    original_frame_1based: int
    candidate_uid: str
    timestamp_seconds: None = None


class FrameOnlyRecognizer:
    def __init__(self,model,anchor,token,*,width,height,initial_frame,initial_box,policy='FULL',capacity=8):
        if policy not in ('FULL','P0','P1','MEAN') or capacity not in (1,4,8) or width<=0 or height<=0:raise ValueError('registered frame-only condition')
        self.model=model.eval();self.anchor=unit(anchor);self.anchor.setflags(write=False)
        self.anchor_sha=hashlib.sha256(self.anchor.tobytes()).hexdigest();self.token=str(token)
        self.width=width;self.height=height;self.last_frame=initial_frame;self.last_box=list(initial_box)
        self.policy=policy;self.capacity=capacity;self.bank=[];self.pending=None

    def step(self,original_frame_1based,rows):
        frame=int(original_frame_1based)
        if frame!=self.last_frame+1:raise ValueError('contiguous actual original frame axis')
        if len({str(r['candidate_uid']) for r in rows})!=len(rows):raise ValueError('duplicate current UID')
        n=max(1,len(rows));candidates=torch.zeros(1,n,len(self.anchor));valid=torch.zeros(1,n,dtype=torch.bool);quality=torch.zeros(1,n,5);vectors=[]
        for i,r in enumerate(rows):
            vector=unit(r['feature']);vectors.append(vector);candidates[0,i]=torch.from_numpy(vector);valid[0,i]=True
            x,y,right,bottom=r['box_xyxy'];w,h=right-x,bottom-y
            if not np.isfinite(r['box_xyxy']).all() or w<=0 or h<=0:raise ValueError('current geometry')
            quality[0,i]=torch.tensor([np.clip(r.get('conf',0.),0,1),np.clip(w*h/(self.width*self.height),0,1),np.clip(w/h,0,4)/4,overlap(r['box_xyxy'],self.last_box),0.])
        entries=self.bank
        if self.policy=='MEAN' and entries:
            entries=[FrameEvidence(unit(np.mean([e.embedding for e in entries],axis=0)),entries[-1].original_frame_1based,'CAUSAL_MEAN_PROTOTYPE')]
        k=max(1,len(entries));bank=torch.zeros(1,k,len(self.anchor));bank_valid=torch.zeros(1,k,dtype=torch.bool)
        for i,e in enumerate(entries):bank[0,i]=torch.from_numpy(e.embedding.copy());bank_valid[0,i]=True
        # Unknown recency, gaps and camera differences are disabled explicitly.
        # No placeholder FPS or seconds are created, stored, or passed in.
        device=next(self.model.parameters()).device
        with torch.inference_mode():
            output=self.model(torch.from_numpy(self.anchor.copy())[None].to(device),candidates.to(device),valid.to(device),quality.to(device),
                bank=bank.to(device),bank_valid=bank_valid.to(device),bank_metadata=torch.zeros(1,k,2,device=device),context=torch.zeros(1,3,device=device))
        joint=output['joint_probabilities'][0].cpu().numpy();index=int(joint.argmax());rank=int(output['candidate_logits'][0].argmax()) if rows else None
        selected=rows[index] if index<len(rows) else None;written=False;current_safe=future_safe=None
        if selected is not None:
            current_safe=float(output['write_logits'][0,index].sigmoid());future_safe=float(output['future_safe_logits'][0,index].sigmoid())
            available=float(output['candidate_valid_probability'][0]);consistent=float(vectors[index]@self.anchor)
            eligible=current_safe>=.95 and future_safe>=.95 and available>=.9 and consistent>=.6
            consecutive=bool(eligible and self.pending and self.pending['frame']==frame-1 and vectors[index]@self.pending['vector']>=.9 and overlap(selected['box_xyxy'],self.pending['box'])>=.3)
            confirmations=self.pending['count']+1 if consecutive else 1
            written=self.policy in ('P1','MEAN') or self.policy=='FULL' and eligible and confirmations>=2
            self.pending={'frame':frame,'vector':vectors[index].copy(),'box':list(selected['box_xyxy']),'count':confirmations} if eligible else None
            self.last_box=list(selected['box_xyxy'])
            if written:
                vector=vectors[index].copy();vector.setflags(write=False)
                self.bank=(self.bank+[FrameEvidence(vector,frame,str(selected['candidate_uid']))])[-self.capacity:]
        else:self.pending=None
        self.last_frame=frame
        if hashlib.sha256(self.anchor.tobytes()).hexdigest()!=self.anchor_sha:raise ValueError('anchor mutated')
        return {'original_frame_1based':frame,'selected_candidate_uid':str(selected['candidate_uid']) if selected else None,
            'rank1_candidate_uid':str(rows[rank]['candidate_uid']) if rank is not None else None,
            'predicted_box_xyxy':list(selected['box_xyxy']) if selected else None,
            'candidate_available_probability':float(output['candidate_valid_probability'][0]),
            'rank1_identity_joint_probability':float(joint[:len(rows)].max()) if rows else 0.,
            'memory_write':bool(written),'memory_write_candidate_uid':str(selected['candidate_uid']) if written else None,
            'machine_bank_size':len(self.bank),'bank_original_frames':[e.original_frame_1based for e in self.bank],
            'current_correctness_probability':current_safe,'future_safe_probability':future_safe,
            'FPS':None,'timestamp_seconds':None,'time_features_disabled':True,'trained_deployment_distribution_shift':True,
            'anonymous_target_token':self.token,'anchor_sha256':self.anchor_sha,
            'extra_clicks':0,'runtime_gt_used':False,'runtime_future_gt_used':False}
