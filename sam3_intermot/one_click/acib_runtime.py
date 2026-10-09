"""Causal GT-free deployment state, separate from frozen T0 empty-bank probe."""
from dataclasses import dataclass
import hashlib
import math
import numpy as np
import torch


def unit(value):
    value=np.array(value,dtype=np.float32,copy=True)
    norm=np.linalg.norm(value)
    if value.ndim!=1 or not np.isfinite(value).all() or norm<1e-8:raise ValueError('valid visual identity vector required')
    value/=norm;return value


def overlap(a,b):
    if a is None or b is None:return 0.
    left=max(a[0],b[0]);top=max(a[1],b[1]);right=min(a[2],b[2]);bottom=min(a[3],b[3])
    area=max(0.,right-left)*max(0.,bottom-top)
    return area/max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


@dataclass(frozen=True)
class Evidence:
    embedding: np.ndarray
    recording_id: str
    frame: int
    camera: str | None
    timestamp_seconds: float
    confidence: float
    source: str
    quality: float
    anchor_consistency: float
    candidate_uid: str | None = None


class ACIBRecognizer:
    """No annotation/future access; exactly one initialization; bounded own bank."""
    def __init__(self,model,anchor,token,*,capacity=8,policy='P0',safe_delay=2):
        if capacity not in (1,4,8) or policy not in ('P0','P1','SAFE_FIXED','NO_DELAY','MEAN') or safe_delay<1:raise ValueError('registered state policy')
        self.model=model.eval();self.anchor=unit(anchor);self.anchor.setflags(write=False)
        self.anchor_sha=hashlib.sha256(self.anchor.tobytes()).hexdigest();self.token=str(token)
        self.capacity=capacity;self.policy=policy;self.safe_delay=safe_delay;self.bank=[];self.recording=None
        self.initial_recording=None;self.initial_frame=None;self.pending=None;self.last_box=None;self.last_accept=None;self.last_frame=None

    def start_recording(self,recording_id,*,fps,width,height,camera=None,initial_frame=None,initial_box=None):
        if not np.isfinite(fps) or fps<=0 or width<=0 or height<=0:raise ValueError('actual recording FPS/geometry required')
        if self.recording==str(recording_id):raise ValueError('not an independent recording transition')
        first=self.initial_recording is None
        if not first and initial_box is not None:raise ValueError('a second independent-recording click is forbidden')
        if initial_frame is None:initial_frame=0 if first else -1
        self.recording=str(recording_id);self.fps=float(fps);self.width=int(width);self.height=int(height);self.camera=camera
        if self.initial_recording is None:self.initial_recording=self.recording;self.initial_frame=int(initial_frame)
        self.pending=None;self.last_box=None;self.last_accept=None;self.last_frame=int(initial_frame)
        if initial_box is not None:
            box=[float(v) for v in initial_box]
            if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:raise ValueError('valid sole initial click geometry')
            self.last_box=box;self.last_accept=int(initial_frame)

    def step(self,frame,rows):
        if self.recording is None or frame<=self.last_frame:raise ValueError('initialized strictly causal frame order required')
        count=max(1,len(rows));vectors=torch.zeros(1,count,len(self.anchor));quality=torch.zeros(1,count,5);valid=torch.zeros(1,count,dtype=torch.bool)
        current=[]
        recency=0. if self.last_accept is None else math.exp(-(frame-self.last_accept)/self.fps/60)
        for i,r in enumerate(rows):
            feature=unit(r['feature']);current.append(feature);vectors[0,i]=torch.from_numpy(feature);valid[0,i]=True
            x,y,right,bottom=r['box_xyxy'];w,h=right-x,bottom-y
            if not np.isfinite([x,y,right,bottom]).all() or w<=0 or h<=0:raise ValueError('valid current geometry required')
            quality[0,i]=torch.tensor([np.clip(r.get('conf',0.),0,1),np.clip(w*h/(self.width*self.height),0,1),np.clip(w/h,0,4)/4,overlap(r['box_xyxy'],self.last_box),recency])
        entries=self.bank
        if self.policy=='MEAN' and entries:
            vector=unit(np.mean([e.embedding for e in entries],axis=0));latest=entries[-1]
            entries=[Evidence(vector,latest.recording_id,latest.frame,latest.camera,latest.timestamp_seconds,
                              float(np.mean([e.confidence for e in entries])),'CAUSAL_MEAN_PROTOTYPE_TIMESTAMP_LATEST_CONTRIBUTOR',
                              float(np.mean([e.quality for e in entries])),float(vector@self.anchor))]
        k=max(1,len(entries));bank=torch.zeros(1,k,len(self.anchor));bank_valid=torch.zeros(1,k,dtype=torch.bool);metadata=torch.zeros(1,k,2)
        for i,e in enumerate(entries):
            bank[0,i]=torch.from_numpy(e.embedding.copy());bank_valid[0,i]=True
            metadata[0,i,0]=math.exp(-(frame-e.frame)/self.fps/60) if e.recording_id==self.recording else 0.
            metadata[0,i,1]=float(self.camera is not None and e.camera is not None and self.camera!=e.camera)
        gap=(frame-self.initial_frame)/self.fps if self.initial_recording==self.recording else 0.
        missing=0. if self.last_accept is None else (frame-self.last_accept)/self.fps
        context=torch.tensor([[math.log1p(gap)/5,math.log1p(missing)/5,float(self.initial_recording!=self.recording)]])
        device=next(self.model.parameters()).device
        with torch.inference_mode():
            output=self.model(torch.from_numpy(self.anchor.copy())[None].to(device),vectors.to(device),valid.to(device),quality.to(device),
                              bank=bank.to(device),bank_valid=bank_valid.to(device),bank_metadata=metadata.to(device),context=context.to(device))
        joint=output['joint_probabilities'][0].cpu().numpy();index=int(joint.argmax());rank=int(output['candidate_logits'][0].argmax()) if rows else None
        selected=rows[index] if index<len(rows) else None;available=float(output['candidate_valid_probability'][0]);written=False
        risk=float(output['write_logits'][0,index].sigmoid()) if selected is not None else None
        anchor_cos=float(current[index]@self.anchor) if selected is not None else None
        consecutive=False;eligible=False
        if selected is not None:
            eligible=risk>=.95 and anchor_cos>=.6 and available>=.9
            consecutive=bool(eligible and self.pending and self.pending['recording']==self.recording and self.pending['frame']==frame-1
                             and current[index]@self.pending['feature']>=.9 and overlap(selected['box_xyxy'],self.pending['box'])>=.3)
            confirmations=self.pending['count']+1 if consecutive else 1
            if self.policy in ('P1','MEAN'):written=True
            elif self.policy in ('SAFE_FIXED','NO_DELAY'):
                written=eligible and (self.policy=='NO_DELAY' or confirmations>=self.safe_delay)
            self.pending={'recording':self.recording,'frame':frame,'count':confirmations,'feature':current[index].copy(),'box':list(selected['box_xyxy'])} if eligible else None
            self.last_box=list(selected['box_xyxy']);self.last_accept=int(frame)
            if written:
                vector=current[index].copy();vector.setflags(write=False)
                self.bank.append(Evidence(vector,self.recording,int(frame),self.camera,frame/self.fps,available,
                    f'{self.policy}_OWN_ACCEPTED_CURRENT_CROP',float(np.clip(selected.get('conf',0.),0,1)),anchor_cos,str(selected['candidate_uid'])))
                self.bank=self.bank[-self.capacity:]
        else:self.pending=None
        self.last_frame=int(frame)
        if hashlib.sha256(self.anchor.tobytes()).hexdigest()!=self.anchor_sha:raise ValueError('immutable anchor changed')
        return {'frame':int(frame),'recording_id':self.recording,'target_token':self.token,
                'selected_candidate_uid':str(selected['candidate_uid']) if selected else None,
                'rank1_candidate_uid':str(rows[rank]['candidate_uid']) if rank is not None else None,
                'predicted_box_xyxy':list(selected['box_xyxy']) if selected else None,
                'candidate_available_probability':available,'target_present_probability':available,'physical_presence_probability':None,
                'presence_probability_semantics':output['probability_semantics'],'memory_write':bool(written),
                'memory_write_candidate_uid':str(selected['candidate_uid']) if written else None,
                'write_correctness_proxy':risk,'write_future_harm_head_trained':False,'anchor_consistency':anchor_cos,
                'eligible_safe_write':bool(eligible),'consecutive_confirmation':bool(consecutive),'machine_bank_size':len(self.bank),
                'anchor_sha256':self.anchor_sha,'bank_source_recordings':[e.recording_id for e in self.bank],
                'unknown_cross_recording_recency_not_concatenated_time':any(e.recording_id!=self.recording for e in self.bank),
                'runtime_gt_used':False,'runtime_future_gt_used':False,'extra_clicks':0}

    def snapshot(self):
        """TRAIN-state caching by sealed candidate references, not giant duplicates."""
        return {'recording_id':self.recording,'last_frame':self.last_frame,'last_accept':self.last_accept,
                'last_box':self.last_box,'anchor_sha256':self.anchor_sha,'anonymous_target_token':self.token,
                'bank':[{'recording_id':e.recording_id,'frame':e.frame,'candidate_uid':e.candidate_uid,
                         'camera':e.camera,'timestamp_seconds':e.timestamp_seconds,'confidence':e.confidence,
                         'source':e.source,'quality':e.quality,'anchor_consistency':e.anchor_consistency,
                         'embedding_sha256':hashlib.sha256(e.embedding.tobytes()).hexdigest()} for e in self.bank]}
