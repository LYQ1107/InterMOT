"""Post-seal TRAIN label adapter for actual own causal state references.

No GT-positive observation is added to the bank. Intermediate before-states
are reconstructed from past runtime events, then checked against stored
snapshots. Thus INNER remains full frame axis, not silently subsampled.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sam3_intermot.one_click.acib_runtime import unit,overlap
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from scripts.n72r21_common import ROOT,OUT,read_json,sha256
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from scripts.n72r21_train_t0 import collate as empty_collate


def vector_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def mixed_policy(episode_uid,ordered_uids):
    return 'P0' if ordered_uids.index(episode_uid)%2==0 else 'P1'


class StateReconstructor:
    """Only preceding anonymous runtime decisions, no labels or future rows."""
    def __init__(self,event,anchor,capacity=8):
        self.event=event;self.anchor=unit(anchor);self.capacity=capacity
        self.state={'recording_id':event['sequence'],'last_frame':event['frame'],
                    'last_accept':event['frame'],'last_box':event['box_xyxy'],
                    'anchor_sha256':vector_sha(self.anchor),'anonymous_target_token':event['episode_uid'],'bank':[]}

    def check(self,snapshot):
        if snapshot is not None and snapshot!=self.state:
            raise ValueError('causal before-state differs from sealed runtime snapshot')

    def features(self,frame,rows,uid_vectors):
        if frame<=self.state['last_frame']:raise ValueError('strictly past state required')
        e=self.event;last=self.state['last_accept']
        recency=0. if last is None else math.exp(-(frame-last)/e['fps']/60)
        q=[]
        for r in rows:
            x,y,right,bottom=r['box_xyxy'];w,h=right-x,bottom-y
            q.append([np.clip(r.get('conf',0.),0,1),np.clip(w*h/(e['width']*e['height']),0,1),
                      np.clip(w/h,0,4)/4,overlap(r['box_xyxy'],self.state['last_box']),recency])
        bank=self.state['bank']
        vectors=[];metadata=[]
        for item in bank:
            if item['frame']>=frame or item['recording_id']!=e['sequence']:raise ValueError('future/unregistered recording in F1 bank')
            vector=uid_vectors[item['candidate_uid']]
            if vector_sha(vector)!=item['embedding_sha256']:raise ValueError('bank UID/vector SHA')
            vectors.append(torch.from_numpy(vector))
            metadata.append([math.exp(-(frame-item['frame'])/e['fps']/60),0.])
        context=torch.tensor([math.log1p((frame-e['frame'])/e['fps'])/5,
                              math.log1p(0. if last is None else (frame-last)/e['fps'])/5,0.])
        return {'quality':torch.tensor(q,dtype=torch.float32).reshape(-1,5),'context':context,
                'bank_vectors':tuple(vectors),'bank_metadata':torch.tensor(metadata,dtype=torch.float32).reshape(-1,2)}

    def advance(self,record,rows,uid_vectors,policy):
        frame=int(record['frame']);current={str(r['candidate_uid']):r for r in rows}
        if frame<=self.state['last_frame'] or record['recording_id']!=self.event['sequence']:raise ValueError('invalid causal runtime order')
        if record['anchor_sha256']!=self.state['anchor_sha256'] or record['runtime_gt_used'] or record['extra_clicks']:raise ValueError('runtime boundary')
        selected=record['selected_candidate_uid']
        if selected is not None:
            if selected not in current or record['predicted_box_xyxy']!=current[selected]['box_xyxy']:raise ValueError('selected UID/box not current')
            self.state['last_box']=record['predicted_box_xyxy'];self.state['last_accept']=frame
        written=bool(record['memory_write'])
        if written!=(policy=='P1' and selected is not None):raise ValueError('registered P0/P1 write semantics')
        if written:
            if record['memory_write_candidate_uid']!=selected:raise ValueError('write not own current accepted crop')
            r=current[selected];vector=uid_vectors[selected]
            self.state['bank'].append({'recording_id':self.event['sequence'],'frame':frame,'candidate_uid':selected,
                'camera':None,'timestamp_seconds':frame/self.event['fps'],'confidence':record['candidate_available_probability'],
                'source':'P1_OWN_ACCEPTED_CURRENT_CROP','quality':float(np.clip(r.get('conf',0.),0,1)),
                'anchor_consistency':float(vector@self.anchor),'embedding_sha256':vector_sha(vector)})
            self.state['bank']=self.state['bank'][-self.capacity:]
        self.state['last_frame']=frame
        if record['machine_bank_size']!=len(self.state['bank']):raise ValueError('bounded bank state reconstruction')


class CausalFrames(Dataset):
    def __init__(self,outer,seed,sequences,condition,*,stride):
        if condition not in ('P0','P1','MIXED') or stride not in (1,5):raise ValueError('registered T1 dataset')
        fitted=read_json(OUT/'training/T0_AMP_R1'/f'{outer}__seed{seed}.json')
        if outer in sequences or not set(sequences)<=set(fitted['schema']['fit_sequences']+[fitted['schema']['inner_sequence']]):raise ValueError('outer/input leakage')
        inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
        if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor source SHA')
        anchors=np.load(inputs['anchor_path'],mmap_mode='r')
        # Shared initialization metadata access is explicitly disclosed.
        identities={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
        ordered=sorted(e['episode_uid'] for e in inputs['inputs'])
        self.examples=[];self.sources={};self.counts=[0,0,0];self.checked_snapshots=0;self.reconstructed_frames=0
        for sequence in sequences:
            events=[e for e in inputs['inputs'] if e['sequence']==sequence]
            directory=ROOT.parent/'InterMOT_N72R20R2_assets/candidates'/sequence;index=read_json(directory/'index.json')
            for name,path in [('metadata',directory/'metadata.jsonl.zst'),('embeddings',directory/'embeddings.f16')]:
                if sha256(path)!=index[name+'_sha256']:raise ValueError('candidate source SHA')
            frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
            frame_rows={int(p['frame']):rows for p,rows in frames}
            uid_vectors={str(r['candidate_uid']):unit(r['feature']) for _,rows in frames for r in rows}
            feature_tensors={f:torch.stack([torch.from_numpy(uid_vectors[str(r['candidate_uid'])]) for r in rows]) if rows else torch.empty(0,512)
                             for f,rows in frame_rows.items()}
            policies=('P0','P1') if condition=='MIXED' else (condition,);sealed={};source_seals={}
            # Verify actual runtime artifacts BEFORE opening future label files.
            for policy in policies:
                path=OUT/'training/causal_states/seals'/f'{outer}__seed{seed}__{policy}__K8'/f'{sequence}.json'
                seal=read_json(path)
                if seal['outer']!=outer or seal['seed']!=seed or seal['policy']!=policy or seal['source_checkpoint_SHA256']!=fitted['best_checkpoint_sha256']:raise ValueError('state source mismatch')
                if seal['source_fit_record_sha256']!=sha256(OUT/'training/T0_AMP_R1'/f'{outer}__seed{seed}.json'):raise ValueError('fit record changed')
                if seal['source_protocol_sha256']!=sha256(OUT/'protocol/CAUSAL_STATE_ROLLOUT.json'):raise ValueError('state protocol changed')
                for p,h in seal['code_sha256'].items():
                    if sha256(ROOT/p)!=h:raise ValueError('state generation source changed')
                for artifact in seal['artifacts']:
                    if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('state artifact changed before labels')
                sealed[policy]={r['episode_uid']:r for r in seal['artifacts']};source_seals[policy]=sha256(path)
            gtroot=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
            gt=dancetrack_annotations(gtroot)
            matched={f:strict_candidate_matching(rows,gt.get(f,[])) for f,rows in frame_rows.items()}
            for event in events:
                policy=mixed_policy(event['episode_uid'],ordered) if condition=='MIXED' else condition
                artifact=sealed[policy][event['episode_uid']];state=StateReconstructor(event,anchors[event['anchor_index']]);n=0
                future_axis=[int(p['frame']) for p,_ in frames if int(p['frame'])>event['frame']]
                for record,frame in zip(read_zstd_jsonl(Path(artifact['path'])),future_axis,strict=True):
                    if int(record['frame'])!=frame:raise ValueError('full state axis mismatch')
                    rows=frame_rows[frame];state.check(record['TRAIN_state_before']);self.checked_snapshots+=record['TRAIN_state_before'] is not None
                    if (frame-event['frame']-1)%stride==0:
                        ids=[matched[frame][str(r['candidate_uid'])] for r in rows];identity=identities[event['episode_uid']]
                        positives=[i for i,value in enumerate(ids) if value==identity]
                        if len(positives)>1:raise ValueError('strict one-to-one positive')
                        visible=any(a['identity']==identity for a in gt.get(frame,[]));availability=0 if positives else 1 if visible else 2
                        sample={'anchor':torch.from_numpy(state.anchor.copy()),'candidates':feature_tensors[frame],
                                'target':positives[0] if positives else -1,'availability':availability,
                                'write_labels':torch.tensor([v==identity for v in ids],dtype=torch.float32),
                                'write_verified':torch.tensor([v is not None for v in ids],dtype=torch.bool)}
                        sample.update(state.features(frame,rows,uid_vectors));self.examples.append(sample);self.counts[availability]+=1
                    state.advance(record,rows,uid_vectors,policy);n+=1;self.reconstructed_frames+=1
                if n!=artifact['frames']:raise ValueError('actual full causal rollout count')
            self.sources[sequence]={'state_seals':source_seals,'GT_sha256':sha256(gtroot/'gt/gt.txt'),
                                    'candidate_index_sha256':sha256(directory/'index.json')}

    def __len__(self):return len(self.examples)
    def __getitem__(self,index):return self.examples[index]


def collate(examples):
    batch=empty_collate(examples);n=len(examples);k=max(1,max(len(e['bank_vectors']) for e in examples))
    batch.update(bank=torch.zeros(n,k,512),bank_valid=torch.zeros(n,k,dtype=torch.bool),bank_metadata=torch.zeros(n,k,2))
    for i,e in enumerate(examples):
        count=len(e['bank_vectors'])
        if count:batch['bank'][i,:count]=torch.stack(e['bank_vectors']);batch['bank_valid'][i,:count]=True;batch['bank_metadata'][i,:count]=e['bank_metadata']
    return batch


def forward(model,batch):
    return model(batch['anchor'],batch['candidates'],batch['valid'],batch['quality'],bank=batch['bank'],
                 bank_valid=batch['bank_valid'],bank_metadata=batch['bank_metadata'],context=batch['context'])
