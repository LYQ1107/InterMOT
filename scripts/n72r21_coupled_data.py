"""T2 post-seal current feature/label adapter; future labels are loss-only."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.n72r21_causal_data import StateReconstructor,collate as state_collate
from scripts.n72r21_collect_coupled_states import CODE
from scripts.n72r21_common import ROOT,OUT,read_json,sha256
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.acib_runtime import unit
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching


class CoupledFrames(Dataset):
    def __init__(self,outer,seed,sequences,*,stride):
        source=read_json(OUT/'training/T1_CAUSAL_V1'/f'{outer}__seed{seed}__MIXED.json')['schema']['configuration']
        if outer in sequences or not set(sequences)<=set(source['fit_sequences']+[source['inner_sequence']]) or stride not in (1,5):raise ValueError('T2 split/stride')
        inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
        if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor source')
        anchors=np.load(inputs['anchor_path'],mmap_mode='r')
        identities={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
        self.examples=[];self.sources={};self.counts=[0,0,0];self.checked_snapshots=self.reconstructed_frames=0
        self.known_risk=self.safe_risk=0;case=f'{outer}__seed{seed}__T1MIXED_P1_K8'
        for sequence in sequences:
            sealpath=OUT/'training/coupled_states/seals'/case/f'{sequence}.json';seal=read_json(sealpath)
            labelpath=OUT/'training/coupled_states/labels'/case/f'{sequence}.json';labels=read_json(labelpath)
            if seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE} or labels['source_seal_sha256']!=sha256(sealpath):raise ValueError('source seal')
            if labels['labeler_code_sha256']!=sha256(ROOT/'scripts/n72r21_label_coupled_states.py'):raise ValueError('label source changed')
            if seal['protocol_sha256']!=sha256(OUT/'protocol/T2_MEMORY_COUPLED_TRAINING.json') or labels['risk_label_protocol_sha256']!=seal['protocol_sha256']:raise ValueError('protocol')
            for artifact in seal['artifacts']:
                for kind in ('states','branches'):
                    if sha256(artifact[kind+'_path'])!=artifact[kind+'_sha256']:raise ValueError('artifact changed')
            directory=ROOT.parent/'InterMOT_N72R20R2_assets/candidates'/sequence;index=read_json(directory/'index.json')
            for name,path in [('metadata',directory/'metadata.jsonl.zst'),('embeddings',directory/'embeddings.f16')]:
                if sha256(path)!=index[name+'_sha256']:raise ValueError('candidate source changed')
            frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
            frame_rows={int(p['frame']):rows for p,rows in frames};vectors={str(r['candidate_uid']):unit(r['feature']) for _,rows in frames for r in rows}
            tensors={f:torch.stack([torch.from_numpy(vectors[str(r['candidate_uid'])]) for r in rows]) if rows else torch.empty(0,512) for f,rows in frame_rows.items()}
            gtroot=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
            if sha256(gtroot/'gt/gt.txt')!=labels['GT_sha256']:raise ValueError('GT label source changed')
            gt=dancetrack_annotations(gtroot);matched={f:strict_candidate_matching(rows,gt.get(f,[])) for f,rows in frame_rows.items()}
            artifact_map={a['episode_uid']:a for a in seal['artifacts']};risk_map={r['episode_uid']:{int(x['frame']):x for x in r['labels']} for r in labels['episodes']}
            for event in [e for e in inputs['inputs'] if e['sequence']==sequence]:
                uid=event['episode_uid'];state=StateReconstructor(event,anchors[event['anchor_index']]);artifact=artifact_map[uid];n=0
                axis=[int(p['frame']) for p,_ in frames if int(p['frame'])>event['frame']]
                for record,frame in zip(read_zstd_jsonl(Path(artifact['states_path'])),axis,strict=True):
                    if record['frame']!=frame:raise ValueError('full-frame causal axis')
                    rows=frame_rows[frame];state.check(record['TRAIN_state_before']);self.checked_snapshots+=record['TRAIN_state_before'] is not None
                    if (frame-event['frame']-1)%stride==0:
                        identity=identities[uid];ids=[matched[frame][str(r['candidate_uid'])] for r in rows]
                        positive=[i for i,v in enumerate(ids) if v==identity]
                        if len(positive)>1:raise ValueError('one-to-one target matching')
                        visible=any(a['identity']==identity for a in gt.get(frame,[]));availability=0 if positive else 1 if visible else 2
                        future_known=torch.zeros(len(rows),dtype=torch.bool);future_safe=torch.zeros(len(rows))
                        risk=risk_map[uid].get(frame)
                        if risk is not None and risk['risk_label_verified']:
                            selected=risk['proposed_candidate_uid'];uids=[str(r['candidate_uid']) for r in rows]
                            if selected!=record['selected_candidate_uid'] or selected not in uids:raise ValueError('risk label not source own proposal')
                            slot=uids.index(selected);future_known[slot]=True;future_safe[slot]=float(risk['future_safe_label']);self.known_risk+=1;self.safe_risk+=int(risk['future_safe_label'])
                        sample={'anchor':torch.from_numpy(state.anchor.copy()),'candidates':tensors[frame],
                            'target':positive[0] if positive else -1,'availability':availability,
                            'write_labels':torch.tensor([v==identity for v in ids],dtype=torch.float32),
                            'write_verified':torch.tensor([v is not None for v in ids],dtype=torch.bool),
                            'future_safe_labels':future_safe,'future_safe_verified':future_known}
                        sample.update(state.features(frame,rows,vectors));self.examples.append(sample);self.counts[availability]+=1
                    state.advance(record,rows,vectors,'P1');n+=1;self.reconstructed_frames+=1
                if n!=artifact['frames']:raise ValueError('state artifact count')
            self.sources[sequence]={'causal_and_branch_seal_sha256':sha256(sealpath),'posthoc_future_label_sha256':sha256(labelpath),
                'candidate_index_sha256':sha256(directory/'index.json'),'GT_sha256':labels['GT_sha256']}

    def __len__(self):return len(self.examples)
    def __getitem__(self,index):return self.examples[index]


def collate(examples):
    batch=state_collate(examples);shape=batch['valid'].shape
    batch.update(future_safe_labels=torch.zeros(shape),future_safe_verified=torch.zeros(shape,dtype=torch.bool))
    for i,e in enumerate(examples):
        n=len(e['candidates']);batch['future_safe_labels'][i,:n]=e['future_safe_labels'];batch['future_safe_verified'][i,:n]=e['future_safe_verified']
    return batch
