"""Official frozen OSTrack model/crop/window/box-head, one-click SOT reference.

Small device-neutral wrapper of the upstream inference equations, not a newly
trained tracker. Original source/model bytes remain unchanged. The original
SOT always emits a box: it has no learned NONE policy or persistent identity
bank. Candidate-free pixel input is reported separately from SAM3 baselines.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl


def checked_device(gpu):
    if gpu is None:return torch.device('cpu')
    response=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    memory,util=[int(v.strip()) for v in response.stdout.strip().split(',')]
    if memory>64 or util:raise RuntimeError('selected GPU is not genuinely idle; do not terminate/share unrelated jobs')
    torch.cuda.set_device(gpu);torch.cuda.set_per_process_memory_fraction(min(.15,(12*(1<<30))/torch.cuda.get_device_properties(gpu).total_memory),gpu)
    return torch.device(f'cuda:{gpu}')


class FrozenSOT:
    def __init__(self,device):
        self.device=device;source=ASSETS/'official_sources/OSTrack/source'
        manifest=read_json(OUT/'baselines/SOT_SOURCE_MANIFEST.json')
        for r in manifest['files']:
            if sha256(r['path'])!=r['sha256']:raise ValueError('official source seal')
        sys.path.insert(0,str(source))
        from lib.models.ostrack import build_ostrack
        from lib.config.ostrack.config import cfg,update_config_from_file
        # Import this SHA-verified pure file directly. Upstream train package
        # __init__ eagerly imports training/admin utilities not needed for
        # inference; avoid rebuilding its full training environment.
        spec=importlib.util.spec_from_file_location('n72r21_upstream_crop',source/'lib/train/data/processing_utils.py')
        crop_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(crop_module)
        sample_target=crop_module.sample_target
        from lib.test.utils.hann import hann2d
        from lib.utils.box_ops import clip_box
        from lib.utils.ce_utils import generate_mask_cond
        self.sample_target=sample_target;self.clip_box=clip_box;self.generate_mask_cond=generate_mask_cond
        update_config_from_file(str(source/'experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml'))
        self.cfg=cfg
        record=read_json(OUT/'baselines/SOT_CHECKPOINT_MANIFEST.json')
        if sha256(record['destination'])!=record['sha256']:raise ValueError('official checkpoint SHA')
        model=build_ostrack(cfg,training=False)
        checkpoint=torch.load(record['destination'],map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['net'],strict=True);self.model=model.eval().to(device)
        for p in self.model.parameters():p.requires_grad_(False)
        self.window=hann2d(torch.tensor([16,16]).long(),centered=True).to(device)
        self.mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
        self.std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        self.initialized=False

    def preprocess(self,image):
        x=torch.as_tensor(np.ascontiguousarray(image),device=self.device).float().permute(2,0,1).unsqueeze(0)
        return ((x/255)-self.mean)/self.std

    def initialize(self,image,box):
        x,y,right,bottom=[float(v) for v in box];self.state=[x,y,right-x,bottom-y]
        patch,resize,_=self.sample_target(image,self.state,2.,output_sz=128)
        self.template=self.preprocess(patch)
        # Official CTR_POINT mask ignores box values and uses the fixed centre
        # of this sole initialized template. No future/template refresh input.
        self.mask=self.generate_mask_cond(self.cfg,1,self.device,None)
        self.template_sha=hashlib.sha256(patch.tobytes()).hexdigest()
        self.initialized=True

    def step(self,image):
        if not self.initialized:raise RuntimeError('single click must initialize')
        patch,resize,_=self.sample_target(image,self.state,4.,output_sz=256)
        with torch.inference_mode():
            out=self.model(template=self.template,search=self.preprocess(patch),ce_template_mask=self.mask)
            response=self.window*out['score_map']
            boxes=self.model.box_head.cal_bbox(response,out['size_map'],out['offset_map']).view(-1,4)
            cx,cy,w,h=(boxes.mean(dim=0)*256/resize).tolist()
            probability=float(out['score_map'].max())
        previous_cx=self.state[0]+.5*self.state[2];previous_cy=self.state[1]+.5*self.state[3]
        half_side=.5*256/resize
        mapped=[cx+previous_cx-half_side-.5*w,cy+previous_cy-half_side-.5*h,w,h]
        self.state=self.clip_box(mapped,image.shape[0],image.shape[1],margin=10)
        x,y,w,h=self.state
        return [x,y,x+w,y+h],probability


def replay(sequences,gpu):
    torch.set_num_threads(1);torch.manual_seed(72101);storage(100<<20)
    protocol=read_json(OUT/'protocol/SOT_REFERENCE_PROTOCOL.json')
    if not set(sequences)<=set(protocol['DanceTrack_train_sequences']):raise ValueError('not frozen SOT TRAIN scope')
    events=read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']
    device=checked_device(gpu);tracker=FrozenSOT(device)
    code={p:sha256(ROOT/p) for p in ['scripts/n72r21_sot.py']}
    for sequence in sequences:
        started=time.monotonic();completed=[]
        source=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
        for event in [e for e in events if e['sequence']==sequence]:
            name=event['episode_uid'];done=OUT/'baselines/sot/runtime_seals'/f'{name}.json'
            if done.exists():
                old=read_json(done)
                if old['code_sha256']!=code or sha256(old['path'])!=old['sha256']:raise ValueError('SOT source/runtime changed: create new run, no overwrite')
                completed.append(old);continue
            # No annotations or target ID enter this process: the immutable
            # click geometry is the only GT-simulated input, labelled explicitly.
            initial=cv2.imread(str(source/'img1'/f"{event['frame']+1:08d}.jpg"))
            if initial is None:raise ValueError('unreadable initialization image')
            tracker.initialize(cv2.cvtColor(initial,cv2.COLOR_BGR2RGB),event['box_xyxy'])
            path=ASSETS/'baselines/sot/runtime'/f'{name}.jsonl.zst';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('retain existing unsealed partial; never overwrite')
            with path.open('xb') as handle:
                proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=handle)
                try:
                    for f in range(event['frame']+1,event['frames']):
                        if time.monotonic()-started>1800:raise RuntimeError('30-minute sequence resource limit; prior episodes have independent resume seals')
                        image=cv2.imread(str(source/'img1'/f'{f+1:08d}.jpg'))
                        if image is None:raise ValueError('unreadable future image')
                        box,p=tracker.step(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                        row={'frame':f,'recording_id':sequence,'predicted_box_xyxy':box,'SOT_response_max':p,
                             'original_SOT_emits_box_always':True,'memory_write':False,'extra_clicks':0,
                             'runtime_gt_used':False,'runtime_future_gt_used':False}
                        proc.stdin.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode())
                    proc.stdin.close()
                    if proc.wait()!=0:raise RuntimeError('SOT trace compression')
                finally:
                    if proc.poll() is None:proc.terminate();proc.wait()
            record={'episode_uid':name,'sequence':sequence,'path':str(path),'sha256':sha256(path),'frames':event['frames']-event['frame']-1,
                    'initialization_frame':event['frame'],'one_click_label':'SIMULATED_ONE_CLICK_FROM_GT',
                    'template_image_SHA256':tracker.template_sha,'code_sha256':code,
                    'official_checkpoint_manifest_sha256':sha256(OUT/'baselines/SOT_CHECKPOINT_MANIFEST.json'),
                    'source_manifest_sha256':sha256(OUT/'baselines/SOT_SOURCE_MANIFEST.json'),
                    'device':str(device),'AMP':False,'no_future_GT':True,'new_training':False,
                    'same_SAM3_candidates':False,'input_scope':'Original SOT pixel-localization reference, separate from candidate-scoring comparisons',
                    'seconds_sequence_cumulative':time.monotonic()-started,
                    'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
            write_json(f'baselines/sot/runtime_seals/{name}.json',record);completed.append(record)
            print(json.dumps({'SOT_sequence':sequence,'episode':name,'sealed_frames':record['frames'],'seconds':round(time.monotonic()-started,1)}),flush=True)
        write_json(f'baselines/sot/sequences/{sequence}.json',{'status':'SEALED_GT_FREE_SOT','artifacts':completed})


def evaluate(sequences):
    from sam3_intermot.one_click.datasets import dancetrack_annotations
    from sam3_intermot.evaluation.one_click_protocol import iou,runs
    events={e['episode_uid']:e for e in read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']}
    labels={e['episode_uid']:e for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    for sequence in sequences:
        records=read_json(OUT/'baselines/sot/sequences'/f'{sequence}.json')['artifacts']
        for r in records:
            if sha256(r['path'])!=r['sha256'] or r['code_sha256']['scripts/n72r21_sot.py']!=sha256(ROOT/'scripts/n72r21_sot.py'):raise ValueError('SOT runtime seal')
        gt=dancetrack_annotations(ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence);results=[]
        for record in records:
            event=events[record['episode_uid']];identity=labels[record['episode_uid']]['target_gt_identity']
            runtime=read_zstd_jsonl(Path(record['path']));visible=[];overlaps=[];distances=[];takeovers=[]
            if [r['frame'] for r in runtime]!=list(range(event['frame']+1,event['frames'])):raise ValueError('incomplete SOT frame axis')
            for r in runtime:
                annotations=gt.get(r['frame'],[]);target=next((a for a in annotations if a['identity']==identity),None)
                visible.append(target is not None);box=r['predicted_box_xyxy']
                overlap=iou(box,target['box']) if target else 0.;overlaps.append(overlap)
                takeovers.append(bool(overlap<.5 and any(iou(box,a['box'])>=.5 for a in annotations if a['identity']!=identity)))
                if target:
                    b=np.asarray(box);t=np.asarray(target['box'])
                    distances.append(float(np.linalg.norm(((b[:2]+b[2:])/2-(t[:2]+t[2:])/2)/(t[2:]-t[:2]))))
            v=np.asarray(visible,bool);o=np.asarray(overlaps);correct=v&(o>=.5)
            gaps=[(start,end) for start,end in runs(~v) if start>0 and end<len(v)]
            recovered=[]
            for a,b in gaps:
                stop=next((i for i in range(b,len(v)) if not v[i]),len(v));indices=np.flatnonzero(correct[b:stop])
                recovered.append({'reappearance_frame':runtime[b]['frame'],'reacquired':bool(len(indices)),
                                  'delay_seconds':float(indices[0]/event['fps']) if len(indices) else None})
            results.append({'episode_uid':record['episode_uid'],'frames':len(runtime),'visible_frames':int(v.sum()),
                'correct_frames_IoU_0_5':int(correct.sum()),'all_visible_target_recall':float(correct.sum()/v.sum()) if v.any() else None,
                'SOT_success_AUC_21_thresholds':float(np.mean([(o[v]>t).mean() for t in np.linspace(0,1,21)])) if v.any() else None,
                'SOT_normalized_precision_at_0_2':float((np.asarray(distances)<=.2).mean()) if distances else None,
                'verified_wrong_person_takeover_frames':int(np.sum(takeovers)),
                'visible_GT_gap_false_presence_frames':int((~v).sum()),'visible_GT_gap_false_presence_rate':1. if (~v).any() else None,
                'reappearance_episodes':recovered,'original_model_has_NONE_head':False,
                'candidate_conditional_recall':None,'candidate_conditional_metric_reason':'SOT uses pixels/local search, not fixed SAM3 candidates',
                'cross_recording_test':False,'development_exposed_TRAIN_only':True,'memory_safety_PASS':False,
                'runtime_sha256':record['sha256']})
        write_json(f'baselines/sot/evaluations/{sequence}.json',{'baseline':'B6_OFFICIAL_OSTRACK_VITB256_CE',
            'episodes':results,'one_click_only':True,'future_gt_only_after_seal':True,'new_training':False,'scientific_success':False})
        print(json.dumps({'SOT_evaluated_sequence':sequence,'targets':len(results)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['replay','evaluate']);p.add_argument('--gpu',type=int)
    p.add_argument('--sequences',nargs='+',default=['dancetrack0001','dancetrack0002']);a=p.parse_args()
    if a.command=='replay':replay(a.sequences,a.gpu)
    else:evaluate(a.sequences)
