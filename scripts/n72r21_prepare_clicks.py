"""Offline simulation of exactly one real-image GT-box click per target.

Future truth identifiers are isolated from runtime input files. This prepares
development smoke, not performance-selected interactions or human-study data.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import numpy as np
import torch
from sam3_intermot.one_click.datasets import dancetrack_info,dancetrack_annotations
from sam3_intermot.identity_probe.dataset import GTBox
from sam3_intermot.identity_probe.crop import load_person_crops
from sam3_intermot.identity_probe.encoders import OSNetEncoder
from scripts.n72r21_common import ROOT,ASSETS,OUT,read_json,write_json,sha256,storage


def run(sequences):
    protocol=read_json(OUT/'protocol/ENGINEERING_SMOKE_PROTOCOL.json')
    if not set(sequences)<=set(protocol['sequences']):raise ValueError('not registered TRAIN development')
    torch.set_num_threads(1)
    checkpoint=ROOT.parent/'InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth'
    if sha256(checkpoint)!='2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154':raise ValueError('encoder SHA')
    storage(1<<20)
    encoder=OSNetEncoder(checkpoint,'cpu')
    runtime_inputs=[];truth_inputs=[];vectors=[]
    for sequence in sequences:
        source=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
        info=dancetrack_info(source);gt=dancetrack_annotations(source)
        initial={}
        for frame,rows in sorted(gt.items()):
            for row in sorted(rows,key=lambda r:r['identity']):
                x,y,right,bottom=row['box'];w,h=right-x,bottom-y
                if row['identity'] not in initial and w>=16 and h>=32 and w*h>=1024 and row['visibility']>=.5:
                    initial[row['identity']]=(frame,row)
        grouped=defaultdict(list)
        for identity,(frame,row) in sorted(initial.items()):grouped[frame].append((identity,row))
        for frame,rows in sorted(grouped.items()):
            boxes=[]
            for identity,row in rows:
                x,y,right,bottom=row['box']
                boxes.append(GTBox(sequence,frame+1,identity,x,y,right-x,bottom-y,row['confidence'],1,row['visibility']))
            image_path=source/'img1'/f'{frame+1:08d}.jpg'
            crops=load_person_crops(image_path,boxes)
            embeddings=encoder.encode(torch.stack(crops)).cpu().numpy()
            for (identity,row),embedding in zip(rows,embeddings):
                key=f'{sequence}__click{len(runtime_inputs):04d}'
                offset=len(vectors);vectors.append(embedding)
                runtime_inputs.append({'episode_uid':key,'sequence':sequence,'recording_id':sequence,
                                       'frame':frame,'anchor_index':offset,'box_xyxy':list(row['box']),**info,
                                       'initialization_label':'SIMULATED_ONE_CLICK_FROM_GT','runtime_future_gt_used':False})
                truth_inputs.append({'episode_uid':key,'sequence':sequence,'target_gt_identity':identity,'posthoc_or_train_truth_only':True})
        print(json.dumps({'sequence':sequence,'simulated_one_click_targets':len(initial),'device':'cpu','no_gpu_claim':True}),flush=True)
    path=ASSETS/'smoke/anchors.npy';path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,np.stack(vectors).astype(np.float32))
    write_json('smoke/RUNTIME_INPUTS.json',{'inputs':runtime_inputs,'anchor_path':str(path),'anchor_sha256':sha256(path),
               'source_protocol_sha256':sha256(OUT/'protocol/ENGINEERING_SMOKE_PROTOCOL.json'),'runtime_GT_identity_fields':False})
    write_json('smoke/INITIALIZATION_TRUTH.json',{'labels':truth_inputs,'runtime_inputs_contain_GT_identity':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequences',nargs='+',default=['dancetrack0001','dancetrack0002']);args=parser.parse_args();run(args.sequences)
