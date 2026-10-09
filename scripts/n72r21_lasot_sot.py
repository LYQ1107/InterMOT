"""Existing TRAIN person windows: real SOT pixel inference, limited box truth."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import torch
from scripts.n72r21_sot import FrozenSOT,checked_device
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl

CODE=['scripts/n72r21_lasot_sot.py','scripts/n72r21_sot.py']


def prepare():
    protocol=read_json(OUT/'protocol/LASOT_TRAIN_TRIM_SOT.json')
    manifest=read_json(OUT/protocol['source_manifest'].removeprefix('outputs/N72R21/'))
    events=[]
    for window in manifest['local_train_windows']:
        if window['name'] not in protocol['windows']:continue
        annotation=ASSETS/'official_sources/LaSOT/annos'/f"{window['name']}.txt"
        if sha256(annotation)!=window['official_annotation_sha256']:raise ValueError('official box SHA')
        boxes=np.loadtxt(annotation,delimiter=',')
        first=window['first_original_frame_1based'];last=window['last_original_frame_1based']
        paths=sorted(Path(window['root']).glob('*.jpg'))
        if [int(p.stem) for p in paths]!=list(range(first,last+1)):raise ValueError('actual contiguous original frame alignment')
        for frame in range(first,last+1):
            x,y,w,h=boxes[frame-1]
            if not np.isfinite([x,y,w,h]).all() or w<16 or h<32 or w*h<1024:continue
            image=cv2.imread(str(Path(window['root'])/f'{frame:08d}.jpg'))
            if image is None:raise ValueError('actual initial RGB unreadable')
            events.append({'name':window['name'],'root':window['root'],'anchor_original_frame_1based':frame,
                           'last_original_frame_1based':last,'box_xyxy':[float(x),float(y),float(x+w),float(y+h)],
                           'width':image.shape[1],'height':image.shape[0],'original_FPS':None,
                           'initialization_label':'SIMULATED_ONE_CLICK_FROM_GT','future_GT_used_by_runtime':False,
                           'annotation_file_for_offline_evaluation_only':str(annotation),'annotation_SHA256':sha256(annotation)})
            break
        else:raise ValueError('no eligible real initial target; never invent a click')
    write_json('development/LASOT_TRAIN_TRIM_CLICKS.json',{'source_protocol_SHA256':sha256(OUT/'protocol/LASOT_TRAIN_TRIM_SOT.json'),'inputs':events,
               'annotation_path_is_manifest_only_runtime_does_not_open_it':True})
    print(json.dumps({'prepared_actual_window_clicks':len(events)}),flush=True)


def replay(gpu):
    torch.set_num_threads(1);storage(50<<20);tracker=FrozenSOT(checked_device(gpu))
    code={p:sha256(ROOT/p) for p in CODE}
    for event in read_json(OUT/'development/LASOT_TRAIN_TRIM_CLICKS.json')['inputs']:
        name=event['name'];done=OUT/'experiments/LaSOT_sot/runtime_seals'/f'{name}.json'
        if done.exists():
            old=read_json(done)
            if old['code_sha256']!=code or sha256(old['path'])!=old['sha256']:raise ValueError('existing runtime seal differs')
            continue
        started=time.monotonic();root=Path(event['root']);initial=event['anchor_original_frame_1based'];last=event['last_original_frame_1based']
        image=cv2.imread(str(root/f'{initial:08d}.jpg'));tracker.initialize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),event['box_xyxy'])
        path=ASSETS/'experiments/LaSOT_sot'/f'{name}.jsonl.zst';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('preserve partial runtime, no overwrite')
        with path.open('xb') as stream:
            proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
            try:
                for f in range(initial+1,last+1):
                    if time.monotonic()-started>1800:raise RuntimeError('bounded window wall limit')
                    image=cv2.imread(str(root/f'{f:08d}.jpg'))
                    if image is None:raise ValueError('unreadable actual future image')
                    box,response=tracker.step(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    row={'original_frame_1based':f,'predicted_box_xyxy':box,'response_max':response,
                         'runtime_gt_used':False,'runtime_future_gt_used':False,'extra_clicks':0,'memory_write':False}
                    proc.stdin.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode())
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError('compression failure')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
        write_json(f'experiments/LaSOT_sot/runtime_seals/{name}.json',{'path':str(path),'sha256':sha256(path),'code_sha256':code,
                   'source_checkpoint_manifest_SHA256':sha256(OUT/'baselines/SOT_CHECKPOINT_MANIFEST.json'),
                   'one_click_inputs_SHA256':sha256(OUT/'development/LASOT_TRAIN_TRIM_CLICKS.json'),
                   'initial_original_frame':initial,'last_original_frame':last,'actual_frames':last-initial,
                   'seconds':time.monotonic()-started,'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(tracker.device) if tracker.device.type=='cuda' else 0,
                   'future_truth_opened':False,'complete_original_LaSOT_benchmark':False})
        print(json.dumps({'real_LaSOT_window_SOT':name,'sealed_frames':last-initial,'seconds':round(time.monotonic()-started,1)}),flush=True)


def evaluate():
    from sam3_intermot.evaluation.one_click_protocol import iou
    results={}
    for event in read_json(OUT/'development/LASOT_TRAIN_TRIM_CLICKS.json')['inputs']:
        name=event['name'];seal=read_json(OUT/'experiments/LaSOT_sot/runtime_seals'/f'{name}.json')
        if sha256(seal['path'])!=seal['sha256'] or seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE}:raise ValueError('runtime SHA before GT read')
        if sha256(event['annotation_file_for_offline_evaluation_only'])!=event['annotation_SHA256']:raise ValueError('official annotation SHA')
        boxes=np.loadtxt(event['annotation_file_for_offline_evaluation_only'],delimiter=',');runtime=read_zstd_jsonl(Path(seal['path']))
        if [r['original_frame_1based'] for r in runtime]!=list(range(seal['initial_original_frame']+1,seal['last_original_frame']+1)):raise ValueError('incomplete actual original-frame axis')
        overlaps=[];distances=[];unknown=[]
        for row in runtime:
            f=row['original_frame_1based'];x,y,w,h=boxes[f-1]
            if not np.isfinite([x,y,w,h]).all() or w<=0 or h<=0:unknown.append(f);continue
            truth=[x,y,x+w,y+h];box=np.asarray(row['predicted_box_xyxy']);overlaps.append(iou(box,truth))
            distances.append(float(np.linalg.norm(((box[:2]+box[2:])/2-np.array([x+w/2,y+h/2]))/np.array([w,h]))))
        values=np.asarray(overlaps)
        results[name]={'actual_runtime_frames':len(runtime),'valid_official_target_box_frames':len(values),'unknown_invalid_box_frames':unknown,
                       'target_recall_IoU_0_5_given_valid_box':float((values>=.5).mean()) if len(values) else None,
                       'SOT_success_AUC_21_thresholds':float(np.mean([(values>t).mean() for t in np.linspace(0,1,21)])) if len(values) else None,
                       'normalized_precision_at_0_2':float((np.asarray(distances)<=.2).mean()) if distances else None,
                       'FPS_verified':False,'seconds_absence_hard_negative_wrong_other_ID_metrics':None,
                       'same_person_across_recordings_claim':False,'pretraining_unseen_claim':False,'raw_runtime_SHA256':seal['sha256']}
    write_json('experiments/LaSOT_sot/BOX_ONLY_WINDOW_DIAGNOSTIC.json',{'status':'ACTUAL_EXISTING_TRAIN_WINDOWS_NOT_FULL_BENCHMARK',
               'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','results':results,'invalid_box_not_absence':True,
               'single_target_GT_not_proof_of_low_crowd_density':True,'no_new_fit_or_threshold_selection':True,
               'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','replay','evaluate']);parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='replay':replay(args.gpu)
    else:evaluate()
