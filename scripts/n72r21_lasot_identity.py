"""Unknown-time single-target domain diagnostics, not cross-recording success."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from sam3_intermot.one_click.frame_only_runtime import FrameOnlyRecognizer
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl

PROTOCOL=OUT/'protocol/LASOT_FRAME_ONLY_IDENTITY.json'
CODE=['scripts/n72r21_lasot_identity.py','sam3_intermot/one_click/frame_only_runtime.py',
      'sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_memory.py']


def load_candidates(name):
    sealpath=OUT/'experiments/LaSOT_identity/candidate_seals'/f'{name}.json';seal=read_json(sealpath)
    if seal['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('candidate protocol changed')
    for p,h in seal['code_sha256'].items():
        if sha256(ROOT/p)!=h:raise ValueError('candidate source changed')
    files={a['kind']:a for a in seal['artifacts']}
    for artifact in files.values():
        if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('candidate artifacts changed')
    n=seal['embedding_count'];features=np.memmap(files['embeddings']['path'],mode='r',dtype=np.float16,shape=(n,512)) if n else np.empty((0,512),dtype=np.float16)
    frames=read_zstd_jsonl(Path(files['metadata']['path']));first=seal['first_original_frame_1based']
    if [r['original_frame_1based'] for r in frames]!=list(range(first,first+seal['frame_count'])):raise ValueError('incomplete actual original-frame candidate axis')
    for frame in frames:
        for row in frame['candidates']:
            row['feature']=np.asarray(features[row['embedding_offset']],np.float32);row['conf']=row['confidence']
    return seal,frames,np.load(files['anchor']['path'])


def replay(name):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL);seal,frames,anchor=load_candidates(name);event=seal['event']
    if name not in protocol['windows']:raise ValueError('frozen window scope')
    models_path=OUT/'validation/FROZEN_MODEL_MANIFEST.json';models=read_json(models_path);code={p:sha256(ROOT/p) for p in CODE}
    for source in models['models']:
        if sha256(source['path'])!=source['sha256'] or sha256(source['actual_fit_record_path'])!=source['actual_fit_record_sha256']:raise ValueError('predetermined fitted model changed')
        checkpoint=torch.load(source['path'],map_location='cpu',weights_only=True);model=ACIBMemoryNetwork();model.load_state_dict(checkpoint['model'],strict=True)
        for policy in protocol['policies']:
            case=f"{policy}_SEED{source['seed']}";done=OUT/'experiments/LaSOT_identity/runtime_seals'/case/f'{name}.json'
            if done.exists():
                old=read_json(done)
                if old['code_sha256']!=code or sha256(old['path'])!=old['sha256']:raise ValueError('sealed frame-only runtime changed')
                continue
            runtime=FrameOnlyRecognizer(model,anchor,name,width=event['width'],height=event['height'],
                initial_frame=event['anchor_original_frame_1based'],initial_box=event['box_xyxy'],policy=policy,capacity=8)
            path=ASSETS/'experiments/LaSOT_identity/runtime'/case/f'{name}.jsonl.zst';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('preserve partial frame-only runtime')
            storage(10<<20);started=time.monotonic();count=0
            with path.open('xb') as handle:
                proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=handle)
                try:
                    for frame in frames:
                        f=frame['original_frame_1based']
                        if f<=event['anchor_original_frame_1based']:continue
                        if time.monotonic()-started>1800:raise RuntimeError('bounded frame-only replay')
                        result=runtime.step(f,frame['candidates']);result.update(initialization_label='SIMULATED_ONE_CLICK_FROM_GT')
                        proc.stdin.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                    proc.stdin.close()
                    if proc.wait()!=0:raise RuntimeError('frame-only compression')
                finally:
                    if proc.poll() is None:proc.terminate();proc.wait()
            write_json(f'experiments/LaSOT_identity/runtime_seals/{case}/{name}.json',{'case':case,'name':name,'path':str(path),'sha256':sha256(path),
                'code_sha256':code,'candidate_seal_sha256':sha256(OUT/'experiments/LaSOT_identity/candidate_seals'/f'{name}.json'),
                'model_sha256':source['sha256'],'frozen_model_manifest_sha256':sha256(models_path),'protocol_sha256':sha256(PROTOCOL),
                'frames':count,'future_truth_opened':False,'time_features_disabled':True,'FPS':None,'next_stage_authorized':False})
            print(json.dumps({'frame_only_identity_runtime_sealed':case,'name':name,'frames':count}),flush=True)


def evaluate(name):
    from sam3_intermot.evaluation.one_click_protocol import iou
    from sam3_intermot.one_click.datasets import strict_candidate_matching
    protocol=read_json(PROTOCOL);models=read_json(OUT/'validation/FROZEN_MODEL_MANIFEST.json')
    seals={f"{p}_SEED{s['seed']}":read_json(OUT/'experiments/LaSOT_identity/runtime_seals'/f"{p}_SEED{s['seed']}"/f'{name}.json') for s in models['models'] for p in protocol['policies']}
    for seal in seals.values():
        if seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE} or sha256(seal['path'])!=seal['sha256']:raise ValueError('runtime changed before GT')
    source,frames,anchor=load_candidates(name);event=source['event']
    if sha256(event['annotation_file_for_offline_evaluation_only'])!=event['annotation_SHA256']:raise ValueError('official target-box source changed')
    boxes=np.loadtxt(event['annotation_file_for_offline_evaluation_only'],delimiter=',');frame_map={r['original_frame_1based']:r['candidates'] for r in frames};results={}
    for case,seal in seals.items():
        trace=read_zstd_jsonl(Path(seal['path']));valid=available=correct=strict_correct=none_reject=writes=verified_target_writes=0;unknown=[];densities=[]
        for row in trace:
            f=row['original_frame_1based'];candidates=frame_map[f];densities.append(len(candidates));x,y,w,h=boxes[f-1]
            if not np.isfinite([x,y,w,h]).all() or w<=0 or h<=0:unknown.append(f);continue
            valid+=1;truth=[x,y,x+w,y+h];matched=strict_candidate_matching(candidates,[{'identity':0,'box':truth}])
            targets={uid for uid,label in matched.items() if label==0};available+=bool(targets)
            correct+=row['predicted_box_xyxy'] is not None and iou(row['predicted_box_xyxy'],truth)>=.5
            strict_correct+=row['selected_candidate_uid'] in targets
            none_reject+=bool(targets) and row['selected_candidate_uid'] is None
            writes+=row['memory_write'];verified_target_writes+=row['memory_write'] and row['memory_write_candidate_uid'] in targets
        def ratio(n,d):return n/d if d else None
        results[case]={'valid_official_target_box_frames':valid,'unknown_invalid_box_frames':unknown,
            'target_recall_IoU_0_5_given_valid_box':ratio(correct,valid),'strict_UID_target_recall_given_valid_box':ratio(strict_correct,valid),
            'candidate_coverage_given_valid_box':ratio(available,valid),'NONE_false_rejection_given_target_candidate':ratio(none_reject,available),
            'valid_box_frame_writes':writes,'verified_target_writes':verified_target_writes,
            'non_target_or_unverified_write_upper_bound':writes-verified_target_writes,
            'conservative_write_error_upper_bound_rate':ratio(writes-verified_target_writes,writes),'strict_correct_observation_retention':ratio(verified_target_writes,available),
            'candidate_density_mean':float(np.mean(densities)),'wrong_other_person_ID_rate':None,
            'physical_absence_FPR_recovery_seconds_or_cross_recording_metrics':None,'scientific_safety_PASS_allowed':False,
            'runtime_sha256':seal['sha256']}
    write_json(f'experiments/LaSOT_identity/evaluations/{name}.json',{'results':results,'protocol_sha256':sha256(PROTOCOL),
        'source_candidate_seal_sha256':sha256(OUT/'experiments/LaSOT_identity/candidate_seals'/f'{name}.json'),
        'FPS':None,'time_features_disabled_distribution_shift':True,'invalid_boxes_not_absence':True,'complete_person_identity_labels':False,
        'single_target_boxes_do_not_prove_sparse_scene_or_wrong_other_identity':True,'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps({'actual_box_only_identity_diagnostic':name,'cases':len(results)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['replay','evaluate']);parser.add_argument('--name',required=True)
    args=parser.parse_args();(replay if args.action=='replay' else evaluate)(args.name)
