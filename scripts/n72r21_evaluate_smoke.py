"""Offline evaluation ONLY AFTER independent GT-free runtime files are sealed."""
from __future__ import annotations
import json
import numpy as np
from sam3_intermot.one_click.datasets import dancetrack_annotations,dancetrack_truth
from sam3_intermot.evaluation.one_click_protocol import evaluate_episode
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256


def run():
    sealed=read_json(OUT/'smoke/RUNTIME_SEAL.json')
    for path,digest in sealed['code_sha256'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('runtime code changed after smoke seal')
    for record in sealed['artifacts']:
        if sha256(record['path'])!=record['sha256']:raise ValueError('runtime changed before GT evaluation')
    inputs={e['episode_uid']:e for e in read_json(OUT/'smoke/RUNTIME_INPUTS.json')['inputs']}
    labels={e['episode_uid']:e for e in read_json(OUT/'smoke/INITIALIZATION_TRUTH.json')['labels']}
    results=[];examples={};negative=positive=reappearance=0
    for sequence in sorted({e['sequence'] for e in inputs.values()}):
        source=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
        gt=dancetrack_annotations(source)
        frames=load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)
        for record in [r for r in sealed['artifacts'] if inputs[r['episode_uid']]['sequence']==sequence]:
            event=inputs[record['episode_uid']];identity=labels[record['episode_uid']]['target_gt_identity']
            runtime=read_zstd_jsonl(record['path'])
            truth=[dancetrack_truth(int(payload['frame']),rows,gt.get(int(payload['frame']),[]),identity)
                   for payload,rows in frames if int(payload['frame'])>event['frame']]
            if not runtime:
                results.append({'episode_uid':record['episode_uid'],'status':'NO_FUTURE_FRAMES_NOT_EVALUATED'});continue
            result=evaluate_episode(runtime,truth,fps=event['fps'],recording_id=sequence)
            result.update({'episode_uid':record['episode_uid'],'initialization_frame':event['frame'],
                           'runtime_sha256':record['sha256'],'simulated_one_click':True,'development_only':True})
            results.append(result)
            positive+=result['target']['visible_frames'];negative+=len(truth)-result['target']['visible_frames']
            reappearance+=len(result['target']['reacquisition_episodes'])
            for decision,t in zip(runtime,truth):
                kind='POSITIVE' if t.target_visible and decision['selected_candidate_uid'] in t.valid_target_candidate_uids else 'ABSENT' if not t.target_visible else None
                if kind is not None and kind not in examples:
                    examples[kind]={'episode_uid':record['episode_uid'],'sequence':sequence,'initialization_frame':event['frame'],
                                    'frame':t.frame,'image_path':str(source/'img1'/f'{t.frame+1:08d}.jpg'),
                                    'runtime_decision':decision,'offline_visible_target':t.target_visible,
                                    'offline_target_box_xyxy':t.target_box_xyxy,'posthoc_only':True,
                                    'selection_rule':'FIRST_IN_PREREGISTERED_SEQUENCE_EPISODE_FRAME_ORDER_NOT_BEST_RESULT'}
            if 'REAPPEARANCE' not in examples and result['target']['reacquisition_episodes']:
                episode=result['target']['reacquisition_episodes'][0]
                decision=next(r for r in runtime if r['frame']==episode['frame'])
                examples['REAPPEARANCE']={'episode_uid':record['episode_uid'],'sequence':sequence,'initialization_frame':event['frame'],
                                         'image_path':str(source/'img1'/f"{episode['frame']+1:08d}.jpg"),
                                         'reappearance_episode':episode,'runtime_decision':decision,'posthoc_only':True,
                                         'absence_claim':'COMPLETE_VISIBLE_GT_GAP; physical out-of-view versus occlusion is not inferred'}
    for name in ('POSITIVE','ABSENT','REAPPEARANCE'):
        write_json(f'smoke/EXAMPLE_{name}.json',examples.get(name,{'status':'NO_REAL_EPISODE_AVAILABLE_NO_SYNTHETIC_SUBSTITUTE'}))
    result={'stage':'N72R21','final_goal_file':'outputs/N72R21/FINAL_GOAL.json',
            'status':'PASS_REAL_DEVELOPMENT_SMOKE' if positive and negative and reappearance and all(k in examples for k in ('POSITIVE','ABSENT','REAPPEARANCE')) else 'INCOMPLETE_REAL_EPISODE_COVERAGE',
            'real_initial_image_crops_encoded':True,'real_SAM3_candidate_tape_read':True,'runtime_written_before_GT_evaluation':True,
            'new_SAM3_inference':False,'new_training':False,'GPU_used':False,'encoder_inference_device':'cpu',
            'episodes':results,'visible_target_frames':positive,'GT_visible_absence_frames':negative,'reappearance_episodes':reappearance,
            'one_frame_one_decision_scope':'PER_INDEPENDENT_ONE_CLICK_TARGET_EPISODE; targets in the same sequence are correlated',
            'absence_semantics':'Missing target in complete visible-person GT, not proved physical out-of-view',
            'scientific_success':False,'final_evaluation':False,'heldout_tuning':False,
            'baseline':'RAW_OSNET_IMMUTABLE_ANCHOR_P0_UNCALIBRATED_SMOKE_THRESHOLDS',
            'limitations':['Only two repeatedly exposed TRAIN sequences','No real human click','No cross-recording identity claim','No safe-memory PASS from zero writes'],
            'runtime_seal_sha256':sha256(OUT/'smoke/RUNTIME_SEAL.json')}
    write_json('smoke/PIPELINE_SMOKE.json',result)
    print(json.dumps({k:result[k] for k in ('status','visible_target_frames','GT_visible_absence_frames','reappearance_episodes','GPU_used','scientific_success')}))


if __name__=='__main__':run()
