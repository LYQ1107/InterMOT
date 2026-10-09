"""TRAIN posthoc discrimination, separated from candidate coverage and NONE.

Uses actual immutable anchors and frozen OSNet / strict R3R2 refit weights.
Other-identity hard negatives require deterministic one-to-one GT matching;
unresolved candidates are never silently labelled another physical person.
This diagnostic does not fit, tune, select or claim cross-recording success.
"""
from collections import defaultdict
import json
import numpy as np
import torch
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256
from scripts.n72r21_baselines import load_adapter,project
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from scripts.n72r21_baselines_geometry_repair import valid_geometry


def finish(rows):
    if not rows:return {'competitive_frames':0,'hard_negative_win_rate':None,'median_margin':None}
    values=np.asarray(rows,float)
    return {'competitive_frames':len(values),'hard_negative_win_rate':float((values>0).mean()),
            'mean_margin':float(values.mean()),'median_margin':float(np.median(values)),
            'P10_P25_P75_margin':np.quantile(values,[.1,.25,.75]).tolist()}


def run():
    torch.set_num_threads(1)
    protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
    if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor SHA')
    anchors=np.load(inputs['anchor_path'],mmap_mode='r')
    labels={e['episode_uid']:e for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    results={}
    for sequence in protocol['sequences']:
        # Verify a complete GT-free reference run before reading future labels.
        seal=read_json(OUT/'baselines/runtime_seals/B0_RAW_ANCHOR'/f'{sequence}.json')
        for r in seal['artifacts']:
            if sha256(r['path'])!=r['sha256']:raise ValueError('reference runtime seal')
        frames=load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in frames]
        gt=dancetrack_annotations(ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence)
        matches=[strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames]
        adapter=load_adapter(sequence);encoded=project(adapter,frames)
        groups={'RAW_OSNET':defaultdict(list),'R3R2_STRICT_ADAPTER_ENSEMBLE':defaultdict(list)}
        counts={'target_visible':0,'target_candidate_available':0,'competitive_true_other_ID':0,'unresolved_candidates_not_verified_negatives':0}
        for event in [e for e in inputs['inputs'] if e['sequence']==sequence]:
            identity=labels[event['episode_uid']]['target_gt_identity']
            anchor=np.array(anchors[event['anchor_index']],copy=True);anchor/=np.linalg.norm(anchor)
            with torch.inference_mode():
                queries=[m.encode_query(torch.as_tensor(anchor)[None]).numpy()[0] for m in adapter.models]
            for payload,rows in frames:
                f=int(payload['frame'])
                if f<=event['frame']:continue
                matched=matches[f];visible=any(a['identity']==identity for a in gt.get(f,[]))
                pos=[i for i,r in enumerate(rows) if matched[str(r['candidate_uid'])]==identity]
                neg=[i for i,r in enumerate(rows) if matched[str(r['candidate_uid'])] not in (None,identity)]
                counts['target_visible']+=visible;counts['target_candidate_available']+=bool(pos)
                counts['unresolved_candidates_not_verified_negatives']+=sum(matched[str(r['candidate_uid'])] is None for r in rows)
                if not pos or not neg:continue
                counts['competitive_true_other_ID']+=1
                raw=np.stack([r['feature'] for r in rows])@anchor
                learned=np.mean([e@q for e,q in zip(encoded[f],queries)],axis=0)
                gap=(f-event['frame'])/event['fps'];time_bin='0_5_SECONDS' if gap<=5 else '5_20_SECONDS' if gap<=20 else 'GT20_SECONDS_WITHIN_RECORDING'
                for name,scores in [('RAW_OSNET',raw),('R3R2_STRICT_ADAPTER_ENSEMBLE',learned)]:
                    margin=float(max(scores[pos])-max(scores[neg]))
                    groups[name]['ALL_FUTURE_COMPETITIVE'].append(margin);groups[name][time_bin].append(margin)
        results[sequence]={'counts':counts,'representations':{name:{g:finish(v) for g,v in groups[name].items()} for name in groups},
                           'actual_adapter_model_manifest':adapter.manifest}
        write_json('evaluation/FROZEN_REPRESENTATION_DIAGNOSTIC.json',{'status':'TRAIN_POSTHOC_NOT_GENERALIZATION',
            'per_sequence':results,'complete_registered_cohort':len(results)==len(protocol['sequences']),
            'candidate_missing_not_counted_as_identity_discrimination_win_or_loss':True,
            'all_visible_tracking_metrics_reported_separately_in_baselines':True,
            'negative_identity_verification':'Current deterministic >=0.5 one-to-one GT matching; unresolved excluded, not guessed',
            'zero_area_geometry_repair_matches_baseline_UID_axis':True,'new_fit_or_threshold_tuning':False,
            'same_person_across_independent_recordings_claimed':False,
            'other_ReID_encoder_and_general_visual_encoder_controls':'PENDING_ACTUAL_FEATURE_EXTRACTION_NOT_REPLACED_BY_A_BIG_CONTROLLER',
            'scientific_success':False})
        print(json.dumps({'sequence':sequence,'posthoc_competitive_frames':counts['competitive_true_other_ID']}),flush=True)


if __name__=='__main__':run()
