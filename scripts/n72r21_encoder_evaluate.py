"""Posthoc hard-negative and rank comparison of sealed frozen real features."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.n72r21_common import ROOT, OUT, read_json, write_json, sha256
from scripts.n72r21_encoder_controls import NAMES
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching


def unit(value):
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1e-8): raise ValueError('invalid frozen features')
    return value/norm


def observation(scores, positives, negatives):
    """Ties are losses; verified rank pessimistically counts tied negatives."""
    positive = float(np.max(scores[positives])); negative = float(np.max(scores[negatives]))
    rank = 1+int(np.sum(scores[negatives] >= positive))
    return positive-negative, rank, 1/rank, 1+int(np.sum(np.delete(scores,positives) >= positive))


def finish(values):
    if not values: return {'competitive_frames':0,'hard_negative_win_rate':None}
    array = np.asarray(values,float); margin,rank,mrr,full_rank = array.T
    return {'competitive_frames':len(array),'hard_negative_wins':int((margin>0).sum()),
            'hard_negative_win_rate':float((margin>0).mean()), 'median_margin':float(np.median(margin)),
            'mean_margin':float(margin.mean()),'P10_P25_P75_margin':np.quantile(margin,[.1,.25,.75]).tolist(),
            'verified_other_ID_Rank1':float((rank==1).mean()),'verified_other_ID_Rank2':float((rank<=2).mean()),
            'verified_other_ID_Rank3':float((rank<=3).mean()),'verified_other_ID_MRR':float(mrr.mean()),
            'all_candidate_UID_Rank1_including_unresolved':float((full_rank==1).mean())}


def run():
    protocol = read_json(OUT/'protocol/FROZEN_ENCODER_CONTROLS.json')
    inputs = read_json(OUT/'development/RUNTIME_INPUTS.json')
    anchors = np.load(inputs['anchor_path'],mmap_mode='r')
    if sha256(inputs['anchor_path']) != inputs['anchor_sha256']: raise ValueError('OSNet anchor SHA')
    labels = {r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    names = ('EXISTING_FROZEN_OSNET',)+NAMES
    output = {}; pooled = {n:defaultdict(list) for n in names}; source_seals = {}
    for sequence in protocol['sequences']:
        path = OUT/'baselines/encoder_controls/seals'/f'{sequence}.json'; seal = read_json(path)
        if seal['protocol_sha256'] != sha256(OUT/'protocol/FROZEN_ENCODER_CONTROLS.json'): raise ValueError('protocol seal')
        for r in seal['artifacts']:
            if sha256(r['path']) != r['sha256']: raise ValueError('actual feature seal')
        for p,h in protocol['code_sha256'].items():
            if sha256(p) != h: raise ValueError('extraction code seal')
        source_seals[sequence] = sha256(path)
        frames = [(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
        for (p,rows),axis in zip(frames,seal['candidate_axis'],strict=True):
            if axis['frame'] != p['frame'] or axis['UIDs'] != [str(r['candidate_uid']) for r in rows]: raise ValueError('UID alignment')
        features = {name:unit(np.load(Path(seal['artifacts'][2*i]['path']),mmap_mode='r')) for i,name in enumerate(NAMES)}
        initial = {name:unit(np.load(Path(seal['artifacts'][2*i+1]['path']),mmap_mode='r')) for i,name in enumerate(NAMES)}
        gt = dancetrack_annotations(ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence)
        matching = [strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames]
        groups = {n:defaultdict(list) for n in names}; counts = {'visible_frames':0,'available_positive_frames':0,'competitive_frames':0}
        events = [e for e in inputs['inputs'] if e['sequence']==sequence]
        if [e['episode_uid'] for e in events] != seal['episodes_in_anchor_order']: raise ValueError('anchor order')
        for i,event in enumerate(events):
            identity = labels[event['episode_uid']]
            query = {'EXISTING_FROZEN_OSNET':unit(anchors[event['anchor_index']]),**{n:initial[n][i] for n in NAMES}}
            for (p,rows),axis,matched in zip(frames,seal['candidate_axis'],matching,strict=True):
                frame = int(p['frame'])
                if frame <= event['frame']: continue
                pos = [j for j,r in enumerate(rows) if matched[str(r['candidate_uid'])]==identity]
                neg = [j for j,r in enumerate(rows) if matched[str(r['candidate_uid'])] not in (None,identity)]
                counts['visible_frames'] += any(r['identity']==identity for r in gt.get(frame,[]))
                counts['available_positive_frames'] += bool(pos)
                if not pos or not neg: continue
                counts['competitive_frames'] += 1
                offset,count = axis['offset'],axis['count']
                matrices = {'EXISTING_FROZEN_OSNET':unit(np.stack([r['feature'] for r in rows])),
                            **{n:features[n][offset:offset+count] for n in NAMES}}
                gap = (frame-event['frame'])/event['fps']
                time_bin = '0_5_SECONDS' if gap<=5 else '5_20_SECONDS' if gap<=20 else 'GT20_SECONDS_WITHIN_RECORDING'
                density = 'SPARSE_0_4' if count<=4 else 'MEDIUM_5_8' if count<=8 else 'CROWDED_GE9'
                for name in names:
                    row = observation(matrices[name]@query[name],pos,neg)
                    for group in ['ALL_COMPETITIVE',time_bin,density]:
                        groups[name][group].append(row); pooled[name][group].append(row)
        output[sequence] = {'denominators':counts,'representations':{n:{g:finish(v) for g,v in groups[n].items()} for n in names}}
        print(json.dumps({'encoder_evaluated_sequence':sequence,'competitive_frames':counts['competitive_frames']}),flush=True)
    rng = np.random.default_rng(72104); draw = rng.integers(0,len(output),(2000,len(output)))
    intervals = {}; base = np.array([s['representations'][names[0]]['ALL_COMPETITIVE']['hard_negative_win_rate'] for s in output.values()])
    for name in names:
        rates = np.array([s['representations'][name]['ALL_COMPETITIVE']['hard_negative_win_rate'] for s in output.values()])
        intervals[name] = {'sequence_macro_win_rate':float(rates.mean()),'sequence_cluster_95pct_CI':np.quantile(rates[draw].mean(axis=1),[.025,.975]).tolist(),
                           'paired_sequence_macro_delta_vs_existing_OSNet':float((rates-base).mean()),
                           'paired_sequence_delta_95pct_CI':np.quantile((rates-base)[draw].mean(axis=1),[.025,.975]).tolist()}
    write_json('evaluation/FROZEN_ENCODER_CONTROLS.json',{
        'status':'COMPLETE_ACTUAL_FROZEN_TRAIN_DIAGNOSTIC_NOT_INDEPENDENT_GENERALIZATION',
        'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','protocol_sha256':sha256(OUT/'protocol/FROZEN_ENCODER_CONTROLS.json'),
        'source_feature_seals_sha256':source_seals,'per_sequence':output,
        'pooled':{n:{g:finish(v) for g,v in pooled[n].items()} for n in names},'sequence_cluster_uncertainty':intervals,
        'verified_other_ID_ranking_excludes_unresolved_candidates':True,'full_UID_rank_also_reported':True,
        'missing_positive_is_coverage_failure_not_conditional_discrimination':True,
        'new_fit_or_threshold_tuning':False,'cross_recording_test':False,'scientific_success':False,
        'no_backbone_selected_from_outer_performance':True,'next_stage_authorized':False})


if __name__=='__main__':run()
