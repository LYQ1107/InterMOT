"""Fixed-cohort summaries, sequence-cluster uncertainty, exact pooled curves."""
import json
from pathlib import Path
import numpy as np
from scripts.n72r21_common import OUT,read_json,write_json,sha256
from scripts.n72r21_validation import cases,case_id
from scripts.n72r21_t2_summary import extended
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.one_click_curves import open_set_metrics_fast

PROTOCOL=OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json'


def clustered(values,bootstrap):
    values=np.asarray(values,dtype=float);ci=None
    if len(values)>1:
        rng=np.random.default_rng(bootstrap['seed']);draw=rng.integers(0,len(values),(bootstrap['replicates'],len(values)))
        ci=np.quantile(values[draw].mean(1),[.025,.975]).tolist()
    return {'sequence_macro_mean':float(values.mean()) if len(values) else None,'sequence_cluster_95pct_CI':ci,'clusters':len(values)}


def run():
    protocol=read_json(PROTOCOL);expected=protocol['sequences'];results={}
    for name,seed,condition in cases(protocol):
        case=case_id(name,seed);records={};sources={}
        for sequence in expected:
            p=OUT/'validation/evaluations'/case/f'{sequence}.json'
            if not p.exists():continue
            record=read_json(p);sealpath=OUT/'validation/runtime_seals'/case/f'{sequence}.json'
            if record['runtime_seal_sha256']!=sha256(sealpath) or record['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('VAL source changed')
            records[sequence]=record['episodes'];sources[sequence]=sha256(p)
        if not records:continue
        by_sequence={s:extended(episodes) for s,episodes in records.items()}
        results[case]={'family':name,'seed':seed,'condition':condition,'per_sequence':by_sequence,
            'pooled_fixed_operating_point':extended([e for episodes in records.values() for e in episodes]),
            'source_evaluation_sha256':sources,'complete_registered_cohort':set(records)==set(expected)}
    families={}
    for name,condition in protocol['cases'].items():
        ids=[case_id(name,seed) for seed in (protocol['seeds'] if condition['family']=='T2' else [None])]
        complete=[s for s in expected if all(case in results and s in results[case]['per_sequence'] for case in ids)]
        metrics={}
        for metric in ('target_recall_all_visible','strict_UID_identity_recall_all_visible','deployed_negative_FPR','reacquisition_recall','wrong_write_rate','correct_observation_retention'):
            eligible=[s for s in complete if all(results[case]['per_sequence'][s][metric] is not None for case in ids)]
            metrics[metric]={**clustered([np.mean([results[c]['per_sequence'][s][metric] for c in ids]) for s in eligible],protocol['bootstrap']),
                'eligible_sequences':eligible,'seed_average_inside_sequence_not_independent_seed_clusters':True}
        families[name]={'all_seed_sequences':complete,'all25_sequences_complete':len(complete)==25,'macro_metrics':metrics}
    baseline=results.get('RAW_OSNET_ANCHOR',{}).get('per_sequence',{});contrasts={}
    for name in protocol['cases']:
        ids=[case_id(name,seed) for seed in (protocol['seeds'] if protocol['cases'][name]['family']=='T2' else [None])]
        shared=[s for s in expected if s in baseline and all(c in results and s in results[c]['per_sequence'] for c in ids)]
        values=[np.mean([results[c]['per_sequence'][s]['target_recall_all_visible'] for c in ids])-baseline[s]['target_recall_all_visible'] for s in shared]
        contrasts[name]={**clustered(values,protocol['bootstrap']),'paired_sequences':shared,
            'complete_registered_cohort':len(shared)==25,'no_selection_from_contrast':True}
    report={'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','protocol_sha256':sha256(PROTOCOL),'cases':results,'families':families,
        'paired_recall_deltas_vs_raw_anchor':contrasts,'not_frame_IID':True,'no_VAL_fit_selection_or_refit':True,
        'historical_benchmark_exposure_not_virgin_test':True,'GT_gap_not_physical_absence':True,
        'T3_complete':False,'scientific_success':False,'next_stage_authorized':False}
    write_json('validation/FROZEN_SUMMARY.json',report)
    print(json.dumps({'scene_cases_evaluated':sum(len(r['per_sequence']) for r in results.values()),'expected':375,
        'complete_families':[name for name,r in families.items() if r['all25_sequences_complete']]}),flush=True)


def pooled_curves():
    protocol=read_json(PROTOCOL);registered=[case_id(n,s) for n,s,_ in cases(protocol)]
    if not all((OUT/'validation/evaluations'/c/f'{s}.json').exists() for c in registered for s in protocol['sequences']):raise RuntimeError('pooled final curves require all375 compared scene-case evaluations')
    arrays={c:{'scores':[],'available':[],'correct':[],'claim_scores':[],'claim_labels':[],'unknown':0} for c in registered};sources={}
    for sequence in protocol['sequences']:
        seals={}
        # Verify every compared runtime before opening this sequence's truth.
        for case in registered:
            path=OUT/'validation/runtime_seals'/case/f'{sequence}.json';seal=read_json(path)
            evaluation=read_json(OUT/'validation/evaluations'/case/f'{sequence}.json')
            if evaluation['runtime_seal_sha256']!=sha256(path):raise ValueError('evaluated runtime seal changed')
            for artifact in seal['artifacts']:
                if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('runtime changed before pooled truth')
            seals[case]=seal;sources[f'{case}/{sequence}']=sha256(path)
        labels=read_json(OUT/'validation/initialization_truth'/f'{sequence}.json');identities={e['episode_uid']:e['target_gt_identity'] for e in labels['labels']}
        gtroot=Path(protocol['dataset_root'])/'val'/sequence
        if sha256(gtroot/'gt/gt.txt')!=labels['GT_sha256']:raise ValueError('GT source changed')
        gt=dancetrack_annotations(gtroot);candidate=Path(protocol['candidate_root'])/'candidates'/sequence
        index=read_json(candidate/'index.json')
        if sha256(candidate/'metadata.jsonl.zst')!=index['metadata_sha256']:raise ValueError('candidate metadata changed')
        # Match current candidate metadata without copying/decoding embeddings.
        frame_rows=read_zstd_jsonl(candidate/'metadata.jsonl.zst')
        matched={int(r['frame']):strict_candidate_matching([c for c in r['candidates'] if valid_geometry(c)],gt.get(int(r['frame']),[])) for r in frame_rows}
        for case in registered:
            seal=seals[case];a=arrays[case]
            for artifact in seal['artifacts']:
                identity=identities[artifact['episode_uid']]
                for row in read_zstd_jsonl(Path(artifact['path'])):
                    axis=matched[int(row['frame'])];uid=row.get('rank1_candidate_uid')
                    if uid is not None and uid not in axis:raise ValueError('rank UID outside candidate axis')
                    label=axis.get(uid);a['scores'].append(row['candidate_available_probability']);a['available'].append(identity in axis.values());a['correct'].append(label==identity)
                    verified=False if uid is None else None if label is None else bool(label==identity)
                    if verified is None:a['unknown']+=1
                    else:a['claim_scores'].append(row['rank1_identity_joint_probability']);a['claim_labels'].append(verified)
        print(json.dumps({'pooled_frozen_VAL_axes_read':sequence}),flush=True)
    results={c:{'primary_candidate_availability':open_set_metrics_fast(a['scores'],a['available'],a['correct']),
        'secondary_verified_identity_claim':open_set_metrics_fast(a['claim_scores'],a['claim_labels'],a['claim_labels']),
        'secondary_UNKNOWN_excluded_frames':a['unknown'],'all_future_frames':len(a['scores'])} for c,a in arrays.items()}
    write_json('validation/POOLED_CURVES.json',{'cases':results,'runtime_seal_sha256':sources,'protocol_sha256':sha256(PROTOCOL),
        'metric_implementation_sha256':sha256(Path(__file__).resolve().parents[1]/'sam3_intermot/evaluation/one_click_curves.py'),
        'correct_identity_not_candidate_availability_calibration':True,'curves_not_deployment_threshold_reselection':True,
        'cohort_complete':True,'no_refitting_or_VAL_based_selection':True,'next_stage_authorized':False})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--pooled-curves',action='store_true');args=parser.parse_args()
    (pooled_curves if args.pooled_curves else run)()
