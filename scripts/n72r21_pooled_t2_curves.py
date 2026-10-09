"""Exact complete-cohort T2 principal-control curves, no model/threshold selection."""
import json
from pathlib import Path
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256
from scripts.n72r21_failure_breakdowns import scope_inputs
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.one_click_curves import open_set_metrics_fast


def run():
    sequences,registered,base,candidate_root,gtroot=scope_inputs('T2')
    if not all((base/'evaluations'/case/f'{sequence}.json').exists() for case in registered for sequence in sequences):
        raise RuntimeError('all fixed eight-scene/three-seed/principal-control evaluations required')
    arrays={case:{'score':[],'available':[],'rank_correct':[],'claim_score':[],'claim_label':[],'UNKNOWN':0} for case in registered}
    sources={};labels=read_json(OUT/'development/INITIALIZATION_TRUTH.json')
    identities={e['episode_uid']:e['target_gt_identity'] for e in labels['labels']}
    for sequence in sequences:
        seals={}
        for case in registered:
            sealpath=base/'runtime_seals'/case/f'{sequence}.json';seal=read_json(sealpath)
            evaluation=read_json(base/'evaluations'/case/f'{sequence}.json')
            if evaluation['runtime_seal_sha256']!=sha256(sealpath):raise ValueError('evaluated runtime changed')
            for file,h in seal['code_sha256'].items():
                if sha256(ROOT/file)!=h:raise ValueError('runtime code changed')
            for a in seal['artifacts']:
                if sha256(a['path'])!=a['sha256']:raise ValueError('sealed runtime artifact changed')
            if sha256(gtroot/sequence/'gt/gt.txt')!=evaluation['GT_sha256']:raise ValueError('evaluated GT changed')
            seals[case]=seal;sources[f'{case}/{sequence}']=sha256(sealpath)
        # All compared runtimes verified before GT labeling for this scene.
        gt=dancetrack_annotations(gtroot/sequence);candidate=candidate_root/'candidates'/sequence
        index=read_json(candidate/'index.json')
        if sha256(candidate/'metadata.jsonl.zst')!=index['metadata_sha256']:raise ValueError('candidate metadata changed')
        rows=read_zstd_jsonl(candidate/'metadata.jsonl.zst')
        matched={int(r['frame']):strict_candidate_matching([c for c in r['candidates'] if valid_geometry(c)],gt.get(int(r['frame']),[])) for r in rows}
        for case in registered:
            a=arrays[case]
            for artifact in seals[case]['artifacts']:
                identity=identities[artifact['episode_uid']]
                for row in read_zstd_jsonl(Path(artifact['path'])):
                    axis=matched[row['frame']];rank=row['rank1_candidate_uid']
                    if rank is not None and rank not in axis:raise ValueError('rank outside actual current UID axis')
                    label=axis.get(rank);a['score'].append(row['candidate_available_probability'])
                    a['available'].append(identity in axis.values());a['rank_correct'].append(label==identity)
                    verified=False if rank is None else None if label is None else bool(label==identity)
                    if verified is None:a['UNKNOWN']+=1
                    else:a['claim_score'].append(row['rank1_identity_joint_probability']);a['claim_label'].append(verified)
        print(json.dumps({'complete_cohort_T2_pooled_curve_axes_read':sequence}),flush=True)
    cases={case:{'primary_candidate_availability':open_set_metrics_fast(a['score'],a['available'],a['rank_correct']),
        'secondary_verified_identity_claim':open_set_metrics_fast(a['claim_score'],a['claim_label'],a['claim_label']),
        'secondary_UNKNOWN_excluded_frames':a['UNKNOWN'],'all_future_frames':len(a['score'])} for case,a in arrays.items()}
    write_json('experiments/T2/POOLED_PRINCIPAL_CURVES.json',{'cases':cases,'runtime_seal_sha256':sources,
        'analysis_source_sha256':sha256(Path(__file__)),'metric_source_sha256':sha256(ROOT/'sam3_intermot/evaluation/one_click_curves.py'),
        'fixed_controls':['FULL_K8','ANCHOR_ONLY_P0','WITHOUT_SAFE_WRITE_P1','MEAN_PROTOTYPE_P1'],
        'scope':'Eight historically exposed TRAIN scenes; all3 seeds, exact pooled curves',
        'primary_availability_is_not_secondary_correct_identity_calibration':True,
        'curves_are_not_deployment_threshold_reselection':True,'new_fitting_or_VAL_selection':False,
        'new_SOT_or_association_work':False,'next_stage_authorized':False})
    print(json.dumps({'all_T2_principal_pooled_curves_complete':len(cases),'scenes':len(sequences),'seeds':3}))


if __name__=='__main__':run()
