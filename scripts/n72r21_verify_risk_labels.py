"""Non-mutating restart verifier for the original sealed T2 labeler.

JSON converts integer dictionary keys to strings. Compare the serialized
representation without modifying the original labeler or its sealed outputs.
"""
import argparse
import json
from pathlib import Path
from scripts.n72r21_common import ROOT,OUT,read_json,sha256,write_json
from scripts.n72r21_label_coupled_states import run as original_run,risk_label
from scripts.n72r21_collect_coupled_states import CODE
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching


def json_equal(left,right):
    # Normalize keys before sorting: numeric and lexical key orders differ.
    def canonical(value):return json.dumps(json.loads(json.dumps(value,allow_nan=False)),sort_keys=True,allow_nan=False)
    return canonical(left)==canonical(right)


def verify_existing(outer,seed,sequence):
    source=read_json(OUT/'training/T1_CAUSAL_V1'/f'{outer}__seed{seed}__MIXED.json')['schema']['configuration']
    if outer==sequence or sequence not in source['fit_sequences']+[source['inner_sequence']]:raise ValueError('outer labels forbidden')
    case=f'{outer}__seed{seed}__T1MIXED_P1_K8'
    sealpath=OUT/'training/coupled_states/seals'/case/f'{sequence}.json';seal=read_json(sealpath)
    if seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE}:raise ValueError('coupled source changed')
    for artifact in seal['artifacts']:
        for kind in ('states','branches'):
            if sha256(artifact[kind+'_path'])!=artifact[kind+'_sha256']:raise ValueError('source artifact changed before GT')
    gtroot=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
    gt=dancetrack_annotations(gtroot)
    frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
    matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
    initial={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    episodes=[]
    for artifact in seal['artifacts']:
        episode=artifact['episode_uid']
        labels=[risk_label(proposal,matched,initial[episode]) for proposal in read_zstd_jsonl(Path(artifact['branches_path']))]
        if len(labels)!=artifact['paired_proposals']:raise ValueError('risk proposal count')
        episodes.append({'episode_uid':episode,'labels':labels})
    expected={'case':case,'outer':outer,'seed':seed,'sequence':sequence,'source_seal_sha256':sha256(sealpath),
        'GT_sha256':sha256(gtroot/'gt/gt.txt'),'labeler_code_sha256':sha256(ROOT/'scripts/n72r21_label_coupled_states.py'),
        'risk_label_protocol_sha256':sha256(OUT/'protocol/T2_MEMORY_COUPLED_TRAINING.json'),'episodes':episodes,
        'future_GT_used_only_after_RUNTIME_seal':True,'risk_labels_enter_online_features':False,
        'verified_proposals':sum(r['risk_label_verified'] for e in episodes for r in e['labels']),
        'verified_safe_proposals':sum(r['future_safe_label'] is True for e in episodes for r in e['labels']),
        'unknown_proposals':sum(not r['risk_label_verified'] for e in episodes for r in e['labels']),
        'scientific_success':False,'next_stage_authorized':False}
    labelpath=OUT/'training/coupled_states/labels'/case/f'{sequence}.json';before=sha256(labelpath)
    original=read_json(labelpath)
    if not json_equal(original,expected):raise ValueError('serialized sealed risk labels truly changed; do not overwrite')
    if sha256(labelpath)!=before:raise ValueError('verification changed sealed labels')
    write_json(f'training/coupled_states/restart_verification/{case}/{sequence}.json',{
        'case':case,'sequence':sequence,'label_sha256_unchanged':before,
        'original_python_dict_equal':original==expected,'canonical_JSON_equal':True,
        'original_labeler_and_supervision_unchanged':True,'verified_proposals':expected['verified_proposals'],
        'verifier_code_sha256':sha256(ROOT/'scripts/n72r21_verify_risk_labels.py'),
        'repair_protocol_sha256':sha256(OUT/'protocol/T2_LABEL_RESTART_SERIALIZATION_REPAIR.json'),
        'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps({'existing_risk_labels_verified':case,'sequence':sequence,'serialized_equal':True,'original_dict_equal':original==expected}),flush=True)


def run(outer,seeds,sequences):
    for seed in seeds:
        for sequence in sequences:
            path=OUT/'training/coupled_states/labels'/f'{outer}__seed{seed}__T1MIXED_P1_K8'/f'{sequence}.json'
            if path.exists():verify_existing(outer,seed,sequence)
            else:original_run(outer,[seed],[sequence])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103]);parser.add_argument('--sequences',nargs='+',required=True)
    args=parser.parse_args();run(args.outer,args.seeds,args.sequences)
