"""Post-seal TRAIN/inner labels for paired real causal memory trajectories."""
import argparse
from pathlib import Path
import numpy as np
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256
from scripts.n72r21_collect_coupled_states import CODE
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching


def outcome(branch,matched,identity):
    margins={};correct=wrong=0
    for r in branch:
        labels=matched[int(r['frame'])];scores=r['current_candidate_logits'];selected=r['selected_candidate_uid']
        if selected is not None and selected not in labels:raise ValueError('future selected UID not on source axis')
        label=labels.get(selected)
        correct+=selected is not None and label==identity
        wrong+=selected is not None and label is not None and label!=identity
        positives=[scores[u] for u,value in labels.items() if value==identity and u in scores]
        negatives=[scores[u] for u,value in labels.items() if value is not None and value!=identity and u in scores]
        if len(positives)>1:raise ValueError('strict one-to-one target candidate')
        if positives and negatives:margins[int(r['frame'])]=float(positives[0]-max(negatives))
    return {'competitive_margins':margins,'correct_future_selections':int(correct),'verified_wrong_future_selections':int(wrong)}


def risk_label(proposal,matched,identity):
    frame=int(proposal['proposal_frame']);uid=proposal['proposed_candidate_uid']
    if uid not in matched[frame]:raise ValueError('proposal UID absent from current candidate axis')
    current=matched[frame][uid];without=outcome(proposal['branches']['without_current_write'],matched,identity)
    with_write=outcome(proposal['branches']['with_current_write'],matched,identity)
    if set(without['competitive_margins'])!=set(with_write['competitive_margins']):raise ValueError('counterfactual competitor frame axes differ')
    frames=sorted(without['competitive_margins']);known=current is not None and bool(frames)
    delta=float(np.mean([with_write['competitive_margins'][f]-without['competitive_margins'][f] for f in frames])) if frames else None
    safe=bool(known and current==identity and delta>=-.02 and with_write['correct_future_selections']>=without['correct_future_selections']
              and with_write['verified_wrong_future_selections']<=without['verified_wrong_future_selections'])
    return {'frame':frame,'proposed_candidate_uid':uid,'risk_label_verified':bool(known),'future_safe_label':safe if known else None,
            'current_identity_verified':current is not None,'current_identity_correct':current==identity if current is not None else None,
            'competitive_future_frames':frames,'mean_projected_margin_delta':delta,'without_current_write':without,'with_current_write':with_write,
            'future_labels_supervision_only_not_online_features':True}


def run(outer,seeds,sequences):
    initial={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    for seed in seeds:
        source=read_json(OUT/'training/T1_CAUSAL_V1'/f'{outer}__seed{seed}__MIXED.json')['schema']['configuration']
        allowed=source['fit_sequences']+[source['inner_sequence']]
        if not set(sequences)<=set(allowed) or outer in sequences:raise ValueError('outer labels forbidden in T2 risk training')
        case=f'{outer}__seed{seed}__T1MIXED_P1_K8'
        for sequence in sequences:
            sealpath=OUT/'training/coupled_states/seals'/case/f'{sequence}.json';seal=read_json(sealpath)
            if seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE}:raise ValueError('source changed')
            for a in seal['artifacts']:
                for kind in ('states','branches'):
                    if sha256(a[kind+'_path'])!=a[kind+'_sha256']:raise ValueError('state/paired branches changed before GT read')
            gtroot=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence;gt=dancetrack_annotations(gtroot)
            frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
            matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
            results=[]
            for a in seal['artifacts']:
                episode=a['episode_uid'];labels=[risk_label(proposal,matched,initial[episode]) for proposal in read_zstd_jsonl(Path(a['branches_path']))]
                if len(labels)!=a['paired_proposals']:raise ValueError('risk proposal count')
                results.append({'episode_uid':episode,'labels':labels})
            record={'case':case,'outer':outer,'seed':seed,'sequence':sequence,'source_seal_sha256':sha256(sealpath),
                'GT_sha256':sha256(gtroot/'gt/gt.txt'),'labeler_code_sha256':sha256(ROOT/'scripts/n72r21_label_coupled_states.py'),
                'risk_label_protocol_sha256':sha256(OUT/'protocol/T2_MEMORY_COUPLED_TRAINING.json'),'episodes':results,
                'future_GT_used_only_after_RUNTIME_seal':True,'risk_labels_enter_online_features':False,
                'verified_proposals':sum(r['risk_label_verified'] for e in results for r in e['labels']),
                'verified_safe_proposals':sum(r['future_safe_label'] is True for e in results for r in e['labels']),
                'unknown_proposals':sum(not r['risk_label_verified'] for e in results for r in e['labels']),
                'scientific_success':False,'next_stage_authorized':False}
            destination=OUT/'training/coupled_states/labels'/case/f'{sequence}.json'
            if destination.exists():
                if read_json(destination)!=record:raise ValueError('sealed labels changed, version explicitly')
            else:write_json(f'training/coupled_states/labels/{case}/{sequence}.json',record)
            print({'risk_posthoc_labeled':case,'sequence':sequence,'known':record['verified_proposals'],'safe':record['verified_safe_proposals'],'unknown':record['unknown_proposals']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103]);parser.add_argument('--sequences',nargs='+',required=True)
    args=parser.parse_args();run(args.outer,args.seeds,args.sequences)
