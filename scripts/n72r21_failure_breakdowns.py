"""Fixed-cohort posthoc failure decomposition; no inference or tuning."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256
from scripts.n72r21_validation import cases,case_id
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.one_click_breakdowns import Counts,frame_facts,gap_bin,recovery_events

PROTOCOL=OUT/'protocol/POSTHOC_FAILURE_BREAKDOWNS.json'
CODE=['scripts/n72r21_failure_breakdowns.py','sam3_intermot/evaluation/one_click_breakdowns.py']


def compact_result(bucket):
    r=bucket.result()
    for key in ('candidate_availability_calibration','verified_identity_claim_calibration'):
        r[key].pop('calibration_bins',None)
    return r


def groups(fact,seconds):
    margin=fact['raw_anchor_hard_negative_margin'];overlap=fact['GT_geometric_overlap_proxy']
    return ['ALL','DENSITY_'+fact['density'],'ANCHOR_GAP_'+gap_bin(seconds),
        'AVAILABLE' if fact['available'] else 'VISIBLE_CANDIDATE_MISSING' if fact['visible'] else 'VISIBLE_GT_GAP_UNRESOLVED',
        'RAW_ANCHOR_WIN' if margin is not None and margin>0 else 'RAW_ANCHOR_LOSS_OR_TIE' if margin is not None else 'RAW_ANCHOR_NO_VERIFIED_COMPARISON',
        'GT_OVERLAP_GE025' if overlap is not None and overlap>=.25 else 'GT_OVERLAP_LT025_OR_TARGET_GT_GAP']


def scope_inputs(scope):
    if scope=='VAL':
        protocol=read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json');registered=[case_id(n,s) for n,s,_ in cases(protocol)]
        return protocol['sequences'],registered,OUT/'validation',Path(protocol['candidate_root']),Path(protocol['dataset_root'])/'val'
    protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json');deploy=read_json(OUT/'protocol/T2_DEPLOYMENT_ABLATIONS.json')
    names=['FULL_K8','ANCHOR_ONLY_P0','WITHOUT_SAFE_WRITE_P1','MEAN_PROTOTYPE_P1']
    registered=[f'T2_{n}_SEED{s}' for s in deploy['seeds'] for n in names]
    return protocol['sequences'],registered,OUT/'experiments/T2',ROOT.parent/'InterMOT_N72R20R2_assets',ROOT.parent/'InterMOT_N72R16_assets/dataset/train'


def run(scope,requested=None):
    sequences,registered,base,candidate_root,gtroot=scope_inputs(scope);code={p:sha256(ROOT/p) for p in CODE}
    if requested:
        if not set(requested)<=set(sequences):raise ValueError('registered cohort scope only')
        sequences=[s for s in sequences if s in requested]
    completed=[];pending=[]
    for sequence in sequences:
        if not all((base/'evaluations'/case/f'{sequence}.json').exists() for case in registered):pending.append(sequence);continue
        source_seals={case:sha256(base/'runtime_seals'/case/f'{sequence}.json') for case in registered}
        done=OUT/'failure_analysis'/scope/f'{sequence}.json'
        if done.exists():
            old=read_json(done)
            if old['code_sha256']!=code or old['protocol_sha256']!=sha256(PROTOCOL) or old['runtime_seal_sha256']!=source_seals:raise ValueError('completed analysis source changed')
            completed.append(sequence);continue
        seals={case:read_json(base/'runtime_seals'/case/f'{sequence}.json') for case in registered}
        evaluations={case:read_json(base/'evaluations'/case/f'{sequence}.json') for case in registered}
        # Verify all compared runtimes, original source code, and evaluation seals
        # before opening this scene's truth. No training/runtime state is mutated.
        for case,seal in seals.items():
            if evaluations[case]['runtime_seal_sha256']!=source_seals[case]:raise ValueError('evaluated runtime changed')
            for p,h in seal['code_sha256'].items():
                if sha256(ROOT/p)!=h:raise ValueError('runtime code changed')
            for artifact in seal['artifacts']:
                if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('runtime artifact changed')
        if scope=='VAL':
            inputs=read_json(OUT/'validation/initialization'/f'{sequence}.json');labels=read_json(OUT/'validation/initialization_truth'/f'{sequence}.json')['labels']
        else:
            inputs=read_json(OUT/'development/RUNTIME_INPUTS.json');labels=read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']
        if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('sole real anchor changed')
        gtpath=gtroot/sequence/'gt/gt.txt';gtsha=sha256(gtpath)
        if any(e['GT_sha256']!=gtsha for e in evaluations.values()):raise ValueError('truth changed')
        gt=dancetrack_annotations(gtroot/sequence);frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(candidate_root,sequence)]
        indexpath=candidate_root/'candidates'/sequence/'index.json';index=read_json(indexpath)
        for kind,file in [('metadata','metadata.jsonl.zst'),('embeddings','embeddings.f16')]:
            if sha256(indexpath.parent/file)!=index[kind+'_sha256']:raise ValueError('candidate source changed')
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        identities={l['episode_uid']:l['target_gt_identity'] for l in labels};anchors=np.load(inputs['anchor_path'],mmap_mode='r');episode_facts={};probe=defaultdict(list)
        events={e['episode_uid']:e for e in inputs['inputs'] if e['sequence']==sequence}
        for uid,event in events.items():
            anchor=np.array(anchors[event['anchor_index']],np.float32);anchor/=np.linalg.norm(anchor);axis=[];facts=[]
            for p,rows in frames:
                f=int(p['frame'])
                if f<=event['frame']:continue
                fact=frame_facts(rows,gt.get(f,[]),matched[f],identities[uid],anchor);axis.append(f);facts.append(fact)
                margin=fact['raw_anchor_hard_negative_margin']
                if margin is not None:
                    for group in ('ALL','DENSITY_'+fact['density'],'ANCHOR_GAP_'+gap_bin((f-event['frame'])/event['fps'])):probe[group].append(margin)
            episode_facts[uid]=(axis,facts)
        results={}
        for case,seal in seals.items():
            if {a['episode_uid'] for a in seal['artifacts']}!=set(events):raise ValueError('incomplete all-target runtime scope')
            buckets=defaultdict(Counts);recoveries=[]
            for artifact in seal['artifacts']:
                uid=artifact['episode_uid'];event=events[uid];trace=read_zstd_jsonl(Path(artifact['path']));axis,facts=episode_facts[uid]
                if [r['frame'] for r in trace]!=axis or len(axis)!=event['frames']-event['frame']-1:raise ValueError('one frame one decision complete axis')
                for row,fact in zip(trace,facts,strict=True):
                    if row.get('runtime_future_gt_used') is not False or row.get('runtime_gt_used') is not False or row.get('extra_clicks',0)!=0:raise ValueError('causal no-extra-click assertion')
                    for group in groups(fact,(row['frame']-event['frame'])/event['fps']):buckets[group].add(row,fact)
                recoveries.append({'episode_uid':uid,**recovery_events(trace,facts,fps=event['fps'])})
            strata={g:compact_result(b) for g,b in buckets.items()}
            primary=sum(e['target']['frames'] for e in evaluations[case]['episodes']);all_counts=strata['ALL']['counts']
            if primary!=all_counts['frames']:raise ValueError('primary/full-frame denominator mismatch')
            if sum(e['secondary_strict_identity_claim']['actual_strict_correct_selected_frames'] for e in evaluations[case]['episodes'])!=all_counts['strict_correct']:raise ValueError('strict identity A/A mismatch')
            if sum(e['memory']['accepted_writes'] for e in evaluations[case]['episodes'])!=all_counts['writes']:raise ValueError('write count A/A mismatch')
            results[case]={'strata':strata,'episodes':recoveries,'original_evaluation_sha256':sha256(base/'evaluations'/case/f'{sequence}.json')}
        raw_probe={g:{'competitive_target_frames':len(v),'win_rate':float(np.mean(np.asarray(v)>0)),
            'mean_margin':float(np.mean(v)),'median_margin':float(np.median(v)),
            'P10_P25_P75_margin':np.quantile(v,[.1,.25,.75]).tolist()} for g,v in probe.items()}
        write_json(f'failure_analysis/{scope}/{sequence}.json',{'scope':scope,'sequence':sequence,'cases':results,
            'raw_frozen_OSNet_anchor_candidate_probe':raw_probe,'raw_probe_not_learned_memory_ranking':True,
            'runtime_seal_sha256':source_seals,'GT_sha256':gtsha,'candidate_index_sha256':sha256(indexpath),
            'code_sha256':code,'protocol_sha256':sha256(PROTOCOL),'all_compared_runtimes_verified_before_truth':True,
            'no_new_model_candidate_policy_threshold_or_checkpoint_selection':True,'cross_recording_or_physical_absence_claim':False,
            'scientific_success':False,'next_stage_authorized':False})
        completed.append(sequence);print(json.dumps({'failure_breakdown_complete':sequence,'scope':scope,'cases':len(results)}),flush=True)
    print(json.dumps({'scope':scope,'completed_sequences':completed,'pending_sequences':pending}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['VAL','T2'],required=True);parser.add_argument('--sequences',nargs='+');args=parser.parse_args();run(args.scope,args.sequences)
