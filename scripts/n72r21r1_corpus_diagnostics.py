"""Offline E/F/C6 audits after complete real joint-runtime sealing."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import numpy as np
from scripts.n72r21r1_common import ROOT,OUT,TRAIN,read_json,write_json,sha256
from scripts.n72r21r1_label_corpus import verify_registered
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.safe_intervention_events import assignment_map,identity_outcome
from sam3_intermot.evaluation.joint_trajectory_labels import memory_safety
from sam3_intermot.evaluation.one_click_protocol import ratio,open_set_metrics
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES


def one_episode(seal,trace,matched,gt):
    event=seal['event']; target=seal['_offline_target']; public=trace[event['frame']]['target_public_id']
    count=Counter(); features=[]; ap=[]; available=[]; rank_correct=[]; cp=[]; known_correct=[]
    effective=[]; claim_risk=[];density=defaultdict(Counter)
    for r in trace[event['frame']+1:]:
        f=r['frame']; truth=matched[f]; decision=r['identity_decision']
        has_target=any(q==target for q in truth.values());visible=any(a['identity']==target for a in gt.get(f,[]))
        uid=assignment_map(r).get(public);outcome=identity_outcome(uid,truth,target)
        count['frames']+=1;count['target_visible']+=visible;count['candidate_available']+=has_target
        count['target_selected']+=uid is not None;count['correct_committed']+=outcome=='TARGET'
        count['verified_other_committed']+=outcome=='VERIFIED_OTHER';count['UNKNOWN_committed']+=outcome=='UNKNOWN_UNMATCHED'
        count['NONE_committed']+=uid is None;count['unavailable_false_accept']+=not has_target and uid is not None
        count['unavailable_frames']+=not has_target
        rank_uid=decision['rank1_candidate_uid'];rank=identity_outcome(rank_uid,truth,target)
        count['rank1_correct_given_available']+=has_target and rank=='TARGET'
        count['rank1_UNKNOWN']+=rank=='UNKNOWN_UNMATCHED'
        count['correct_proposed_observations']+=identity_outcome(decision['proposed_candidate_uid'],truth,target)=='TARGET'
        count['actor_NONE']+=decision['proposed_candidate_uid'] is None
        ap.append(float(decision['candidate_available_probability']));available.append(has_target);rank_correct.append(rank=='TARGET')
        if rank!='UNKNOWN_UNMATCHED':cp.append(r['cached_rank1_joint_probability']);known_correct.append(rank=='TARGET')
        if r['joint_identity_memory_write']:
            assert r['joint_memory_write_candidate_uid']==uid
            count['write_'+outcome]+=1
        a=r['authority'];count['effective_interventions']+=bool(a.get('effective_assignment_change'))
        if a['features'] is not None:features.append([a['features'][k] for k in FEATURE_NAMES])
        if a['features'] is not None:
            effective.append(bool(a.get('effective_assignment_change')))
            own=truth.get(a['own_KEEP_uid'])==target;actual=outcome=='TARGET'
            claim_risk.append(bool(own and not actual))
        n=len(r['outputs']);group='0-4' if n<=4 else '5-8' if n<=8 else '9+'
        density[group]['frames']+=1;density[group]['correct']+=outcome=='TARGET';density[group]['candidate_available']+=has_target
    scores=np.array(features,float)
    return {'counts':dict(count),'candidate_coverage_visible':ratio(count['candidate_available'],count['target_visible']),
        'correct_UID_recall_visible':ratio(count['correct_committed'],count['target_visible']),
        'correct_UID_recall_given_candidate':ratio(count['correct_committed'],count['candidate_available']),
        'Rank1_given_available':ratio(count['rank1_correct_given_available'],count['candidate_available']),
        'actual_unavailable_false_accept_rate':ratio(count['unavailable_false_accept'],count['unavailable_frames']),
        'candidate_relative_NONE_not_physical_absence':True,'UNKNOWN_separate_not_verified_other':True,
        'availability_curve':open_set_metrics(ap,available,rank_correct),
        'known_identity_claim_curve':open_set_metrics(cp,known_correct,known_correct),
        'identity_claim_UNKNOWN_excluded':count['rank1_UNKNOWN'],
        'feature_distribution':{k:{'mean':float(scores[:,i].mean()),'std':float(scores[:,i].std()),
            'P10':float(np.quantile(scores[:,i],.1)),'P90':float(np.quantile(scores[:,i],.9))} for i,k in enumerate(FEATURE_NAMES)},
        'density_groups':{k:dict(v) for k,v in density.items()},
        'memory':memory_safety(count['write_TARGET'],count['write_VERIFIED_OTHER'],count['write_UNKNOWN_UNMATCHED'],count['correct_proposed_observations']) | {
            'denominator':'Correct actor-proposed current candidate observations, including those rejected by full joint association.',
            'correct_write_retention_given_correct_committed':ratio(count['write_TARGET'],count['correct_committed']),
            'correct_write_retention_given_candidate_available':ratio(count['write_TARGET'],count['candidate_available'])},
        'actual_direct_harm_given_intervention':ratio(sum(a and b for a,b in zip(effective,claim_risk,strict=True)),sum(effective)),
        'initial_clicked_UID_verified_target':matched[event['frame']].get(event['clicked_candidate_uid'])==target,
        'actual_generating_full_MOT_trace_not_teacher_forced':True}


def run():
    inputs,seals,failed=verify_registered()  # Complete all hashes before future GT.
    identities={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']}
    episodes={}; sources=defaultdict(list)
    for sequence in sorted({s['event']['sequence'] for _,s in seals}):
        frames,index_SHA=checked_frames(sequence);gt=dancetrack_annotations(TRAIN/sequence)
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        for seal_path,s in [(p,s) for p,s in seals if s['event']['sequence']==sequence]:
            source=s['source_distribution'];key=source+'/'+s['event']['episode_uid']
            trace=read_zstd_jsonl(Path(next(a['path'] for a in s['artifacts'] if a['kind']=='trace')))
            assert [r['frame'] for r in trace]==list(range(len(frames)))
            result=one_episode({**s,'_offline_target':identities[s['event']['episode_uid']]},trace,matched,gt)
            episodes[key]={'sequence':sequence,'role':s['event']['diagnostic_role'],'source':source,
                'runtime_seal_SHA':sha256(seal_path),'candidate_index_SHA':index_SHA,**result}
            sources[source].append(result)
        print(json.dumps({'offline_corpus_diagnostic_complete':sequence}),flush=True)
    aggregate={}
    for source,rows in sources.items():
        valid=[r for r in rows if r['initial_clicked_UID_verified_target']];counts=Counter()
        for r in valid:counts.update(r['counts'])
        aggregate[source]={'valid_initial_episode_count':len(valid),'invalid_initial_episode_count':len(rows)-len(valid),
            'counts':dict(counts),'Rank1_given_available':ratio(counts['rank1_correct_given_available'],counts['candidate_available']),
            'correct_UID_recall_given_candidate':ratio(counts['correct_committed'],counts['candidate_available']),
            'actual_unavailable_false_accept_rate':ratio(counts['unavailable_false_accept'],counts['unavailable_frames']),
            'memory':memory_safety(counts['write_TARGET'],counts['write_VERIFIED_OTHER'],counts['write_UNKNOWN_UNMATCHED'],counts['correct_proposed_observations']),
            'frame_weighted_diagnostic_not_sequence_cluster_estimate':True}
    manifest=read_json(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json');raw_rows=read_zstd_jsonl(Path(manifest['path']))
    trajectory=Counter()
    for r in raw_rows:
        if not r['initial_clicked_UID_verified_target']:continue
        current=r['raw_trajectory_labels']['time_zero'];future=r['raw_trajectory_labels']['future']['H100']
        if not future['complete']:continue
        trajectory['complete_action_rows']+=1
        trajectory['current_target_gain_but_future_risk']+=bool(current['N01'] and future['risk_label'])
        trajectory['current_no_target_gain_but_future_benefit']+=bool(not current['N01'] and future['benefit_label'])
        trajectory['current_target_gain']+=bool(current['N01'])
        trajectory['future_benefit']+=future['benefit_label'];trajectory['future_risk']+=future['risk_label']
    write_json('diagnostics/JOINT_CORPUS_E_F_STATE_AUDIT.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_OFFLINE_CORPUS_DIAGNOSTICS',
        'source_evaluator_SHA':sha256(Path(__file__)),'episodes':episodes,'state_sources':aggregate,'trajectory_reward_disagreement':dict(trajectory),
        'real_rollouts':len(seals),'failed_initializations':failed,'supervision_SHA':manifest['sha256'],
        'candidate_relative_NONE_not_physical_absence':True,'known_identity_UNKNOWN_excluded_separately':True,
        'delayed_source_actual_delay_frames':1,'delayed_source_is_not_distinct_delayed_write_experiment':True,
        'historical_development_not_confirmation':True,'no_VAL_test_used':True,'not_scientific_success':True})


if __name__=='__main__':run()
