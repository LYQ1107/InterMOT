"""Sealed outer-only diagnostics; never a checkpoint/approval selector."""
from collections import Counter,defaultdict
from scipy.stats import rankdata
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_fit import causal_features,decode_sequence,ValueEnsemble


def auc(labels,scores):
    y=np.asarray(labels,dtype=bool);n=int(y.sum());m=len(y)-n
    return float((rankdata(scores)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def run_fold(heldout):
    path=OUT/'opportunity/posthoc_outer'/f'{heldout}.json'
    if path.exists():return read_json(path)
    frozen=read_json(OUT/'authority/frozen_outer'/f'{heldout}.json');adapter=strict_ensemble(heldout);coverage={};rows=None
    for k in (1,3,5):
        examples,audit=causal_features(heldout,adapter,k=k,heldout=heldout,role='posthoc_outer');coverage[str(k)]=audit
        if k==frozen['proposal_K']:rows=examples
    er,actions,summary=decode_sequence(heldout);controllers={};training=read_json(OUT/'authority/training'/f'{heldout}.json')
    for key,model_record in training['models'].items():
        controller=ValueEnsemble(model_record['selected'],heldout);pred=[controller.predict(r['features']) for r in rows];policy=frozen['selected'][key]['policy']
        approved=[i for i,p in enumerate(pred) if p['beneficial']>=policy['beneficial_min'] and p['harmful']<=policy['harmful_max'] and p['value']>policy['value_min']]
        groups=defaultdict(list)
        for i in approved:groups[tuple(rows[i]['group_key'])].append(i)
        chosen=[max(indices,key=lambda i:pred[i]['value']) for indices in groups.values()]
        labels=[r['H5']['beneficial'] for r in rows]
        controllers[key]={'baseline_state_isolated_counterfactual_diagnostic_not_full_online_replay':True,'examples':len(rows),
            'beneficial_AUC':auc(labels,[p['beneficial'] for p in pred]),'harm_AUC':auc([r['H5']['harmful'] for r in rows],[p['harmful'] for p in pred]),
            'value_MAE_clipped3':float(np.mean([abs(p['value']-np.clip(r['H5']['value'],-3,3)) for r,p in zip(rows,pred)])),
            'approved_actions':len(approved),'selected_action_groups':len(chosen),'selected_naturally_beneficial':sum(rows[i]['H5']['beneficial'] for i in chosen),
            'selected_harmful':sum(rows[i]['H5']['harmful'] for i in chosen),'selected_actual_mean_H5_value':float(np.mean([rows[i]['H5']['value'] for i in chosen])) if chosen else None,
            'natural_beneficial_proposal_recall':sum(rows[i]['H5']['beneficial'] for i in chosen)/max(1,sum(labels)),
            'thresholds_frozen_from_inner':policy,'no_outer_selection':True}
    beneficial=[r for records in actions.values() for r in records if r['H5']['beneficial']]
    positive_parts={'all_natural_beneficial':len(beneficial),'action_families':dict(Counter(r['action']['family'] for r in beneficial)),
        'correct_available_identity_positive':sum(r['H5']['raw_frames'][0]['target_available'] and r['H5']['raw_frames'][0]['strict_identity_correct'] for r in beneficial),
        'valid_NONE_availability_positive':sum(not r['H5']['raw_frames'][0]['target_available'] and r['action']['candidate_uid'] is None for r in beneficial),
        'remaining_positive_not_immediate_identity_recovery':sum(not (r['H5']['raw_frames'][0]['target_available'] and r['H5']['raw_frames'][0]['strict_identity_correct']) and not (not r['H5']['raw_frames'][0]['target_available'] and r['action']['candidate_uid'] is None) for r in beneficial)}
    event_rows=defaultdict(list)
    for r in rows:event_rows[tuple(r['group_key'])].append(r)
    rank_audit={'competitive_events':0,'misleading_adapter_rank_events':0,'correct_candidate_covered_in_runtime_union':0,'wrong_baseline_and_candidate_available':0}
    for (identity,frame),event in er.items():
        available=event['oracle_correct_uids']
        if not available:continue
        group=event_rows[(heldout,identity,frame)];correct=[r for r in group if r['action']['candidate_uid'] in available]
        wrong=[r for r in group if r['action']['candidate_uid'] is not None and r['action']['candidate_uid'] not in available]
        rank_audit['correct_candidate_covered_in_runtime_union']+=int(bool(correct))
        rank_audit['wrong_baseline_and_candidate_available']+=int(event['kind']!='T0_ALREADY_CORRECT')
        if correct and wrong:
            rank_audit['competitive_events']+=1
            rank_audit['misleading_adapter_rank_events']+=int(max(r['features'][0] for r in correct)<=max(r['features'][0] for r in wrong))
    result={'outer':heldout,'strict_adapter_manifest':adapter.manifest,'coverage':coverage,'controller_value_diagnostics':controllers,
        'natural_positive_partition':positive_parts,'T6_rank_audit_runtime_proposals_only':rank_audit,'full_corpus_source_SHA':summary['artifact']['sha256'],
        'created_only_after_frozen_outer_complete':True,'never_used_for_training_or_policy_selection':True}
    write_json(path,result);return result


if __name__=='__main__':
    import argparse
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--heldouts',nargs='+',default=list(SEQUENCES));args=p.parse_args()
    for s in args.heldouts:run_fold(s)
