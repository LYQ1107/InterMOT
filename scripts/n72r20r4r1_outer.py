"""Strict frozen outer replay; no outer observations can select a policy."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_inner import controller_objects,reliability_objects,metric_key
from scripts.n72r20r4r1_evaluate import run_runtime,posthoc,baseline_trace,project,EvaluationBatch
from scripts.n72r20r4r1_fit import ValueEnsemble
from sam3_intermot.association.opportunity_tracker import InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy


def freeze_selection(heldout):
    path=OUT/'authority/frozen_final'/f'{heldout}.json'
    if path.exists():return read_json(path)
    family_path=OUT/'authority/frozen_outer'/f'{heldout}.json';joint_path=OUT/'authority/frozen_joint'/f'{heldout}.json'
    f=read_json(family_path);j=read_json(joint_path)
    if sha256(family_path)!=j['source_frozen_family_sha256']:raise ValueError('family freeze changed')
    # Include every preregistered family, native-only control and joint winner.
    # Same fixed inner metric criterion; no new grid and no outer evaluation.
    candidates=[]
    for key,case in f['selected'].items():
        if key.startswith('P'):continue
        candidates.append((case,metric_key(case['name'],f['inner_metrics'],f['inner_metrics']['BASELINE'],f['inner_statistics']), 'family'))
    for case in j['cases']:
        candidates.append((case,metric_key(case['name'],j['metrics'],j['metrics']['JOINT_BASELINE'],j['statistics']), 'joint'))
    selected,key,source=max(candidates,key=lambda item:item[1])
    result={'outer':heldout,'inner':f['inner'],'fit':f['fit'],'selected':selected,'selection_key':key,'selected_source':source,
        'source_family_sha256':sha256(family_path),'source_joint_sha256':sha256(joint_path),
        'selection_rule_file':'outputs/N72R20R4R1/FINAL_SELECTION_PROTOCOL.json','selection_rule_sha256':sha256(OUT/'FINAL_SELECTION_PROTOCOL.json'),
        'frozen_before_outer':True,'outer_labels_or_metrics_used':False,'VAL_used':False}
    write_json(path,result);return result


def cases_for_fold(f,j,final):
    cases=[]
    def add(name,case,**changes):
        c=deepcopy(case);c.update(changes);c['name']=name;cases.append(c)
    selected=f['selected'];base=selected['C0'];fixed=selected['C1']
    add('BASELINE',base)
    add('ADAPTER_P0',base)
    for p in ('P1','P2','P3','P4','P5','P6'):add('MEMORY_'+p,selected[p])
    for key in ('C1','C2_L0','C3_L0','C4_L0','C5_L1','C6_L0','C6_L1','C6_L2','C6_L3'):add(key,selected[key])
    add('NATIVE_GLOBAL',selected['NATIVE_GLOBAL'])
    if 'NATIVE_IDENTITY' in selected:add('NATIVE_IDENTITY',selected['NATIVE_IDENTITY'])
    add('POSITIVE_SOFT_CONTROL',base,policy={**base['policy'],'positive_discount':0.})
    # Strength, K and approvals stay the inner-selected fixed control values.
    add('RAW_IDENTITY',fixed,policy={**fixed['policy'],'source':'raw'})
    add('SHUFFLED_IDENTITY',fixed,policy={**fixed['policy'],'source':'shuffled'})
    for c in j['cases']:
        if c['name']!='JOINT_BASELINE':add('JOINT_'+c['name'],c)
    add('FINAL_SELECTED',final['selected'])
    for key in ('C6_L0','C5_L1'):
        for seed in SEEDS:add(f'{key}_SEED_{seed}',selected[key],single_seed=seed)
    return cases


def run_fold(heldout):
    done=OUT/'authority/outer'/f'{heldout}.json'
    if done.exists():return read_json(done)
    check_storage(.1);final=freeze_selection(heldout)
    family_path=OUT/'authority/frozen_outer'/f'{heldout}.json';joint_path=OUT/'authority/frozen_joint'/f'{heldout}.json'
    f=read_json(family_path);j=read_json(joint_path)
    if sha256(family_path)!=final['source_family_sha256'] or sha256(joint_path)!=final['source_joint_sha256']:raise ValueError('frozen policy changed')
    training=read_json(OUT/'authority/training'/f'{heldout}.json');controllers=controller_objects(heldout,training)
    memory_predictor,native_predictor=reliability_objects(heldout)
    cases=cases_for_fold(f,j,final)
    write_json(OUT/'authority/outer_plan'/f'{heldout}.json',{'cases':cases,'final_freeze_sha256':sha256(OUT/'authority/frozen_final'/f'{heldout}.json'),
        'frozen_before_outer_input_open':True,'models_trained_only_on':f['fit'],'GT_free_runtime':True})
    adapter=strict_ensemble(heldout);frames=load_frames(heldout);event=events()[heldout];base,_=baseline_trace(heldout);encoded=project(adapter,frames)
    stats={};batch=EvaluationBatch('outer/'+heldout,[heldout]);cache={}
    try:
        for case in cases:
            name=case['name'];key=json.dumps({k:v for k,v in case.items() if k!='name'},sort_keys=True)
            if key in cache:
                cached=cache[key];batch.names.append(name);batch.manifests[name]=deepcopy(batch.manifests[cached])
                for manifest in batch.manifests[name].values():manifest['name']=name;manifest['exact_policy_alias_of']=cached
                source=batch.root/'trackers'/cached/'data'/f'{heldout}.txt';target=batch.root/'trackers'/name/'data'/f'{heldout}.txt'
                target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
                stats[name]=deepcopy(stats[cached]);continue
            controller=controllers.get(case['controller_key'])
            if 'single_seed' in case:
                records=[r for r in training['models'][case['controller_key']]['selected'] if r['seed']==case['single_seed']]
                if len(records)!=1:raise ValueError('individual seed checkpoint missing')
                controller=ValueEnsemble(records,heldout)
            trace,profile=run_runtime(heldout,frames,event,adapter,InterventionPolicy(**case['policy']),MemoryPolicy(**case['memory']),
                controller=controller,memory_predictor=memory_predictor if case['memory_predictor'] else None,
                native_predictor=native_predictor if case['native_predictor'] else None,encoded=encoded,calibration_config=f['calibration'])
            batch.add(name,heldout,trace,profile);stats[name]=posthoc(heldout,frames,event,trace,base);cache[key]=name
            print(json.dumps({'outer_fold':heldout,'case':name,'seconds':profile['seconds'],'N01':stats[name]['target']['N01'],'N10':stats[name]['target']['N10']}),flush=True)
        evaluated=batch.evaluate()
    finally:batch.close()
    result={'outer':heldout,'fit':f['fit'],'inner':f['inner'],'statistics':stats,'metrics':evaluated['metrics'],
        'frozen_final_sha256':sha256(OUT/'authority/frozen_final'/f'{heldout}.json'),'policy_selection_uses_outer':False,
        'evaluation_file':str(OUT/'evaluations/outer'/f'{heldout}.json'),'complete_non_target_track_outputs':True,'status':'COMPLETE'}
    write_json(done,result);return result


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--heldouts',nargs='+',default=list(SEQUENCES));args=p.parse_args()
    if not set(args.heldouts)<=set(SEQUENCES):raise ValueError('unregistered outer')
    for s in args.heldouts:run_fold(s)
