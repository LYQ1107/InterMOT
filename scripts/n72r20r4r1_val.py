"""Train-side policy freeze and one conditional, historically exposed VAL run."""
from collections import Counter
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_memory_fit import example,RELIABILITY_FEATURES
from scripts.n72r20r4r1_fit import fit_model
from scripts.n72r20r4r1_supervision import prepare_anchors
from scripts.n72r20r3_common import gt_by_frame
from scripts.n72r20r4r1_evaluate import run_runtime,posthoc,baseline_trace,project,EvaluationBatch
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.opportunity_models import ActionValueModel
from sam3_intermot.association.identity_authority import AuthorityConfig,AdapterEnsemble


def select_train_side():
    path=OUT/'val/TRAIN_SIDE_SELECTION.json'
    if path.exists():return read_json(path)
    gate=read_json(OUT/'dev_trackeval/FINAL_SELECTED.json')['gate']
    if not gate['numerical_DEVELOPMENT_gate_pass']:raise ValueError('VAL not authorized without preregistered DEV point signal')
    families={s:read_json(OUT/'authority/frozen_outer'/f'{s}.json') for s in SEQUENCES};joints={s:read_json(OUT/'authority/frozen_joint'/f'{s}.json') for s in SEQUENCES}
    keys=['FAMILY_'+k for k in families[SEQUENCES[0]]['selected']]+[c['name'] for c in joints[SEQUENCES[0]]['cases']];rows=[]
    for key in keys:
        configs=[];delta=[];assa=[];idsw=[];guards=[]
        for s in SEQUENCES:
            if key.startswith('FAMILY_'):
                source=families[s];c=source['selected'][key[7:]];m=source['inner_metrics'][c['name']];base=source['inner_metrics']['BASELINE']
            else:
                source=joints[s];c=next(c for c in source['cases'] if c['name']==key);m=source['metrics'][key];base=source['metrics']['JOINT_BASELINE']
            configs.append({k:v for k,v in c.items() if k!='name'});delta.append(m['HOTA']-base['HOTA']);assa.append(m['AssA']-base['AssA']);idsw.append(m['IDSW']-base['IDSW']);guards.append(m['DetA']>=base['DetA']-.005)
        capacity=sum((0 if c['controller_key'] is None else 3 if c['controller_key'].startswith('C2') else 75 if c['controller_key'].startswith('C3') else 2101 if c['controller_key'].startswith('C6') else 2065)+(33 if c['native_predictor'] else 0)+(33 if c['memory_predictor'] else 0) for c in configs)/8
        rows.append({'family':key,'all_DetA_guards':all(guards),'mean_inner_HOTA_delta':float(np.mean(delta)),'mean_inner_AssA_delta':float(np.mean(assa)),'mean_inner_IDSW_delta':float(np.mean(idsw)),'mean_parameters':capacity,'configs':configs})
    best=max(rows,key=lambda r:(r['all_DetA_guards'],r['mean_inner_HOTA_delta'],r['mean_inner_AssA_delta'],-r['mean_inner_IDSW_delta'],-r['mean_parameters']))
    counts=Counter(json.dumps(c,sort_keys=True) for c in best['configs']);modal=max(counts,key=lambda k:(counts[k],k))
    result={'winner':best['family'],'configuration':json.loads(modal),'all_family_inner_statistics':rows,'source_inner_SHA':{s:sha256(OUT/'authority/frozen_outer'/f'{s}.json') for s in SEQUENCES},
        'no_outer_metrics_used_for_selection':True,'no_VAL_used_for_selection':True,'eligibility_gate':gate,'VAL_closure_protocol_sha256':sha256(OUT/'VAL_CLOSURE_PROTOCOL.json')}
    write_json(path,result);return result


def fit_all_train_native():
    path=OUT/'val/ALL_TRAIN_NATIVE_FIT.json'
    if path.exists():return read_json(path)
    data=[];sources={};digest=hashlib.sha256()
    for s in SEQUENCES:
        frames=load_frames(s);gt=gt_by_frame(DATASET/'train'/s/'gt/gt.txt');anchors,matches=prepare_anchors(s,frames,gt);count=positive=0
        for identity,prepared in sorted(anchors.items()):
            tracker=OpportunityTracker(config=AuthorityConfig(memory='P1',source='raw'),event=prepared['event'],bank=make_bank(),memory_policy=MemoryPolicy('P1'),intervention_policy=InterventionPolicy(source='raw'),audit_hashes=False)
            for payload,rows in frames:
                f=payload['frame'];features=[]
                if f>prepared['event']['event_frame'] and tracker.target_public is not None and rows:
                    state=tracker.states[tracker.target_public];query=tracker.bank.records[tracker.target_public].current_state;raw=np.asarray([np.dot(query,r['feature']) for r in rows])
                    for i,r in enumerate(rows):
                        if state.last_native_tid==int(r['native_tid']) and state.last_native_scope==r.get('native_scope'):features.append((str(r['candidate_uid']),tracker.native_features(state,r,rows,i,raw,query,f)))
                d=tracker.step(rows,f)
                digest.update(json.dumps({'sequence':s,'identity':identity,'frame':f,'assignments':d['assignments'],'memory':d['memory']},sort_keys=True,separators=(',',':')).encode())
                for uid,feature in features:
                    correct=matches[f].get(uid)==identity;data.append(example(s,identity,f,uid,feature,correct));count+=1;positive+=int(correct)
            print(json.dumps({'VAL_fit_train_only':s,'identity':identity}),flush=True)
        sources[s]={'identities':len(anchors),'observations':count,'positive':positive,'negative':count-positive}
    if not all(sum(r[k] for r in sources.values())>0 for k in ('positive','negative')):raise ValueError('unidentifiable all-train native data')
    records=[]
    for seed in SEEDS:
        model,diagnostics=fit_model('C3','L0',data,seed,30);p=ASSETS/'models'/f'VAL_TRAIN8__NATIVE_RELIABILITY__seed{seed}.pt'
        if p.exists():
            r=read_json(p.with_suffix('.json'))
            if sha256(p)!=r['sha256'] or r['training_trace_digest']!=digest.hexdigest():raise ValueError('resumed all-train native lineage differs')
        else:
            check_storage(.02);torch.save({'stage':STAGE,'family':'C3','purpose':'VAL_ALL_TRAIN_NATIVE_RELIABILITY','feature_names':RELIABILITY_FEATURES,'parameters':model.parameter_count,'state_dict':model.state_dict(),'actual_training_sequences':list(SEQUENCES),'forbidden_splits':['val','test'],'seed':seed,'epochs':30,'training_trace_digest':digest.hexdigest()},p)
            r={'path':str(p),'sha256':sha256(p),'seed':seed,'parameters':model.parameter_count,'training_trace_digest':digest.hexdigest(),'diagnostics':diagnostics};write_json(p.with_suffix('.json'),r)
        records.append(r)
    result={'models':records,'sources':sources,'actual_training_sequences':list(SEQUENCES),'native_features_independent_of_Adapter':True,'raw_P1_teacher_same_consensus_assignment_GRU_updates_as_Adapter_P1':True,'GT_labels_attached_only_after_causal_frame_commit':True,'VAL_accessed_for_fit':False,'trace_digest':digest.hexdigest()}
    write_json(path,result);return result


class AllTrainNative:
    def __init__(self,records):
        self.manifest=records;self.models=[]
        for r in records:
            if sha256(Path(r['path']))!=r['sha256']:raise ValueError('VAL native model SHA mismatch')
            c=torch.load(r['path'],map_location='cpu',weights_only=False)
            if c['actual_training_sequences']!=list(SEQUENCES) or c['forbidden_splits']!=['val','test'] or tuple(c['feature_names'])!=RELIABILITY_FEATURES:raise ValueError('VAL native model train lineage mismatch')
            model=ActionValueModel('C3',10);model.load_state_dict(c['state_dict']);model.eval()
            for p in model.parameters():p.requires_grad_(False)
            self.models.append(model)
    def predict(self,features):
        predictions=[m.predict(features) for m in self.models]
        return {k:float(np.mean([r[k] for r in predictions])) for k in predictions[0]}


def freeze():
    path=OUT/'val/FROZEN_POLICY.json'
    if path.exists():return read_json(path)
    selected=select_train_side();case=selected['configuration']
    if case['controller_key'] is not None or not case['native_predictor'] or case['memory']['family']!='P0':raise ValueError('this actual inner winner requires a different train refit; do not silently substitute')
    fitted=fit_all_train_native();source=read_json(R4OUT/'val/FROZEN_POLICY.json');adapters=source['adapter_checkpoints']
    AdapterEnsemble([Path(r['path']) for r in adapters],expected_shas=[r['sha256'] for r in adapters])
    for r in adapters:
        c=torch.load(r['path'],map_location='cpu',weights_only=False)
        actual=c.get('actual_training_sequences',c.get('parameter_fit_sequences'))
        if set(actual)!=set(SEQUENCES):raise ValueError('VAL Adapter not trained only on eight train-development sequences')
    policy={'stage':STAGE,'goal_file':'outputs/N72R20R4R1/FINAL_GOAL.json','development_gate_pass':True,'point_estimate_signal_passed':True,'DEV_CI_excludes_zero':False,
        'configuration':case,'calibration':source['configuration'],'native_models':fitted['models'],'adapter_checkpoints':adapters,'training_sequences':list(SEQUENCES),
        'VAL_used_for_selection_or_training':False,'frozen_before_new_VAL_input_access':True,'VAL_previously_accessed_in_history':True,'test_accessed':False,
        'source_inner_selection_sha256':sha256(OUT/'val/TRAIN_SIDE_SELECTION.json'),'source_all_train_native_fit_sha256':sha256(OUT/'val/ALL_TRAIN_NATIVE_FIT.json'),
        'scientific_PASS_not_implied_by_eligibility':True,'zero_write_memory_safety_pass':False}
    write_json(path,policy);return policy


def run():
    policy=freeze();source=OUT/'val/FROZEN_POLICY.json';policy_sha=sha256(source)
    adapter=AdapterEnsemble([Path(r['path']) for r in policy['adapter_checkpoints']],expected_shas=[r['sha256'] for r in policy['adapter_checkpoints']]);native=AllTrainNative(policy['native_models'])
    ev=events('val');sequences=sorted(ev)
    if len(sequences)!=25:raise ValueError('VAL sequence count differs')
    batch=EvaluationBatch('VAL_FROZEN',sequences,split='val');stats={}
    try:
        for s in sequences:
            frames=load_frames(s,'val');encoded=project(adapter,frames);base,_=baseline_trace(s,'val');stats[s]={}
            for name,predictor in [('BASELINE',None),('TREATMENT',native)]:
                p=InterventionPolicy(**policy['configuration']['policy']) if name=='TREATMENT' else InterventionPolicy(top_k=policy['configuration']['policy']['top_k'])
                trace,profile=run_runtime(s,frames,ev[s],adapter,p,MemoryPolicy(),native_predictor=predictor,encoded=encoded,calibration_config=policy['calibration'])
                batch.add(name,s,trace,profile);stats[s][name]=posthoc(s,frames,ev[s],trace,base,split='val')
            if sha256(source)!=policy_sha:raise RuntimeError('VAL policy changed during evaluation')
            print(json.dumps({'frozen_VAL_completed':s,'N01':stats[s]['TREATMENT']['target']['N01'],'N10':stats[s]['TREATMENT']['target']['N10']}),flush=True)
        evaluated=batch.evaluate()
    finally:batch.close()
    m=evaluated['metrics'];keys=('HOTA','AssA','DetA','IDF1','IDSW');indices=np.random.default_rng(720401).integers(0,25,size=(2000,25));bootstrap={}
    for k in keys:
        delta=np.asarray([m['TREATMENT']['per_sequence'][s][k]-m['BASELINE']['per_sequence'][s][k] for s in sequences]);bootstrap[k]={'macro_delta':float(delta.mean()),'CI95':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist(),'official_combined_delta':m['TREATMENT'][k]-m['BASELINE'][k]}
    result={'status':'COMPLETE_FROZEN_ONE_PASS','metrics':m,'paired_bootstrap':bootstrap,'statistics':stats,'policy_sha256':policy_sha,'VAL_tuning':False,'test_accessed':False,'historically_exposed_not_virgin':True}
    write_json(OUT/'val/RESULT.json',result);write_json(OUT/'val/BASELINE.json',m['BASELINE']);write_json(OUT/'val/TREATMENT.json',m['TREATMENT']);write_json(OUT/'val/DELTA.json',bootstrap)
    return result


if __name__=='__main__':torch.set_num_threads(1);run()
