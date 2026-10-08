"""Posthoc native-only scope/OOD control; not a heldout policy selector."""
from dataclasses import replace
from collections import Counter
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_inner import reliability_objects
from scripts.n72r20r4r1_evaluate import run_runtime,posthoc,baseline_trace,project,EvaluationBatch
from scripts.n72r20r4r1_memory_fit import RELIABILITY_FEATURES
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.identity_authority import AuthorityConfig


def run():
    done=OUT/'opportunity/NATIVE_SCOPE_POSTHOC_MECHANISM.json'
    if done.exists():return read_json(done)
    stats={};ood={};batch=EvaluationBatch('posthoc_native_scope',SEQUENCES)
    try:
        for s in SEQUENCES:
            frozen=read_json(OUT/'authority/frozen_outer'/f'{s}.json');formal=read_json(OUT/'authority/outer'/f'{s}.json')
            if formal['status']!='COMPLETE':raise ValueError('mechanism diagnostic before completed formal evaluation')
            adapter=strict_ensemble(s);frames=load_frames(s);event=events()[s];encoded=project(adapter,frames);base,_=baseline_trace(s)
            p=InterventionPolicy(**frozen['selected']['NATIVE_IDENTITY']['policy'])
            policy=replace(p,native_discount=0.,discount_scope='target')
            trace,profile=run_runtime(s,frames,event,adapter,policy,MemoryPolicy(),encoded=encoded,calibration_config=frozen['calibration'])
            batch.add('POSTHOC_TARGET_NATIVE_ZERO',s,trace,profile);stats[s]=posthoc(s,frames,event,trace,base)
            # Default target scope cannot act before initialization because
            # no public state has the target key before the click.
            if any(d['outputs']!=b['outputs'] for d,b in zip(trace[:event['event_frame']+1],base[:event['event_frame']+1])):raise RuntimeError('diagnostic changed pre-click outputs')
            _,predictor=reliability_objects(s);cfg=AuthorityConfig(**frozen['calibration'])
            tracker=OpportunityTracker(config=cfg,event=event,adapter=adapter,bank=make_bank(),intervention_policy=p,memory_policy=MemoryPolicy(),native_predictor=predictor)
            native_trace=[];counts=np.zeros(10,dtype=int);observations=0;probs=[]
            mean=predictor.models[0].mean.detach().numpy();scale=predictor.models[0].scale.detach().numpy()
            for payload,rows in frames:
                f=int(payload['frame'])
                if f>event['event_frame'] and tracker.target_public is not None and rows:
                    state=tracker.states[tracker.target_public];query=tracker.bank.records[tracker.target_public].current_state
                    raw=np.asarray([float(np.dot(query,r['feature'])) for r in rows])
                    for i,r in enumerate(rows):
                        if state.last_native_tid==int(r['native_tid']) and state.last_native_scope==r.get('native_scope'):
                            feature=tracker.native_features(state,r,rows,i,raw,query,f);z=(np.asarray(feature)-mean)/scale
                            counts+=abs(z)>5;observations+=1;probs.append(predictor.predict(feature)['beneficial'])
                d=tracker.step(rows,f,encoded_candidates=encoded[f],collect_proposals=True);native_trace.append(d)
            manifest=read_json(OUT/'evaluations/outer'/f'{s}.json')['manifests']['NATIVE_IDENTITY'][s]
            digest=hashlib.sha256(trajectory_text(native_trace).encode()).hexdigest()
            if digest!=manifest['trajectory_sha256']:raise RuntimeError('native-only observational replay differs from frozen formal MOT')
            ood[s]={'actual_native_active_observations':observations,'abs_standardized_feature_gt5_counts':dict(zip(RELIABILITY_FEATURES,counts.tolist())),
                'abs_standardized_feature_gt5_fraction':dict(zip(RELIABILITY_FEATURES,(counts/max(1,observations)).tolist())),
                'native_reliability_probability_quantiles':np.quantile(probs,[0,.1,.5,.9,1]).tolist() if probs else None,
                'fitted_on_actual_P1_states_deployed_on_P0':True,'formal_native_MOT_SHA_reproduced':True}
            print(json.dumps({'posthoc_native_scope':s,'observations':observations,'N01':stats[s]['target']['N01'],'N10':stats[s]['target']['N10']}),flush=True)
        evaluated=batch.evaluate()
    finally:batch.close()
    result={'posthoc_only':True,'not_used_for_selection_or_scientific_gate':True,'same_preregistered_formal_outputs_unchanged':True,
        'constant_target_zero_metrics':evaluated['metrics']['POSTHOC_TARGET_NATIVE_ZERO'],'statistics':stats,'native_classifier_state_shift':ood,
        'protocol_sha256':sha256(OUT/'POSTHOC_NATIVE_SCOPE_PROTOCOL.json')}
    write_json(done,result);return result


if __name__=='__main__':torch.set_num_threads(1);run()
