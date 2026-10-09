"""Prospective current-UID/NONE/UNKNOWN fitting; INNER-only selective point."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.open_set_verifier import OpenSetVerifierHead,predictions,choose,simple_predictions

PROTOCOL=OUT/'protocol/CURRENT_AXIS_VERIFIER_FITS_V1.json'
CODE=['scripts/n72r21r1_train_open_set.py','sam3_intermot/one_click/open_set_verifier.py',
      'sam3_intermot/one_click/intervention_features.py','sam3_intermot/one_click/joint_intervention_primitives.py']
SIMPLE=['E0_FROZEN','E1_CALIBRATED','E2_RELATIVE_NONE','E3_ANCHOR','E4_BASE','E5_TEMPORAL']


def freeze():
    path=OUT/'availability/current_axis_v1/SUPERVISION_MANIFEST.json';m=read_json(path)
    assert sha256(m['path'])==m['sha256'] and m['all22_source_seals_verified_before_truth_opened']
    write_json('protocol/CURRENT_AXIS_VERIFIER_FITS_V1.json',{'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_new_fits_and_full_policy_effects':True,'source_code_SHA':{p:sha256(ROOT/p) for p in CODE},
        'supervision_manifest_path':str(path),'supervision_manifest_SHA':sha256(path),'data_SHA':m['sha256'],
        'families':['SCALAR','LOGISTIC','MLP'],'seeds':[72111,72112,72113],'simple_controls':SIMPLE,
        'FIT_sequences':['dancetrack0023','dancetrack0024','dancetrack0039','dancetrack0057','dancetrack0062','dancetrack0072'],
        'INNER_sequences':['dancetrack0002'],'exposed_diagnostic_only':['dancetrack0001'],
        'training':{'max_epochs':60,'patience':8,'batch_size':256,'learning_rate':.001,'weight_decay':.0001,'max_seconds_per_fit':180},
        'loss':'Inverse FIT class-frequency correctness3class CE + physical candidate-availability BCE; equal FIT scenes and then equal frame groups; UNKNOWN separate class',
        'normalization':'FIT-only mean/std, floor.05','temperatures':[.5,1.,2.],
        'probability_cutoffs':[.5,.8,.9,.95,.98],'margin_cutoffs':[0.,.05,.15],
        'unknown_max':.02,'current_claim_risk_max':.02,'min_INNER_accepted_groups':3,
        'selection':'INNER temperature minimum current3class NLL, then max correct target recall with empirical claim-risk<=2%, >=3 selected groups; ties lower error then higher cutoff. No qualifying point => CALIBRATION_ABSTAIN.',
        'diagnostic_operating_point':{'probability_min':.5,'margin_min':0.,'unknown_max':.5,'status':'UNCONSTRAINED_DIAGNOSTIC_NOT_SELECTED_OR_DEPLOYABLE'},
        'runtime_global_regret_max':.2,'runtime_owner_protection':'No other-owner or displaced-public overrides; applies to all E0-E6 equally; unchanged full exact solver',
        'runtime_memory':'Frozen ACIB P0 immutable anchor, no accepted bank writes; only actually committed feedback',
        'runtime_full_scope':['dancetrack0001','dancetrack0002'],'runtime_own_history':'Fresh rollout per policy; C0-source to treatment-state shift explicitly reported',
        'not_trajectory_safety_training':True,'no_VAL_TEST_fresh_confirmation':True,'threads_per_worker':1,'max_CPU_workers':2,
        'reserve_GiB':60,'next_stage_authorized':False})


def data():
    p=read_json(PROTOCOL);manifest=read_json(p['supervision_manifest_path'])
    assert sha256(p['supervision_manifest_path'])==p['supervision_manifest_SHA'] and sha256(manifest['path'])==p['data_SHA']
    rows=read_zstd_jsonl(Path(manifest['path']));fit=[];inner=[];excluded=Counter()
    for r in rows:
        if r['role'] not in ('FIT','INNER'):excluded[r['role']]+=1;continue
        assert r['sequence'] in p[r['role']+'_sequences'] and r['current_GT_labels_not_features'] and r['UNKNOWN_not_verified_negative']
        assert r['axis'][-1]['candidate_uid'] is None
        for a in r['axis']:
            assert len(a['feature_vector'])==32 and np.isfinite(a['feature_vector']).all()
            if a['class']==2:assert a['current_outcome']=='UNKNOWN_UNMATCHED' and not a['verified_identity_negative']
        (fit if r['role']=='FIT' else inner).append(r)
    assert sorted({r['sequence'] for r in fit})==p['FIT_sequences'] and sorted({r['sequence'] for r in inner})==p['INNER_sequences']
    return p,fit,inner,dict(excluded)


def flat(groups):
    seq=Counter(r['sequence'] for r in groups);x=[];c=[];a=[];w=[];spans=[]
    for r in groups:
        start=len(x)
        for row in r['axis']:
            x.append(row['feature_vector']);c.append(row['class']);a.append(float(r['target_candidate_available_label']))
            w.append(1/seq[r['sequence']]/len(r['axis']))
        spans.append((start,len(x)))
    weights=np.array(w,np.float32);weights/=weights.sum()
    return np.array(x,np.float32),torch.tensor(c),torch.tensor(a),weights,spans


def operating_points(groups,predicted,p,simple_family=None):
    result=[]
    for cutoff in p['probability_cutoffs']:
        for margin in p['margin_cutoffs']:
            selection={'status':'CALIBRATED_CURRENT_IDENTITY_DIAGNOSTIC','probability_min':cutoff,
                'margin_min':margin,'unknown_max':p['unknown_max'],'temperature':1.,'global_regret_max':p['runtime_global_regret_max'],
                'simple_family':simple_family}
            c=Counter();c['groups']=len(groups)
            for r,pred in zip(groups,predicted,strict=True):
                choice=choose(r['axis'],pred,selection);row=r['axis'][choice['axis_index']]
                c['available_groups']+=bool(r['target_candidate_available_label'])
                c['verified_other_candidate_rows']+=sum(a['current_outcome']=='VERIFIED_OTHER' for a in r['axis'])
                if not choice['accepted']:continue
                c['selected_groups']+=1;c['correct_claims']+=row['class']==0
                c['wrong_or_UNKNOWN_claims']+=row['class']!=0
                c['UNKNOWN_claims']+=row['class']==2;c['verified_other_claims']+=row['current_outcome']=='VERIFIED_OTHER'
                c['correct_target_claims']+=row['current_outcome']=='TARGET';c['correct_NONE_claims']+=row['current_outcome']=='CORRECT_NONE'
                c['incorrect_NONE_claims']+=row['current_outcome']=='INCORRECT_NONE'
            n=c['selected_groups'];risk=c['wrong_or_UNKNOWN_claims']/n if n else None
            result.append({**selection,'counts':dict(c),'claim_risk':risk,'coverage':n/len(groups),
                'target_recall_given_available':c['correct_target_claims']/c['available_groups'] if c['available_groups'] else None,
                'verified_other_row_FPR':c['verified_other_claims']/c['verified_other_candidate_rows'] if c['verified_other_candidate_rows'] else None,
                'safe_nonvacuous':n>=p['min_INNER_accepted_groups'] and risk is not None and risk<=p['current_claim_risk_max'],
                'empirical_historical_INNER_not_confidence_bound_or_future_safety':True})
    good=[r for r in result if r['safe_nonvacuous']]
    selected=sorted(good,key=lambda r:(-r['counts'].get('correct_target_claims',0),r['claim_risk'],-r['probability_min'],-r['margin_min']))[0] if good else {
        'status':'CALIBRATION_ABSTAIN','probability_min':1.01,'margin_min':1.,'unknown_max':p['unknown_max'],
        'temperature':1.,'global_regret_max':p['runtime_global_regret_max'],'reason':'NO_CURRENT_IDENTITY_SAFE_NONVACUOUS_INNER_POINT'}
    return selected,result


def run(family,seed):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    p,fit,inner,excluded=data();assert family in p['families'] and seed in p['seeds']
    assert all(sha256(ROOT/k)==v for k,v in p['source_code_SHA'].items())
    uid=family+'__seed'+str(seed);record_path=OUT/'training/current_axis_verifier'/ (uid+'.json')
    if record_path.exists():
        r=read_json(record_path);assert r['protocol_SHA']==sha256(PROTOCOL) and sha256(r['checkpoint_path'])==r['checkpoint_SHA'];return
    destination=ASSETS/'training/current_axis_verifier'/uid
    if destination.exists():raise FileExistsError('preserve partial verifier fit')
    storage(32<<20);destination.mkdir(parents=True)
    raw,c,a,w,_=flat(fit);iraw,ic,ia,iw,spans=flat(inner)
    mean=(raw*w[:,None]).sum(0);std=np.maximum(np.sqrt(((raw-mean)**2*w[:,None]).sum(0)),.05)
    x=torch.from_numpy((raw-mean)/std);ix=torch.from_numpy((iraw-mean)/std)
    weights=torch.from_numpy(w*len(w));iweights=torch.from_numpy(iw*len(iw))
    freq=torch.bincount(c,minlength=3).float();cw=freq.sum()/freq.clamp_min(1)/3
    model=OpenSetVerifierHead(family);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=p['training']['learning_rate'],weight_decay=p['training']['weight_decay'])
    def loss(output,classes,available):
        return F.cross_entropy(output[:,:3],classes,weight=cw,reduction='none')+F.binary_cross_entropy_with_logits(output[:,3],available,reduction='none')
    began=time.monotonic();best=None;best_loss=float('inf');steps=0;gradients=0;epochs=[];patience=0;max_grad=0.
    log_path=destination/'epochs.jsonl'
    with log_path.open('x') as log:
        for epoch in range(1,p['training']['max_epochs']+1):
            if time.monotonic()-began>p['training']['max_seconds_per_fit']:raise RuntimeError('verifier fit time cap; preserve partial artifacts')
            model.train();fit_losses=[]
            for ids in torch.randperm(len(x)).split(p['training']['batch_size']):
                opt.zero_grad(set_to_none=True);l=(loss(model(x[ids]),c[ids],a[ids])*weights[ids]).mean()
                if not torch.isfinite(l):raise RuntimeError('nonfinite open-set loss')
                l.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.))
                if not np.isfinite(norm):raise RuntimeError('nonfinite gradient')
                gradients+=norm>0;max_grad=max(max_grad,norm);opt.step();steps+=1;fit_losses.append(float(l.detach()))
            model.eval()
            with torch.inference_mode():v=float((loss(model(ix),ic,ia)*iweights).mean())
            if not np.isfinite(v):raise RuntimeError('nonfinite INNER loss')
            checkpoint=destination/('epoch'+str(epoch)+'.pt')
            torch.save({'schema':'N72R21R1_CURRENT_AXIS_OPEN_SET_V1','family':family,'model':model.state_dict(),
                'feature_names':list(FEATURE_NAMES),'FIT_mean':mean.tolist(),'FIT_std':std.tolist(),
                'seed':seed,'epoch':epoch,'selection':{'status':'UNCALIBRATED_EPOCH_NOT_DEPLOYABLE'},'protocol_SHA':sha256(PROTOCOL)},checkpoint)
            row={'epoch':epoch,'FIT_loss':float(np.mean(fit_losses)),'INNER_loss':v,'optimizer_steps':steps,
                 'checkpoint_path':str(checkpoint),'checkpoint_SHA':sha256(checkpoint)}
            epochs.append(row);log.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');log.flush()
            if v<best_loss:
                best_loss=v;best_epoch=epoch;best={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
            else:patience+=1
            if patience>=p['training']['patience']:break
    model.load_state_dict(best,strict=True);model.eval()
    with torch.inference_mode():logits=model(ix)
    temperatures=[(float(F.cross_entropy(logits[:,:3]/t,ic)),t) for t in p['temperatures']]
    temperature=min(temperatures)[1]
    predicted=[predictions(logits[start:stop],temperature) for start,stop in spans]
    selection,points=operating_points(inner,predicted,p);selection={**selection,'temperature':temperature}
    checkpoint=destination/'selected_calibrated.pt'
    torch.save({'schema':'N72R21R1_CURRENT_AXIS_OPEN_SET_V1','family':family,'model':best,'feature_names':list(FEATURE_NAMES),
        'FIT_mean':mean.tolist(),'FIT_std':std.tolist(),'selection':selection,'seed':seed,'epoch':best_epoch,'protocol_SHA':sha256(PROTOCOL)},checkpoint)
    changed=sum(not torch.equal(v,initial[k]) for k,v in best.items());assert gradients and changed
    record={'stage':'N72R21R1','status':'COMPLETE_ACTUAL_CURRENT_AXIS_VERIFIER_OPTIMIZATION','family':family,'seed':seed,
        'checkpoint_path':str(checkpoint),'checkpoint_SHA':sha256(checkpoint),'strict_state_dict_schema':True,
        'FIT_sequences':p['FIT_sequences'],'INNER_sequences':p['INNER_sequences'],'FIT_groups':len(fit),'INNER_groups':len(inner),
        'FIT_rows':len(x),'INNER_rows':len(ix),'FIT_class_counts':dict(Counter(c.tolist())),'INNER_class_counts':dict(Counter(ic.tolist())),
        'excluded_groups':excluded,'epochs_completed':len(epochs),'selected_epoch':best_epoch,'optimizer_steps':steps,
        'nonzero_gradient_steps':gradients,'changed_state_tensors':changed,'max_preclip_gradient_norm':max_grad,
        'parameter_count':sum(v.numel() for v in model.parameters()),'seconds':time.monotonic()-began,
        'epoch_log_path':str(log_path),'epoch_log_SHA':sha256(log_path),'all_epoch_checkpoints':epochs,'selection':selection,
        'INNER_temperature_NLL':temperatures,'INNER_all_operating_points':points,
        'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':p['source_code_SHA'],'supervision_manifest_SHA':p['supervision_manifest_SHA'],
        'FIT_only_normalization':True,'UNKNOWN_separate_class_not_verified_negative':True,'runtime_GT_feature':False,
        'no_VAL_TEST_confirmation_selection':True,'not_closed_loop_MOT_evaluation':True,'not_future_trajectory_safety_PASS':True}
    write_json('training/current_axis_verifier/'+uid+'.json',record)
    print(json.dumps({'actual_current_axis_fit':uid,'epochs':len(epochs),'steps':steps,'calibration':selection['status']}),flush=True)


def calibrate_simple():
    p,fit,inner,_=data();result={}
    for family in SIMPLE:
        temps=[1.] if family=='E0_FROZEN' else p['temperatures']
        scored=[]
        for t in temps:
            pred=[simple_predictions(r['axis'],family,t) for r in inner]
            nll=float(np.mean([-np.log(max(sum(s['correct'] for a,s in zip(r['axis'],pr,strict=True) if a['class']==0),1.e-12)) for r,pr in zip(inner,pred,strict=True)]))
            scored.append((nll,t,pred))
        _,t,pred=min(scored,key=lambda x:(x[0],x[1]))
        # Apply the control-specific fixed causal filters at every point too.
        selected,points=operating_points(inner,pred,p,simple_family=family)
        if family=='E0_FROZEN':
            selected={'status':'FROZEN_UNCALIBRATED_REFERENCE','probability_min':0.,'margin_min':0.,'unknown_max':1.,'global_regret_max':p['runtime_global_regret_max']}
        result[family]={'selection':{**selected,'temperature':t,'simple_family':family},'INNER_all_operating_points':points,
            'temperature_NLL':[(n,t) for n,t,_ in scored],'not_model_training':True,'not_future_safety':True}
    write_json('availability/current_axis_verifier/SIMPLE_CALIBRATION_V1.json',{'stage':'N72R21R1','protocol_SHA':sha256(PROTOCOL),'controls':result})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--simple',action='store_true')
    parser.add_argument('--family');parser.add_argument('--seed',type=int);a=parser.parse_args()
    if a.freeze:freeze()
    elif a.simple:calibrate_simple()
    else:run(a.family,a.seed)
