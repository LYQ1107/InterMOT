"""Actual correction optimization on matched learned-policy own histories."""
import argparse
from collections import Counter
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r1_train_authority import supervision,tensors,loss_components,calibrate
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.on_policy_authority import OnPolicyAuthorityHead,decoded_predictions

PROTOCOL=OUT/'protocol/ON_POLICY_CORRECTION_HEADS_V1.json'
CODE=['scripts/n72r21r1_train_on_policy.py','sam3_intermot/one_click/on_policy_authority.py','scripts/n72r21r1_train_authority.py']


def data(contrast):
    manifests=[OUT/'on_policy/round1/SUPERVISION_MANIFEST.json']
    if contrast=='ORIGINAL_MIXED_PLUS_ON_POLICY':manifests.insert(0,OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json')
    fit=[];inner=[];sources=[];excluded=Counter()
    for path in manifests:
        manifest=read_json(path);assert sha256(manifest['path'])==manifest['sha256']
        label_sha=manifest.get('label_component_SHA',manifest.get('label_components_SHA'))
        assert sha256(ROOT/'sam3_intermot/evaluation/joint_trajectory_labels.py')==label_sha
        assert sha256(OUT/'protocol/TRAJECTORY_ORIGIN_DEFINITION.json')==manifest['origin_definition_SHA']
        sources.append({'manifest_path':str(path),'manifest_SHA':sha256(path),'data_SHA':manifest['sha256']})
        for row in read_zstd_jsonl(Path(manifest['path'])):
            if row['role'] not in ('FIT','INNER'):continue
            if not row['initial_clicked_UID_verified_target']:excluded['unverified_init']+=1;continue
            target=supervision(row,'H100_GLOBAL_RISK')
            if target is None:excluded['incomplete_H100']+=1;continue
            h=row['raw_trajectory_labels']['future']['H100']
            record={'x':np.array(row['causal_previous_feature_vectors']+[row['feature_vector']],np.float32),
                'class':target[0],'value':target[1],'sequence':row['sequence'],'role':row['role'],
                'group':(row['episode_uid'],row['state_source'],row['frame']),'action':row['action'],
                'components':[float(np.clip((h['N01']-h['N10'])/100,-3,3)),float(h['non_target_damage']>0),float(h['N10']>0)]}
            assert record['x'].shape==(4,32) and np.isfinite(record['x']).all()
            (fit if row['role']=='FIT' else inner).append(record)
    if not fit or not inner or not any(r['class']==1 for r in fit) or not any(r['class']==2 for r in fit):
        raise RuntimeError('insufficient verified complete benefit/harm diversity')
    return fit,inner,sources,dict(excluded)


def objective(output,classes,values,components,class_weights,family):
    loss=loss_components(output,classes,values,class_weights,family=family)
    if family=='GLOBAL_RISK':
        loss=loss+F.smooth_l1_loss(output[:,4],components[:,0],reduction='none')
        loss=loss+F.binary_cross_entropy_with_logits(output[:,5],components[:,1],reduction='none')
        loss=loss+F.binary_cross_entropy_with_logits(output[:,6],components[:,2],reduction='none')
    return loss


def run(family,contrast,seed):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);protocol=read_json(PROTOCOL)
    assert family in protocol['families'] and contrast in protocol['data_contrasts'] and seed in protocol['seeds']
    uid='__'.join((family,contrast,'seed'+str(seed)));record_path=OUT/'training/on_policy_correction'/ (uid+'.json')
    if record_path.exists():
        r=read_json(record_path);assert r['source_code_SHA']=={p:sha256(ROOT/p) for p in CODE}
        assert r['protocol_SHA']==sha256(PROTOCOL) and sha256(r['checkpoint_path'])==r['checkpoint_SHA']
        print(json.dumps({'verified_existing_actual_correction_fit':uid}),flush=True);return
    destination=ASSETS/'training/on_policy_correction'/uid
    if destination.exists():raise FileExistsError('preserve partial correction fit, explicit versioned recovery required')
    storage(32<<20);destination.mkdir(parents=True)
    fit,inner,sources,excluded=data(contrast);counts=Counter(r['sequence'] for r in fit)
    weights=np.array([1/counts[r['sequence']] for r in fit],np.float32);weights/=weights.sum()
    raw=np.stack([r['x'] for r in fit]);mean=(raw[:,-1]*weights[:,None]).sum(0)
    std=np.maximum(np.sqrt((((raw[:,-1]-mean)**2)*weights[:,None]).sum(0)),.05)
    x,classes,values=tensors(fit,mean,std);ix,iclasses,ivalues=tensors(inner,mean,std)
    component=torch.tensor([r['components'] for r in fit]);icomponent=torch.tensor([r['components'] for r in inner])
    freq=torch.bincount(classes,minlength=3).float();cw=freq.sum()/freq.clamp_min(1)/3
    sw=torch.from_numpy(weights*len(fit));model=OnPolicyAuthorityHead(family)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['training']['learning_rate'],weight_decay=protocol['training']['weight_decay'])
    began=time.monotonic();best_loss=float('inf');best=None;best_epoch=0;patience=0;steps=0;gradients=0;max_grad=0.;epochs=[]
    log_path=destination/'epochs.jsonl'
    with log_path.open('x') as log:
        for epoch in range(protocol['training']['max_epochs']):
            if time.monotonic()-began>protocol['training']['max_seconds_per_fit']:
                raise RuntimeError('correction fit time budget exceeded; preserve partial fit')
            model.train();losses=[]
            for ids in torch.randperm(len(fit)).split(protocol['training']['batch_size']):
                opt.zero_grad(set_to_none=True)
                loss=(objective(model(x[ids]),classes[ids],values[ids],component[ids],cw,family)*sw[ids]).mean()
                if not torch.isfinite(loss):raise RuntimeError('non-finite correction loss')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.))
                if not np.isfinite(norm):raise RuntimeError('non-finite correction gradient')
                gradients+=norm>0;max_grad=max(max_grad,norm);opt.step();steps+=1;losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                prediction=model(ix)
                inner_loss=float(objective(prediction,iclasses,ivalues,icomponent,cw,family).mean())
                accuracy=float((prediction[:,:3].argmax(1)==iclasses).float().mean())
            if not np.isfinite(inner_loss):raise RuntimeError('non-finite INNER correction loss')
            epoch_path=destination/('epoch'+str(epoch+1)+'.pt')
            torch.save({'schema':'N72R21R1_ON_POLICY_AUTHORITY_V1','family':family,'model':model.state_dict(),
                'feature_names':list(FEATURE_NAMES),'FIT_mean':mean.tolist(),'FIT_std':std.tolist(),
                'selection':{'status':'UNCALIBRATED_EPOCH_NOT_DEPLOYABLE'},'seed':seed,'epoch':epoch+1,
                'supervision_sources':sources,'protocol_SHA':sha256(PROTOCOL)},epoch_path)
            item={'epoch':epoch+1,'optimizer_steps':steps,'FIT_loss':float(np.mean(losses)),
                'INNER_loss':inner_loss,'INNER_accuracy':accuracy,'checkpoint_path':str(epoch_path),'checkpoint_SHA':sha256(epoch_path)}
            epochs.append(item);log.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');log.flush()
            if inner_loss<best_loss:
                best_loss=inner_loss;best_epoch=epoch+1;best={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
            else:patience+=1
            if patience>=protocol['training']['patience']:break
    assert steps and gradients and best is not None
    model.load_state_dict(best,strict=True);model.eval()
    with torch.inference_mode():predicted=decoded_predictions(model(ix))
    calibration_protocol=read_json(OUT/'protocol/LEARNED_AUTHORITY_V1.json')
    selection,points=calibrate(inner,predicted,calibration_protocol)
    checkpoint=destination/'selected_calibrated.pt'
    torch.save({'schema':'N72R21R1_ON_POLICY_AUTHORITY_V1','family':family,'model':best,
        'feature_names':list(FEATURE_NAMES),'FIT_mean':mean.tolist(),'FIT_std':std.tolist(),'selection':selection,
        'seed':seed,'epoch':best_epoch,'supervision_sources':sources,'protocol_SHA':sha256(PROTOCOL)},checkpoint)
    changed=sum(not torch.equal(v,initial[k]) for k,v in best.items());assert changed
    record={'stage':'N72R21R1','status':'COMPLETE_ACTUAL_ON_POLICY_CORRECTION_OPTIMIZATION','fit_uid':uid,
        'family':family,'data_contrast':contrast,'reward':'H100_GLOBAL_RISK','seed':seed,'device':'cpu',
        'checkpoint_path':str(checkpoint),'checkpoint_SHA':sha256(checkpoint),'strict_state_dict_schema':True,
        'FIT_sequences':sorted(counts),'INNER_sequences':sorted({r['sequence'] for r in inner}),
        'FIT_records':len(fit),'INNER_records':len(inner),'FIT_class_counts':dict(Counter(r['class'] for r in fit)),
        'INNER_class_counts':dict(Counter(r['class'] for r in inner)),'excluded_records':excluded,
        'epochs_completed':len(epochs),'selected_epoch':best_epoch,'optimizer_steps':steps,
        'nonzero_gradient_steps':gradients,'max_preclip_gradient_norm':max_grad,'changed_state_tensors':changed,
        'parameter_count':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-began,
        'epoch_log_path':str(log_path),'epoch_log_SHA':sha256(log_path),'all_epoch_checkpoints':epochs,
        'selection':selection,'INNER_all18_operating_points':points,'supervision_sources':sources,
        'protocol_SHA':sha256(PROTOCOL),'calibration_protocol_SHA':sha256(OUT/'protocol/LEARNED_AUTHORITY_V1.json'),
        'source_code_SHA':{p:sha256(ROOT/p) for p in CODE},'FIT_only_normalization':True,
        'runtime_GT_feature':False,'no_VAL_test_confirmation_selection':True,'historical_INNER_not_virgin':True,
        'not_closed_loop_MOT_evaluation':True,'not_scientific_success':True}
    write_json('training/on_policy_correction/'+uid+'.json',record)
    print(json.dumps({'actual_correction_optimization_complete':uid,'epochs':len(epochs),'steps':steps,
        'calibration_status':selection['status'],'INNER_selected_events':selection['accepted_events'],
        'seconds':round(record['seconds'],2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--family',required=True)
    parser.add_argument('--contrast',required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();run(args.family,args.contrast,args.seed)
