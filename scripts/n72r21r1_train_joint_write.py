"""Actual joint-committed current correctness and masked paired future risk."""
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
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.joint_write_authority import JointWriteHead

PROTOCOL=OUT/'protocol/JOINT_WRITE_HEADS_V1.json'
CODE=['scripts/n72r21r1_train_joint_write.py','sam3_intermot/one_click/joint_write_authority.py']


def load_data():
    path=OUT/'memory/curriculum/SUPERVISION_MANIFEST.json';manifest=read_json(path)
    assert sha256(manifest['path'])==manifest['sha256']
    assert sha256(ROOT/'sam3_intermot/evaluation/joint_trajectory_labels.py')==manifest['label_component_SHA']
    assert sha256(OUT/'protocol/TRAJECTORY_ORIGIN_DEFINITION.json')==manifest['origin_definition_SHA']
    records=[];excluded=Counter();all_available=Counter()
    for row in read_zstd_jsonl(Path(manifest['path'])):
        if row['role'] not in ('FIT','INNER'):continue
        assert row['initial_clicked_UID_verified_target'] and row['current_or_future_GT_not_feature']
        all_available[row['role']]+=row['target_candidate_available']
        if not row['eligible_current_crop']:excluded[row['role']+'/NONE_no_writable_crop']+=1;continue
        future=row['paired_future'];h=None if future is None else future['raw_trajectory_labels']['future']['H100']
        complete=bool(h is not None and h['complete'])
        records.append({'x':np.asarray(row['feature_vector'],np.float32),'correct':row['current_outcome']=='TARGET',
            'outcome':row['current_outcome'],'future_risk':[h['non_target_damage']>0,h['N10']>0] if complete else None,
            'sequence':row['sequence'],'episode_uid':row['episode_uid'],'frame':row['frame'],'role':row['role'],
            'pair_complete':complete,'candidate_available':row['target_candidate_available']})
    fit=[r for r in records if r['role']=='FIT'];inner=[r for r in records if r['role']=='INNER']
    if not fit or not inner or len({r['correct'] for r in fit})<2:raise RuntimeError('no nondegenerate actual joint writer supervision')
    if not any(r['pair_complete'] for r in fit):raise RuntimeError('no actual complete FIT write-risk pairs')
    return fit,inner,manifest,dict(excluded),dict(all_available)


def tensors(rows,mean,std):
    return (torch.from_numpy((np.stack([r['x'] for r in rows])-mean)/std),
        torch.tensor([r['correct'] for r in rows],dtype=torch.float32),
        torch.tensor([r['future_risk'] if r['pair_complete'] else [False,False] for r in rows],dtype=torch.float32),
        torch.tensor([r['pair_complete'] for r in rows],dtype=torch.bool))


def loss(output,correct,future,mask,weights,pos_weight):
    current=(F.binary_cross_entropy_with_logits(output[:,0],correct,pos_weight=pos_weight,reduction='none')*weights).mean()
    auxiliary=F.binary_cross_entropy_with_logits(output[mask,1:],future[mask]) if mask.any() else output.sum()*0.
    return current+auxiliary


def curve(rows,probabilities,threshold,available_denominator):
    selected=(probabilities[:,0]>=threshold)&(probabilities[:,1:].max(1)<=1-threshold)
    correct=sum(r['correct'] for r in rows);accepted=int(selected.sum());good=sum(bool(k and r['correct']) for k,r in zip(selected,rows,strict=True))
    pair_selected=sum(bool(k and r['pair_complete']) for k,r in zip(selected,rows,strict=True))
    pair_harm=sum(bool(k and r['pair_complete'] and any(r['future_risk'])) for k,r in zip(selected,rows,strict=True))
    return {'score_threshold':threshold,'accepted':accepted,'correct':good,'wrong_or_UNKNOWN':accepted-good,
        'wrong_or_UNKNOWN_rate':(accepted-good)/accepted if accepted else None,
        'correct_committed_crop_denominator':correct,'correct_committed_retention':good/correct if correct else None,
        'candidate_available_denominator':available_denominator,'candidate_available_retention':good/available_denominator if available_denominator else None,
        'coverage':accepted/len(rows),'selected_complete_future_pairs':pair_selected,'selected_future_pair_harm':pair_harm,
        'population_2percent_risk_guarantee':False,'correlated_frames_not_independent_people':True}


def run(seed):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);protocol=read_json(PROTOCOL);assert seed in protocol['seeds']
    uid='JOINT_COMMITTED_CURRENT_AND_FUTURE_RISK_MLP__seed'+str(seed);record_path=OUT/'training/joint_write'/(uid+'.json')
    if record_path.exists():
        r=read_json(record_path);assert r['source_code_SHA']=={p:sha256(ROOT/p) for p in CODE}
        assert r['protocol_SHA']==sha256(PROTOCOL) and sha256(r['checkpoint_path'])==r['checkpoint_SHA'];return
    destination=ASSETS/'training/joint_write'/uid
    if destination.exists():raise FileExistsError('preserve partial actual joint write fit')
    storage(32<<20);destination.mkdir(parents=True);fit,inner,manifest,excluded,available=load_data()
    counts=Counter(r['sequence'] for r in fit);weights=np.array([1/counts[r['sequence']] for r in fit],np.float32);weights/=weights.sum()
    raw=np.stack([r['x'] for r in fit]);mean=(raw*weights[:,None]).sum(0)
    std=np.maximum(np.sqrt((((raw-mean)**2)*weights[:,None]).sum(0)),.05)
    x,c,f,m=tensors(fit,mean,std);ix,ic,ifuture,im=tensors(inner,mean,std)
    sw=torch.from_numpy(weights*len(fit));positive=float(c.sum());pw=torch.tensor((len(fit)-positive)/positive)
    model=JointWriteHead();initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['training']['learning_rate'],weight_decay=protocol['training']['weight_decay'])
    began=time.monotonic();steps=0;gradients=0;max_grad=0.;best_loss=float('inf');best=None;best_epoch=0;patience=0;epochs=[]
    log_path=destination/'epochs.jsonl'
    with log_path.open('x') as log:
        for epoch in range(protocol['training']['max_epochs']):
            if time.monotonic()-began>protocol['training']['max_seconds_per_fit']:raise RuntimeError('joint writer fit time cap')
            model.train();losses=[]
            for ids in torch.randperm(len(fit)).split(protocol['training']['batch_size']):
                optimizer.zero_grad(set_to_none=True);value=loss(model(x[ids]),c[ids],f[ids],m[ids],sw[ids],pw)
                if not torch.isfinite(value):raise RuntimeError('non-finite joint writer loss')
                value.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.))
                if not np.isfinite(norm):raise RuntimeError('non-finite joint writer gradient')
                gradients+=norm>0;max_grad=max(max_grad,norm);optimizer.step();steps+=1;losses.append(float(value.detach()))
            model.eval()
            with torch.inference_mode():
                output=model(ix);inner_loss=float(loss(output,ic,ifuture,im,torch.ones(len(inner)),pw))
                accuracy=float(((output[:,0].sigmoid()>=.5)==ic.bool()).float().mean())
            if not np.isfinite(inner_loss):raise RuntimeError('non-finite INNER joint writer loss')
            epoch_path=destination/('epoch'+str(epoch+1)+'.pt')
            torch.save({'schema':'N72R21R1_JOINT_COMMIT_WRITE_V1','model':model.state_dict(),
                'feature_names':list(FEATURE_NAMES),'FIT_mean':mean.tolist(),'FIT_std':std.tolist(),
                'selection':{'status':'UNCALIBRATED_EPOCH_NOT_DEPLOYABLE'},'seed':seed,'epoch':epoch+1,
                'supervision_SHA':manifest['sha256'],'protocol_SHA':sha256(PROTOCOL)},epoch_path)
            item={'epoch':epoch+1,'optimizer_steps':steps,'FIT_loss':float(np.mean(losses)),
                'INNER_loss':inner_loss,'INNER_correctness_accuracy':accuracy,'checkpoint_path':str(epoch_path),'checkpoint_SHA':sha256(epoch_path)}
            epochs.append(item);log.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');log.flush()
            if inner_loss<best_loss:
                best_loss=inner_loss;best_epoch=epoch+1;best={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
            else:patience+=1
            if patience>=protocol['training']['patience']:break
    assert best is not None and steps and gradients
    model.load_state_dict(best,strict=True);model.eval()
    with torch.inference_mode():probabilities=model(ix).sigmoid().numpy()
    points=[curve(inner,probabilities,t,available['INNER']) for t in protocol['risk_coverage_score_thresholds']]
    point=next(p for p in points if p['score_threshold']==.98)
    legal=bool(point['accepted'] and point['wrong_or_UNKNOWN_rate']<=.02 and point['correct_committed_retention'] is not None
        and point['correct_committed_retention']>=.6 and point['selected_complete_future_pairs']>0 and point['selected_future_pair_harm']==0)
    selection={'status':'CALIBRATED_DIAGNOSTIC' if legal else 'CALIBRATION_ABSTAIN','fixed_runtime_point':point,
        'population_2percent_risk_guarantee':False,'not_memory_usefulness_or_MOT_PASS':True}
    checkpoint=destination/'selected_calibrated.pt'
    torch.save({'schema':'N72R21R1_JOINT_COMMIT_WRITE_V1','model':best,'feature_names':list(FEATURE_NAMES),
        'FIT_mean':mean.tolist(),'FIT_std':std.tolist(),'selection':selection,'seed':seed,'epoch':best_epoch,
        'supervision_SHA':manifest['sha256'],'protocol_SHA':sha256(PROTOCOL)},checkpoint)
    changed=sum(not torch.equal(v,initial[k]) for k,v in best.items());assert changed
    record={'stage':'N72R21R1','status':'COMPLETE_ACTUAL_JOINT_COMMITTED_WRITE_OPTIMIZATION','fit_uid':uid,'seed':seed,
        'checkpoint_path':str(checkpoint),'checkpoint_SHA':sha256(checkpoint),'strict_state_dict_schema':True,
        'FIT_sequences':sorted(counts),'INNER_sequences':sorted({r['sequence'] for r in inner}),'FIT_records':len(fit),'INNER_records':len(inner),
        'FIT_outcomes':dict(Counter(r['outcome'] for r in fit)),'INNER_outcomes':dict(Counter(r['outcome'] for r in inner)),
        'FIT_complete_paired_future_records':sum(r['pair_complete'] for r in fit),'INNER_complete_paired_future_records':sum(r['pair_complete'] for r in inner),
        'future_labels_missing_and_masked_not_imputed':True,'excluded_records':excluded,'candidate_available_counts':available,
        'epochs_completed':len(epochs),'selected_epoch':best_epoch,'optimizer_steps':steps,'nonzero_gradient_steps':gradients,
        'max_preclip_gradient_norm':max_grad,'changed_state_tensors':changed,'parameter_count':sum(p.numel() for p in model.parameters()),
        'seconds':time.monotonic()-began,'epoch_log_path':str(log_path),'epoch_log_SHA':sha256(log_path),'all_epoch_checkpoints':epochs,
        'selection':selection,'INNER_risk_coverage_points':points,'source_code_SHA':{p:sha256(ROOT/p) for p in CODE},
        'supervision_SHA':manifest['sha256'],'supervision_manifest_SHA':sha256(OUT/'memory/curriculum/SUPERVISION_MANIFEST.json'),
        'protocol_SHA':sha256(PROTOCOL),'runtime_GT_feature':False,'no_VAL_TEST_confirmation_selection':True,
        'device':'cpu','not_closed_loop_memory_or_MOT_evaluation':True,'not_scientific_success':True}
    write_json('training/joint_write/'+uid+'.json',record)
    print(json.dumps({'actual_joint_write_fit_complete':uid,'epochs':len(epochs),'steps':steps,'calibration_status':selection['status'],
        'INNER_fixed_point':point,'seconds':round(record['seconds'],2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();run(args.seed)
