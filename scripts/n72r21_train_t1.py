"""Actual T1 fits on sealed own P0/P1/mixed states, never oracle banks.

T1 is frozen-behavior current-state training. T2 must separately refresh
states using learned T1 models and learn/test future memory risk.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sam3_intermot.one_click.acib import ACIBNetwork,training_losses
from scripts.n72r21_causal_data import CausalFrames,collate,forward
from scripts.n72r21_train_t0 import rng_state,restore_rng,atomic_save,split
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21_sot import checked_device

RUN_NAME='T1_CAUSAL_V1'
CODE=['scripts/n72r21_train_t1.py','scripts/n72r21_causal_data.py','scripts/n72r21_train_t0.py',
      'sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_runtime.py']


def current_loss(output,batch):
    return training_losses(output,batch['target_slot'],batch['availability_label'],batch['write_labels'],batch['write_verified'])


def epoch(model,loader,device,optimizer,scaler,loss_fn=current_loss):
    training=optimizer is not None;model.train(training);counts=0;totals={};gradients=[];overflows=[];fallbacks=0
    correct=available=accepted=negative=false_accept=0
    for source in loader:
        batch={k:v.to(device) for k,v in source.items()};retry=0
        while True:
            if training:optimizer.zero_grad(set_to_none=True)
            amp=device.type=='cuda' and retry<5
            with torch.set_grad_enabled(training),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                output=forward(model,batch);parts=loss_fn(output,batch)
            if not all(torch.isfinite(v) for v in parts.values()):raise FloatingPointError('nonfinite actual T1 loss')
            if not training:break
            scaler.scale(parts['total']).backward();scaler.unscale_(optimizer)
            invalid=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if invalid:
                overflows.append({'batch_start_sample':counts,'retry':retry,'scale':float(scaler.get_scale()),'AMP':amp,'parameter_names':invalid})
                if not amp:raise FloatingPointError('nonfinite FP32 same-minibatch gradient; no invalid update')
                scaler.step(optimizer);scaler.update();retry+=1;continue
            gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),25.,error_if_nonfinite=True)))
            scaler.step(optimizer);scaler.update();fallbacks+=int(device.type=='cuda' and not amp);break
        n=len(batch['anchor']);counts+=n
        for k,value in parts.items():totals[k]=totals.get(k,0.)+float(value.detach())*n
        prediction=output['joint_probabilities'].argmax(-1);positive=batch['availability_label']==0
        predicted=prediction!=batch['candidates'].shape[1]
        correct+=int(((prediction==batch['target_slot'])&positive).sum());available+=int(positive.sum())
        accepted+=int(predicted.sum());negative+=int((~positive).sum());false_accept+=int((predicted&~positive).sum())
    if not counts:raise ValueError('no actual fitting/inner samples')
    return {'samples':counts,'loss_components':{k:v/counts for k,v in totals.items()},'candidate_available_samples':available,
        'correct_candidate_recall':correct/available if available else None,'accepted_samples':accepted,
        'negative_samples':negative,'negative_false_accept_rate':false_accept/negative if negative else None,
        'gradient_norm_mean':float(np.mean(gradients)) if gradients else None,'gradient_norm_max':max(gradients) if gradients else None,
        'accepted_optimizer_gradients_finite':True,'AMP_overflow_events':overflows,'same_minibatch_FP32_fallbacks':fallbacks,
        'all_NONE':accepted==0,'all_present':accepted==counts}


def data_summary(data):
    return {'sources':data.sources,'samples':len(data),'availability_class_counts':data.counts,
            'full_causal_frames_reconstructed':data.reconstructed_frames,'sealed_snapshots_checked_exactly':data.checked_snapshots}


def fit(outer,seeds,conditions,gpu):
    torch.set_num_threads(1);protocol_path=OUT/'protocol/T1_CAUSAL_STATE_TRAINING.json';protocol=read_json(protocol_path)
    sequences=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'];fit_sequences,inner=split(outer,sequences)
    if not set(seeds)<=set(protocol['seed_values']) or not set(conditions)<=set(protocol['conditions']):raise ValueError('frozen T1 seeds/conditions')
    code={p:sha256(ROOT/p) for p in CODE};device=checked_device(gpu)
    for seed in seeds:
        source_path=OUT/'training/T0_AMP_R1'/f'{outer}__seed{seed}.json';source=read_json(source_path)
        if not source['completed'] or sha256(source['best_checkpoint_path'])!=source['best_checkpoint_sha256']:raise ValueError('source T0 incomplete/changed')
        initial=torch.load(source['best_checkpoint_path'],map_location='cpu',weights_only=True)
        for condition in conditions:
            tag=f'{outer}__seed{seed}__{condition}';directory=ASSETS/'training'/RUN_NAME/tag;done=OUT/'training'/RUN_NAME/f'{tag}.json'
            basic={'T1_protocol_sha256':sha256(protocol_path),'code_sha256':code,'fit_sequences':fit_sequences,'inner_sequence':inner,
                'outer_sequence':outer,'seed':seed,'training_condition':condition,'capacity':protocol['capacity'],
                'source_T0_fit_sha256':sha256(source_path),'initial_checkpoint_sha256':source['best_checkpoint_sha256'],
                'future_outer_GT_pixels_used_in_fitting':False,'shared_all_click_initialization_metadata_parsed':True,
                'physical_absence_supervision_available':False,'cross_recording_training_claim':False,
                'behavior_source':'T0_AMP_R1 own predictions, not updated T1 policy or oracle positives'}
            if done.exists() and read_json(done).get('completed'):
                old=read_json(done)
                if old['schema']['configuration']!=basic or sha256(old['best_checkpoint_path'])!=old['best_checkpoint_sha256']:raise ValueError('sealed T1 changed')
                print(json.dumps({'T1_reused_completed':tag}),flush=True);continue
            storage(100<<20);started=time.monotonic()
            training_data=CausalFrames(outer,seed,fit_sequences,condition,stride=protocol['fit_stride'])
            inner_data=CausalFrames(outer,seed,[inner],condition,stride=protocol['inner_stride'])
            if set(training_data.sources)&{outer,inner} or set(inner_data.sources)!={inner}:raise ValueError('split leakage')
            schema={'configuration':basic,'fit_data':data_summary(training_data),'inner_data':data_summary(inner_data)}
            print(json.dumps({'T1_data_ready':tag,'fit_samples':len(training_data),'inner_full_frame_samples':len(inner_data),
                              'reconstructed_frames':training_data.reconstructed_frames+inner_data.reconstructed_frames,'seconds':round(time.monotonic()-started,1)}),flush=True)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            generator=torch.Generator().manual_seed(seed);model=ACIBNetwork().to(device);model.load_state_dict(initial['model'],strict=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
            scaler=torch.amp.GradScaler('cuda',init_scale=protocol['AMP_initial_scale'],enabled=device.type=='cuda')
            latest=directory/'latest.pt';best=directory/'best.pt';logs=[];oom=[];start_epoch=0;stale=0;best_loss=math.inf;batch_size=protocol['batch_size']
            if latest.exists():
                saved=torch.load(latest,map_location='cpu',weights_only=True)
                if saved['schema']!=schema:raise ValueError('resume schema mismatch')
                model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
                restore_rng(saved['rng'],generator);start_epoch=saved['epoch']+1;stale=saved['stale'];best_loss=saved['best_loss']
                logs=saved['logs'];oom=saved['OOM'];batch_size=saved['batch_size']
            for number in range(start_epoch,protocol['max_epochs']):
                if stale>=protocol['patience']:break
                if time.monotonic()-started>1800:raise RuntimeError('30-minute T1 run ceiling, resume checkpoint retained')
                storage(100<<20)
                recovery={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'optimizer':copy.deepcopy(optimizer.state_dict()),'scaler':copy.deepcopy(scaler.state_dict()),'rng':rng_state(generator)}
                while True:
                    try:
                        train_loader=DataLoader(training_data,batch_size=batch_size,shuffle=True,generator=generator,num_workers=0,collate_fn=collate)
                        inner_loader=DataLoader(inner_data,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=collate)
                        training=epoch(model,train_loader,device,optimizer,scaler);validation=epoch(model,inner_loader,device,None,scaler);break
                    except torch.cuda.OutOfMemoryError:
                        oom.append({'epoch':number,'batch_size':batch_size,'action':'restore_epoch_start_then_halve_same_sample_batch'})
                        if batch_size<=32:raise
                        model.load_state_dict(recovery['model']);optimizer.load_state_dict(recovery['optimizer']);scaler.load_state_dict(recovery['scaler'])
                        restore_rng(recovery['rng'],generator);torch.cuda.empty_cache();batch_size//=2
                current=validation['loss_components']['total'];improved=current<best_loss-1e-6
                if improved:best_loss=current;stale=0
                else:stale+=1
                logs.append({'epoch':number,'fit':training,'inner':validation,'best_selected_on_inner_loss':improved,
                             'batch_size':batch_size,'seconds':time.monotonic()-started,'AMP':device.type=='cuda'})
                saved={'schema':schema,'seed':seed,'epoch':number,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'scaler':scaler.state_dict(),'rng':rng_state(generator),'best_loss':best_loss,'stale':stale,'logs':logs,'OOM':oom,'batch_size':batch_size}
                if improved:atomic_save(best,saved)
                atomic_save(latest,saved)
                record={'completed':False,'schema':schema,'seed':seed,'logs':logs,'OOM':oom,'model_parameters':sum(p.numel() for p in model.parameters()),
                    'best_checkpoint_path':str(best),'best_checkpoint_sha256':sha256(best),'latest_checkpoint_path':str(latest),'latest_checkpoint_sha256':sha256(latest),
                    'runtime_device':str(device),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                    'T2_T3_complete':False,'scientific_success':False,'next_stage_authorized':False}
                write_json(f'training/{RUN_NAME}/{tag}.json',record)
                print(json.dumps({'T1':tag,'epoch':number,'fit_loss':training['loss_components']['total'],'inner_loss':current,
                    'inner_candidate_recall':validation['correct_candidate_recall'],'inner_negative_FPR':validation['negative_false_accept_rate'],
                    'AMP_overflows':len(training['AMP_overflow_events']),'best':improved,'seconds':round(time.monotonic()-started,1)}),flush=True)
            record=read_json(done);record.update(completed=True,early_stopped=stale>=protocol['patience'])
            write_json(f'training/{RUN_NAME}/{tag}.json',record)
            del model,optimizer,scaler,training_data,inner_data
            if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103])
    parser.add_argument('--conditions',nargs='+',default=['P0','P1','MIXED']);parser.add_argument('--gpu',type=int)
    args=parser.parse_args();fit(args.outer,args.seeds,args.conditions,args.gpu)
