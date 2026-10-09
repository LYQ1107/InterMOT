"""Memory-coupled current-input training from sealed paired causal branches."""
import argparse
import copy
import json
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork,memory_training_losses
from scripts.n72r21_coupled_data import CoupledFrames,collate
from scripts.n72r21_train_t1 import epoch,data_summary
from scripts.n72r21_train_t0 import rng_state,restore_rng,atomic_save,split
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21_sot import checked_device

RUN_NAME='T2_COUPLED_V1'
CODE=['scripts/n72r21_train_t2.py','scripts/n72r21_coupled_data.py','scripts/n72r21_label_coupled_states.py',
      'scripts/n72r21_train_t1.py','scripts/n72r21_causal_data.py','scripts/n72r21_train_t0.py',
      'sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_memory.py','sam3_intermot/one_click/acib_runtime.py']


def fit(outer,seeds,gpu):
    torch.set_num_threads(1);protocol_path=OUT/'protocol/T2_MEMORY_COUPLED_TRAINING.json';protocol=read_json(protocol_path)
    sequences=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'];fit_sequences,inner=split(outer,sequences)
    if not set(seeds)<=set(protocol['seeds']):raise ValueError('registered seeds')
    code={p:sha256(ROOT/p) for p in CODE};device=checked_device(gpu)
    for seed in seeds:
        source_path=OUT/'training/T1_CAUSAL_V1'/f'{outer}__seed{seed}__MIXED.json';source=read_json(source_path)
        if not source['completed'] or sha256(source['best_checkpoint_path'])!=source['best_checkpoint_sha256']:raise ValueError('source T1 incomplete/changed')
        initial=torch.load(source['best_checkpoint_path'],map_location='cpu',weights_only=True)
        tag=f'{outer}__seed{seed}';directory=ASSETS/'training'/RUN_NAME/tag;done=OUT/'training'/RUN_NAME/f'{tag}.json'
        basic={'protocol_sha256':sha256(protocol_path),'code_sha256':code,'fit_sequences':fit_sequences,'inner_sequence':inner,
            'outer_sequence':outer,'seed':seed,'source_T1_fit_record_sha256':sha256(source_path),'initial_T1_checkpoint_sha256':source['best_checkpoint_sha256'],
            'outer_future_GT_features_pixels_used_for_fit':False,'shared_initial_click_metadata_parsed':True,
            'future_trajectory_supervision_not_online_features':True,'behavior_model':'T1_MIXED_OWN_P1_WITH_REAL_PAIRED_WRITE_BRANCHES',
            'physical_absence_or_cross_recording_training_claim':False}
        if done.exists() and read_json(done).get('completed'):
            old=read_json(done)
            if old['schema']['configuration']!=basic or sha256(old['best_checkpoint_path'])!=old['best_checkpoint_sha256']:raise ValueError('sealed T2 changed')
            print(json.dumps({'T2_reused_completed':tag}),flush=True);continue
        storage(100<<20);started=time.monotonic()
        training_data=CoupledFrames(outer,seed,fit_sequences,stride=protocol['current_loss_sample_stride_fit'])
        inner_data=CoupledFrames(outer,seed,[inner],stride=protocol['current_loss_sample_stride_inner'])
        if not training_data.known_risk or not inner_data.known_risk:raise ValueError('insufficient verified TRAIN/inner future-risk supervision; no invented labels')
        schema={'configuration':basic,'fit_data':{**data_summary(training_data),'known_risk':training_data.known_risk,'safe_risk':training_data.safe_risk},
                'inner_data':{**data_summary(inner_data),'known_risk':inner_data.known_risk,'safe_risk':inner_data.safe_risk}}
        print(json.dumps({'T2_data_ready':tag,'fit_samples':len(training_data),'inner_full_frame_samples':len(inner_data),
            'fit_risk':training_data.known_risk,'inner_risk':inner_data.known_risk,'seconds':round(time.monotonic()-started,1)}),flush=True)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        generator=torch.Generator().manual_seed(seed);model=ACIBMemoryNetwork().to(device);model.load_base(initial['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',init_scale=protocol['AMP_initial_scale'],enabled=device.type=='cuda')
        latest=directory/'latest.pt';best=directory/'best.pt';logs=[];oom=[];start_epoch=0;stale=0;best_loss=math.inf;batch_size=protocol['batch_size']
        if latest.exists():
            saved=torch.load(latest,map_location='cpu',weights_only=True)
            if saved['schema']!=schema:raise ValueError('T2 resume schema changed')
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            restore_rng(saved['rng'],generator);start_epoch=saved['epoch']+1;stale=saved['stale'];best_loss=saved['best_loss'];logs=saved['logs'];oom=saved['OOM'];batch_size=saved['batch_size']
        for number in range(start_epoch,protocol['max_epochs']):
            if stale>=protocol['patience']:break
            if time.monotonic()-started>1800:raise RuntimeError('30-minute T2 fit ceiling; resume checkpoint retained')
            storage(100<<20)
            recovery={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'optimizer':copy.deepcopy(optimizer.state_dict()),'scaler':copy.deepcopy(scaler.state_dict()),'rng':rng_state(generator)}
            while True:
                try:
                    train_loader=DataLoader(training_data,batch_size=batch_size,shuffle=True,generator=generator,collate_fn=collate,num_workers=0)
                    inner_loader=DataLoader(inner_data,batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=0)
                    training=epoch(model,train_loader,device,optimizer,scaler,loss_fn=memory_training_losses)
                    validation=epoch(model,inner_loader,device,None,scaler,loss_fn=memory_training_losses);break
                except torch.cuda.OutOfMemoryError:
                    oom.append({'epoch':number,'batch_size':batch_size,'action':'restore_epoch_start_and_halve_same_sample_batch'})
                    if batch_size<=32:raise
                    model.load_state_dict(recovery['model']);optimizer.load_state_dict(recovery['optimizer']);scaler.load_state_dict(recovery['scaler'])
                    restore_rng(recovery['rng'],generator);torch.cuda.empty_cache();batch_size//=2
            current=validation['loss_components']['total'];improved=current<best_loss-1e-6
            if improved:best_loss=current;stale=0
            else:stale+=1
            logs.append({'epoch':number,'fit':training,'inner':validation,'best_selected_on_inner_loss':improved,'batch_size':batch_size,
                         'seconds':time.monotonic()-started,'AMP':device.type=='cuda'})
            saved={'schema':schema,'seed':seed,'epoch':number,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'rng':rng_state(generator),'best_loss':best_loss,'stale':stale,'logs':logs,'OOM':oom,'batch_size':batch_size}
            if improved:atomic_save(best,saved)
            atomic_save(latest,saved)
            record={'completed':False,'schema':schema,'seed':seed,'logs':logs,'OOM':oom,'model_parameters':sum(p.numel() for p in model.parameters()),
                'best_checkpoint_path':str(best),'best_checkpoint_sha256':sha256(best),'latest_checkpoint_path':str(latest),'latest_checkpoint_sha256':sha256(latest),
                'runtime_device':str(device),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                'future_safe_head_trained_on_verified_paired_supervision':True,'T3_complete':False,'scientific_success':False,'next_stage_authorized':False}
            write_json(f'training/{RUN_NAME}/{tag}.json',record)
            print(json.dumps({'T2':tag,'epoch':number,'fit_loss':training['loss_components']['total'],'inner_loss':current,
                'fit_future_safe_loss':training['loss_components']['future_safe'],'inner_future_safe_loss':validation['loss_components']['future_safe'],
                'inner_candidate_recall':validation['correct_candidate_recall'],'inner_negative_FPR':validation['negative_false_accept_rate'],
                'AMP_overflows':len(training['AMP_overflow_events']),'best':improved,'seconds':round(time.monotonic()-started,1)}),flush=True)
        record=read_json(done);record.update(completed=True,early_stopped=stale>=protocol['patience']);write_json(f'training/{RUN_NAME}/{tag}.json',record)
        del model,optimizer,scaler,training_data,inner_data
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103]);parser.add_argument('--gpu',type=int)
    args=parser.parse_args();fit(args.outer,args.seeds,args.gpu)
