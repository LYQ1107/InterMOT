"""Real bounded three-seed T0 fit, with strict sequence splits and resume.

Labels enter only this TRAIN/inner fitting process. They never enter the
network feature schema; independent runtime will consume the sealed weights.
T0 is not claimed as full memory-coupled or physical-absence training.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21_sot import checked_device
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.one_click.acib import ACIBNetwork,training_losses,AVAILABILITY_CLASSES,EVIDENCE_SCHEMA

RUN_NAME='T0_AMP_R1'


def split(outer,sequences):
    if len(sequences)!=8 or len(set(sequences))!=8 or outer not in sequences: raise ValueError('registered eight-sequence fold')
    inner=sequences[(sequences.index(outer)+1)%8]
    return [s for s in sequences if s not in (outer,inner)],inner


class T0Frames(Dataset):
    def __init__(self,sequences,*,stride):
        self.examples=[];self.sources={};self.counts=[0,0,0]
        inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
        if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor SHA')
        anchors=np.load(inputs['anchor_path'],mmap_mode='r')
        labels={e['episode_uid']:e['target_gt_identity'] for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
        for sequence in sequences:
            directory=ROOT.parent/'InterMOT_N72R20R2_assets/candidates'/sequence
            index=read_json(directory/'index.json')
            for name,path in [('metadata',directory/'metadata.jsonl.zst'),('embeddings',directory/'embeddings.f16')]:
                if sha256(path)!=index[name+'_sha256']:raise ValueError('candidate source SHA')
            gtpath=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
            gt=dancetrack_annotations(gtpath)
            frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
            events=[e for e in inputs['inputs'] if e['sequence']==sequence]
            prepared=[]
            for p,rows in frames:
                feature=np.stack([r['feature'] for r in rows]).astype(np.float32) if rows else np.empty((0,512),np.float32)
                if len(rows):feature/=np.linalg.norm(feature,axis=1,keepdims=True).clip(1e-8)
                matched=strict_candidate_matching(rows,gt.get(p['frame'],[]))
                qualities=[]
                for r in rows:
                    x,y,right,bottom=r['box_xyxy'];w,h=right-x,bottom-y
                    qualities.append([np.clip(r.get('conf',0.),0,1),np.clip(w*h/(events[0]['width']*events[0]['height']),0,1),
                                      np.clip(w/h,0,4)/4,0.,0.])
                prepared.append((torch.from_numpy(feature),torch.tensor(qualities,dtype=torch.float32).reshape(-1,5),
                                 [matched[str(r['candidate_uid'])] for r in rows]))
            for event in events:
                identity=labels[event['episode_uid']]
                anchor=np.array(anchors[event['anchor_index']],copy=True);anchor/=np.linalg.norm(anchor)
                for (p,rows),(feature,quality,ids) in zip(frames,prepared,strict=True):
                    frame=int(p['frame'])
                    if frame<=event['frame'] or (frame-event['frame']-1)%stride:continue
                    positives=[i for i,value in enumerate(ids) if value==identity]
                    if len(positives)>1:raise ValueError('one-to-one target label')
                    visible=any(a['identity']==identity for a in gt.get(frame,[]))
                    availability=0 if positives else 1 if visible else 2
                    self.counts[availability]+=1
                    self.examples.append({'anchor':torch.from_numpy(anchor),'candidates':feature,'quality':quality,
                        'context':torch.tensor([math.log1p((frame-event['frame'])/event['fps'])/5,0.,0.]),
                        'target':positives[0] if positives else -1,'availability':availability,
                        'write_labels':torch.tensor([v==identity for v in ids],dtype=torch.float32),
                        'write_verified':torch.tensor([v is not None for v in ids],dtype=torch.bool)})
            self.sources[sequence]={'candidate_index_sha256':sha256(directory/'index.json'),'GT_sha256':sha256(gtpath/'gt/gt.txt'),
                                    'initialization_manifest_sha256':sha256(OUT/'development/RUNTIME_INPUTS.json')}
    def __len__(self):return len(self.examples)
    def __getitem__(self,index):return self.examples[index]


def collate(examples):
    batch=len(examples);count=max(1,max(len(e['candidates']) for e in examples))
    values={'anchor':torch.stack([e['anchor'] for e in examples]),'candidates':torch.zeros(batch,count,512),
            'valid':torch.zeros(batch,count,dtype=torch.bool),'quality':torch.zeros(batch,count,5),
            'context':torch.stack([e['context'] for e in examples]),'target_slot':torch.tensor([e['target'] if e['target']>=0 else count for e in examples]),
            'availability_label':torch.tensor([e['availability'] for e in examples]),'write_labels':torch.zeros(batch,count),
            'write_verified':torch.zeros(batch,count,dtype=torch.bool)}
    for i,e in enumerate(examples):
        n=len(e['candidates']);values['candidates'][i,:n]=e['candidates'];values['valid'][i,:n]=True
        values['quality'][i,:n]=e['quality'];values['write_labels'][i,:n]=e['write_labels'];values['write_verified'][i,:n]=e['write_verified']
    return values


def forward(model,batch):
    return model(batch['anchor'],batch['candidates'],batch['valid'],batch['quality'],context=batch['context'])


def loss(output,batch):
    return training_losses(output,batch['target_slot'],batch['availability_label'],batch['write_labels'],batch['write_verified'])


def epoch(model,loader,device,optimizer,scaler):
    training=optimizer is not None;model.train(training);counts=0;totals={k:0. for k in ('total','decision','availability','write_correctness_proxy')}
    correct=available=accepted=negative=false_accept=0;gradients=[];overflows=[];fallbacks=0
    for source in loader:
        batch={k:v.to(device) for k,v in source.items()}
        retry=0
        while True:
            if training:optimizer.zero_grad(set_to_none=True)
            use_amp=device.type=='cuda' and retry<5
            with torch.set_grad_enabled(training),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                output=forward(model,batch);parts=loss(output,batch)
            if not all(torch.isfinite(v) for v in parts.values()):raise FloatingPointError('nonfinite actual training/inner loss')
            if not training:break
            scaler.scale(parts['total']).backward();scaler.unscale_(optimizer)
            invalid=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if invalid:
                overflows.append({'batch_start_sample':counts,'retry':retry,'AMP':use_amp,'scale':float(scaler.get_scale()),'nonfinite_parameter_names':invalid})
                if not use_amp:raise FloatingPointError('FP32 same-minibatch gradient nonfinite; stop with no invalid update')
                # GradScaler's checked found_inf path skips this optimizer step
                # and lowers the scale. Retry the same data/unchanged parameters.
                scaler.step(optimizer);scaler.update();retry+=1;continue
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),25.,error_if_nonfinite=True)
            gradients.append(float(norm));scaler.step(optimizer);scaler.update()
            if device.type=='cuda' and not use_amp:fallbacks+=1
            break
        n=len(batch['anchor']);counts+=n
        for k,v in parts.items():totals[k]+=float(v.detach())*n
        prediction=output['joint_probabilities'].argmax(-1);is_positive=batch['availability_label']==0
        none_slot=batch['candidates'].shape[1];predicted=prediction!=none_slot
        correct+=int(((prediction==batch['target_slot'])&is_positive).sum());available+=int(is_positive.sum())
        accepted+=int(predicted.sum());negative+=int((~is_positive).sum());false_accept+=int((predicted&~is_positive).sum())
    return {'samples':counts,'loss_components':{k:v/counts for k,v in totals.items()},'candidate_available_samples':available,
            'correct_candidate_recall':correct/available if available else None,'accepted_samples':accepted,
            'negative_samples':negative,'negative_false_accept_rate':false_accept/negative if negative else None,
            'gradient_norm_mean':float(np.mean(gradients)) if gradients else None,'gradient_norm_max':max(gradients) if gradients else None,
            'all_loss_and_gradient_finite':not overflows,'accepted_optimizer_gradients_finite':True,
            'AMP_overflow_events':overflows,'same_minibatch_FP32_fallbacks':fallbacks,
            'all_NONE':accepted==0,'all_present':accepted==counts}


def rng_state(generator):
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':(state[0],state[1].tolist(),state[2],state[3],state[4]),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'loader':generator.get_state()}


def restore_rng(state,generator):
    random.setstate(state['python']);ns=state['numpy'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),*ns[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def atomic_save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def fit(outer,seeds,gpu):
    torch.set_num_threads(1)
    protocol=read_json(OUT/'protocol/F1_ACIB_TRAINING.json')
    sequences=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'];fit_sequences,inner=split(outer,sequences)
    if not set(seeds)<=set(protocol['new_seeds']):raise ValueError('registered seeds only')
    # Mandatory controls must be actually complete, not replaced by interfaces.
    controls=read_json(OUT/'evaluation/FROZEN_ENCODER_CONTROLS.json')
    if set(controls['per_sequence'])!=set(sequences):raise ValueError('actual encoder controls incomplete')
    for s,h in controls['source_feature_seals_sha256'].items():
        if sha256(OUT/'baselines/encoder_controls/seals'/f'{s}.json')!=h:raise ValueError('control seals changed')
    if read_json(OUT/'baselines/SOT_DEVELOPMENT_SUMMARY.json')['pooled']['episodes']!=52:raise ValueError('actual B6 prerequisite')
    storage(100<<20)
    fit_data=T0Frames(fit_sequences,stride=protocol['T0_fit_stride_frames']);inner_data=T0Frames([inner],stride=1)
    if set(fit_data.sources)&{outer,inner} or set(inner_data.sources)!={inner}:raise ValueError('split leakage')
    device=checked_device(gpu)
    code={p:sha256(ROOT/p) for p in ['scripts/n72r21_train_t0.py','sam3_intermot/one_click/acib.py']}
    schema={'T0_protocol_sha256':sha256(OUT/'protocol/F1_ACIB_TRAINING.json'),'F1_protocol_sha256':sha256(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json'),
            'run_name':RUN_NAME,'numerical_repair_protocol_SHA256':sha256(OUT/'protocol/T0_AMP_NUMERICAL_REPAIR.json'),
            'code_sha256':code,'fit_sequences':fit_sequences,'inner_sequence':inner,'outer_sequence':outer,
            'fit_sources':fit_data.sources,'inner_sources':inner_data.sources,'outer_GT_or_pixels_read_by_fitting':False,
            'input_dim':512,'hidden_dim':128,'evidence_schema':list(EVIDENCE_SCHEMA),'availability_classes':list(AVAILABILITY_CLASSES),
            'physical_absence_supervision_available':False,'T0_empty_machine_bank':True,
            'fixed_OSNet_lineage_not_selected_from_outer_results':True}
    for seed in seeds:
        tag=f'{outer}__seed{seed}';directory=ASSETS/'training'/RUN_NAME/tag;done=OUT/'training'/RUN_NAME/f'{tag}.json'
        if done.exists() and read_json(done).get('completed'):
            record=read_json(done)
            if record['schema']!=schema or sha256(record['best_checkpoint_path'])!=record['best_checkpoint_sha256']:raise ValueError('sealed fit changed')
            print({'T0_reused_completed':tag},flush=True);continue
        started=time.monotonic();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        generator=torch.Generator().manual_seed(seed)
        model=ACIBNetwork().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.,enabled=device.type=='cuda')
        latest=directory/'latest.pt';best=directory/'best.pt';batch_size=protocol['batch_size'];start_epoch=0;best_loss=math.inf;stale=0;logs=[];oom=[]
        if latest.exists():
            saved=torch.load(latest,map_location='cpu',weights_only=True)
            if saved['schema']!=schema or saved['seed']!=seed:raise ValueError('resume lineage differs; version new run')
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            restore_rng(saved['rng'],generator);start_epoch=saved['epoch']+1;best_loss=saved['best_loss'];stale=saved['stale'];logs=saved['logs'];batch_size=saved['batch_size'];oom=saved['OOM']
        for number in range(start_epoch,protocol['max_epochs']):
            if stale>=protocol['patience']:break
            if time.monotonic()-started>protocol['max_seconds_each_fit']:raise RuntimeError('30-minute fit limit; latest checkpoint retained for resume')
            storage(100<<20)
            # Epoch-start recovery state allows identical selected samples when
            # batch size is reduced after an actual OOM, not GT-driven deletion.
            recovery={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                      'optimizer':copy.deepcopy(optimizer.state_dict()),'scaler':copy.deepcopy(scaler.state_dict()),'rng':rng_state(generator)}
            while True:
                fit_loader=DataLoader(fit_data,batch_size=batch_size,shuffle=True,generator=generator,collate_fn=collate,num_workers=0)
                inner_loader=DataLoader(inner_data,batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=0)
                try:
                    training=epoch(model,fit_loader,device,optimizer,scaler);validation=epoch(model,inner_loader,device,None,scaler);break
                except torch.cuda.OutOfMemoryError:
                    oom.append({'epoch':number,'batch_size':batch_size,'action':'restore_epoch_start_and_halve_same_sample_batch'})
                    if batch_size<=32:raise
                    model.load_state_dict(recovery['model']);optimizer.load_state_dict(recovery['optimizer']);scaler.load_state_dict(recovery['scaler'])
                    restore_rng(recovery['rng'],generator);torch.cuda.empty_cache();batch_size//=2
            current=validation['loss_components']['total'];improved=current<best_loss-1e-6
            if improved:best_loss=current;stale=0
            else:stale+=1
            row={'epoch':number,'fit':training,'inner':validation,'best_selected_on_inner_loss':improved,
                 'batch_size':batch_size,'seconds':time.monotonic()-started,'AMP':device.type=='cuda'};logs.append(row)
            snapshot={'schema':schema,'seed':seed,'epoch':number,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                      'scaler':scaler.state_dict(),'rng':rng_state(generator),'best_loss':best_loss,'stale':stale,'logs':logs,'batch_size':batch_size,'OOM':oom}
            if improved:atomic_save(best,snapshot)
            atomic_save(latest,snapshot)
            write_json(f'training/{RUN_NAME}/{tag}.json',{'completed':False,'schema':schema,'seed':seed,'logs':logs,'OOM':oom,
                'model_parameters':sum(p.numel() for p in model.parameters()),'best_checkpoint_path':str(best),
                'best_checkpoint_sha256':sha256(best),'latest_checkpoint_path':str(latest),'latest_checkpoint_sha256':sha256(latest),
                'runtime_device':str(device),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                'no_scientific_or_full_memory_training_claim':True})
            print(json.dumps({'T0':tag,'epoch':number,'fit_loss':training['loss_components']['total'],'inner_loss':current,
                              'inner_candidate_recall':validation['correct_candidate_recall'],'inner_negative_FPR':validation['negative_false_accept_rate'],
                              'accepted_gradient_finite':training['accepted_optimizer_gradients_finite'],
                              'AMP_overflow_count':len(training['AMP_overflow_events']),'best':improved,'seconds':round(time.monotonic()-started,1)}),flush=True)
        record=read_json(done);record.update(completed=True,early_stopped=stale>=protocol['patience'],
             fit_availability_class_counts=fit_data.counts,inner_availability_class_counts=inner_data.counts,
             full_T1_T2_T3_complete=False,outer_evaluation_executed=False,next_stage_authorized=False)
        write_json(f'training/{RUN_NAME}/{tag}.json',record)
        del model,optimizer,scaler
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103]);parser.add_argument('--gpu',type=int)
    args=parser.parse_args();fit(args.outer,args.seeds,args.gpu)
