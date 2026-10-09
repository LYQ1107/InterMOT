"""Real frozen person candidates on in-place legal TRAIN windows, GT-free."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21_sot import checked_device
from scripts.n72r20r1_candidate_stream_smoke import MachineOSNet,feature_sha256

PROTOCOL=OUT/'protocol/LASOT_FRAME_ONLY_IDENTITY.json'
CODE=['scripts/n72r21_lasot_candidates.py','scripts/n72r20r1_candidate_stream_smoke.py',
      'sam3_intermot/backend/sam3_backend.py','sam3_intermot/backend/sam3_compat.py']


def valid_crop(box,width,height):
    values=np.asarray(box,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all() or values[2]<=values[0] or values[3]<=values[1]:return False
    x,y,right,bottom=[int(round(v)) for v in values]
    return min(width,max(0,right))>min(width,max(0,x)) and min(height,max(0,bottom))>min(height,max(0,y))


def generate(name,gpu):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL)
    if name not in protocol['windows']:raise ValueError('frozen TRAIN window scope')
    fits=list((OUT/'training/T2_COUPLED_V1').glob('*.json'))
    if len(fits)!=24 or not all(read_json(p).get('completed') for p in fits):raise RuntimeError('wait for entire T2 GPU fitting phase')
    event=next(e for e in read_json(OUT/'development/LASOT_TRAIN_TRIM_CLICKS.json')['inputs'] if e['name']==name)
    root=Path(event['root']);paths=sorted(root.glob('*.jpg'));first=int(paths[0].stem)
    if [int(p.stem) for p in paths]!=list(range(first,first+len(paths))):raise ValueError('original contiguous source-frame axis')
    checkpoint=read_json(OUT/'datasets/LASOT_REUSED_SAM3_CHECKPOINT_VERIFICATION.json')
    if sha256(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('SAM3 source SHA')
    osnet=ROOT.parent/'InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth'
    if sha256(osnet)!='2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154':raise ValueError('OSNet source SHA')
    done=OUT/'experiments/LaSOT_identity/candidate_seals'/f'{name}.json';code={p:sha256(ROOT/p) for p in CODE}
    if done.exists():
        seal=read_json(done)
        if seal['code_sha256']!=code or seal['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('sealed candidate source changed')
        for artifact in seal['artifacts']:
            if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('sealed candidates changed')
        return
    storage(128<<20);device=checked_device(gpu);started=time.monotonic()
    from sam3_intermot.backend.sam3_backend import Sam3Backend
    backend=Sam3Backend(checkpoint_path=checkpoint['path'],max_num_objects=16,multiplex_count=16,use_fa3=False,use_rope_real=True,
        compile=False,warm_up=False,output_prob_thresh=.30,async_loading_frames=False,trim_past_non_cond_mem_for_eval=True,device=str(device))
    encoder=MachineOSNet(osnet,str(device));directory=ASSETS/'experiments/LaSOT_identity/candidates'/name;directory.mkdir(parents=True,exist_ok=True)
    meta=directory/'metadata.jsonl.zst';embedding=directory/'embeddings.f16';anchor=directory/'anchor.npy'
    if any(p.exists() for p in (meta,embedding,anchor)):raise FileExistsError('retain unsealed partial candidate data')
    seen=set();count=0;last=[];sessions=[];pixel_digest=[]
    try:
        initial_path=root/f"{event['anchor_original_frame_1based']:08d}.jpg"
        vector=encoder.encode(initial_path,[event['box_xyxy']])[0];np.save(anchor,vector.astype(np.float32))
        with meta.open('xb') as handle,embedding.open('xb') as features:
            proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=handle)
            try:
                def consume(frame,observations):
                    nonlocal count,last
                    frame=int(frame)
                    if frame in seen:return
                    if not 0<=frame<len(paths):raise ValueError('SAM3 frame outside real window')
                    if time.monotonic()-started>protocol['max_seconds_per_window']:raise RuntimeError('bounded window wall-clock ceiling')
                    retained=[o for o in observations if valid_crop(o.box_xyxy,event['width'],event['height'])]
                    last=[o.copy() for o in retained];boxes=[np.asarray(o.box_xyxy).tolist() for o in retained]
                    vectors=encoder.encode(paths[frame],boxes);backend._output_cache[frame]=[o.copy() for o in retained]
                    try:exported=backend.export_frame_candidates(frame,embeddings=vectors,include_masks=False,include_raw_provenance=True)
                    finally:backend._output_cache.pop(frame,None)
                    rows=[]
                    for i,c in enumerate(exported):
                        v=np.asarray(c['embedding'],np.float32)
                        if v.shape!=(512,) or not np.isfinite(v).all():raise ValueError('current OSNet vector')
                        uid=f"{name}:{first+frame}:{i}:{c.get('raw_native_id',c['native_tid'])}"
                        rows.append({'candidate_uid':uid,'box_xyxy':np.asarray(c['box_xyxy']).tolist(),
                            'confidence':float(c.get('confidence',0.)),'native_tid':int(c['native_tid']),
                            'raw_native_id':c.get('raw_native_id'),'embedding_offset':count,'feature_sha256_float32_before_quantization':feature_sha256(v)})
                        features.write(v.astype(np.float16).tobytes());count+=1
                    row={'local_frame_0based':frame,'original_frame_1based':first+frame,'candidates':rows,
                        'raw_candidate_count':len(observations),'invalid_or_clipped_empty_ROIs':len(observations)-len(retained),
                        'runtime_gt_used':False,'runtime_future_gt_used':False}
                    proc.stdin.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode());seen.add(frame)
                    pixel_digest.append(sha256(paths[frame]))
                for start in range(0,len(paths),160):
                    stop=min(len(paths)-1,start+159);backend.close();backend.start_video(str(root));propagation_start=start
                    if start and last:
                        seeds=[];ids=set()
                        for observation in last:
                            identity=int(observation.sam_object_id)
                            if identity in ids:
                                raw=getattr(observation,'raw_sam_object_id',None)
                                if raw is None or int(raw) in ids:continue
                                identity=int(raw)
                            ids.add(identity);seeds.append((identity,np.asarray(observation.box_xyxy,dtype=float)))
                        rebound=backend.rebind_past_state_boxes(start-1,seeds)
                        if rebound.get('recovered_count',0)>0:propagation_start=start-1
                        else:consume(start,backend.detect_concept(start,'person'))
                    else:consume(start,backend.detect_concept(start,'person'))
                    backend.propagate(propagation_start,stop,start_frame_index=propagation_start,keep_masks=False,cache_outputs=False,output_callback=consume)
                    sessions.append({'start':start,'stop':stop,'past_runtime_rebind':propagation_start==start-1,'GT_used':False})
                    print(json.dumps({'real_LaSOT_SAM3_chunk':name,'original_frame_end':first+stop,'seen_frames':len(seen),'seconds':round(time.monotonic()-started,1)}),flush=True)
                if seen!=set(range(len(paths))):raise ValueError('missing real window candidate decisions')
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError('compression failure')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
        artifacts=[{'kind':kind,'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size} for kind,p in [('metadata',meta),('embeddings',embedding),('anchor',anchor)]]
        if sum(a['bytes'] for a in artifacts)>128<<20:raise RuntimeError('bounded candidate-cache budget exceeded')
        write_json(f'experiments/LaSOT_identity/candidate_seals/{name}.json',{'name':name,'event':event,'artifacts':artifacts,
            'frame_count':len(paths),'embedding_count':count,'first_original_frame_1based':first,
            'code_sha256':code,'protocol_sha256':sha256(PROTOCOL),'source_checkpoint_sha256':checkpoint['sha256'],
            'pixel_sha256_in_original_frame_order':pixel_digest,'bounded_session_records':sessions,
            'loader_and_actual_pixel_forward_completed':True,'future_GT_opened':False,'one_click_initial_crop_only':True,
            'device':str(device),'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(device),'seconds':time.monotonic()-started,
            'new_pixels_copied_or_downloaded':False,'next_stage_authorized':False})
    except Exception as error:
        write_json(f'experiments/LaSOT_identity/failures/{name}.json',{'type':type(error).__name__,'reason':str(error),
            'partial_frames':len(seen),'partial_candidate_count':count,'partial_artifacts_retained':[str(p) for p in (meta,embedding,anchor) if p.exists()],
            'source_code_sha256':code,'no_partial_scientific_PASS':True,'other_jobs_terminated':False})
        raise
    finally:
        backend.close();del backend,encoder;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();generate(args.name,args.gpu)
