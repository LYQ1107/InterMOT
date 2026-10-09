"""Explicit resource-only R1 wrapper, original generator/source untouched."""
import argparse
import os
from pathlib import Path
import subprocess
import torch
from scripts import n72r21_lasot_candidates as original
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256

REPAIR=OUT/'protocol/LASOT_GPU_ALLOCATION_REPAIR_R1.json'


def checked_budget_device(gpu):
    if gpu!=0:raise ValueError('sole registered GPU0')
    result=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    memory,util=[int(v.strip()) for v in result.stdout.strip().split(',')]
    if memory>64 or util:raise RuntimeError('GPU not genuinely idle; no unrelated job terminated')
    torch.cuda.set_device(gpu);total=torch.cuda.get_device_properties(gpu).total_memory
    fraction=min(1.,11*(1<<30)/total);torch.cuda.set_per_process_memory_fraction(fraction,gpu)
    write_json('storage/LASOT_GPU_RESOURCE_R1.json',{'gpu':gpu,'total_bytes':total,'allocator_fraction':fraction,
        'allocator_budget_bytes':min(total,11*(1<<30)),'hard_process_budget_GiB':12,'non_allocator_headroom_GiB':1,
        'initial_memory_MiB':memory,'initial_utilization':util,'original_frozen_cap_not_increased':True,
        'repair_protocol_sha256':sha256(REPAIR)})
    return torch.device(f'cuda:{gpu}')


def recover_original_partials():
    recovery=OUT/'experiments/LaSOT_identity/ORIGINAL_GPU_OOM_RECOVERY.json'
    if recovery.exists():
        for row in read_json(recovery)['moved']:
            if sha256(row['recoverable_destination'])!=row['sha256']:raise ValueError('original failure evidence changed')
        return
    failure=read_json(OUT/'experiments/LaSOT_identity/failures/person-2.json')
    if failure['partial_frames']!=0 or failure['partial_candidate_count']!=0 or failure['type']!='OutOfMemoryError':raise ValueError('exact zero-frame GPU failure required')
    directory=ASSETS/'experiments/LaSOT_identity/candidates/person-2';names=['metadata.jsonl.zst','embeddings.f16','anchor.npy']
    if set(failure['partial_artifacts_retained'])!={str(directory/name) for name in names}:raise ValueError('exact partial targets differ')
    destination=ASSETS/'diagnostics/LaSOT_GPU_ALLOC_ORIGINAL/person-2';moved=[]
    for name in names:
        path=directory/name;target=destination/name
        if path.is_symlink() or path.resolve()!=path or path.stat().st_uid!=os.getuid() or target.exists():raise ValueError('validated current-stage owner/realpath/recovery destination required')
        moved.append({'original_path':str(path),'recoverable_destination':str(target),'sha256':sha256(path),'bytes':path.stat().st_size})
    destination.mkdir(parents=True,exist_ok=True)
    for row in moved:
        Path(row['original_path']).rename(row['recoverable_destination'])
        if sha256(row['recoverable_destination'])!=row['sha256']:raise ValueError('recovery changed bytes')
    write_json('experiments/LaSOT_identity/ORIGINAL_GPU_OOM_RECOVERY.json',{'moved':moved,'material_bytes_deleted':0,
        'old_datasets_weights_or_historical_assets_changed':False,'repair_protocol_sha256':sha256(REPAIR)})


def run(name,gpu):
    protocol=read_json(REPAIR)
    if sha256(ROOT/'scripts/n72r21_lasot_candidates.py')!=protocol['original_candidate_worker_sha256']:raise ValueError('original generator changed')
    recover_original_partials()
    # Versioned dependency override is only allocation policy. Both the
    # original worker SHA and this wrapper/protocol SHA enter every seal.
    original.checked_device=checked_budget_device
    original.CODE=[*original.CODE,'scripts/n72r21_lasot_gpu_repair.py','outputs/N72R21/protocol/LASOT_GPU_ALLOCATION_REPAIR_R1.json']
    original.generate(name,gpu)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();run(args.name,args.gpu)
