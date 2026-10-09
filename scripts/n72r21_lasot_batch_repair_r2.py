"""Versioned official batch1 grounding, original code/weights retained."""
import argparse
import os
from pathlib import Path
import shutil
from scripts import n72r21_lasot_candidates as original
from scripts.n72r21_lasot_gpu_repair import checked_budget_device
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256

REPAIR=OUT/'protocol/LASOT_GROUNDING_BATCH_REPAIR_R2.json'
PARTIALS={
    'anchor.npy':(2176,'c79a7e59ec7f93ffe5992a8e4f3da91bcf2c48d4a834e765fa2c8477f853d909'),
    'embeddings.f16':(16384,'3c76bde70055ea33854c63cc49d6f1ae9a405036c46ccfc0d793416b7ada8386'),
    'metadata.jsonl.zst':(0,'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')}


def recover_R1_partials():
    receipt=OUT/'experiments/LaSOT_identity/R1_GPU_OOM_RECOVERY.json'
    if receipt.exists():
        old=read_json(receipt)
        if sha256(old['exact_failure_backup'])!=old['exact_failure_sha256']:raise ValueError('R1 failure evidence changed')
        for row in old['moved']:
            if sha256(row['recoverable_destination'])!=row['sha256']:raise ValueError('R1 partial evidence changed')
        return
    failurepath=OUT/'experiments/LaSOT_identity/failures/person-2.json';failure=read_json(failurepath)
    if (failure['type'],failure['partial_frames'],failure['partial_candidate_count'])!=('OutOfMemoryError',1,16):raise ValueError('exact inspected R1 failure required')
    if failure['source_code_sha256'].get('scripts/n72r21_lasot_gpu_repair.py')!=sha256(ROOT/'scripts/n72r21_lasot_gpu_repair.py'):raise ValueError('R1 source changed')
    directory=ASSETS/'experiments/LaSOT_identity/candidates/person-2';destination=ASSETS/'diagnostics/LaSOT_GPU_ALLOC_R1_PARTIALS/person-2'
    if set(failure['partial_artifacts_retained'])!={str(directory/name) for name in PARTIALS}:raise ValueError('exact partial scope required')
    moved=[]
    for name,(size,digest) in PARTIALS.items():
        p=directory/name;target=destination/name
        if p.is_symlink() or p.resolve()!=p or p.stat().st_uid!=os.getuid() or target.exists():raise ValueError('owned realpath and unused recovery target required')
        if p.stat().st_size!=size or sha256(p)!=digest:raise ValueError('inspected partial bytes changed')
        moved.append({'original_path':str(p),'recoverable_destination':str(target),'bytes':size,'sha256':digest})
    backup=OUT/'experiments/LaSOT_identity/R1_ONE_FRAME_GPU_OOM.json'
    if backup.exists():raise FileExistsError('preserve prior R1 failure backup')
    shutil.copy2(failurepath,backup)
    if sha256(backup)!=sha256(failurepath):raise ValueError('exact R1 failure copy mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    for row in moved:
        Path(row['original_path']).rename(row['recoverable_destination'])
        if sha256(row['recoverable_destination'])!=row['sha256']:raise ValueError('partial recovery changed bytes')
    write_json('experiments/LaSOT_identity/R1_GPU_OOM_RECOVERY.json',{'moved':moved,'material_bytes_deleted':0,
        'exact_failure_backup':str(backup),'exact_failure_sha256':sha256(backup),'repair_protocol_sha256':sha256(REPAIR),
        'datasets_weights_or_historical_stage_assets_changed':False})


def run(name,gpu):
    protocol=read_json(REPAIR)
    if sha256(ROOT/'scripts/n72r21_lasot_candidates.py')!=protocol['original_candidate_worker_sha256']:raise ValueError('original source changed')
    recover_R1_partials()
    import sam3_intermot.backend.sam3_backend as backend_module
    base=backend_module.Sam3Backend

    class BatchOneBackend(base):
        def __init__(self,*args,**kwargs):
            if 'official_batched_grounding_batch_size' in kwargs:raise ValueError('ambiguous batch resource policy')
            super().__init__(*args,official_batched_grounding_batch_size=1,**kwargs)

    backend_module.Sam3Backend=BatchOneBackend
    original.checked_device=checked_budget_device
    original.CODE=[*original.CODE,'scripts/n72r21_lasot_gpu_repair.py','scripts/n72r21_lasot_batch_repair_r2.py',
        'outputs/N72R21/protocol/LASOT_GPU_ALLOCATION_REPAIR_R1.json',
        'outputs/N72R21/protocol/LASOT_GROUNDING_BATCH_REPAIR_R2.json']
    original.generate(name,gpu)
    candidatepath=OUT/'experiments/LaSOT_identity/candidate_seals'/f'{name}.json'
    write_json(f'experiments/LaSOT_identity/R2_EFFECTIVE_CANDIDATE_PROTOCOL/{name}.json',{
        'name':name,'candidate_seal_sha256':sha256(candidatepath),'repair_protocol_sha256':sha256(REPAIR),
        'lineage':protocol['effective_candidate_lineage'],'official_batched_grounding_batch_size':1,
        'batch16_numeric_equivalence_proven':False,'source_weights_or_identity_models_changed':False,
        'original_generator_sha256':sha256(ROOT/'scripts/n72r21_lasot_candidates.py'),
        'next_stage_authorized':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();run(args.name,args.gpu)
