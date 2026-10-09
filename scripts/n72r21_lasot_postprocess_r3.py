"""Explicit official grounding/postprocessing batch1; bounded domain diagnostic."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts import n72r21_lasot_candidates as original
from scripts.n72r21_lasot_gpu_repair import checked_budget_device
from scripts.n72r21_lasot_ready import other_CPU_runtime_workers
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256

REPAIR=OUT/'protocol/LASOT_POSTPROCESS_BATCH_REPAIR_R3.json'


def recover_R2_partials():
    receipt=OUT/'experiments/LaSOT_identity/R2_GPU_OOM_RECOVERY.json'
    if receipt.exists():
        old=read_json(receipt)
        if sha256(old['exact_failure_backup'])!=old['exact_failure_sha256']:raise ValueError('R2 failure changed')
        for row in old['moved']:
            if sha256(row['recoverable_destination'])!=row['sha256']:raise ValueError('R2 partial changed')
        return
    failurepath=OUT/'experiments/LaSOT_identity/failures/person-2.json';failure=read_json(failurepath)
    if (failure['type'],failure['partial_frames'],failure['partial_candidate_count'])!=('OutOfMemoryError',128,2048):raise ValueError('exact inspected R2 failure required')
    if failure['source_code_sha256'].get('scripts/n72r21_lasot_batch_repair_r2.py')!=sha256(ROOT/'scripts/n72r21_lasot_batch_repair_r2.py'):raise ValueError('R2 worker changed')
    directory=ASSETS/'experiments/LaSOT_identity/candidates/person-2';destination=ASSETS/'diagnostics/LaSOT_GROUNDING_R2_PARTIALS/person-2';names=['anchor.npy','embeddings.f16','metadata.jsonl.zst']
    if set(failure['partial_artifacts_retained'])!={str(directory/name) for name in names}:raise ValueError('exact owned partial scope required')
    moved=[]
    for name in names:
        p=directory/name;target=destination/name
        if p.is_symlink() or p.resolve()!=p or p.stat().st_uid!=os.getuid() or target.exists():raise ValueError('owned realpath and unused recovery target required')
        moved.append({'original_path':str(p),'recoverable_destination':str(target),'bytes':p.stat().st_size,'sha256':sha256(p)})
    backup=OUT/'experiments/LaSOT_identity/R2_128_FRAME_GPU_OOM.json'
    if backup.exists():raise FileExistsError('preserve exact failure backup')
    shutil.copy2(failurepath,backup)
    if sha256(backup)!=sha256(failurepath):raise ValueError('failure backup mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    for row in moved:
        Path(row['original_path']).rename(row['recoverable_destination'])
        if sha256(row['recoverable_destination'])!=row['sha256']:raise ValueError('recovered bytes changed')
    write_json('experiments/LaSOT_identity/R2_GPU_OOM_RECOVERY.json',{'moved':moved,'material_bytes_deleted':0,
        'exact_failure_backup':str(backup),'exact_failure_sha256':sha256(backup),'repair_protocol_sha256':sha256(REPAIR),
        'old_assets_datasets_or_weights_changed':False})


def worker(name,gpu):
    protocol=read_json(REPAIR)
    if sha256(ROOT/'scripts/n72r21_lasot_candidates.py')!=protocol['original_candidate_worker_sha256']:raise ValueError('original generator changed')
    recover_R2_partials()
    import sam3_intermot.backend.sam3_backend as backend_module
    base=backend_module.Sam3Backend

    class BatchOneBackend(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,official_batched_grounding_batch_size=1,**kwargs)

        def _ensure_model(self):
            super()._ensure_model()
            model=self._predictor.model
            if not hasattr(model,'postprocess_batch_size'):raise RuntimeError('pinned official postprocess batch control unavailable')
            model.postprocess_batch_size=1

    backend_module.Sam3Backend=BatchOneBackend
    original.checked_device=checked_budget_device
    original.CODE=[*original.CODE,'scripts/n72r21_lasot_gpu_repair.py','scripts/n72r21_lasot_postprocess_r3.py',
        'outputs/N72R21/protocol/LASOT_GPU_ALLOCATION_REPAIR_R1.json',
        'outputs/N72R21/protocol/LASOT_POSTPROCESS_BATCH_REPAIR_R3.json']
    original.generate(name,gpu)
    candidatepath=OUT/'experiments/LaSOT_identity/candidate_seals'/f'{name}.json'
    write_json(f'experiments/LaSOT_identity/R3_EFFECTIVE_CANDIDATE_PROTOCOL/{name}.json',{
        'name':name,'candidate_seal_sha256':sha256(candidatepath),'repair_protocol_sha256':sha256(REPAIR),
        'lineage':protocol['effective_candidate_lineage'],'official_grounding_batch_size':1,'official_postprocess_batch_size':1,
        'default_batch_numeric_equivalence_proven':False,'source_weights_thresholds_or_identity_models_changed':False,
        'original_generator_sha256':sha256(ROOT/'scripts/n72r21_lasot_candidates.py'),'next_stage_authorized':False})


def driver(gpu):
    windows=read_json(OUT/'protocol/LASOT_FRAME_ONLY_IDENTITY.json')['windows']
    for name in windows:
        log=ASSETS/'diagnostics/LaSOT_POSTPROCESS_R3_logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        if log.exists():raise FileExistsError('preserve R3 logs, no automatic second attempt')
        with log.open('xb') as stream:
            result=subprocess.run([sys.executable,'scripts/n72r21_lasot_postprocess_r3.py','--worker','--name',name,'--gpu',str(gpu)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        write_json(f'experiments/LaSOT_identity/R3_LOGS/{name}.json',{'path':str(log),'sha256':sha256(log),'bytes':log.stat().st_size,
            'returncode':result.returncode,'repair_protocol_sha256':sha256(REPAIR)})
        print({'LaSOT_R3_candidate_result':name,'exit_code':result.returncode,'full_log_retained':str(log)},flush=True)
        if result.returncode:raise RuntimeError(f'R3 failed for {name}; retain evidence, stop this version')
    started=time.monotonic()
    while other_CPU_runtime_workers()>=4:
        if time.monotonic()-started>7200:raise RuntimeError('bounded CPU-slot wait; sealed candidates retained')
        time.sleep(20)
    for name in windows:
        for action in ('replay','evaluate'):
            subprocess.run([sys.executable,'scripts/n72r21_lasot_identity.py',action,'--name',name],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--name');parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();(worker(args.name,args.gpu) if args.worker else driver(args.gpu))
