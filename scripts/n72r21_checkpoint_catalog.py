"""Full actual N72R21 fitted-checkpoint catalog; no fitting, copying or selection."""
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import subprocess
import torch
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256,utcnow
from sam3_intermot.evaluation.checkpoint_audit import verify_checkpoint
from sam3_intermot.one_click.acib import ACIBNetwork
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork

STAGES={'T0':4,'T0_AMP_R1':24,'T1_CAUSAL_V1':72,'T2_COUPLED_V1':24}


def forward_contract(model):
    anchor=torch.zeros((1,512));anchor[0,0]=1
    candidates=anchor[:,None,:];quality=torch.zeros((1,1,5))
    result={}
    with torch.inference_mode():
        for tag,valid in [('one_candidate',True),('empty_candidates',False)]:
            output=model(anchor,candidates,torch.tensor([[valid]]),quality)
            joint=output['joint_probabilities']
            if not torch.isfinite(joint).all() or not torch.allclose(joint.sum(-1),torch.ones(1),atol=1e-6):
                raise ValueError('checkpoint forward probability contract')
            if not valid and float(joint[0,-1])!=1:raise ValueError('structural empty set must yield NONE')
            result[tag]={'joint_probability_shape':list(joint.shape),'finite':True,'sum':float(joint.sum()),
                'NONE_probability':float(joint[0,-1]),'synthetic_loader_contract_not_scientific_metric':True}
    return result


def run():
    torch.set_num_threads(1);records=[];fits=[];sources={}
    for stage,expected in STAGES.items():
        paths=sorted((OUT/'training'/stage).glob('*.json'))
        if len(paths)!=expected:raise ValueError('all registered fits including archived original runs required')
        for path in paths:
            record=read_json(path)
            if not record['completed']:raise ValueError('catalog cannot treat an incomplete fit as complete')
            schema=record['schema'];configuration=schema.get('configuration',schema)
            if configuration['outer_sequence'] in configuration['fit_sequences'] or configuration['outer_sequence']==configuration['inner_sequence']:
                raise ValueError('outer sequence in fitting or INNER')
            if len(configuration['fit_sequences'])!=6:raise ValueError('six actual FIT scenes required')
            source_proof={}
            for file,h in configuration['code_sha256'].items():
                actual=sha256(ROOT/file)
                if actual==h:
                    source_proof[file]={'sha256':h,'verification':'CURRENT_SOURCE_EXACT_MATCH'}
                    continue
                # The original four runs were already archived before the
                # preregistered numerical repair; never revert the live trainer
                # or overwrite their checkpoint/schema just to match hashes.
                repair=read_json(OUT/'protocol/T0_AMP_NUMERICAL_REPAIR.json')
                if stage!='T0' or file!='scripts/n72r21_train_t0.py' or h!=repair['original_training_source_SHA256']:
                    raise ValueError('non-archived fitted code changed')
                commit=repair['original_source_commit']
                blob=subprocess.run(['git','show',f'{commit}:{file}'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=10).stdout
                if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('original archived training source changed')
                source_proof[file]={'sha256':h,'verification':'EXACT_PRE_REPAIR_ARCHIVED_GIT_BLOB',
                    'original_commit':commit,'current_repaired_source_sha256':actual,'original_source_or_weights_rewritten':False}
            log=record['logs']
            if not log:raise ValueError('actual fitted epoch logs missing')
            for epoch in log:
                fit=epoch['fit']
                if not fit.get('accepted_optimizer_gradients_finite',fit.get('all_loss_and_gradient_finite',False)):
                    raise ValueError('accepted optimizer gradients are not verified finite')
                if not math.isfinite(fit['gradient_norm_max']) or not math.isfinite(fit['gradient_norm_mean']):
                    raise ValueError('finite actual gradient norms required')
            source=str(path.relative_to(ROOT));sources[source]=sha256(path)
            role='ARCHIVED_PRE_REPAIR_RUN_NOT_ELIGIBLE_FOR_REPAIRED_COMPARISONS' if stage=='T0' else 'ACTUAL_N72R21_TRAIN_ONLY_FIT'
            fits.append({'stage':stage,'fit_record':source,'sha256':sources[source],'role':role,'seed':record['seed'],
                'outer_sequence':configuration['outer_sequence'],'fit_sequences':configuration['fit_sequences'],
                'inner_sequence':configuration['inner_sequence'],'epochs_completed':len(log),'early_stopped':record['early_stopped'],
                'peak_GPU_allocated_bytes':record['peak_GPU_allocated_bytes'],'OOM':record['OOM'],
                'accepted_gradients_finite':True,'training_source_proof':source_proof,
                'last_epoch_fit_metrics':log[-1]['fit'],'last_epoch_inner_metrics':log[-1]['inner']})
            for kind in ('best','latest'):
                model=ACIBMemoryNetwork() if stage=='T2_COUPLED_V1' else ACIBNetwork()
                item=verify_checkpoint(record[f'{kind}_checkpoint_path'],record[f'{kind}_checkpoint_sha256'],schema,model)
                if item['model_parameters']!=record['model_parameters']:raise ValueError('actual parameter count differs')
                item.update(stage=stage,kind=kind,role=role,fit_record=source,fit_record_sha256=sources[source],
                    CPU_loader_forward=forward_contract(model),checkpoint_selection='INNER loss only; latest is resume state, not a selected evaluation checkpoint')
                records.append(item)
        print(json.dumps({'stage_checkpoint_SHA_strict_load_and_contract_complete':stage,'fits':len(paths),'checkpoints':2*len(paths)}),flush=True)
    base=read_json(OUT/'historical_audit/REUSABLE_ASSETS.json');foundation=[]
    items=[('OSNet',base['OSNet']['path'],base['OSNet']['sha256'],'historical_audit/REUSABLE_ASSETS.json'),
        ('SAM3',None,None,'datasets/LASOT_REUSED_SAM3_CHECKPOINT_VERIFICATION.json'),
        ('SECOND_REID',None,None,'baselines/SECOND_REID_CHECKPOINT_MANIFEST.json'),
        ('OSTRACK_HISTORICAL_SOT_DEFERRED',None,None,'baselines/SOT_CHECKPOINT_MANIFEST.json')]
    for name,path,h,manifest in items:
        provenance=read_json(OUT/manifest)
        path=path or provenance.get('path',provenance.get('destination'));h=h or provenance['sha256']
        if sha256(path)!=h:raise ValueError('existing foundation checkpoint changed')
        foundation.append({'role':name,'path':path,'sha256':h,'bytes':Path(path).stat().st_size,
            'manifest_file':f'outputs/N72R21/{manifest}','manifest_sha256':sha256(OUT/manifest),
            'checksum_reverified_now':True,'new_loader_or_inference':False,'new_download_or_copy':False,
            'vendor_checksum_verified':provenance.get('vendor_checksum_verified',False),'license_file':'outputs/N72R21/datasets/LICENSES.md',
            'public_binary_republication':False})
    general=read_json(OUT/'baselines/ENCODER_CONTROLS_LOADER_SMOKE.json')['lineage']['general']
    if sha256(general['checkpoint_path'])!=general['checkpoint_sha256']:raise ValueError('reused ImageNet cache changed')
    foundation.append({'role':'GENERAL_IMAGENET_CONTROL','path':general['checkpoint_path'],'sha256':general['checkpoint_sha256'],
        'bytes':Path(general['checkpoint_path']).stat().st_size,'new_download_or_copy':False,'new_loader_or_inference':False,
        'checksum_reverified_now':True,'public_binary_republication':False,'license_file':'outputs/N72R21/datasets/LICENSES.md'})
    result={'stage':'N72R21','final_goal_file':'outputs/N72R21/FINAL_GOAL.json','utc':utcnow(),
        'catalog_source_sha256':sha256(Path(__file__)),'loader_source_sha256':sha256(ROOT/'sam3_intermot/evaluation/checkpoint_audit.py'),
        'fits':fits,'fitted_checkpoints':records,'existing_foundation_checkpoints':foundation,'fit_record_source_sha256':sources,
        'actual_fit_counts_by_stage':dict(Counter(f['stage'] for f in fits)),
        'actual_checkpoints_verified':len(records),'actual_fitted_checkpoint_bytes':sum(r['bytes'] for r in records),
        'archived_original_runs_not_mixed_with_repaired_science':True,'all_fitting_in_TRAIN_only':True,
        'no_new_fitting_or_VAL_selection':True,'no_new_SOT_work':True,'weights_published_to_Git':False,'next_stage_authorized':False}
    write_json('checkpoints/SHA256_MANIFEST.json',result)
    write_json('training/CHECKPOINTS.json',{'canonical_manifest':'outputs/N72R21/checkpoints/SHA256_MANIFEST.json',
        'sha256':sha256(OUT/'checkpoints/SHA256_MANIFEST.json'),'actual_checkpoints_verified':len(records),
        'actual_fit_counts_by_stage':result['actual_fit_counts_by_stage'],'original4_runs_archived':True,'next_stage_authorized':False})
    print(json.dumps({'all_actual_N72R21_checkpoint_SHA_strict_load_complete':len(records),'actual_bytes':result['actual_fitted_checkpoint_bytes'],'foundation_checkpoints':len(foundation)}))


if __name__=='__main__':run()
