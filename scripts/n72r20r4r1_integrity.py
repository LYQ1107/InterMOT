"""Verify checkpoints, immutable historical evidence and closed-lineage tapes."""
from scripts.n72r20r4r1_common import *
from sam3_intermot.association.opportunity_tracker import FEATURE_NAMES
from scripts.n72r20r4r1_memory_fit import RELIABILITY_FEATURES


def run(seal=False):
    records=[];val_records=[]
    for path in sorted((ASSETS/'models').glob('*.pt')):
        record=read_json(path.with_suffix('.json'))
        if sha256(path)!=record['sha256']:raise RuntimeError('learned checkpoint SHA changed')
        c=torch.load(path,map_location='cpu',weights_only=False)
        if c.get('purpose')=='VAL_ALL_TRAIN_NATIVE_RELIABILITY':
            if c['actual_training_sequences']!=list(SEQUENCES) or c['forbidden_splits']!=['val','test']:raise RuntimeError('all-train native checkpoint split leak')
            if tuple(c['feature_names'])!=RELIABILITY_FEATURES or c['parameters']!=33:raise RuntimeError('all-train native feature/capacity mismatch')
            if c['seed'] not in SEEDS or c['epochs']!=30:raise RuntimeError('all-train native fit protocol mismatch')
            val_records.append({'path':str(path),'sha256':record['sha256'],'parameters':c['parameters'],'fit':list(SEQUENCES),
                'purpose':c['purpose'],'seed':c['seed'],'epochs':c['epochs'],'training_trace_digest':c['training_trace_digest']})
            continue
        outer=c.get('outer')
        if outer is None:outer=record['outer']
        fit,inner=fold_split(outer)
        if c['actual_training_sequences']!=fit or set(c['actual_training_sequences'])&{inner,outer}:raise RuntimeError('checkpoint training fold leak')
        names=RELIABILITY_FEATURES if 'purpose' in c else FEATURE_NAMES
        if tuple(c['feature_names'])!=names:raise RuntimeError('checkpoint feature allowlist changed')
        if c['parameters']>=50000:raise RuntimeError('model capacity exceeds frozen cap')
        records.append({'path':str(path),'sha256':record['sha256'],'parameters':c['parameters'],'fit':fit,'inner':inner,'outer':outer,
            'family':c['family'],'purpose':c.get('purpose','GLOBAL_ACTION_VALUE'),'seed':c['seed'],'epochs':c.get('epochs',30),
            'frozen_GRU_sha256':c.get('frozen_GRU_sha256',MEMORY_SHA)})
    if len(records)!=432:raise RuntimeError(f'expected 384 action plus48 reliability checkpoints, got {len(records)}')
    val_policy=OUT/'val/FROZEN_POLICY.json';val_adapters=[]
    if val_policy.exists():
        policy=read_json(val_policy)
        if len(val_records)!=3 or {r['seed'] for r in val_records}!=set(SEEDS):raise RuntimeError('incomplete all-train VAL native ensemble')
        if {r['sha256'] for r in val_records}!={r['sha256'] for r in policy['native_models']}:raise RuntimeError('VAL native freeze changed')
        val_adapters=policy['adapter_checkpoints']
        if val_adapters!=read_json(R4OUT/'val/FROZEN_POLICY.json')['adapter_checkpoints']:raise RuntimeError('source VAL Adapter seal mismatch')
        for r in val_adapters:
            if sha256(Path(r['path']))!=r['sha256']:raise RuntimeError('source VAL Adapter SHA changed')
            c=torch.load(r['path'],map_location='cpu',weights_only=False)
            if set(c.get('actual_training_sequences',c.get('parameter_fit_sequences')))!=set(SEQUENCES):raise RuntimeError('source VAL Adapter training split leak')
        if not policy['frozen_before_new_VAL_input_access'] or policy['VAL_used_for_selection_or_training']:raise RuntimeError('invalid VAL freeze')
        for s in sorted(events('val')):
            index=read_json(VAL_ROOT/'candidates'/s/'index.json')
            if sha256(Path(index['metadata']))!=index['metadata_sha256'] or sha256(Path(index['embeddings']))!=index['embeddings_sha256']:raise RuntimeError('VAL candidate SHA changed')
    elif val_records:raise RuntimeError('all-train models exist without final VAL policy freeze')
    if sha256(MEMORY_CHECKPOINT)!=MEMORY_SHA:raise RuntimeError('frozen GRU mutated')
    if source_hashes()!=read_json(OUT/'audit/HISTORICAL_HASHES_BEFORE.json'):raise RuntimeError('historical output mutated')
    source=read_json(OUT/'audit/CHECKPOINT_LINEAGE.json');adapters={s:strict_ensemble(s).manifest for s in SEQUENCES}
    source_adapters={r['path']:r['sha256'] for records in source['strict_adapter_models'].values() for r in records}
    if any(source_adapters.get(r['path'])!=r['sha256'] for records in adapters.values() for r in records):raise RuntimeError('source strict Adapter lineage changed')
    for s in SEQUENCES:
        index=read_json(DEV_ROOT/'candidates'/s/'index.json')
        if sha256(Path(index['metadata']))!=index['metadata_sha256'] or sha256(Path(index['embeddings']))!=index['embeddings_sha256']:raise RuntimeError('candidate input SHA changed')
    payload={'status':'COMPLETE_ACTUAL_CHECKPOINT_VERIFICATION','source_24_strict_six_fit_adapters':adapters,
        'frozen_GRU':{'path':str(MEMORY_CHECKPOINT),'sha256':MEMORY_SHA},'new_models':records,'new_action_model_count':384,'new_native_write_model_count':48,
        'all_fit_inner_outer_exclusions_verified':True,'all_SHA_verified':True,'encoder_sha256':ENCODER_SHA,'no_historical_files_modified':True}
    payload.update(new_all_train_VAL_native_models=val_records,source_three_all_train_VAL_adapters=val_adapters,
        new_model_total=len(records)+len(val_records),VAL_model_split_exclusions_verified=bool(val_records))
    write_json(OUT/'checkpoints/CHECKPOINT_MANIFEST.json',payload)
    if seal:
        for s in SEQUENCES:
            if read_json(OUT/'authority/outer'/f'{s}.json')['status']!='COMPLETE':raise RuntimeError('incomplete outer fold')
            for path in (OUT/'authority/frozen_outer'/f'{s}.json',OUT/'authority/frozen_joint'/f'{s}.json',OUT/'authority/frozen_final'/f'{s}.json'):
                if not read_json(path)['frozen_before_outer']:raise RuntimeError('outer freeze invalid')
        files=[p for p in OUT.rglob('*.json') if p.name not in ('SEALED_EVIDENCE.json','stage_status.json')]
        evidence={'checkpoint_manifest_sha256':sha256(OUT/'checkpoints/CHECKPOINT_MANIFEST.json'),'code_SHA':code_manifest(),
            'report_json_SHA':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)},'historical_outputs_unchanged':True,
            'all_trajectory_reconstruction_and_runtime_noGT_proofs':'evaluations/combined_DEV.json','next_stage_authorization_not_implied':True}
        if val_policy.exists():
            if read_json(OUT/'val/RESULT.json')['policy_sha256']!=sha256(val_policy):raise RuntimeError('evaluated VAL policy no longer frozen')
            evidence['VAL_trajectory_reconstruction_and_runtime_noGT_proofs']='evaluations/VAL_FROZEN.json'
        evidence['markdown_report_SHA']={str(p.relative_to(ROOT)):sha256(p) for p in sorted(OUT.rglob('*.md'))}
        evidence['final_project_report_SHA']=sha256(ROOT/'docs/N72R20R4R1_FINAL_REPORT.md')
        write_json(OUT/'checkpoints/SEALED_EVIDENCE.json',evidence)
    print(json.dumps({'new_checkpoint_SHA_verified':len(records)+len(val_records),'strict_Adapter_verified':24,'VAL_Adapter_verified':len(val_adapters),'historical_mutation':False,'sealed':seal}),flush=True)
    return payload


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');args=p.parse_args();run(args.seal)
