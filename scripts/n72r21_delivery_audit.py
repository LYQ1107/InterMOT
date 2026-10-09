"""Evidence-backed intermediate audit; file existence alone never closes Goal."""
import json
import os
from pathlib import Path
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256,utcnow

NAMED_ARTIFACTS=[
    'FINAL_GOAL.json','PREREGISTRATION.json','stage_status.json','EXECUTION_LOG.md',
    'historical_audit/R20_LESSONS.md','historical_audit/REUSABLE_ASSETS.json',
    'datasets/DATASET_SOURCE_AUDIT.json','datasets/DOWNLOAD_MANIFEST.json','datasets/DATASET_INTEGRITY.json',
    'datasets/CHIRLA_METADATA_AUDIT.json','datasets/LASOT_PERSON_AUDIT.json','datasets/LICENSES.md',
    'storage/STORAGE_BEFORE.json','storage/CLEANUP_DRY_RUN.json','storage/CLEANUP_EXECUTED.json','storage/STORAGE_AFTER.json',
    'protocol/SPLIT_MANIFEST.json','protocol/ONE_CLICK_INITIALIZATION.json','protocol/WITHIN_VIDEO_PROTOCOL.json',
    'protocol/CROSS_SESSION_PROTOCOL.json','protocol/ABSENCE_PROTOCOL.json','protocol/EVALUATION_SPECIFICATION.json',
    'smoke/PIPELINE_SMOKE.json','smoke/EXAMPLE_POSITIVE.json','smoke/EXAMPLE_ABSENT.json','smoke/EXAMPLE_REAPPEARANCE.json',
    'baselines/FROZEN_ANCHOR.json','baselines/RAW_OSNET.json','baselines/R3R2_ADAPTER.json','baselines/R4_MEMORY.json','baselines/LONG_TERM_SOT.json',
    'training/TRAINING_MANIFEST.json','training/IDENTITY_REPRESENTATION.json','training/AVAILABILITY.json','training/SAFE_MEMORY.json','training/CHECKPOINTS.json',
    'experiments/WITHIN_VIDEO.json','experiments/REAPPEARANCE.json','experiments/CROSS_CAMERA.json','experiments/CROSS_DAY.json',
    'experiments/CROSS_SESSION.json','experiments/LOW_DENSITY.json','experiments/HIGH_DENSITY.json',
    'ablations/ANCHOR_VS_BANK.json','ablations/NONE.json','ablations/SAFE_WRITE.json','ablations/STATE_MATCHING.json','ablations/VISUAL_ENCODER.json',
    'evaluation/TARGET_METRICS.json','evaluation/OPEN_SET_METRICS.json','evaluation/LONG_TERM_METRICS.json','evaluation/MEMORY_SAFETY.json',
    'evaluation/MOT_SUPPLEMENTARY.json','evaluation/UNCERTAINTY.json','checkpoints/SHA256_MANIFEST.json','FINAL_RESULT.json','FINAL_REPORT.md']


def run():
    files={p:{'present':(OUT/p).is_file(),'sha256':sha256(OUT/p) if (OUT/p).is_file() else None} for p in NAMED_ARTIFACTS}
    registered=read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json');val_complete=True;missing=[]
    from scripts.n72r21_validation import cases,case_id
    for name,seed,_ in cases(registered):
        case=case_id(name,seed)
        for sequence in registered['sequences']:
            if not (OUT/'validation/evaluations'/case/f'{sequence}.json').exists():missing.append(f'{case}/{sequence}')
    val_complete=not missing
    live=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
            if len(args)>1 and Path(args[1]).name in {'n72r21_validation_ready.py','n72r21_validation.py'}:
                live.append({'pid':int(p.name),'script':Path(args[1]).name,'phase':next((a for a in args[2:] if a in ('replay','evaluate')),None)})
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    catalog=read_json(OUT/'checkpoints/SHA256_MANIFEST.json');mot=read_json(OUT/'mot_pilot/RESULT_VERIFICATION.json')
    evidence={
        'previous_goal_turn':{'classification':'PROGRESS','evidence_commit':'39e63581a2d4b49164a9c9a1d12b278e50cb1d8b','work':'MOT direction override; all32 complete joint trajectories, official evaluation, preserved negative results, exact nonforce publication'},
        'fitting_and_checkpoint_delivery':{'actual_eligible_fits':120,'archived_original_fits':4,'actual_checkpoints':catalog['actual_checkpoints_verified'],
            'all124_fit_provenance_and_248_strict_model_loads_checked':catalog['actual_checkpoints_verified']==248,'manifest_sha256':sha256(OUT/'checkpoints/SHA256_MANIFEST.json')},
        'full_MOT_transfer_diagnostic':{'all32_verified':len(mot['all32_sealed_runtime_sources_and_artifacts_verified'])==32,
            'all16_cases_and_all3_seeds_negative_retained':mot['diagnostic_findings'],'independent_VAL_MOT_generalization_established':False},
        'frozen_VAL':{'complete_all375':val_complete,'missing_scene_cases':missing,'confirmed_live_owned_processes':live,'no_restart_or_selection_from_wait':True},
        'required_final_tables_A_to_E':{'status':'NOT_YET_ASSEMBLED_FINAL_COHORT','missing_input':'Full frozen VAL and pooled curves if incomplete; otherwise final assembler'},
        'cross_recording':{'status':'NOT_EVALUABLE_NO_LAWFUL_LOCAL_MEDIA','official_metadata_and_lineages_audited':True,
            'actual_cross_recording_episodes':0,'resume_plan':'outputs/N72R21/datasets/CROSS_RECORDING_RESUME_PLAN.json','scientific_cross_day_success':False},
        'SOT':{'status':'DEFERRED_BY_EXPLICIT_LATEST_USER_DIRECTION','prior_outputs_preserved':True,'new_work_authorized':False},
        'scientific_Gate0':{'status':'NOT_PROVEN_END_TO_END_ONLINE_PIXEL_PIPELINE','reason':'Actual GT-free current/past replay contract does not prove cached SAM3 upstream future-pixel causality/latency'},
        'scientific_Gate1_Gate2':{'status':'FINAL_FROZEN_COHORT_PENDING' if not val_complete else 'FINAL_GATE_ASSEMBLY_PENDING'},
        'scientific_Gate3':{'status':'NOT_EVALUABLE_NO_MEDIA','PASS':False},
        'scientific_Gate4':{'status':'FAIL_TRAIN_MEMORY_SAFETY_AND_USEFULNESS_FINAL_VAL_PENDING','zero_write_PASS_allowed':False},
        'final_scientific_decision_and_Git_delivery':{'status':'INCOMPLETE','requires':'Final files, semantic requirement-by-requirement audit, actual complete regressions, history/storage receipts, exact local/remote HEAD and clean worktree'},
    }
    write_json('COMPLETION_REQUIREMENTS_AUDIT.json',{'stage':'N72R21','utc':utcnow(),'audit_source_sha256':sha256(Path(__file__)),
        'status':'INCOMPLETE_FINAL_DELIVERY','named_artifacts':files,'evidence':evidence,
        'frozen_original_scope_preserved_except_explicit_latest_SOT_deferral':True,
        'artifact_presence_is_not_completion_proof':True,'Goal_completion_proven':False,'next_stage_authorized':False})
    print(json.dumps({'canonical_named_artifacts_present':sum(r['present'] for r in files.values()),'expected':len(files),
        'frozen_VAL_missing_scene_cases':len(missing),'confirmed_live_VAL_processes':len(live),'Goal_completion_proven':False}))


if __name__=='__main__':run()
