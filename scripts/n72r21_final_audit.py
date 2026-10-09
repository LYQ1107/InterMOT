"""Semantic delivery checks; scientific failure is not engineering completion."""
import json
from pathlib import Path
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256,utcnow
from scripts.n72r21_delivery_audit import NAMED_ARTIFACTS


def verified_reference(r):
    return sha256(ROOT/r['file'])==r['sha256']


def run():
    files={p:{'present':(OUT/p).is_file(),'sha256':sha256(OUT/p) if (OUT/p).is_file() else None} for p in NAMED_ARTIFACTS}
    if not all(r['present'] for r in files.values()): raise ValueError('named delivery artifacts incomplete')
    final=read_json(OUT/'FINAL_RESULT.json');tables=read_json(OUT/'evaluation/TABLES_A_TO_E.json')
    if not all(verified_reference(r) for r in final['source_evidence']+tables['source_evidence']): raise ValueError('final evidence changed after assembly')
    if final['next_stage_authorized'] is not False or final['scientific_success'] is not False: raise ValueError('no scientific PASS or next stage may be inferred')
    if sha256(ROOT/'docs/N72R21_FINAL_REPORT.md')!=sha256(OUT/'FINAL_REPORT.md'): raise ValueError('report mirrors disagree')
    history=read_json(OUT/'historical_audit/HISTORY_DELIVERY_CHECK.json');catalog=read_json(OUT/'checkpoints/SHA256_MANIFEST.json')
    videos=read_json(OUT/'visualizations/DELIVERY_VIDEO_RECEIPT.json');tests=read_json(OUT/'tests/REGRESSION_CURRENT.json')
    categories=[r['category'] for r in tests['failures']]
    if categories.count('PINNED_THIRD_PARTY_TRACKEVAL_CLI_SEQMAP_LIST_VS_PATH')!=4 or categories.count('HISTORICAL_LITERAL_BRANCH_ASSERTION_NOT_RESEARCH_RUNTIME_FAILURE')!=1:
        raise ValueError('actual regression failure classification differs; inspect before closure')
    if sha256(tests['actual_XML_path'])!=tests['actual_XML_sha256'] or sha256(tests['focused_actual_XML_path'])!=tests['focused_actual_XML_sha256']:
        raise ValueError('regression XML source changed')
    requirements=[
        {'item':1,'status':'DELIVERED_WITH_EXPLICIT_CAUSAL_REPLAY_LIMIT','evidence':['smoke/PIPELINE_SMOKE.json','protocol/WITHIN_VIDEO_PROTOCOL.json','mot_pilot/RESULT_VERIFICATION.json'],
         'scope':'Runnable causal current/past GT-free one-click identity/NONE replay and joint full-MOT TRAIN path; upstream cached-SAM3 online pixel causality not proven'},
        {'item':2,'status':'DELIVERED_RESTRICTED_LAWFUL_DATA_AUDIT','evidence':['datasets/DATASET_INTEGRITY.json','datasets/CHIRLA_METADATA_AUDIT.json','datasets/LASOT_PERSON_AUDIT.json','datasets/CROSS_RECORDING_RESUME_PLAN.json'],
         'scope':'Actual DanceTrack40TRAIN25VAL reused; official CHIRLA metadata only, media unavailable; previous LaSOT trim auxiliary retained, current SOT deferred'},
        {'item':3,'status':'DELIVERED','evidence':['datasets/DOWNLOAD_MANIFEST.json','datasets/DATASET_SOURCE_AUDIT.json','datasets/LICENSES.md'],
         'scope':'Actual lawful sources/direct-proxy history/licenses, no guessed vendor hash or provider permission bypass'},
        {'item':4,'status':'DELIVERED_NO_MATERIAL_CLEANUP','evidence':['storage/STORAGE_BEFORE.json','storage/CLEANUP_DRY_RUN.json','storage/CLEANUP_EXECUTED.json','storage/STORAGE_AFTER.json'],
         'scope':'Actual personal mount reserve/inodes and deletion receipts; no other-project/unique-historical data deletion'},
        {'item':5,'status':'DELIVERED_SCOPE_FROZEN','evidence':['protocol/SPLIT_MANIFEST.json','protocol/ONE_CLICK_INITIALIZATION.json','protocol/ABSENCE_PROTOCOL.json','protocol/CROSS_SESSION_PROTOCOL.json','protocol/EVALUATION_SPECIFICATION.json'],
         'scope':'Derived views reference original preregistration, official split/global-ID scope, no final query/VAL fit, no cross-session identity/FPS fiction'},
        {'item':6,'status':'DELIVERED_ALL_REGISTERED_COHORTS','evidence':['baselines/DEVELOPMENT_SUMMARY.json','validation/FROZEN_SUMMARY.json','mot_pilot/RESULT.json'],
         'scope':'All9 historical8TRAIN baselines, all15 frozen25VAL controls and all16 full-MOT TRAIN cases. Unregistered historical-adapter VAL comparison is NA'},
        {'item':7,'status':'DELIVERED_REAL_TRAIN_ONLY_FITTED_MODEL_SCIENTIFIC_FAIL','evidence':['training/TRAINING_MANIFEST.json','training/IDENTITY_REPRESENTATION.json','training/AVAILABILITY.json','training/SAFE_MEMORY.json'],
         'scope':'120 eligible actual T0/T1/T2 fits plus4 archived T0; no T3 cross-recording fit without lawful media'},
        {'item':8,'status':'DELIVERED_ACTUAL_SHA_AND_STRICT_LOAD','evidence':['checkpoints/SHA256_MANIFEST.json','training/CHECKPOINTS.json'],
         'scope':'Actual248 best/latest weights, full provenance/schema/finite state/strict load, CPU probability contract, training/gradient logs; no synthetic metrics or binary Git publication'},
        {'item':9,'status':'DELIVERED_WITH_EXPLICIT_UNAVAILABLE_CROSS_RECORDING','evidence':['experiments/WITHIN_VIDEO.json','experiments/REAPPEARANCE.json','evaluation/OPEN_SET_METRICS.json','experiments/CROSS_SESSION.json'],
         'scope':'All375 frozen scene-cases and all-frame negative/recovery diagnostics; cross-recording metrics null,0 actual episodes, resumable official-media plan'},
        {'item':10,'status':'DELIVERED_ALL_REGISTERED_CONTROLS','evidence':['ablations/ANCHOR_VS_BANK.json','ablations/NONE.json','ablations/SAFE_WRITE.json','ablations/STATE_MATCHING.json','ablations/VISUAL_ENCODER.json'],
         'scope':'All264 T2 module/capacity cases,120 T1 deployment replays and4 frozen representation controls; state-shift diagnostics not separately retrained-architecture causal proof'},
        {'item':11,'status':'DELIVERED_LOCAL_LICENSED_REAL_EXAMPLES','evidence':['visualizations/DELIVERY_VIDEO_RECEIPT.json'],
         'scope':'Actual16 readable clips incl4 inspected full-MOT examples; failures/NONE/recovery/unsafe writes retained; no invented cross-recording clip or public pixels'},
        {'item':12,'status':'DELIVERED_HONEST_ACTUAL_REGRESSIONS','evidence':['tests/REGRESSION_CURRENT.json','tests/DELIVERY_R1_LAUNCH_ENV_DIAGNOSTIC.json'],
         'scope':f"Actual{tests['passed']} passed/{tests['failed']} failed; four pinned CLI and one historical-branch failure; launch error/retry preserved, no hidden failures"},
        {'item':13,'status':'DELIVERED_RESTRICTED_SCIENTIFIC_FAIL','evidence':['FINAL_RESULT.json'],
         'scope':'Explicit gated current-system failure, missing-media and end-to-end causality limits, no next stage'},
        {'item':14,'status':'DELIVERED_FIVE_COHORT_SCOPED_TABLES','evidence':['FINAL_REPORT.md','evaluation/TABLES_A_TO_E.json'],
         'scope':'All five tables and full seed/sequence/failure scopes, mirrored docs, zeros versus undefined/NA distinguished'},
        {'item':15,'status':'PENDING_EXACT_CLEAN_NONFORCE_GIT_PUBLICATION','evidence':[],
         'scope':'No file-presence or scientific-result claim closes this item; verify localHEAD==remoteHEAD and fsmonitor-disabled clean worktree after publication'},
    ]
    checks={'all375_VAL':final['complete_frozen_VAL_scene_cases']==375,
            'all248_actual_checkpoints':catalog['actual_checkpoints_verified']==248,
            'all795_history_and463_models_unchanged':history['all_history_and_models_unchanged'] and history['metadata_and_historical_code_checked']==795 and history['historical_model_count']==463,
            'all16_real_readable_examples':videos['actual_video_count']==16,
            'all_five_tables_present':set(tables['tables'])=={f'Table_{s}' for s in 'ABCDE'},
            'no_unavailable_cross_recording_metric_fabricated':all(r['target_recall'] is None and r['valid_episodes']==0 for r in tables['tables']['Table_C']['cross_recording']),
            'MOT_primary_SOT_deferred':final['research_primary_track']=='MOT' and final['SOT_work_status']=='DEFERRED_BY_USER',
            'new_stage_not_authorized':final['next_stage_authorized'] is False}
    if not all(checks.values()):raise ValueError('semantic completion check failed')
    for r in requirements:
        r['evidence_SHA256']={p:sha256(OUT/p) for p in r['evidence']}
    report={'stage':'N72R21','utc':utcnow(),'status':'FEASIBLE_RESTRICTED_DELIVERY_VERIFIED_GIT_PUBLICATION_PENDING',
            'source_sha256':sha256(Path(__file__)),'named_artifacts':files,'semantic_requirements':requirements,
            'checks':checks,'scientific_decision':final['scientific_decision'],
            'missing_cross_recording_media_is_NOT_EVALUABLE_not_PASS':True,'SOT_expansion_deferred_by_latest_user':True,
            'artifact_presence_alone_is_not_completion_proof':True,'Goal_completion_proven':False,'next_stage_authorized':False}
    write_json('COMPLETION_REQUIREMENTS_AUDIT.json',report)
    status=read_json(OUT/'stage_status.json')
    status.update(status='RESTRICTED_SCIENTIFIC_FAIL_MOT_PRIMARY_FEASIBLE_DELIVERY_COMPLETE_GIT_PENDING',
                  scientific_success=False,scientific_decision=final['scientific_decision'],next_stage_authorized=False,
                  final_result_file='outputs/N72R21/FINAL_RESULT.json',final_result_sha256=sha256(OUT/'FINAL_RESULT.json'),
                  final_report_file='docs/N72R21_FINAL_REPORT.md',final_report_sha256=sha256(ROOT/'docs/N72R21_FINAL_REPORT.md'),
                  semantic_delivery_audit_file='outputs/N72R21/COMPLETION_REQUIREMENTS_AUDIT.json',
                  all375_frozen_VAL_scene_cases_completed=True,T3_complete=False,
                  T3_disposition='NOT_EVALUABLE_NO_LAWFUL_CROSS_RECORDING_MEDIA',
                  app_Goal_completion_proven=False,new_SOT_work_authorized=False)
    write_json('stage_status.json',status)
    print(json.dumps({'semantic_items_1_through14':'VERIFIED_WITH_EXPLICIT_RESTRICTED_SCIENCE_SCOPES','item15':'PENDING_GIT','checks':checks,'Goal_complete':False}))


if __name__=='__main__':run()
