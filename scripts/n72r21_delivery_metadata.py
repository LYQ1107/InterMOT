"""Canonical delivery views of existing evidence, never new preregistration.

No model, immutable protocol, historical result, source data or runtime is
changed. Missing cross-recording media and the user's SOT deferral stay explicit.
"""
from collections import Counter
import configparser
import json
import os
from pathlib import Path
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256,utcnow,storage


def reference(relative):
    path=OUT/relative
    return {'file':f'outputs/N72R21/{relative}','sha256':sha256(path)}


def view(relative,content,sources):
    write_json(relative,{'stage':'N72R21','final_goal_file':'outputs/N72R21/FINAL_GOAL.json',
        'research_direction_file':'outputs/N72R21/RESEARCH_DIRECTION_OVERRIDE.json',
        'kind':'DERIVED_DELIVERY_VIEW_NOT_A_NEW_OR_CHANGED_PREREGISTRATION','assembled_utc':utcnow(),
        'source_evidence':[reference(p) for p in sources],**content,'next_stage_authorized':False})


def dataset_integrity():
    original=read_json(OUT/'datasets/DATASET_DISCOVERY.json');root=Path(original['dancetrack_root']);splits={}
    for split,expected in [('train',40),('val',25)]:
        records=[]
        for directory in sorted((root/split).iterdir()):
            if not directory.is_dir() or not directory.name.startswith('dancetrack'):continue
            info=configparser.ConfigParser();info.read(directory/'seqinfo.ini');section=info['Sequence']
            frames=int(section['seqLength']);fps=float(section['frameRate'])
            names=sorted(p.name for p in (directory/'img1').iterdir() if p.suffix.lower()=='.jpg')
            if names!=[f'{f:08d}.jpg' for f in range(1,frames+1)]:raise ValueError('dataset frame axis incomplete')
            gt=directory/'gt/gt.txt'
            if not os.access(gt,os.R_OK) or gt.stat().st_size<=0 or fps<=0:raise ValueError('GT or actual seqinfo unavailable')
            records.append({'sequence':directory.name,'frames':frames,'FPS_from_actual_seqinfo':fps,
                'all_original_JPG_frame_names_present':True,'GT_file_readable_nonempty':True,'GT_content_parsed_by_this_audit':False,
                'seqinfo_sha256':sha256(directory/'seqinfo.ini')})
        if len(records)!=expected:raise ValueError('registered train40/val25 root incomplete')
        splits[split]=records
    development=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json');validation=read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json')
    if not set(development['sequences'])<={r['sequence'] for r in splits['train']} or not set(validation['sequences'])<={r['sequence'] for r in splits['val']}:
        raise ValueError('registered sequence missing from actual data')
    view('datasets/DATASET_INTEGRITY.json',{'status':'CURRENT_DANCETRACK_TRAIN40_VAL25_FRAME_AXIS_AND_FILE_METADATA_VERIFIED',
        'DANCETRACK_ROOT':str(root),'splits':splits,'new_data_copies_or_downloads':False,
        'GT_semantics_not_redefined_by_metadata_check':True,
        'CHIRLA_media':'UNAVAILABLE; metadata is not video/annotations/crops',
        'LaSOT_current_work':'DEFERRED_BY_USER; previous incomplete/auxiliary outputs retained'},
        ['datasets/DATASET_DISCOVERY.json','datasets/CHIRLA_METADATA_AUDIT.json','datasets/LASOT_PERSON_AUDIT.json'])
    return splits


def run():
    direction=read_json(OUT/'RESEARCH_DIRECTION_OVERRIDE.json')
    if direction['research_primary_track']!='INTERACTIVE_MULTI_OBJECT_TRACKING' or direction['new_SOT_download_inference_training_or_resource_retry'] is not False:
        raise ValueError('latest explicit user MOT direction required')
    splits=dataset_integrity();development=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    validation=read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json');catalog=read_json(OUT/'checkpoints/SHA256_MANIFEST.json')
    if sha256(ROOT/'scripts/n72r21_checkpoint_catalog.py')!=catalog['catalog_source_sha256']:
        raise ValueError('catalog generator changed after checkpoint verification')
    if sha256(ROOT/'sam3_intermot/evaluation/checkpoint_audit.py')!=catalog['loader_source_sha256']:
        raise ValueError('catalog loader changed after verification')
    original_sources=['protocol/ENGINEERING_SMOKE_PROTOCOL.json','protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json',
        'protocol/F1_ACIB_TRAINING.json','protocol/T1_CAUSAL_STATE_TRAINING.json','protocol/T2_MEMORY_COUPLED_TRAINING.json',
        'protocol/T2_DEPLOYMENT_ABLATIONS.json','protocol/DANCETRACK_FROZEN_VALIDATION.json','protocol/MOT_TRAIN_PILOT.json']
    view('protocol/SPLIT_MANIFEST.json',{'TRAIN_development_sequences':development['sequences'],
        'VAL_sequences_excluded_from_fitting':validation['sequences'],'TRAIN_VAL_sequence_overlap':sorted(set(development['sequences'])&set(validation['sequences'])),
        'fit_folds':[{k:f[k] for k in ('stage','seed','fit_sequences','inner_sequence','outer_sequence','role','fit_record','sha256')} for f in catalog['fits']],
        'VAL_model_choice':validation['source_selection'],'VAL_checkpoint_selection':validation['checkpoint_selection'],
        'exposure':validation['exposure'],'global_identity_disjointness_claim':False},original_sources+['checkpoints/SHA256_MANIFEST.json'])
    view('protocol/ONE_CLICK_INITIALIZATION.json',{'initialization_label':'SIMULATED_ONE_CLICK_FROM_GT',
        'initialization_rule':validation['initialization'],'runtime':validation['runtime'],'initialization_failure_must_not_replace_target':True,
        'initialization_truth_access_disclosure':'Initialization identity metadata is distinct from future fitting/evaluation truth; all-click TRAIN initialization metadata parsing is explicitly disclosed',
        'additional_human_runtime_clicks':0,'immutable_anchor':True},
        ['protocol/ENGINEERING_SMOKE_PROTOCOL.json','protocol/DANCETRACK_FROZEN_VALIDATION.json','protocol/MOT_TRAIN_PILOT.json','training/INITIALIZATION_METADATA_ACCESS_DISCLOSURE.json'])
    view('protocol/WITHIN_VIDEO_PROTOCOL.json',{'development_protocol':reference('protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json'),
        'frozen_independent_sequence_validation':reference('protocol/DANCETRACK_FROZEN_VALIDATION.json'),
        'complete_joint_MOT_TRAIN_transfer':reference('protocol/MOT_TRAIN_PILOT.json'),
        'individual_target_episodes_are_not_full_MOT':True,'physical_absence_or_cross_recording_claim':False},original_sources)
    view('protocol/ABSENCE_PROTOCOL.json',{'primary_negative':'TARGET_VALID_CANDIDATE_UNAVAILABLE',
        'visible_target_without_candidate_is_separate_failure':True,'visible_GT_gap':'Physical exit versus full occlusion unresolved; do not call missing visible GT verified physical absence',
        'secondary_identity_calibration':'Verified rank1 target/other plus structural empty set; unmatched UNKNOWN excluded only from secondary calibration, never all-visible recall',
        'frozen_operating_point_not_reselected_from_VAL':True,'true_cross_recording_absence_evaluable':False},
        ['protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json','protocol/T2_VERIFIED_CLAIM_CALIBRATION_REPAIR.json','protocol/DANCETRACK_FROZEN_VALIDATION.json'])
    membership=read_json(OUT/'protocol/CHIRLA_OFFICIAL_MEMBERSHIP.json')
    view('protocol/CROSS_SESSION_PROTOCOL.json',{'status':'OFFICIAL_METADATA_SCOPE_AUDITED_RUNTIME_UNAVAILABLE_WITHOUT_LAWFUL_MEDIA',
        'official_membership':reference('protocol/CHIRLA_OFFICIAL_MEMBERSHIP.json'),
        'official_scenarios':list(membership['ReID_protocols']),
        'separate_Tracking_and_ReID_lineages':True,'local_ID_coincidence_is_not_global_ID':True,
        'TRAIN_gallery_overlap_must_not_be_hidden_by_path_prefix':True,'Gallery_is_anchor_only_never_parameter_fit':True,
        'Query_must_never_tune_models_policies_or_thresholds':True,
        'session_reset':'Reset motion/native tracklets/absolute frame context; preserve only eligible immutable identity anchor and trusted bank, without another human click',
        'actual_video_FPS_and_PTS_required_before_time_horizon_evaluation':True,
        'cross_video_stitching_or_guessed_identity_mapping_forbidden':True,
        'runtime_episodes_executed':0,'scientific_metrics':None,'reason':'No locally available authorized CHIRLA media/dense annotation; official public metadata is not a runtime episode'},
        ['datasets/CHIRLA_METADATA_AUDIT.json','protocol/CHIRLA_OFFICIAL_MEMBERSHIP.json','datasets/MEDIA_ACCESS_RECHECK.json','datasets/SCIENCEDB_RECHECK_20261009.json'])
    view('protocol/EVALUATION_SPECIFICATION.json',{'preexisting_quantitative_gates':{
        'minimum_usefulness':development['minimum_usefulness'],'effect':development['development_effect_gate'],'memory':development['memory_joint_gate']},
        'bootstrap':validation['bootstrap'],'seed_average_within_sequence_not_independent_seed_clusters':True,
        'primary_and_secondary_calibration_distinct':True,'posthoc_strata_are_descriptive_not_new_gates':True,
        'full_MOT_requires_joint_one_to_one_complete_trajectories':True,
        'cached_candidate_replay_is_not_end_to_end_online_SAM3_causality_or_latency_proof':True,
        'whole_stage_scientific_decision_not_from_two_TRAIN_scenes':True},
        original_sources+['protocol/POSTHOC_FAILURE_BREAKDOWNS.json','mot_pilot/EVALUATOR_ASSERT_REPAIR_R1.json'])
    view('datasets/CROSS_RECORDING_RESUME_PLAN.json',{'status':'NO_GATED_DOWNLOAD_OR_CONTACT_TERMS_ACCEPTED',
        'official_HF':'https://huggingface.co/datasets/bdager/CHIRLA',
        'official_ScienceDB':'https://www.scidb.cn/en/detail?dataSetId=2247f442a9784b5c959e7bead89c0313',
        'official_metadata_commit':'fcb6f53359d5888b6e8fb745b65a411697dcc22c',
        'resume_steps':[
            'Obtain authorized official access; user handles any login/contact-sharing agreement in the official UI; never send passwords/cookies/tokens',
            'List official files and size first; select only a metadata-supported bounded TRAIN pilot, not a full snapshot or final queries',
            'Respect original direct/proxy attempt records, actual free space and 60GiB reserve; checksum every obtained asset',
            'Verify authoritative global IDs, dense boxes, timestamps/FPS and official Tracking versus ReID memberships before constructing runtime episodes',
            'Freeze actual candidate, split, initialization and cross-session manifests before final query input access; no inferred cross-video IDs or extra clicks'],
        'new_SOT_work':False,'media_download_started_by_this_plan':False,'current_cross_recording_result':None},
        ['datasets/DOWNLOAD_MANIFEST.json','datasets/CHIRLA_METADATA_AUDIT.json','protocol/CHIRLA_OFFICIAL_MEMBERSHIP.json','datasets/PUBLIC_CATALOGUE_RETRY.json'])
    # Baseline aliases retain the complete original cohort and all negative results.
    baseline_files={'FROZEN_ANCHOR.json':'B0_RAW_ANCHOR','RAW_OSNET.json':'B0_RAW_ANCHOR','R3R2_ADAPTER.json':'B1_R3R2_ADAPTER',
        'R4_MEMORY.json':'B2_R4_CAUSAL_P1'}
    base=read_json(OUT/'baselines/DEVELOPMENT_SUMMARY.json')['baseline_results']
    # Names are validated against the actual frozen baselines, never fabricated.
    for filename,name in baseline_files.items():
        if name not in base:raise ValueError('canonical baseline alias requires the exact frozen case')
        view('baselines/'+filename,{'case':name,'scope':'Eight historically exposed TRAIN scenes, not25-sequence VAL or full-MOT evidence',
            'result':base[name],'missing_VAL_comparator_results_not_invented':True},['baselines/DEVELOPMENT_SUMMARY.json'])
    view('baselines/LONG_TERM_SOT.json',{'current_work_status':'DEFERRED_BY_USER','existing_reference_preserved':True,
        'completed_historical_box_only_reference':reference('baselines/SOT_DEVELOPMENT_SUMMARY.json'),
        'incomplete_SAM3_candidate_attempts_do_not_establish_identity_success':True,'new_work_started':False},
        ['baselines/SOT_DEVELOPMENT_SUMMARY.json','RESEARCH_DIRECTION_OVERRIDE.json'])
    view('training/TRAINING_MANIFEST.json',{'eligible_actual_fits':120,'archived_original_fits':4,'actual_fit_counts':catalog['actual_fit_counts_by_stage'],
        'actual_verified_checkpoint_count':catalog['actual_checkpoints_verified'],'T3_cross_recording_evaluation_executed':False,
        'no_cross_recording_or_physical_absence_supervision_claim':True,'best_checkpoint_selection':'TRAIN INNER only; all three seeds retained',
        'checkpoint_catalog':reference('checkpoints/SHA256_MANIFEST.json')},original_sources+['checkpoints/SHA256_MANIFEST.json'])
    for filename,scope in [('IDENTITY_REPRESENTATION.json','Actual T0_AMP_R1 and T1 causal-state parameter fitting'),
        ('AVAILABILITY.json','Actual learned candidate-availability/NONE supervision, not verified physical presence'),
        ('SAFE_MEMORY.json','Actual T2 paired write/no-write future supervision with independent future-safe head; training labels never future runtime input')]:
        view('training/'+filename,{'training_scope':scope,'actual_fits_and_gradient_records':reference('checkpoints/SHA256_MANIFEST.json'),
            'VAL_fitting_or_selection':False,'scientific_success_from_training_or_loader':False},
            ['training/TRAINING_MANIFEST.json','checkpoints/SHA256_MANIFEST.json'])
    print(json.dumps({'canonical_delivery_metadata_materialized':True,'train_sequences':len(splits['train']),'val_sequences':len(splits['val']),
        'protocol_definitions_or_frozen_sources_changed':False,'new_model_or_SOT_work':False}))


if __name__=='__main__':run()
