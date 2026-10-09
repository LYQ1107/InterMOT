"""Read-only input audit and fixed, all-case MOT diagnostic summary."""
import json
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,sha256,utcnow


def run():
    resultpath=OUT/'mot_pilot/RESULT.json';result=read_json(resultpath)
    if sha256(ROOT/'scripts/n72r21_mot_pilot_evaluate.py')!=result['evaluator_source_sha256']:
        raise ValueError('completed repaired evaluator source changed')
    if sha256(ROOT/'sam3_intermot/evaluation/mot_detection_audit.py')!=result['detection_audit_source_sha256']:
        raise ValueError('completed detection-audit source changed')
    models={};verified=[]
    for key,digest in result['runtime_seal_sha256'].items():
        case,sequence=key.split('/');path=OUT/'mot_pilot/runtime_seals'/case/f'{sequence}.json'
        if sha256(path)!=digest:raise ValueError('completed runtime seal changed')
        seal=read_json(path)
        for source,h in seal['code_sha256'].items():
            if sha256(ROOT/source)!=h:raise ValueError('runtime source changed')
        for artifact in seal['artifacts']:
            if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('MOT artifact changed')
        source=seal['source_model']
        if source is not None and source['path'] not in models:
            if sha256(source['path'])!=source['sha256'] or sha256(source['fit_record_path'])!=source['fit_record_sha256']:
                raise ValueError('frozen weights or fit record changed')
            fit=read_json(source['fit_record_path']);configuration=fit['schema']['configuration']
            if sequence in configuration['fit_sequences'] or sequence==configuration['inner_sequence']:
                raise ValueError('pilot outer sequence used in model fitting')
            for file,h in configuration['code_sha256'].items():
                if sha256(ROOT/file)!=h:raise ValueError('training source changed')
            models[source['path']]=source
        verified.append(key)
    if len(verified)!=32 or len(models)!=6 or result['TrackEval_returncode']!=0:
        raise ValueError('all preregistered runtime, model and evaluation cases required')
    if sha256(result['TrackEval_log']['path'])!=result['TrackEval_log']['sha256']:
        raise ValueError('actual official evaluator log changed')
    statistics={}
    for case,per_sequence in result['target_statistics'].items():
        rows=list(per_sequence.values());writes=sum(r['memory']['accepted_writes'] for r in rows)
        wrong=sum(r['memory']['wrong_writes'] for r in rows)
        visible=sum(r['target']['visible_frames'] for r in rows)
        correct=sum(r['secondary_strict_identity_claim']['actual_strict_correct_selected_frames'] for r in rows)
        eligible=sum(r['memory']['eligible_correct_observations'] for r in rows)
        statistics[case]={
            'strict_correct_future_target_frames':correct,'all_visible_future_target_frames':visible,
            'strict_UID_identity_recall_all_visible':correct/visible if visible else None,
            'accepted_writes':writes,'conservative_non_target_or_unverified_writes':wrong,
            'wrong_write_rate_upper_bound':wrong/writes if writes else None,
            'correct_written_observation_retention':(writes-wrong)/eligible if eligible else None,
            'UNKNOWN_unmatched_writes':sum(r['UNKNOWN_unmatched_memory_writes'] for r in rows),
            'verified_other_identity_writes':sum(r['verified_other_identity_memory_writes'] for r in rows),
            'N01_current_strict_corrections':sum(r['N01'] for r in rows),
            'N10_current_strict_regressions':sum(r['N10'] for r in rows),
            'non_target_initial_origin_correct_frame_damage_proxy':sum(r['non_target_initial_origin_correct_frame_damage'] for r in rows),
            'all_H100_future_frames_strict_benefit':sum(r['strict_future_identity_benefit']['H100']['all_future_frames_strict_target_benefit'] for r in rows),
        }
    baseline=result['actual_TrackEval_metrics']['CLICK_C0']
    learned={c:m for c,m in result['actual_TrackEval_metrics'].items() if c.startswith('ACIB_')}
    p0_full_same={}
    for seed in (72101,72102,72103):
        for sequence in result['metadata']:
            seals=[read_json(OUT/'mot_pilot/runtime_seals'/f'ACIB_{name}_SEED{seed}'/f'{sequence}.json') for name in ('FULL','ANCHOR_P0')]
            hashes=[next(a['sha256'] for a in s['artifacts'] if a['kind']=='trajectory') for s in seals]
            p0_full_same[f'{sequence}/seed{seed}']=hashes[0]==hashes[1]
    receipt={'utc':utcnow(),'RESULT_sha256':sha256(resultpath),'verifier_source_sha256':sha256(ROOT/'scripts/n72r21_mot_pilot_verify.py'),
        'all32_sealed_runtime_sources_and_artifacts_verified':verified,'all6_frozen_model_sources_verified':models,
        'target_aggregates_all16_cases':statistics,'FULL_vs_P0_exported_trajectory_byte_identity':p0_full_same,
        'diagnostic_findings':{
            'all12_learned_cases_lower_HOTA_AssA_IDF1_than_CLICK_C0':all(all(m[k]<baseline[k] for k in ('HOTA','AssA','IDF1')) for m in learned.values()),
            'all12_learned_cases_higher_IDSW_than_CLICK_C0':all(m['IDSW']>baseline['IDSW'] for m in learned.values()),
            'all16_cases_zero_registered_strict_H100_benefit':all(s['all_H100_future_frames_strict_benefit']==0 for s in statistics.values()),
        },
        'scope':'Two TRAIN scenes; failed frozen-weight MOT transfer diagnostic, not independent generalization or final scientific decision',
        'runtime_weights_thresholds_and_association_unchanged':True,'new_fitting_or_SOT_work':False,'next_stage_authorized':False}
    write_json('mot_pilot/RESULT_VERIFICATION.json',receipt)
    print(json.dumps({'verified_scene_cases':len(verified),'verified_frozen_models':len(models),'findings':receipt['diagnostic_findings']}))


if __name__=='__main__':run()
