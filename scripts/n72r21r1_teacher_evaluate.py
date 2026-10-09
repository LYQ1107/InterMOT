"""Post-seal C1/C6 diagnostics; oracle curves are not online MOT results."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.n72r21r1_common import OUT,read_json,write_json,sha256,update_status
from scripts.n72r21r1_teacher_diagnostic import PROTOCOL,MODES
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.evaluation.one_click_protocol import open_set_metrics


def describe(values):
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,
        'P10_P25_P50_P75_P90':np.quantile(a,[.1,.25,.5,.75,.9]).tolist() if len(a) else None}


def statistics(records,mode):
    rows=[r['modes'][mode] for r in records];available=[r for r in rows if r['candidate_available']]
    competitive=[r for r in rows if r['competitive_verified_other']]
    denominator=len(available)
    result={'frames':len(rows),'available_frames':denominator,
        'visible_frames':sum(r['target_visible_GT_diagnostic_only'] for r in records),
        'candidate_coverage_visible':denominator/sum(r['target_visible_GT_diagnostic_only'] for r in records) if any(r['target_visible_GT_diagnostic_only'] for r in records) else None,
        'rank1_given_available':sum(r['rank1_correct'] for r in available)/denominator if denominator else None,
        'MRR_given_available':sum(r['MRR'] for r in available)/denominator if denominator else None,
        'rank2_given_available':sum(r['rank2'] for r in available)/denominator if denominator else None,
        'rank3_given_available':sum(r['rank3'] for r in available)/denominator if denominator else None,
        'verified_other_competitive_frames':len(competitive),
        'positive_beats_verified_hard_negative_rate':sum(r['hard_negative_margin']>0 for r in competitive)/len(competitive) if competitive else None,
        'verified_hard_negative_margin':describe([r['hard_negative_margin'] for r in competitive]),
        'open_set_availability':open_set_metrics([r['candidate_available_probability'] for r in rows],
            [r['candidate_available'] for r in rows],[r['rank1_correct'] for r in rows]) if rows else None,
        'causal_feature_distributions':{key:describe([r[key] for r in rows if r[key] is not None]) for key in
            ['NONE_probability','candidate_available_probability','rank1_joint_probability','top1_top2_joint_margin','joint_entropy','bank_size_before_prediction']},
        'strict_selected_UID_given_available':sum(r['selected_correct'] for r in available)/denominator if denominator else None,
        'oracle_not_a_representation_ceiling_or_online_MOT_PASS':mode.startswith('POSTHOC_ORACLE'),
        'UNKNOWN_candidates_not_verified_hard_negatives':True}
    return result


def run():
    protocol=read_json(PROTOCOL);sources=[];all_rows=[];correction_count=0
    for event in protocol['events']:
        path=OUT/'diagnostics/teacher_state_v1'/event['episode_uid']/'seal.json';seal=read_json(path)
        assert seal['protocol_SHA']==sha256(PROTOCOL)
        assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts'])
        assert seal['C0_full_outputs_states_and_actual_write_training_features_equal']
        rows=read_zstd_jsonl(Path(next(a['path'] for a in seal['artifacts'] if a['kind']=='diagnostic')))
        assert len(rows)==seal['diagnostic_frames'] and all(set(r['modes'])==set(MODES) for r in rows)
        all_rows.extend(rows);correction_count+=seal['corrected_cache_fields'];sources.append({'path':str(path),'sha256':sha256(path)})
    sequences=sorted({r['sequence'] for r in all_rows});by_sequence={};by_role={};density={};gaps={}
    boundaries=read_json(OUT/'data/DENSITY_GROUPS.json')
    assert boundaries['definition'].startswith('Current valid real candidate count')
    for sequence in sequences:
        records=[r for r in all_rows if r['sequence']==sequence]
        by_sequence[sequence]={m:statistics(records,m) for m in MODES}
    for role in sorted({r['role'] for r in all_rows}):
        records=[r for r in all_rows if r['role']==role];by_role[role]={m:statistics(records,m) for m in MODES}
    for name,(low,high) in ((k,boundaries[k]) for k in ['sparse','medium','crowded']):
        records=[r for r in all_rows if low<=r['candidate_count'] and (high is None or r['candidate_count']<=high)]
        density[name]={'boundaries':[low,high],'available_records':len(records),
            'statistics':{m:statistics(records,m) for m in MODES},
            'full_MOT_HOTA_AssA_IDSW':'NOT_COMPUTED_BY_TARGET_REPRESENTATION_DIAGNOSTIC'}
    for name,lo,hi in [('1_5',1,5),('6_20',6,20),('21_50',21,50),('51_100',51,100),('101_plus',101,None)]:
        records=[r for r in all_rows if lo<=r['time_gap_original_frames'] and (hi is None or r['time_gap_original_frames']<=hi)]
        gaps[name]={m:statistics(records,m) for m in MODES}
    result={'stage':'N72R21R1','status':'COMPLETE_ACTUAL_C1_C6_POSTHOC_STATE_DIAGNOSTIC',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','independent_exposed_TRAIN_sequences':len(sequences),
        'actual_episodes':len(sources),'actual_postclick_decisions_per_mode':len(all_rows),'sources':sources,
        'protocol_SHA':sha256(PROTOCOL),'aggregate_by_mode':{m:statistics(all_rows,m) for m in MODES},
        'per_sequence':by_sequence,'per_role':by_role,'density':density,'time_gap':gaps,
        'current_probability_cache_metadata_corrections':correction_count,
        'all_original_full_C0_states_outputs_and_32_committed_training_features_equal':True,
        'original22_runtime_and_3_joint_writer_fits_not_modified':True,
        'cache_field_not_read_by_write_labels_or_fits':True,
        'posthoc_oracle_never_online_deployment_or_TrackEval_success':True,
        'fresh_confirmation_VAL_TEST_untouched':True,'next_stage_authorized':False}
    write_json('diagnostics/teacher_state_v1/RESULT.json',result)
    update_status(teacher_state_diagnostic_complete=True,current_cache_diagnostic_corrections=correction_count)
    print(json.dumps({'teacher_episodes':len(sources),'postclick_frames':len(all_rows),'cache_corrections':correction_count,
        'modes':{m:result['aggregate_by_mode'][m]['rank1_given_available'] for m in MODES}}),flush=True)


if __name__=='__main__':run()
