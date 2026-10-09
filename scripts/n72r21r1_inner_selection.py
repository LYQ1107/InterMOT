"""Post-fit INNER-only safety-first method closure, never best outer effects.

All input files contain exposed development results; selection reads only
0002 per-sequence metrics/statistics. This is exploratory historical INNER,
not a virgin validation or preregistered claim about an optimal architecture.
"""
from pathlib import Path
import json
from scripts.n72r21r1_common import OUT,read_json,write_json,sha256,update_status


def candidates(relative,kind):
    path=OUT/relative;d=read_json(path);fixed=relative.startswith('pilot/')
    metrics=d['all_actual_TrackEval_metrics_0to1'] if fixed else d['TrackEval']
    statistics=d['offline_target_event_statistics'] if fixed else d['statistics'];result=[]
    for case,m in metrics.items():
        if case in ('NO_HUMAN_C0','CLICK_C0') or 'SHADOW' in case or 'SHUFFLED' in case:continue
        if kind=='SIMPLE' and not (case.startswith(('RAW_', 'CONFIDENCE_', 'RECOVERY_', 'GLOBAL_REGRET_', 'PROTECTED_', 'DELAYED_', 'ABSTAIN_', 'E0_', 'E1_', 'E2_', 'E3_', 'E4_', 'E5_'))):continue
        if kind=='NEW' and case.startswith(('E0_', 'E1_', 'E2_', 'E3_', 'E4_', 'E5_')):continue
        s=statistics[case+'/dancetrack0002'];ev=s.get('target_events',{})
        n01=ev.get('N01_frames',s.get('N01',0));n10=ev.get('N10_frames',s.get('N10',0))
        damage=s.get('other_person_C0_origin_damage_frames',s.get('non_target_C0_initial_origin_damage_frames',0))
        harm=ev.get('direct_harmful_decisions_against_own_KEEP',s.get('direct_harmful_own_KEEP_decisions',0))
        benefit=ev.get('direct_beneficial_decisions_against_own_KEEP',s.get('direct_beneficial_own_KEEP_decisions',0))
        current_memory=s.get('memory',{});write_ok=not current_memory.get('accepted') or current_memory.get('full_safety_and_usefulness_pass',current_memory.get('safety_and_usefulness_pass',False))
        effective=s['effective_interventions'];safe=bool(effective>0 and n01>n10 and damage==0 and benefit>harm and write_ok)
        result.append({'case':case,'source_result_path':str(path),'source_result_SHA':sha256(path),'pool':kind,
            'INNER_sequence':'dancetrack0002','INNER_full_TrackEval':m['per_sequence']['dancetrack0002'],
            'INNER_effective_actions':effective,'INNER_N01':n01,'INNER_N10':n10,'INNER_other_person_damage_frames':damage,
            'INNER_current_beneficial_decisions':benefit,'INNER_current_harmful_decisions':harm,
            'current_decisions_not_independent_causal_roots':True,'INNER_memory_gate_for_nonzero_writes':write_ok,
            'qualified_nonvacuous_direction':safe,'combined_or_0001_metrics_not_read_for_selection':True})
    return result


def rank(r):
    return (-int(r['qualified_nonvacuous_direction']),r['INNER_current_harmful_decisions'],
        r['INNER_other_person_damage_frames'],-(r['INNER_N01']-r['INNER_N10']),
        -round(r['INNER_full_TrackEval']['AssA'],12),-round(r['INNER_full_TrackEval']['HOTA'],12),r['case'])


def run():
    prereg=read_json(OUT/'PREREGISTRATION.json')
    assert prereg['search']['selection'].startswith('INNER safety lexicographic')
    simple=candidates('pilot/FIXED_GATE_RESULT.json','SIMPLE')+candidates('availability/current_axis_pilot/RESULT.json','SIMPLE')
    new=[]
    for path in ['learned_pilot/RESULT.json','on_policy/correction_pilot/RESULT.json','memory/pilot/RESULT.json',
                 'availability/current_axis_pilot/RESULT.json','trajectory_gate/RESULT.json']:
        new+=candidates(path,'NEW')
    qualified_simple=sorted([r for r in simple if r['qualified_nonvacuous_direction']],key=rank)
    qualified_new=sorted([r for r in new if r['qualified_nonvacuous_direction']],key=rank)
    simple_selected=qualified_simple[0] if qualified_simple else {'case':'CLICK_C0','source_result_path':str(OUT/'pilot/FIXED_GATE_RESULT.json'),
        'source_result_SHA':sha256(OUT/'pilot/FIXED_GATE_RESULT.json'),'selection_status':'BASELINE_FALLBACK_NO_QUALIFIED_NONVACUOUS_SIMPLE'}
    if qualified_new:new_selected=qualified_new[0]
    else:
        # A retained model control, not "the best useful algorithm". Choosing
        # a lexical H100 representative avoids promoting an outer lucky run.
        representatives=sorted([r for r in new if r['source_result_path'].endswith('learned_pilot/RESULT.json') and '__MIXED__H100_GLOBAL_RISK__' in r['case']],key=lambda r:r['case'])
        assert representatives;new_selected={**representatives[0],
            'selection_status':'NO_QUALIFIED_NEW_METHOD_LEXICAL_H100_ABSTAIN_DIAGNOSTIC_ONLY'}
    record={'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','status':'COMPLETE_EXPLORATORY_HISTORICAL_INNER_ONLY_METHOD_SELECTION',
        'preregistration_SHA':sha256(OUT/'PREREGISTRATION.json'),'rule':prereg['search']['selection'],
        'operational_rule_detail':'Require nonzero actions, N01>N10, zero other-person proxy damage, more current beneficial than harmful decisions, nonzero-write safety/usefulness; then harmlessness,balance,AssA,HOTA,lexical tie. Current decision counts do not prove independent future beneficial onsets.',
        'simple_candidates':simple,'new_candidates':new,'qualified_simple_candidates':len(qualified_simple),'qualified_new_candidates':len(qualified_new),
        'BEST_SIMPLE_BASELINE':simple_selected,'BEST_NEW_METHOD_DIAGNOSTIC_REFERENCE':new_selected,
        'no_claim_of_optimal_new_method_if_none_qualified':True,'historical_INNER_exposed_not_virgin':True,
        'selection_before_fresh_confirmation_or_VAL_TEST':True,'source_selector_SHA':sha256(__file__),'next_stage_authorized':False}
    write_json('selection/INNER_ONLY_METHOD_SELECTION_V1.json',record)
    update_status(best_method_INNER_selection_complete=True,INNER_qualified_simple_methods=len(qualified_simple),INNER_qualified_new_methods=len(qualified_new))
    print(json.dumps({'INNER_qualified_simple':len(qualified_simple),'INNER_qualified_new':len(qualified_new),
        'simple':simple_selected['case'],'new_reference':new_selected['case']}),flush=True)


if __name__=='__main__':run()
