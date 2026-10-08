"""Frozen selection and independent correction/damage counts (fixtures only)."""
from copy import deepcopy
import pytest
from scripts import n72r20r4r1_evaluate as evaluation
from scripts import n72r20r4r1_outer as outer


def test_N01_and_N10_are_independent_even_without_controller_approval(monkeypatch):
    monkeypatch.setattr(evaluation,'gt_by_frame',lambda p:{0:[],1:[],2:[]})
    monkeypatch.setattr(evaluation,'target_truth',lambda event,gt:1)
    monkeypatch.setattr(evaluation,'match_frame',lambda rows,gt:{'a':1,'b':2})
    frames=[({},[{'candidate_uid':'a'},{'candidate_uid':'b'}]) for _ in range(3)]
    def row(f,uid):
        return {'frame':f,'target_public_id':100001,'target_uid':uid,'state_after':str(uid),
            'outputs':[{'public_id':100001,'candidate_uid':uid}], 'identity_scores':[.9,.1],
            'memory':{'frame':f,'eligible':True,'accepted':False,'candidate_uid':uid},
            'proposals':[],'selected_action':None}
    baseline=[row(0,'a'),row(1,'a'),row(2,'b')]
    treatment=[row(0,'a'),row(1,'b'),row(2,'a')]
    stats=evaluation.posthoc('s',frames,{'event_frame':0},treatment,baseline)
    assert stats['target']['N01']==stats['target']['N10']==1
    assert stats['target']['globally_changed_output_frames']==2
    assert stats['funnel'].get('authority_approved_frames',0)==0


def test_outer_cannot_open_frames_without_all_policy_freezes(tmp_path,monkeypatch):
    monkeypatch.setattr(outer,'OUT',tmp_path);monkeypatch.setattr(outer,'check_storage',lambda n:None)
    monkeypatch.setattr(outer,'load_frames',lambda s:pytest.fail('outer input opened before freeze'))
    with pytest.raises(FileNotFoundError):outer.run_fold('dancetrack0062')


def test_formal_controls_keep_inner_parameters_and_include_individual_seeds():
    case={'name':'x','policy':{'family':'C0','source':'adapter','strength':2.,'top_k':3},'memory':{'family':'P0'},'controller_key':None,'memory_predictor':False,'native_predictor':False}
    keys=['C0','C1','NATIVE_GLOBAL','NATIVE_IDENTITY','C2_L0','C3_L0','C4_L0','C5_L1','C6_L0','C6_L1','C6_L2','C6_L3',*[f'P{i}' for i in range(1,7)]]
    family={'selected':{k:deepcopy(case) for k in keys}}
    cases=outer.cases_for_fold(family,{'cases':[]},{'selected':case});by_name={c['name']:c for c in cases}
    assert by_name['RAW_IDENTITY']['policy']['top_k']==3
    assert by_name['SHUFFLED_IDENTITY']['policy']['strength']==2.
    assert len([c for c in cases if 'single_seed' in c])==6
    assert 'FINAL_SELECTED' in by_name and 'POSITIVE_SOFT_CONTROL' in by_name
