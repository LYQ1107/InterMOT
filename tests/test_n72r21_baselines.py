import numpy as np
import pytest
from scripts.n72r21_baselines import click_uid, overlap, to_contract
from scripts.n72r21_baselines_geometry_repair import valid_geometry


def test_click_mapping_uses_only_current_geometry_and_uid_ties():
    rows=[{'candidate_uid':'b','box_xyxy':[0,0,10,10]},
          {'candidate_uid':'a','box_xyxy':[0,0,10,10]}]
    assert click_uid(rows,[0,0,10,10])=='a'
    assert click_uid(rows,[100,100,110,110]) is None
    assert overlap([0,0,10,10],[5,0,15,10])==pytest.approx(1/3)


def test_trace_adapter_does_not_equate_identity_probability_with_presence():
    row={'candidate_uid':'a','box_xyxy':[0,0,10,10]}
    d={'frame':1,'target_uid':'a','identity_scores':[.6],
       'memory':{'accepted':True,'candidate_uid':'a'}}
    r=to_contract(d,[row],{'recording_id':'clip'},'immutable','B4_P1')
    assert r['memory_write_candidate_uid']=='a'
    assert r['presence_probability_semantics'].startswith('UNCALIBRATED')
    assert r['runtime_future_gt_used'] is False
    d['target_uid']=None;d['memory']['accepted']=False
    r=to_contract(d,[row],{'recording_id':'clip'},'immutable','B5_ONLINE_NO_LTM')
    assert r['rank1_candidate_uid'] is None and r['candidate_available_probability']==0
    assert r['predicted_box_xyxy'] is None and not r['memory_write']


def test_geometry_repair_never_filters_by_identity_score_quality_or_future():
    assert valid_geometry({'box_xyxy':[0,0,1,1],'confidence':0,'identity_score':-1})
    assert not valid_geometry({'box_xyxy':[0,0,0,1]})
    assert not valid_geometry({'box_xyxy':[0,0,1,0]})
    assert not valid_geometry({'box_xyxy':[0,0,float('nan'),1]})
