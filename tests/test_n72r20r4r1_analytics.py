"""Aggregate endpoint integrity; artificial fixtures are not research labels."""
from scripts.n72r20r4r1_analytics import memory_aggregate
from scripts.n72r20r4r1_combine import paired_bootstrap
from scripts.n72r20r4r1_common import SEQUENCES,write_json,sha256
from scripts import n72r20r4r1_outer as outer


def test_no_zero_write_safety_and_no_cross_sequence_cascade():
    empty={'memory':{'eligible_write_opportunities':5,'accepted_writes':0,'correct_writes':0,'wrong_writes':0,'correct_opportunities':5,'contamination_cascade_length_accepted_writes':0}}
    assert not memory_aggregate([empty,empty])['safety_pass']
    wrong={'memory':{**empty['memory'],'accepted_writes':2,'wrong_writes':2,'contamination_cascade_length_accepted_writes':2}}
    m=memory_aggregate([wrong,wrong])
    assert m['wrong_write_rate']==1. and m['contamination_cascade_max_within_sequence']==2


def test_paired_CI_is_sequence_macro_not_nonlinear_combined_delta():
    fields={'HOTA':.5,'AssA':.5,'DetA':.6,'IDF1':.4,'IDSW':5}
    base={**fields,'per_sequence':{s:fields for s in SEQUENCES}}
    treatment={**fields,'HOTA':.9,'per_sequence':{s:{**fields,'HOTA':.51} for s in SEQUENCES}}
    result=paired_bootstrap({'BASELINE':base,'v':treatment})['variants']['v']['HOTA']
    assert abs(result['official_combined_delta']-.4)<1e-12
    assert abs(result['macro_delta']-.01)<1e-12
    assert abs(result['CI95'][0]-.01)<1e-12


def test_final_pool_includes_native_only_without_rewriting_family_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(outer,'OUT',tmp_path)
    family=tmp_path/'authority/frozen_outer/s.json';joint=tmp_path/'authority/frozen_joint/s.json'
    metrics={'HOTA':.5,'AssA':.5,'DetA':.6,'IDSW':1};native={**metrics,'HOTA':.6}
    stats={'memory':{'safety_pass':False,'wrong_write_rate':None,'correct_write_retention':0.},'funnel':{}}
    base_case={'name':'BASELINE'};native_case={'name':'NATIVE_IDENTITY'}
    write_json(family,{'fit':[],'inner':'i','selected':{'C0':base_case,'NATIVE_IDENTITY':native_case},'inner_metrics':{'BASELINE':metrics,'NATIVE_IDENTITY':native},'inner_statistics':{'BASELINE':stats,'NATIVE_IDENTITY':stats}})
    before=sha256(family)
    write_json(joint,{'source_frozen_family_sha256':before,'cases':[{'name':'JOINT_BASELINE'}],'metrics':{'JOINT_BASELINE':metrics},'statistics':{'JOINT_BASELINE':stats}})
    write_json(tmp_path/'FINAL_SELECTION_PROTOCOL.json',{'fixture':True})
    result=outer.freeze_selection('s')
    assert result['selected']==native_case and sha256(family)==before
