import json
from scripts import n72r21r1_inner_selection as module


def payload(outer_value):
    return {'TrackEval':{'LOGISTIC__MIXED__H100_GLOBAL_RISK__seed1':{'per_sequence':{
        'dancetrack0001':{'HOTA':outer_value,'AssA':outer_value},'dancetrack0002':{'HOTA':.4,'AssA':.2}}}},
        'statistics':{'LOGISTIC__MIXED__H100_GLOBAL_RISK__seed1/dancetrack0002':{
            'effective_interventions':0,'target_events':{'N01_frames':0,'N10_frames':0},'memory':{'accepted':0}}}}


def test_selection_values_never_use_exposed_outer_metrics(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'OUT',tmp_path);path=tmp_path/'result.json';path.write_text(json.dumps(payload(100.)))
    first=module.candidates('result.json','NEW')[0]
    path.write_text(json.dumps(payload(-100.)));second=module.candidates('result.json','NEW')[0]
    assert module.rank(first)==module.rank(second)
    assert first['INNER_full_TrackEval']==second['INNER_full_TrackEval']
    assert not first['qualified_nonvacuous_direction']


def test_INNER_only_benefit_with_nonzero_unsafe_memory_cannot_qualify(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'OUT',tmp_path);d=payload(1.)
    s=next(iter(d['statistics'].values()));s.update(effective_interventions=3,
        target_events={'N01_frames':10,'N10_frames':1,'direct_beneficial_decisions_against_own_KEEP':2,'direct_harmful_decisions_against_own_KEEP':1},
        memory={'accepted':2,'full_safety_and_usefulness_pass':False})
    (tmp_path/'result.json').write_text(json.dumps(d))
    assert not module.candidates('result.json','NEW')[0]['qualified_nonvacuous_direction']
