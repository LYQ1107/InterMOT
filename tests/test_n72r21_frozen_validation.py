from scripts.n72r21_validation import cases,case_id
from scripts.n72r21_common import OUT,read_json


def test_all_registered_validation_sequences_and_seeds_not_selected_from_results():
    protocol=read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json')
    assert len(protocol['sequences'])==25 and len(set(protocol['sequences']))==25
    fit=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences']
    assert not set(protocol['sequences'])&set(fit)
    assert protocol['model_source_outer_configuration']==sorted(fit)[0]
    assert protocol['seeds']==[72101,72102,72103]
    assert not protocol['val_threshold_fit_checkpoint_or_method_selection']


def test_rule_cases_once_and_every_T2_seed_required():
    protocol=read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json');registered=cases(protocol)
    assert len(registered)==15
    assert len({case_id(name,seed) for name,seed,_ in registered})==15
    assert sum(seed is None for _,seed,_ in registered)==3
    assert sum(name=='ACIB_FULL' for name,_,_ in registered)==3
