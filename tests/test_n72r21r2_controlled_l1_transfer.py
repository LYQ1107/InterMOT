import pytest
from scripts import n72r21r2_main_policy as engine
from scripts.n72r21r2_controlled_l1_mot_v1 import scoped_engine,PROTOCOL,execute
from scripts.n72r21r2_controlled_l1_mot_driver_v1 import slot_ready,TERMINAL


def terminal():
    return {'status':TERMINAL,'active':None,'completed':[str(i) for i in range(192)],'failed_retained':[],
            'group_status':{'BASELINE_STATE':'NOT_RUN_ACTUAL_SUPERVISION_FAILURE_NOT_A_WEIGHT'}}


def test_live_original_state_process_even_with_terminal_marker_never_frees_slot():
    assert not slot_ready(terminal(),True)


def test_missing_process_does_not_turn_stale_active_marker_into_completion():
    m=terminal();m['status']='ACTIVE_FINITE9_MATCHED_STATE_OWN_FULL_MOT'
    assert not slot_ready(m,False)


@pytest.mark.parametrize('bad',['missing','duplicate','failure','lost_no_fit'])
def test_incomplete_or_failed_state_is_not_bypassed(bad):
    m=terminal()
    if bad=='missing':m['completed'].pop()
    elif bad=='duplicate':m['completed'][0]=m['completed'][1]
    elif bad=='failure':m['failed_retained']=[{'actual_exit':1}]
    else:m['group_status']={}
    with pytest.raises(RuntimeError):slot_ready(m,False)


def test_complete_state_dead_owner_is_only_slot_ready_not_MOT_SUCCESS():
    assert slot_ready(terminal(),False)


def test_process_local_adapter_restores_all_symbols_on_failure():
    names=('PROTOCOL','PREFIX','BASE','CODE','inputs','write_json');old={n:getattr(engine,n) for n in names}
    with pytest.raises(RuntimeError):
        with scoped_engine():
            assert engine.PROTOCOL==PROTOCOL
            raise RuntimeError('diagnostic test failure')
    assert all(getattr(engine,n) is v for n,v in old.items())


def test_unregistered_action_cannot_requery_train_or_open_confirmation():
    with pytest.raises(ValueError):execute('requery',None,None,None,None)


def test_runtime_binds_dataset_hash_metadata_without_reading_target_payload(monkeypatch):
    from scripts import n72r21r2_controlled_l1_mot_v1 as m
    condition=m.CONDITIONS[0];seed=730101;fitpath=m.fit_path(condition,seed)
    p={'goal':m.GOAL,'source_sha256':{n:'frozen' for n in m.CODE},'conditions':list(m.CONDITIONS),
       'seeds':[seed],'points':{'P0':{}},'split':{'fit':[],'inner':['s']},'fit_protocol_sha256':'frozen',
       'inherited_MAIN_protocol_sha256':'frozen','source_dataset_sha256':'label-payload-digest',
       'fit_refs':{str(fitpath):{'fit_sha256':'frozen','checkpoint_sha256':'frozen'}}}
    r={'condition':condition,'seed':seed,'family':'SMALL_MLP','objective':m.OBJECTIVE,'checkpoint_path':'/registered-weight.pt',
       'checkpoint_sha256':'frozen','protocol_sha256':'frozen','source_dataset_sha256':'label-payload-digest',
       'nonzero_gradient_steps':2,'changed_weight_elements_by_tensor':{'w':1},'actual_model_generated_on_policy':False,
       'next_association_stage_authorized':False}
    def digest(path):
        assert path!=m.DATA, 'Runtime must not read source GT label payload even just to hash it'
        return 'frozen'
    monkeypatch.setattr(m,'sha256',digest);monkeypatch.setattr(m,'read_json',lambda path:p if path==m.PROTOCOL else r)
    monkeypatch.setattr(m,'development_sequence',lambda sequence:None)
    assert m.inputs('s',condition,m.OBJECTIVE,seed,'P0')[2] is r
