from copy import deepcopy
import pytest
from sam3_intermot.evaluation.controlled_unit_objective import CONDITIONS, paired_targets


def labels():
    current={"outcome":"TARGET","KEEP_outcome":"VERIFIED_OTHER","N01":1,"N10":0,"target_correct":1,
             "non_target_damage":0,"verified_OTHER_writes":0,"UNKNOWN_writes":0}
    future={"H100":{"complete":True,"benefit_label":False,"risk_label":False,"normalized_target_value_label":0.}}
    correction={"current_error_unit_corrected":1}
    return current,future,correction


def test_correction_not_future_gain_and_risk_value_are_exactly_shared():
    c,f,r=labels();before=deepcopy((c,f,r));targets=paired_targets(c,f,r)
    assert targets[CONDITIONS[0]]["benefit"]==0 and targets[CONDITIONS[1]]["benefit"]==1
    for key in ["risk","value"]:assert targets[CONDITIONS[0]][key]==targets[CONDITIONS[1]][key]
    assert (c,f,r)==before


def test_correcting_target_does_not_hide_collateral_harm():
    c,f,r=labels();c["non_target_damage"]=2;t=paired_targets(c,f,r)
    assert t[CONDITIONS[1]]["benefit"]==1 and t[CONDITIONS[1]]["risk"]==t[CONDITIONS[0]]["risk"]==1


def test_future_recovery_cannot_be_a_current_unit_correction():
    c,f,r=labels();c.update(outcome="NONE",N01=0,target_correct=0);r["current_error_unit_corrected"]=0
    f["H100"].update(benefit_label=True,normalized_target_value_label=.3)
    t=paired_targets(c,f,r);assert t[CONDITIONS[0]]["benefit"]==1 and t[CONDITIONS[1]]["benefit"]==0


def test_truncated_future_and_unknown_or_duplicate_null_are_not_negative():
    c,f,r=labels();f["H100"]["complete"]=False;assert paired_targets(c,f,r) is None
    c,f,_=labels();c["outcome"]="UNKNOWN";assert paired_targets(c,f,None) is None


@pytest.mark.parametrize("bad",["correct_reference","unknown","wrong_target","no_N01","N10","boolean_label"])
def test_unproved_or_inconsistent_correction_fails_closed(bad):
    c,f,r=labels()
    if bad=="correct_reference":c["KEEP_outcome"]="TARGET"
    elif bad=="unknown":c["outcome"]="UNKNOWN"
    elif bad=="wrong_target":r["current_error_unit_corrected"]=0
    elif bad=="no_N01":c["N01"]=0
    elif bad=="N10":c["N10"]=1
    else:r["current_error_unit_corrected"]=True
    with pytest.raises(ValueError):paired_targets(c,f,r)


def test_background_is_no_current_error_not_a_future_negative():
    c,f,r=labels();c.update(KEEP_outcome="TARGET",N01=0);f["H100"]["benefit_label"]=True
    t=paired_targets(c,f,None,correct_background=True)
    assert t[CONDITIONS[0]]["benefit"]==1 and t[CONDITIONS[1]]["benefit"]==0
    with pytest.raises(ValueError):paired_targets(c,f,r,correct_background=True)


def runtime(anchor=0.):
    from sam3_intermot.one_click.intervention_features import FEATURE_NAMES,feature_vector
    f={n:0. for n in FEATURE_NAMES};f['anchor_cosine']=anchor
    return {'features':f,'feature_vector':feature_vector(f).tolist(),
            'causal_previous_feature_vectors':{'H3':[[0.]*32 for _ in range(3)]}}


def test_pack_uses_actual_paired_KEEP_not_non_KEEP_features_with_renamed_branch():
    from scripts.n72r21r2_controlled_l1_source_v1 import pack
    row={'sequence':'s','episode_uid':'click','branch':'RAW_IDENTITY_TOP','frame':5,'role':'FIT','runtime_features':runtime(.9)}
    result=pack(row,runtime(.2),{},'CONTROLLED_CURRENT_ERROR','unit','event')
    assert result['x'][:32]!=result['keep'][:32]


def test_GT_root_ID_cannot_enter_runtime_feature_axis():
    from sam3_intermot.one_click.event_authority_learning import encode_runtime_input
    r=runtime();r['unit_uid']='offline_only'
    with pytest.raises(ValueError):encode_runtime_input(r,'KEEP')


def test_pack_rejects_a_borrowed_or_different_KEEP_history():
    from scripts.n72r21r2_controlled_l1_source_v1 import pack
    row={'sequence':'s','episode_uid':'click','branch':'KEEP','frame':5,'role':'FIT','runtime_features':runtime()}
    keep=runtime();keep['causal_previous_feature_vectors']['H3'][0][0]=1.
    with pytest.raises(AssertionError):pack(row,keep,{},'BACKGROUND','g','e')
