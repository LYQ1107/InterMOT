from scripts.n72r21_label_coupled_states import risk_label


def proposal(with_scores,without_scores,selected='p'):
    def row(scores):return {'frame':3,'selected_candidate_uid':selected,'current_candidate_logits':scores}
    return {'proposal_frame':2,'proposed_candidate_uid':'p','branches':{'with_current_write':[row(with_scores)],'without_current_write':[row(without_scores)]}}


def test_unknown_unmatched_future_competitor_never_becomes_safe():
    p=proposal({'p':2.,'q':1.},{'p':2.,'q':1.})
    result=risk_label(p,{2:{'p':1},3:{'p':1,'q':None}},1)
    assert not result['risk_label_verified'] and result['future_safe_label'] is None


def test_projected_hard_negative_margin_harm_is_unsafe():
    p=proposal({'p':2.,'q':1.5},{'p':2.,'q':1.})
    result=risk_label(p,{2:{'p':1},3:{'p':1,'q':2}},1)
    assert result['risk_label_verified'] and not result['future_safe_label']
    assert result['mean_projected_margin_delta']==-.5


def test_wrong_current_identity_cannot_be_marked_safe_by_good_future():
    p=proposal({'p':3.,'q':1.},{'p':2.,'q':1.})
    result=risk_label(p,{2:{'p':2},3:{'p':1,'q':2}},1)
    assert result['risk_label_verified'] and result['future_safe_label'] is False


def test_same_identity_non_harming_write_can_have_verified_safe_label():
    p=proposal({'p':3.,'q':1.},{'p':2.,'q':1.})
    result=risk_label(p,{2:{'p':1},3:{'p':1,'q':2}},1)
    assert result['risk_label_verified'] and result['future_safe_label'] is True
