import numpy as np
import pytest
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor, qualified_current_candidate


POINT = dict(claim_min=.8, risk_max=.02, global_regret_max=.2, anchor_advantage_min=.1, confirmation_delay=3)


def features():
    return dict(global_regret=.1, displaced_count=0, proposal_NONE=0., anchor_advantage_vs_KEEP=.2, quality=.9,
                anchor_cosine=.8, NONE_probability=.99, NONE_advantage=.2, KEEP_anchor_cosine=.5)


def test_uncalibrated_model_cannot_acquire_deployment_authority():
    model = EventAuthorityHead("LOGISTIC_RISK")
    saved = dict(schema="N72R21R2_EVENT_AUTHORITY_V1", family="LOGISTIC_RISK", hidden=32, model=model.state_dict(),
                 authority_status="TRAINED_UNCALIBRATED_NOT_DEPLOYABLE", FIT_mean=[0.] * 32, FIT_scale=[1.] * 32)
    with pytest.raises(ValueError):
        EventAuthorityPredictor(saved)
    predictor = EventAuthorityPredictor(saved, development_diagnostic=True)
    assert not any(p.requires_grad for p in predictor.model.parameters())


def test_feasibility_actual_change_competitor_and_confirmation_precede_prediction():
    prediction = dict(beneficial=.99, harmful=.001, value=.1)
    f = features()
    assert qualified_current_candidate(f, prediction, POINT, feasible=True, changed=True, confirmations=3)
    for feasibility, change, count in [(False, True, 3), (True, False, 3), (True, True, 2)]:
        assert not qualified_current_candidate(f, prediction, POINT, feasible=feasibility, changed=change, confirmations=count)
    f["displaced_count"] = 1
    assert not qualified_current_candidate(f, prediction, POINT, feasible=True, changed=True, confirmations=3)


def test_NONE_is_selective_not_always_none_or_relaxed_risk():
    f, prediction = features(), dict(beneficial=.99, harmful=.001, value=.1)
    f["proposal_NONE"] = 1.
    # NONE confirmations count decisions, never fake feature/box observations.
    assert not qualified_current_candidate(f, prediction, POINT, feasible=True, changed=True, confirmations=0)
    assert qualified_current_candidate(f, prediction, POINT, feasible=True, changed=True, confirmations=3)
    f["NONE_probability"] = .8
    assert not qualified_current_candidate(f, prediction, POINT, feasible=True, changed=True, confirmations=3)
