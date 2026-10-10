from copy import deepcopy
import numpy as np
import pytest
from sam3_intermot.one_click import event_authority_runtime as original
from sam3_intermot.one_click.authority_support_ablation import POLICIES, ablated_qualification, scoped_qualification
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from scripts.n72r21r2_support_ablation_pilot import route, FAMILY
from sam3_intermot.one_click.event_authority_models import FAMILIES

POINT = dict(claim_min=.8, risk_max=.02, global_regret_max=.2, anchor_advantage_min=.1, confirmation_delay=3)
PRED = dict(beneficial=.8, harmful=.02, value=.1)


def features():
    f = {k: 0. for k in FEATURE_NAMES}
    f.update(quality=.5, anchor_cosine=.6, anchor_advantage_vs_KEEP=.1)
    return f


def test_named_ceiling_removals_leave_real_features_and_risk_confirmation_unchanged():
    f = features(); f.update(global_regret=3., displaced_count=1, anchor_advantage_vs_KEEP=0.)
    before = deepcopy(f)
    kw = dict(feasible=True, changed=True, confirmations=3)
    assert [ablated_qualification(f, PRED, POINT, policy=p, **kw) for p in POLICIES] == [False, False, False, True]
    assert f == before
    for policy in POLICIES:
        assert not ablated_qualification(f, PRED, POINT, policy=policy, **{**kw, "confirmations": 0})
        assert not ablated_qualification(f, PRED, POINT, policy=policy, **{**kw, "feasible": False})
        assert not ablated_qualification(f, PRED, POINT, policy=policy, **{**kw, "changed": False})
        assert not ablated_qualification(f, {**PRED, "harmful": .03}, POINT, policy=policy, **kw)


def test_legacy_parity_scope_restore_and_no_truth_fields():
    rng = np.random.default_rng(730104)
    for _ in range(100):
        f = {k: float(rng.uniform(-1, 1)) for k in FEATURE_NAMES}
        kw = dict(feasible=True, changed=True, confirmations=3)
        assert ablated_qualification(f, PRED, POINT, policy=POLICIES[0], **kw) == original.qualified_current_candidate(f, PRED, POINT, **kw)
    old = original.qualified_current_candidate
    with pytest.raises(RuntimeError):
        with scoped_qualification(POLICIES[-1]):
            assert original.qualified_current_candidate is not old
            raise RuntimeError("Restore even on failed own-policy step")
    assert original.qualified_current_candidate is old
    f = features(); f["gt_id"] = 3
    with pytest.raises(ValueError):
        ablated_qualification(f, PRED, POINT, feasible=True, changed=True, confirmations=3, policy=POLICIES[-1])


def test_none_constraints_and_output_routing_are_not_relaxed():
    assert FAMILY in FAMILIES and FAMILY == "LOGISTIC_RISK"
    f = features(); f.update(proposal_NONE=1, NONE_probability=.95, NONE_advantage=.05, KEEP_anchor_cosine=.59)
    for policy in POLICIES:
        assert ablated_qualification(f, PRED, POINT, feasible=True, changed=True, confirmations=3, policy=policy)
    f["KEEP_anchor_cosine"] = .6
    assert not ablated_qualification(f, PRED, POINT, feasible=True, changed=True, confirmations=3, policy=POLICIES[-1])
    assert route("training/pilot_policy_results/UID/video.json") == "training/support_ablation_v1/results/UID/video.json"
    assert route("training/event_authority/UID.json") == "training/event_authority/UID.json"
    assert route("data/initialization/video.json") == "data/initialization/video.json"
