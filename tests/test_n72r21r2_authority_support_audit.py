from copy import deepcopy
import numpy as np
import pytest
from sam3_intermot.evaluation.authority_support_audit import hard_filter_violations, safe_direct_source_row
from sam3_intermot.one_click.event_authority_runtime import qualified_current_candidate
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES

POINT = dict(claim_min=.8, risk_max=.02, global_regret_max=.2, anchor_advantage_min=.1, confirmation_delay=3)


def test_hard_filter_ceiling_matches_frozen_runtime_for_both_candidate_axes():
    rng = np.random.default_rng(730104)
    for _ in range(500):
        f = {k: float(rng.uniform(-1., 1.)) for k in FEATURE_NAMES}
        f.update(proposal_NONE=int(rng.integers(2)), displaced_count=int(rng.integers(3)))
        for count in (0, 3, 8):
            expected = qualified_current_candidate(f, dict(beneficial=1., harmful=0., value=1.), POINT,
                                                   feasible=True, changed=True, confirmations=count)
            assert expected == (not hard_filter_violations(f, POINT, confirmations=count))


def test_exact_boundaries_confirmation_shift_and_truth_rejection():
    f = {k: 0. for k in FEATURE_NAMES}
    f.update(global_regret=.2, anchor_advantage_vs_KEEP=.1, quality=.5, anchor_cosine=.6)
    assert hard_filter_violations(f, POINT, confirmations=3) == []
    assert hard_filter_violations(f, POINT, confirmations=0) == ["CONFIRMATION_DELAY"]
    f["displaced_count"] = 1
    assert hard_filter_violations(f, POINT, confirmations=3) == ["ANY_DISPLACED_PUBLIC_ID"]
    f["target_gt_identity"] = 4
    with pytest.raises(ValueError):
        hard_filter_violations(f, POINT, confirmations=3)
    f.pop("target_gt_identity")
    f["quality"] = float("nan")
    with pytest.raises(ValueError):
        hard_filter_violations(f, POINT, confirmations=3)


def test_direct_scope_excludes_delayed_incomplete_harm_and_ineffective_rows():
    row = {"branch": "RAW_IDENTITY_TOP", "frame": 4, "action": {"candidate_uid": "uid"},
           "offline_supervision_labels": {"future": {"H100": dict(complete=True, benefit_label=True, N01=3, N10=0)},
                "current_t": dict(outcome="TARGET", N01=1), "H100_any_harm_including_current_t_label": False,
                "effective_direct_action_frames": [4]}}
    assert safe_direct_source_row(row)
    bad = deepcopy(row); bad["branch"] = "DELAYED_CHALLENGER"
    assert not safe_direct_source_row(bad)
    bad = deepcopy(row); bad["offline_supervision_labels"]["future"]["H100"]["complete"] = False
    assert not safe_direct_source_row(bad)
    bad = deepcopy(row); bad["offline_supervision_labels"]["H100_any_harm_including_current_t_label"] = True
    assert not safe_direct_source_row(bad)
    bad = deepcopy(row); bad["offline_supervision_labels"]["effective_direct_action_frames"] = [6]
    assert not safe_direct_source_row(bad)
