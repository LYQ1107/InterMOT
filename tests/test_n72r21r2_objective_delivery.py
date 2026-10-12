from copy import deepcopy
import numpy as np
import pytest
from scripts import n72r21r2_objective_delivery_v1 as audit


def row(sequence="a", event="a1", branch="KEEP"):
    return {"sequence": sequence, "episode_uid": sequence + "_click",
        "event_uid": event, "branch": branch,
        "correlation_group_not_proven_causal_origin": "interval",
        "x": np.zeros(40, np.float32), "past": np.zeros((3, 32), np.float32),
        "keep": np.zeros(40, np.float32),
        "target": {"benefit": 0., "risk": 0., "value": 0., "scope": "NOT_INDEPENDENT"}}


def line(r):
    return {**{k: r[k] for k in audit.KEYS}, "offline_target": deepcopy(r["target"]),
        "not_own_policy_outcome": True,
        "prediction": {"beneficial": .1, "harmful": .2, "value": -.3}}


def test_exact_original_validation_row_order_and_target_readback():
    rows = [row(), row(event="a2")]
    audit.verify_predictions([line(r) for r in rows], rows)
    with pytest.raises(ValueError):
        audit.verify_predictions([line(r) for r in reversed(rows)], rows)
    with pytest.raises(ValueError):
        audit.verify_predictions([line(rows[0])], rows)


@pytest.mark.parametrize("change", [
    {"event_uid": "future"}, {"offline_target": {"benefit": 1.}},
    {"not_own_policy_outcome": False},
    {"prediction": {"beneficial": .1, "harmful": .2, "future_GT": .3}},
    {"prediction": {"beneficial": np.nan, "harmful": .2, "value": 0.}},
    {"prediction": {"beneficial": 1.1, "harmful": .2, "value": 0.}}])
def test_target_metadata_teacher_state_or_invalid_predictions_rejected(change):
    r = row()
    with pytest.raises(ValueError):
        audit.verify_predictions([{**line(r), **change}], [r])


def test_hierarchical_correlation_weights_do_not_become_independent_roots():
    rows = [row(), row(event="a2"), row(sequence="b", event="b1")]
    summary = audit.summarize(rows)
    assert summary["per_video_weight_mass"] == {"a": .5, "b": .5}
    assert summary["observed_events_NOT_independent_roots"] == 3
    assert summary["independent_causal_root_count"] is None
    assert summary["scope_counts"] == {"NOT_INDEPENDENT": 3}


@pytest.mark.parametrize("name,value", [("benefit", .5), ("risk", -1.), ("value", np.inf)])
def test_nonbinary_or_nonfinite_supervision_cannot_enter_delivery(name, value):
    r = row(); r["target"][name] = value
    with pytest.raises(ValueError):
        audit.summarize([r])


def test_support_contrast_retains_l0_extra_rows_and_exact_l1_l3_targets():
    conditions = {name: [row()] for name in audit.OBJECTIVES}
    conditions[audit.OBJECTIVES[0]].append(row(event="a2"))
    result = audit.support_contrast(conditions)
    assert result["all7_common_rows_same_runtime_input"] == 1
    assert not result["same_eligible_support_across_all7"]
    assert result["L1_L3_same_eligible_rows_and_targets"]
    assert not result["loss_only_comparison_on_identical_support_across_all7_claimed"]


@pytest.mark.parametrize("bad", ["duplicate", "changed_current", "changed_past", "wrong_dtype"])
def test_support_comparison_rejects_duplicates_or_silent_runtime_axis_changes(bad):
    conditions = {name: [row()] for name in audit.OBJECTIVES}
    r = conditions[audit.OBJECTIVES[1]][0]
    if bad == "duplicate":
        conditions[audit.OBJECTIVES[1]].append(deepcopy(r))
    elif bad == "changed_current":
        r["x"][0] = 1.
    elif bad == "changed_past":
        r["past"][0, 0] = 1.
    else:
        r["keep"] = r["keep"].astype(np.float64)
    with pytest.raises(ValueError):
        audit.support_contrast(conditions)


def test_required_three_horizon_reports_do_not_imply_three_horizon_fits():
    assert audit.NAMED["H20"] == audit.NAMED["H50"] == audit.NAMED["H100"] == [audit.OBJECTIVES[2]]
