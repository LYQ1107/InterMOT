import pytest
from scripts.n72r21r2_baseline_delivery_v3 import pilot_no_intervention


def preclick(frame=0):
    return {"frame": frame, "authority": {"family": "shadow", "approved": False, "features": None, "reason": ["NO_INTERVENTION"]}, "selected_action": None, "joint_identity_memory_write": False}


def postclick():
    return {"frame": 1, "authority": {"effective_assignment_change": False, "approved": False}, "selected_action": {"family": "KEEP"}, "joint_identity_memory_write": False}


def test_exact_preclick_and_explicit_postclick_zero_action_schemas():
    assert pilot_no_intervention(preclick(), 0)
    assert pilot_no_intervention(preclick(3), 4)
    assert pilot_no_intervention(postclick(), 0)


@pytest.mark.parametrize("corruption", ("missing_postclick_flag", "preclick_schema_after_click", "effective_change", "approved", "wrong_action", "write", "preclick_action", "preclick_schema_extra"))
def test_missing_or_nonzero_pilot_evidence_rejected(corruption):
    row = postclick()
    if corruption == "missing_postclick_flag": del row["authority"]["effective_assignment_change"]
    elif corruption == "preclick_schema_after_click": row = preclick(1)
    elif corruption == "effective_change": row["authority"]["effective_assignment_change"] = True
    elif corruption == "approved": row["authority"]["approved"] = True
    elif corruption == "wrong_action": row["selected_action"]["family"] = "SWITCH_TO_CANDIDATE"
    elif corruption == "write": row["joint_identity_memory_write"] = True
    elif corruption == "preclick_action": row = preclick(); row["selected_action"] = {"family": "KEEP"}
    else: row = preclick(); row["authority"]["extra"] = True
    with pytest.raises(ValueError): pilot_no_intervention(row, 0)
