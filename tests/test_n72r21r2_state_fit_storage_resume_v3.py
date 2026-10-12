import pytest
from scripts import n72r21r2_state_fit_storage_resume_v3 as recovery


def baseline():
    return {"status": "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY", "mode": "BASELINE_STATE",
            "seed": 730101, "optimizer_steps": 0, "FIT_rows": 517, "INNER_rows": 1212,
            "FIT_target_counts": {"benefit": 0, "risk": 336}, "protocol_sha256": "protocol"}


def test_only_exact_original_measured_zero_benefit_no_optimizer_disposition_skips_refit():
    assert recovery.valid_no_fit(baseline(), 730101, "protocol")


@pytest.mark.parametrize("changed", [
    {"status": "COMPLETE"}, {"mode": "TREATMENT_STATE"}, {"seed": 730102},
    {"optimizer_steps": 1}, {"FIT_rows": 516}, {"INNER_rows": 1200},
    {"FIT_target_counts": {"benefit": 1, "risk": 336}}, {"protocol_sha256": "changed"},
])
def test_unverified_or_different_disposition_never_skip_authority(changed):
    assert not recovery.valid_no_fit({**baseline(), **changed}, 730101, "protocol")


def test_state_recovery_owner_is_new_but_model_protocol_and_entrypoint_stay_original():
    assert recovery.FRESH != recovery.MARKER
    assert recovery.study.PROTOCOL.name == "STATE_SOURCE_FITS_PROTOCOL_V1.json"
    assert recovery.study.MODES == ("BASELINE_STATE", "TREATMENT_STATE", "MIXED_STATE")
