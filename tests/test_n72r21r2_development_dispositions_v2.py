from copy import deepcopy
import pytest
from scripts.n72r21r2_development_dispositions_v2 import enrich, status_key


def fixture():
    groups = {"MAIN/group"+str(i): {"status": "PENDING"} for i in range(14)}
    groups.update({"STATE/"+m+"__L3_HARM_AWARE": {"status": "PENDING"} for m in ("BASELINE_STATE", "TREATMENT_STATE", "MIXED_STATE")})
    failures = [{"status": "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY", "mode": "BASELINE_STATE", "seed": seed,
                 "optimizer_steps": 0, "FIT_rows": 517, "INNER_rows": 1212,
                 "FIT_target_counts": {"benefit": 0, "risk": 336}, "protocol_sha256": "protocol"} for seed in (1,2,3)]
    driver = {"group_status": {"BASELINE_STATE": "NOT_RUN_ACTUAL_SUPERVISION_FAILURE_NOT_A_WEIGHT"}}
    return {"all17_group_census": groups}, driver, failures


def test_state_mode_key_and_all17_preserved_without_zero_metrics_or_fake_MOT_completion():
    assert status_key("STATE", "BASELINE_STATE__L3_HARM_AWARE") == "BASELINE_STATE"
    assert status_key("MAIN", "SMALL_MLP__L3_HARM_AWARE") == "SMALL_MLP__L3_HARM_AWARE"
    original, driver, failures = fixture(); before = deepcopy(original)
    result = enrich(original, driver, failures, [1,2,3], "protocol")
    assert original == before and len(result) == 17
    row = result["STATE/BASELINE_STATE__L3_HARM_AWARE"]
    assert row["MOT_metrics"] is None and not row["counts_as_complete_full_MOT_group"]
    assert row["actual_optimizer_steps"] == row["actual_models"] == 0


@pytest.mark.parametrize("invalid", ["missing_seed", "optimization", "benefit", "wrong_mode_key", "fake_complete"])
def test_unverified_negative_disposition_or_fake_completion_rejected(invalid):
    snapshot, driver, failures = fixture()
    if invalid == "missing_seed": failures.pop()
    if invalid == "optimization": failures[0]["optimizer_steps"] = 1
    if invalid == "benefit": failures[0]["FIT_target_counts"]["benefit"] = 1
    if invalid == "wrong_mode_key": driver["group_status"] = {"BASELINE_STATE__L3_HARM_AWARE": "NOT_RUN_ACTUAL_SUPERVISION_FAILURE_NOT_A_WEIGHT"}
    if invalid == "fake_complete": snapshot["all17_group_census"]["STATE/BASELINE_STATE__L3_HARM_AWARE"]["status"] = "COMPLETE_ACTUAL_FAKE"
    with pytest.raises(ValueError): enrich(snapshot, driver, failures, [1,2,3], "protocol")
