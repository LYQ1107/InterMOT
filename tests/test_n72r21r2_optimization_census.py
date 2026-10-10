import pytest
from sam3_intermot.evaluation.optimization_census import assess_fit, census, SUCCESS_STATUSES, NO_OPTIMIZATION


def record():
    return {"status": "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE", "protocol_sha256": "protocol",
            "optimizer_steps": 100, "nonzero_gradient_steps": 99, "changed_weight_elements": 12,
            "checkpoint_path": "weights", "checkpoint_sha256": "weights-sha", "source_sha256": {"code": "code-sha"}}


def assess(value, *, weight="weights-sha", source="code-sha"):
    return assess_fit(value, protocol_sha256="protocol", artifact_sha256=lambda p: weight, source_sha256=lambda p: source)


@pytest.mark.parametrize("status", sorted(SUCCESS_STATUSES))
def test_actual_steps_changed_weights_and_lineage_are_required(status):
    value = record()
    value["status"] = status
    result = assess(value)
    assert result["verified_actual_optimization"]
    assert not result["deployment_authorized"] and not result["scientific_success_inferred"]


@pytest.mark.parametrize("schema", ("integer", "per_tensor", "main_per_tensor"))
def test_all_three_real_changed_weight_schemas(schema):
    value = record()
    if schema == "per_tensor":
        value["changed_weight_elements"] = {"weight": 9, "bias": 3}
    elif schema == "main_per_tensor":
        del value["changed_weight_elements"]
        value["changed_weight_elements_by_tensor"] = {"weight": 9, "bias": 3}
        value["source_freeze"] = value.pop("source_sha256")
    assert assess(value)["changed_weight_elements"] == 12


def test_failed_state_supervision_is_an_attempt_not_a_trained_model():
    failed = {"status": NO_OPTIMIZATION, "protocol_sha256": "protocol", "optimizer_steps": 0}
    diagnostic = assess(failed)
    assert diagnostic["classification"] == "MEASURED_NO_OPTIMIZATION"
    counts = census([assess(record()), diagnostic])
    assert counts["attempt_records"] == 2
    assert counts["verified_actual_optimizer_models"] == 1
    assert counts["measured_no_optimization_records"] == 1
    assert counts["unverified_records"] == 0


@pytest.mark.parametrize("field,value", (("optimizer_steps", 0), ("optimizer_steps", True),
    ("nonzero_gradient_steps", 0), ("nonzero_gradient_steps", 101), ("changed_weight_elements", 0),
    ("changed_weight_elements", {"bias": True}), ("checkpoint_path", None), ("source_sha256", {}),
    ("protocol_sha256", "changed-protocol"), ("status", "ACTIVE")))
def test_unproven_success_never_counts(field, value):
    item = record()
    item[field] = value
    result = assess(item)
    assert result["classification"] == "UNVERIFIED_RECORD"
    assert not result["verified_actual_optimization"]
    assert census([result])["verified_actual_optimizer_models"] == 0


def test_modified_weight_or_code_hash_never_counts():
    assert assess(record(), weight="modified")["classification"] == "UNVERIFIED_RECORD"
    assert assess(record(), source="modified")["classification"] == "UNVERIFIED_RECORD"


def test_contradictory_failure_and_unregistered_class_are_not_accepted():
    failed = {"status": NO_OPTIMIZATION, "protocol_sha256": "protocol", "optimizer_steps": 0, "checkpoint_path": "weight"}
    assert assess(failed)["classification"] == "UNVERIFIED_RECORD"
    with pytest.raises(ValueError):
        census([{"classification": "pretend-trained"}])
