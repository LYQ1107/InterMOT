"""Attempt records are not trained models, nor deployment qualification."""

SUCCESS_STATUSES = frozenset((
    "COMPLETE_ACTUAL_OPTIMIZATION_UNCALIBRATED_NO_DEPLOYMENT_AUTHORITY",
    "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE",
    "COMPLETE_ACTUAL_FRESH_CURRENT_AXIS_FIT_AND_INNER_CALIBRATION",
    "COMPLETE_ACTUAL_SAME_BANK_CURRENT_WRITE_RISK_FIT",
))
NO_OPTIMIZATION = "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY"


def nonnegative_integer(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Actual nonnegative integer optimization counts required")
    return value


def changed_elements(record):
    value = record.get("changed_weight_elements_by_tensor", record.get("changed_weight_elements"))
    if isinstance(value, dict):
        if not value:
            raise ValueError("Empty tensor-change evidence")
        return sum(nonnegative_integer(v) for v in value.values())
    return nonnegative_integer(value)


def assess_fit(record, *, protocol_sha256, artifact_sha256, source_sha256):
    """Check sealed optimization evidence; abstention/failure never counts.

    Hash callbacks resolve ONLY the caller's scoped, authorized paths.
    Three distinct changed-weight schemas are intentionally supported.
    """
    base = {"record_status": record.get("status"), "verified_actual_optimization": False,
            "scientific_success_inferred": False, "deployment_authorized": False}
    if record.get("protocol_sha256") != protocol_sha256:
        return {**base, "classification": "UNVERIFIED_RECORD", "reason": "FROZEN_PROTOCOL_HASH_MISMATCH"}
    if record.get("status") == NO_OPTIMIZATION:
        if record.get("optimizer_steps") == 0 and not isinstance(record.get("optimizer_steps"), bool) and not record.get("checkpoint_path") and not record.get("checkpoint_sha256"):
            return {**base, "classification": "MEASURED_NO_OPTIMIZATION", "reason": NO_OPTIMIZATION,
                    "optimizer_steps": 0, "changed_weight_elements": 0}
        return {**base, "classification": "UNVERIFIED_RECORD", "reason": "FAILURE_RECORD_CONTRADICTS_NO_OPTIMIZATION"}
    if record.get("status") not in SUCCESS_STATUSES:
        return {**base, "classification": "UNVERIFIED_RECORD", "reason": "NO_REGISTERED_SUCCESS_STATUS"}
    try:
        steps = nonnegative_integer(record.get("optimizer_steps"))
        nonzero = nonnegative_integer(record.get("nonzero_gradient_steps"))
        changed = changed_elements(record)
        if not 0 < nonzero <= steps or changed <= 0:
            raise ValueError("No nonzero-gradient and changed-weight proof")
        checkpoint, digest = record.get("checkpoint_path"), record.get("checkpoint_sha256")
        if not checkpoint or not digest or artifact_sha256(checkpoint) != digest:
            raise ValueError("Selected weight artifact missing or hash differs")
        sources = record.get("source_sha256", record.get("source_freeze"))
        if not isinstance(sources, dict) or not sources or any(source_sha256(p) != h for p, h in sources.items()):
            raise ValueError("Actual optimizer code lineage missing or hash differs")
    except (ValueError, TypeError, KeyError, OSError) as error:
        return {**base, "classification": "UNVERIFIED_RECORD", "reason": str(error)}
    return {**base, "classification": "VERIFIED_ACTUAL_OPTIMIZATION", "verified_actual_optimization": True,
            "optimizer_steps": steps, "nonzero_gradient_steps": nonzero, "changed_weight_elements": changed,
            "checkpoint_sha256": digest}


def census(records):
    """Successful models, measured failures, and unverifiable attempts differ."""
    classes = ("VERIFIED_ACTUAL_OPTIMIZATION", "MEASURED_NO_OPTIMIZATION", "UNVERIFIED_RECORD")
    if any(r.get("classification") not in classes for r in records):
        raise ValueError("Unknown fit-evidence classification")
    return {"attempt_records": len(records),
            "verified_actual_optimizer_models": sum(r["classification"] == classes[0] for r in records),
            "measured_no_optimization_records": sum(r["classification"] == classes[1] for r in records),
            "unverified_records": sum(r["classification"] == classes[2] for r in records),
            "deployment_or_research_success_implied": False}
