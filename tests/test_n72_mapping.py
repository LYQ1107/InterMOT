from sam3_intermot.provenance.mapping import (
    canonical_candidate_uid,
    resolve_exact_mapping,
    validate_mapping_batch,
)


def _row(raw=17, public=None):
    result = resolve_exact_mapping(
        {
            "sequence": "toy",
            "frame": 4,
            "raw_native_id": raw,
            "adapter_external_id": 9001,
            "segment_local_id": "chunk0:17",
            "sequence_global_id": "toy:g17",
        },
        exact_sources=([] if public is None else [{"source": "direct_user_public_id", "public_id": public}]),
        public_assignment_absent=public is None,
    )
    return result


def test_candidate_uid_is_axis_sensitive_and_deterministic():
    args = dict(
        sequence="toy",
        frame=4,
        raw_native_id=17,
        adapter_external_id=9001,
        segment_local_id="chunk0:17",
        sequence_global_id="toy:g17",
    )
    assert canonical_candidate_uid(**args) == canonical_candidate_uid(**args)
    assert canonical_candidate_uid(**args) != canonical_candidate_uid(**{**args, "raw_native_id": 18})


def test_exact_public_source_is_required_for_exact_status():
    assert _row(public=101)["status"] == "EXACT"
    assert _row(public=101)["public_id"] == 101
    assert _row()["status"] == "PUBLIC_ASSIGNMENT_ABSENT"


def test_conflicting_exact_sources_are_ambiguous_not_heuristically_resolved():
    result = resolve_exact_mapping(
        {
            "sequence": "toy",
            "frame": 4,
            "raw_native_id": 17,
            "adapter_external_id": 9001,
            "segment_local_id": "chunk0:17",
            "sequence_global_id": "toy:g17",
        },
        exact_sources=(
            {"source": "identity_registry_binding", "public_id": 101},
            {"source": "direct_user_public_id", "public_id": 102},
        ),
    )
    assert result["status"] == "AMBIGUOUS_ONE_TO_MANY"
    assert result["public_id"] is None


def test_batch_detects_duplicate_candidate_and_public_collision():
    first = _row(public=101)
    duplicate = dict(first)
    duplicate["public_id"] = 101
    audit = validate_mapping_batch([first, duplicate], require_raw_coverage=True, require_public_assignment=True)
    codes = {item["code"] for item in audit["errors"]}
    assert audit["status"] == "FAIL_MAPPING_INTEGRITY"
    assert "DUPLICATE_CANDIDATE_UID" in codes
    assert "DUPLICATE_RAW_ID_IN_FRAME" in codes
    assert "PUBLIC_ID_COLLISION_IN_FRAME" in codes


def test_axis_mismatch_is_not_recovered_from_public_source():
    result = resolve_exact_mapping(
        {"sequence": "toy", "frame": 4, "raw_native_id": None, "local_id": "x", "global_id": "toy:g"},
        exact_sources=({"source": "direct_user_public_id", "public_id": 101},),
    )
    assert result["status"] == "AXIS_MISMATCH"
    assert result["public_id"] is None
