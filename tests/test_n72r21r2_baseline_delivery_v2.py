import pytest
from scripts.n72r21r2_baseline_delivery_v2 import historical_index_binding


def test_actual_old_manifest_chain_without_fabricated_index_field():
    old = {"initialization_manifest_sha256": "manifest", "GT_parsed_by_runtime": False}
    assert historical_index_binding(old, {"candidate_index_sha256": "index"}, "manifest") == "index"
    assert "candidate_index_sha256" not in old


@pytest.mark.parametrize("corruption", ("manifest_missing", "manifest_wrong", "runtime_GT", "runtime_GT_missing", "explicit_index_wrong"))
def test_missing_or_contradictory_historical_binding_rejected(corruption):
    old = {"initialization_manifest_sha256": "manifest", "GT_parsed_by_runtime": False}
    if corruption == "manifest_missing": del old["initialization_manifest_sha256"]
    elif corruption == "manifest_wrong": old["initialization_manifest_sha256"] = "wrong"
    elif corruption == "runtime_GT": old["GT_parsed_by_runtime"] = True
    elif corruption == "runtime_GT_missing": del old["GT_parsed_by_runtime"]
    else: old["candidate_index_sha256"] = "wrong"
    with pytest.raises(ValueError): historical_index_binding(old, {"candidate_index_sha256": "index"}, "manifest")
