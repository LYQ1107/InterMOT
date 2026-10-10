"""Fill actual all24 data deliverables; inventory every full-task requirement.

No new extraction, runtime rollout, truth access, model fitting or selection.
Read and reverify original candidate tapes using the frozen integrity audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, preregistration, append_log, storage
from scripts.n72r21r2_candidates import integrity
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.task_delivery_evidence import (
    REQUIRED, TABLE_REQUIREMENTS, TEST_REQUIREMENTS, coverage_row, unopened_row, named_requirement,
)

CODE = ("scripts/n72r21r2_data_delivery_v1.py", "sam3_intermot/evaluation/task_delivery_evidence.py")


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path)}


def prepare():
    p = preregistration()
    required = p["split"]["fit"] + p["split"]["inner"]
    frozen = read_json(OUT / "data/FROZEN_SPLITS.json")
    assert frozen["split"] == p["split"] and not frozen["reshuffled"]
    assert frozen["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    manifest_path = OUT / "data/CANDIDATE_EXTRACTION_MANIFEST.json"
    manifest = read_json(manifest_path)
    assert manifest["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    assert manifest["configuration"] == p["candidate_generation"]
    lineage = read_json(OUT / "audit/CHECKPOINT_LINEAGE.json")
    weights = {key: ref(value["path"]) for key, value in lineage["weights"].items()}
    assert all(weights[k]["sha256"] == value["sha256"] for k, value in lineage["weights"].items())
    scene_path = OUT / "data/SCENE_CHARACTERISTICS_OFFLINE_V1.json"
    scenes = read_json(scene_path)
    assert scenes["status"] == "COMPLETE_ALL24_OFFLINE_DESCRIPTIVE_ORIGINAL_SCENES_NOT_POLICY_CLOSURE"
    assert not scenes["CONFIRM_VAL_TEST_SOT_accessed"] and scenes["new_runtime_rollouts_or_optimizer_steps"] == 0
    assert scenes["source_sha256"] == {name: sha256(ROOT / name) for name in scenes["source_sha256"]}
    for reference in scenes["source_receipts"]:
        assert sha256(reference["path"]) == reference["sha256"]
    scene_rows = {r["sequence"]: r for r in scenes["all24_census"]}
    assert set(scene_rows) == set(required) and len(scenes["all24_census"]) == 24
    extracted, rows = [], {}
    for sequence in required:
        # This frozen audit is idempotent: differing old seals raise, never overwrite.
        with runtime_file_guard():
            candidate = integrity(sequence)
        seal_path = OUT / "data/candidate_integrity" / (sequence + ".json")
        assert candidate["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_candidates.py")
        assert candidate["inherited_generator_sha256"] == manifest["inherited_generator_sha256"]
        index = read_json(candidate["index_path"])
        done = Path(candidate["index_path"]).parent / "done.json"
        init_path = OUT / "data/initialization" / (sequence + ".json")
        base_path = OUT / "mot/baseline_results" / (sequence + ".json")
        init, base = read_json(init_path), read_json(base_path)
        assert sha256(init["anchor_path"]) == init["anchor_sha256"]
        assert base["GT_labels_offline_only"] and base["density_integrity_sha256"] == sha256(seal_path)
        role = "fit" if sequence in p["split"]["fit"] else "inner"
        rows[sequence] = coverage_row(sequence, role, init, candidate, base, scene_rows[sequence])
        rows[sequence]["source_receipts"] = [ref(path) for path in (seal_path, init_path, base_path)]
        extracted.append({"sequence": sequence, "role": role, "frames": candidate["frames"],
            "candidates": candidate["candidate_count"], "inherited_schema_stage_NOT_old_observation": candidate["inherited_payload_schema_stage"],
            "index": ref(candidate["index_path"]), "done": ref(done), "integrity": ref(seal_path),
            "metadata": ref(index["metadata"]), "embeddings": ref(index["embeddings"]),
            "original_density": candidate["whole_video_density"], "newly_generated_real_candidates": True})
        print({"original_data_reverified": sequence, "frames": candidate["frames"], "candidates": candidate["candidate_count"]}, flush=True)
    common = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "scope": "Exactly original FIT16+INNER8; no historical substitution, failed initialization replacement, or CONFIRM/VAL/TEST/SOT access",
        "configuration_redefined": False, "new_extraction_or_runtime_rollouts_or_optimizer_steps": 0,
        "scientific_success": None, "next_stage_authorized": False}
    extraction = {**common, "status": "COMPLETE_ACTUAL_ALL24_ORIGINAL_CANDIDATE_EXTRACTION",
        "original_frozen_manifest": ref(manifest_path), "frozen_configuration": manifest["configuration"],
        "actual_weights_reverified": weights, "completed_sequences": required, "actual_sequence_records": extracted,
        "actual_total_frames": sum(r["frames"] for r in extracted), "actual_total_candidates": sum(r["candidates"] for r in extracted),
        "earlier_empty_manifest_is_PREPARATION_NOT_current_completion": True}
    write_json("data/CANDIDATE_EXTRACTION.json", extraction)
    write_json("data/CANDIDATE_INTEGRITY.json", {**common, "status": "COMPLETE_ACTUAL_ALL24_ORIGINAL_UID_FEATURE_AXIS_INTEGRITY",
        "verified_sequences": required, "original_frame_axis_zero_to_n_minus_one": True,
        "unique_UIDs_and_embedding_offsets_finite_features_complete_current_candidate_sets": True,
        "all24_existing_integrity_receipts_recomputed_without_change": True,
        "GT_read_during_new_candidate_reverification": False, "f16_storage_not_original_f32_feature_hash_reproduction": True,
        "source_extraction_summary": ref(OUT / "data/CANDIDATE_EXTRACTION.json"), "actual_sequence_records": extracted})
    census = []
    for role, sequences in p["split"].items():
        census.extend(rows[s] if s in rows else unopened_row(s, role) for s in sequences)
    write_json("tables/TABLE1_DATA_COVERAGE_PREPARATION_V1.json", {**common,
        "status": "ALL40_DATA_ROLES_ACTUAL_FRESH24_COVERAGE_CAUSAL_ROOT_COLUMNS_UNRESOLVED",
        "source_scene_summary": ref(scene_path), "source_extraction_summary": ref(OUT / "data/CANDIDATE_EXTRACTION.json"),
        "all40_census": census, "independent_people_and_correction_event_columns_NOT_filled_with_GT_IDs_or_frames": True,
        "is_final_complete_Table1": False, "remaining_proof": "Independent causal correction/N10 roots; actual source audit scope for historical rows. Confirmation stays unopened until qualified."})
    append_log("ACTUAL_ALL24_DATA_DELIVERABLES_AND_ALL40_COVERAGE", source_sha256=common["source_sha256"], actual_sequences=24, census=40)


def ledger(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Narrow immutable ledger suffix required")
    p = preregistration()
    proofs = {}
    if (OUT / "data/CANDIDATE_EXTRACTION.json").exists() and (OUT / "data/CANDIDATE_INTEGRITY.json").exists():
        integrity_report = read_json(OUT / "data/CANDIDATE_INTEGRITY.json")
        extraction_report = read_json(OUT / "data/CANDIDATE_EXTRACTION.json")
        assert integrity_report["status"] == "COMPLETE_ACTUAL_ALL24_ORIGINAL_UID_FEATURE_AXIS_INTEGRITY"
        assert extraction_report["status"] == "COMPLETE_ACTUAL_ALL24_ORIGINAL_CANDIDATE_EXTRACTION"
        for report in (integrity_report, extraction_report):
            assert report["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
            assert report["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
        assert integrity_report["verified_sequences"] == p["split"]["fit"] + p["split"]["inner"]
        assert integrity_report["source_extraction_summary"] == ref(OUT / "data/CANDIDATE_EXTRACTION.json")
        assert integrity_report["actual_sequence_records"] == extraction_report["actual_sequence_records"]
        for row in integrity_report["actual_sequence_records"]:
            for key in ("index", "done", "integrity", "metadata", "embeddings"):
                assert sha256(row[key]["path"]) == row[key]["sha256"]
        proofs.update({key: "All24 actual original candidate inputs/artifacts/axes reverified; not event/model/MOT science completion"
                       for key in ("data/CANDIDATE_EXTRACTION.json", "data/CANDIDATE_INTEGRITY.json")})
    frozen = read_json(OUT / "data/FROZEN_SPLITS.json")
    assert frozen["split"] == p["split"] and frozen["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    goal = read_json(OUT / "FINAL_GOAL.json")
    assert goal["goal"] == GOAL and goal["goal_frozen"]
    task_sha = sha256(goal["task_source"])
    assert task_sha == read_json(OUT / "audit/SOURCE_AUDIT.json")["task_source_SHA"]
    proofs.update({"FINAL_GOAL.json": "Same frozen full-task MOT Goal", "PREREGISTRATION.json": "Frozen original40-role protocol and gates",
                   "data/FROZEN_SPLITS.json": "Exactly inherited frozen split; cross-scene person independence remains unproven"})
    records = [named_requirement(relative, (OUT / relative).exists(), sha256(OUT / relative) if (OUT / relative).exists() else None,
                                proofs.get(relative)) for relative in REQUIRED]
    closure_path = OUT / "mot/development_closure_v1/latest.json"
    closure = read_json(closure_path)
    assert sha256(closure["path"]) == closure["sha256"]
    report = {"stage": "N72R21R2", "goal": GOAL, "task_scope": "Full original M0-M11; do not redefine around existing reports",
        "original_full_task_source_sha256": task_sha,
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
        "required_named_files": records, "required_named_file_count": len(REQUIRED),
        "present_named_file_count": sum(r["present"] for r in records),
        "verified_narrow_preparation_count": sum(bool(r["narrow_proof"]) for r in records),
        "required_six_final_tables": {name: {"requirements": list(fields), "completion_proven": False} for name, fields in TABLE_REQUIREMENTS.items()},
        "required35_test_semantics": {name: "SEPARATE_ACTUAL_TEST_AND_RUNTIME_SCOPE_PROOF_REQUIRED; pass totals alone insufficient" for name in TEST_REQUIREMENTS},
        "latest_complete_group_census": closure, "remaining_group_closure": "All14 MAIN and3 STATE groups, all original videos/points/seeds; not just successful cells",
        "conditional_branches": "Model-generated on-policy/staged rounds, M-B, relational and confirmation must each have actual qualification or evidence-backed NOT_RUN; unresolved now, not silently skipped",
        "overall_completion_proven": False, "scientific_decision": "PENDING", "next_stage_authorized": False,
        "presence_only_or_placeholder_NEVER_completion": True}
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    relative = "audit/FULL_TASK_DELIVERABLE_LEDGER_V1" + suffix + ".json"
    write_json(relative, {**report, "input_bundle_sha256": digest})
    append_log("FULL_TASK_REQUIREMENT_LEDGER_NO_COMPLETION_CLAIM", path=relative, required_files=len(REQUIRED), present_files=report["present_named_file_count"])
    print({"ledger": relative, "required_files": len(REQUIRED), "present": report["present_named_file_count"], "narrow_verified": report["verified_narrow_preparation_count"], "full_task_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "ledger"))
    parser.add_argument("--suffix")
    args = parser.parse_args()
    storage(8 << 20)
    prepare() if args.action == "prepare" else ledger(args.suffix)
