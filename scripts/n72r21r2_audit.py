"""Read-only inheritance/data audit; never parse CONFIRM truth or run policies."""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

from scripts.n72r21r2_common import ROOT, OUT, R1, HISTORY, ASSETS, TRAIN, OLD_CANDIDATES, PYTHON, GOAL, sha256, read_json, write_json, storage, preregistration, append_log, update_status, utcnow
from sam3_intermot.one_click.datasets import dancetrack_info


def checked_record(path, expected=None):
    path = Path(path)
    value = {"path": str(path), "exists": path.is_file()}
    if not value["exists"]:
        raise FileNotFoundError(path)
    value.update(bytes=path.stat().st_size, sha256=sha256(path))
    if expected is not None and value["sha256"] != expected:
        raise ValueError("Historical SHA mismatch: " + str(path))
    return value


def run():
    protocol = preregistration()
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == protocol["branch"]
    for repo, expected in [(R1, protocol["source_head"]), (HISTORY, "813be1e34a644ffeecf8b9ad0404aeefd6bd604b")]:
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == expected
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    # Reverify unuploaded R1 evidence against the committed compact catalog.
    catalog_path = R1 / "docs/N72R21R1_LOCAL_EVIDENCE_SHA.json"
    catalog = read_json(catalog_path)
    records = [checked_record(R1 / "outputs/N72R21R1" / name, expected) for name, expected in catalog["local_artifact_SHA256"].items()]
    old_manifest = read_json(R1 / "outputs/N72R21R1/source_audit/HISTORY_SHA_MANIFEST.json")
    old_records = [checked_record(path, digest) for path, digest in old_manifest["files"].items()]
    assert len(old_records) == 371
    inherited_split = R1 / "outputs/N72R21R1/data/NEW_SEQUENCE_SPLIT.json"
    assert read_json(inherited_split)["split"] == protocol["split"]
    write_json("data/FROZEN_SPLITS.json", {"stage": "N72R21R2", "split": protocol["split"], "inherited_manifest": checked_record(inherited_split),
              "original_manifest_created_time_available": False, "original_ordering_rule": read_json(inherited_split)["rule"],
              "frozen_R2_utc": protocol["created_utc"], "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
              "reshuffled": False, "person_disjointness_unproven": True, "confirmation_policy_evaluation_authorized_now": False})
    dataset = []
    sample_hashes = defaultdict(list)
    groups = {s: k for k, names in protocol["split"].items() for s in names}
    for sequence in sorted(groups):
        directory = TRAIN / sequence
        info = dancetrack_info(directory)
        images = sorted((directory / "img1").glob("*.jpg"))
        assert [p.name for p in images] == [f"{i:08d}.jpg" for i in range(1, info["frames"] + 1)], sequence
        assert all(p.stat().st_size > 0 for p in images), sequence
        gt = checked_record(directory / "gt/gt.txt")  # Integrity bytes only; no CONFIRM label parsing.
        index_path = OLD_CANDIDATES / "candidates" / sequence / "index.json"
        index = read_json(index_path) if index_path.exists() else None
        if index:
            assert not index.get("runtime_gt_read") and not index.get("runtime_future_gt_used")
            assert index["frame_count"] == info["frames"]
            for field in ("metadata", "embeddings"):
                checked_record(index[field], index[field + "_sha256"])
        samples = []
        for i in sorted({0, len(images) // 2, len(images) - 1}):
            record = checked_record(images[i])
            samples.append(record)
            sample_hashes[record["sha256"]].append({"sequence": sequence, "frame": i})
        dataset.append({"sequence": sequence, "role": groups[sequence], "root": str(directory), **info,
                        "all_original_images_present": True, "images": len(images), "image_bytes": sum(p.stat().st_size for p in images),
                        "GT_integrity": gt, "GT_labels_parsed": False, "image_sample_SHA": samples,
                        "existing_candidate_index": checked_record(index_path) if index else None,
                        "existing_candidate_count": index["candidate_count"] if index else None,
                        "fresh_candidate_generation_required": index is None})
    assert len(list(TRAIN.glob("dancetrack*"))) == 40
    sample_duplicates = [v for v in sample_hashes.values() if len({x["sequence"] for x in v}) > 1]
    write_json("data/EXISTING_DATASETS.json", {"stage": "N72R21R2", "root": str(TRAIN), "train_sequences": 40,
              "actual_frames": sum(r["frames"] for r in dataset), "sequences": dataset,
              "exact_cross_video_sample_duplicates": sample_duplicates, "no_duplicate_found_is_not_recording_or_person_disjointness_proof": True,
              "statistical_cluster": "Video sequence, not frames/seeds/clicks; repeated people/recordings not independently verified",
              "VAL_TEST_SOT_opened": False, "no_data_or_environment_copy": True})
    old_assets = read_json(R1 / "outputs/N72R20R2/asset_manifest.json")
    weights = {"sam3": checked_record(old_assets["sam3_checkpoint"]["path"], old_assets["sam3_checkpoint"]["sha256"]),
               "osnet": checked_record(old_assets["frozen_identity_assets"]["osnet_x1_0_market1501"]["path"], old_assets["frozen_identity_assets"]["osnet_x1_0_market1501"]["sha256"])}
    identity = read_json(R1 / "outputs/N72R21R1/corpus/RUNTIME_INPUTS.json")["frozen_identity_model"]
    checked_record(identity["path"], identity["sha256"])
    checked_record(identity["fit_record_path"], identity["fit_record_sha256"])
    assert set(identity["fit_sequences"] + [identity["inner_sequence"], identity["outer_sequence"]]) <= set(protocol["split"]["historical_development"])
    write_json("audit/HISTORICAL_ASSET_MANIFEST.json", {"stage": "N72R21R2", "source_head": protocol["source_head"],
              "R1_evidence_catalog": checked_record(catalog_path), "R1_local_records": records,
              "old_history_371_reverified": old_records, "old_scientific_decision": catalog["scientific_decision"],
              "old_tests": catalog["regression"], "weights": weights, "no_relabeling_old_runs_as_new": True})
    write_json("audit/CHECKPOINT_LINEAGE.json", {"stage": "N72R21R2", "weights": weights, "common_frozen_identity_model": identity,
              "R1_new_checkpoint_counts": catalog["new_head_counts"], "R1_checkpoint_catalogs": [r for r in records if "/checkpoints/" in r["path"]],
              "old_fit_exposure_8_only": True, "new_strict_loads_not_yet_claimed": True, "new_fits_completed": 0})
    # Plan from full maximum16 candidates/frame, not optimistic historical density.
    split_frames = {role: sum(r["frames"] for r in dataset if r["role"] == role) for role in groups.values()}
    candidate_estimates = {role: frames * 16 * 1400 for role, frames in split_frames.items() if role in ("fit", "inner")}
    estimate = {"candidate_max_16_per_frame_bytes": candidate_estimates, "checkpoints_bytes": 1 << 30,
                "TrackEval_logs_traces_events_bytes": 4 << 30, "temporary_decode_bytes": 128 << 20,
                "no_full_decoded_video_or_crop_copies": True, "initial_budget_bytes": 8 << 30}
    assert sum(candidate_estimates.values()) + estimate["checkpoints_bytes"] + estimate["TrackEval_logs_traces_events_bytes"] + estimate["temporary_decode_bytes"] < estimate["initial_budget_bytes"]
    write_json("data/STORAGE_AUDIT.json", {"stage": "N72R21R2", "actual_filesystem": storage(8 << 30), "estimate": estimate,
              "personal_tree_audit_human": "du -x -h --max-depth=1 completed:875GiB; quota -s reports none. Shared filesystem free2.0TiB is not personal allowance.",
              "personal_cap_approximate_user_1TB": True, "initial_new_stage_budget_GiB": 8,
              "no_other_project_cleanup_authorized": True, "cleanup_dry_run": [], "source_worktree_not_a_dataset_copy": True})
    candidate_config = protocol["candidate_generation"]
    write_json("data/CANDIDATE_EXTRACTION_MANIFEST.json", {"stage": "N72R21R2", "DANCETRACK_ROOT": str(TRAIN.parent),
              "sam3_checkpoint": old_assets["sam3_checkpoint"], "frozen_identity_assets": old_assets["frozen_identity_assets"],
              "configuration": candidate_config, "inherited_generator_sha256": sha256(ROOT / candidate_config["source"].split(";")[0]),
              "pilot_sequences": protocol["fresh_candidate_pilot"], "dataset": {"test_present_or_used": False},
              "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "asset_root": str(ASSETS), "completed_sequences": []})
    write_json("audit/SOURCE_AUDIT.json", {"stage": "N72R21R2", "source_HEAD": protocol["source_head"], "original_HEAD": "813be1e34a644ffeecf8b9ad0404aeefd6bd604b",
              "historical_worktrees_clean": True, "applicable_AGENTS_found": [], "remote_fresh_read": protocol["source_remote_fresh_read"],
              "ordinary_fetch": "TLS handshake failed", "connector_fetch_attempts": "Two transport failures, retained honestly",
              "source_report_SHA": sha256(R1 / "docs/N72R21R1_FINAL_REPORT.md"), "task_source_SHA": sha256(read_json(OUT / "FINAL_GOAL.json")["task_source"]),
              "python": str(PYTHON), "no_new_environment": True, "goal_SHA": sha256(OUT / "FINAL_GOAL.json"), "preregistration_SHA": sha256(OUT / "PREREGISTRATION.json")})
    update_status(status="ACTIVE_M0_ACTUAL_REPRODUCTION_AND_M1_FRESH_INPUT_PREPARATION", source_asset_audit_complete=True,
                  actual_TRAIN_sequences=40, historical_artifacts_SHA_reverified=371, fresh_candidates_required=24,
                  confirmation_GT_labels_parsed=False, resource_snapshot=storage())
    append_log("M0_SOURCE_DATA_WEIGHT_AUDIT_COMPLETE", prior_scientific_success=False, fresh_preparation_required=True,
               source_sha256=sha256(Path(__file__)), old_evidence_count=len(records), history_artifacts_verified=len(old_records), split_frame_counts=split_frames)
    print({"source_audit": "PASS", "train_sequences": 40, "old_candidate_sequences": sum(r["existing_candidate_index"] is not None for r in dataset),
           "fresh_sequences_required": 24, "candidate_estimate_GiB": sum(candidate_estimates.values()) / (1 << 30), "storage": storage()})


if __name__ == "__main__":
    run()
