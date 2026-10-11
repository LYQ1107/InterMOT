"""All24 original raw M-A write/global trajectory/CSV independent readback.

No tracker/encoder rollout, training, new threshold, GT annotation parse or
association authorization. Stored offline write outcomes are NOT relabeled.
Legacy tensor fingerprints are not complete tracker-attribute equality.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, utcnow, storage, preregistration
from scripts.n72r21r2_memory_frontier import checked_result, CODE as FRONTIER_CODE, METRICS
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import parse_trackeval, trackeval_summary
from scripts.n72r20r4_run_causal_tracker import trajectory_text

CODE = ("scripts/n72r21r2_memory_frontier_readback_v1.py", *FRONTIER_CODE,
        "scripts/n72r20r1_build_base_score_tape.py", "scripts/n72r20r3r2r3_pipeline.py",
        "scripts/n72r20r4_run_causal_tracker.py", "scripts/n72r21r2_events.py")


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Immutable narrow evidence version required")
    relative = "audit/MEMORY_ALL24_RAW_FRONTIER_READBACK_V1" + suffix + ".json"
    if (OUT / relative).exists():
        raise FileExistsError("Preserve earlier actual readback attempts")
    torch.set_num_threads(1)
    storage(8 << 20)
    source = {name: sha256(ROOT / name) for name in CODE}
    p = preregistration()
    sequences = p["split"]["fit"] + p["split"]["inner"]
    pointer = read_json(OUT / "memory/frontier_v1/latest.json")
    assert sha256(pointer["path"]) == pointer["sha256"]
    frontier = read_json(pointer["path"])
    assert frontier["all24_fixed_and_risk_complete"] and frontier["counts"] == {"fixed": 24, "fresh_risk": 24}
    refs = {pointer["path"]: pointer["sha256"]}
    census = {}
    exposures = case_count = writes_count = 0

    def hashed(path):
        path = Path(path)
        if not path.resolve().is_relative_to(OUT.resolve()) and not path.resolve().is_relative_to(ASSETS):
            raise ValueError("Raw memory evidence escaped its stage roots")
        digest = sha256(path)
        if str(path) in refs:
            assert refs[str(path)] == digest
        refs[str(path)] = digest
        return digest

    def checked(path):
        hashed(path)
        return read_json(path)

    for sequence in sequences:
        init = checked(OUT / "data/initialization" / (sequence + ".json"))
        events = {event["episode_uid"]: event for event in init["inputs"] if not event["initialization_failure"]}
        base = {}
        frozen = {}
        for episode in events:
            seal = checked(OUT / "mot/baseline_runtime/CLICK_C0" / (episode + ".json"))
            for item in seal["artifacts"]: assert hashed(item["path"]) == item["sha256"]
            base[episode] = read_zstd_jsonl(artifact(seal, "trace"))
            seal = checked(OUT / "memory/M_A/runtime/FROZEN" / (episode + ".json"))
            for item in seal["artifacts"]: assert hashed(item["path"]) == item["sha256"]
            frozen[episode] = read_zstd_jsonl(artifact(seal, "trace"))
        details = []
        for kind, prefix in (("fixed", "memory/M_A"), ("fresh_risk", "memory/risk_v1")):
            original = checked_result(sequence, kind)
            assert original is not None
            rows, evidence, valid, failures = original
            assert valid == len(events) and failures == sum(event["initialization_failure"] for event in init["inputs"])
            for path, digest in evidence: assert hashed(path) == digest
            result = checked(OUT / prefix / "results" / (sequence + ".json"))
            if not events:
                assert not result["identity_and_write_results"] and not result["all_nine_actual_metrics"]
                details.append({"kind": kind, "status": "NOT_RUN_NO_VALID_ORIGINAL_CLICK_RETAINED", "actual_cases": 0})
                continue
            invocation = result["receipt"]
            assert invocation["returncode"] == 0 and hashed(invocation["log_path"]) == invocation["log_sha256"]
            args = invocation["command"]
            evaluation = Path(args[args.index("--OUTPUT_FOLDER") + 1])
            seqmap = Path(args[args.index("--SEQMAP_FILE") + 1])
            hashed(seqmap)
            assert seqmap.read_text().splitlines() == ["name", sequence]
            for tracker, expected in result["all_nine_actual_metrics"].items():
                for pattern in ("*.csv", "*_summary.txt"):
                    for path in (evaluation / tracker).rglob(pattern): hashed(path)
                actual = trackeval_summary(parse_trackeval(evaluation, tracker, [sequence]))
                assert set(actual["per_sequence"]) == {sequence}
                assert all(actual[metric] == actual["per_sequence"][sequence][metric] == expected[metric] for metric in METRICS)
            for key, saved in sorted(result["identity_and_write_results"].items()):
                case, episode = key.split("/", 1)
                event = events[episode]
                seal = checked(OUT / prefix / "runtime" / case / (episode + ".json"))
                assert seal["event"] == event and seal["candidate_index_sha256"] == init["candidate_index_sha256"]
                for item in seal["artifacts"]: assert hashed(item["path"]) == item["sha256"]
                trace = read_zstd_jsonl(artifact(seal, "trace"))
                assert len(trace) == len(base[episode]) == len(frozen[episode]) == event["frames"]
                accepted = []
                trajectory = hashlib.sha256()
                previous_bank = None
                for frame, (row, c0, reference) in enumerate(zip(trace, base[episode], frozen[episode], strict=True)):
                    assert row["frame"] == c0["frame"] == reference["frame"] == frame
                    assert all(row[field] == c0[field] for field in ("outputs", "target_uid", "state_before", "state_after"))
                    assert all(row[field] == reference[field] for field in (
                        "full_tracker_state_before_sha256", "full_tracker_state_after_sha256"))
                    assert row["runtime_gt_read"] is False and row["runtime_future_gt_used"] is False
                    assert row["extra_clicks"] == 0 and row["association_authority_zero"] is True
                    assert row["global_assignment_changed"] is False and row["authority"]["approved"] is False
                    outputs = row["outputs"]
                    uids = [output["candidate_uid"] for output in outputs]
                    publics = [output["public_id"] for output in outputs]
                    assert len(outputs) == len(set(uids)) == len(set(publics))
                    assert {output["public_id"]: output["candidate_uid"] for output in outputs}.get(row["target_public_id"]) == row["target_uid"]
                    before, after = row["bank_before_current_score"], row["bank_after_current_commit"]
                    if previous_bank is not None: assert before == previous_bank
                    assert before["anchor_sha256"] == after["anchor_sha256"]
                    assert all(entry["frame"] < frame for entry in before["bank"])
                    previous_bank = after
                    if row["joint_identity_memory_write"]:
                        uid = row["joint_memory_write_candidate_uid"]
                        assert frame > event["frame"] and uid == row["target_uid"] and uid in uids
                        assert row["identity_current_score_captured_before_write"] is True if kind == "fixed" else row["identity_score_before_current_write"] is True
                        accepted.append({"frame": frame, "candidate_uid": uid})
                    else:
                        assert row["joint_memory_write_candidate_uid"] is None
                    trajectory.update(trajectory_text([row]).encode())
                assert accepted == [{"frame": item["frame"], "candidate_uid": item["candidate_uid"]}
                                    for item in saved["accepted_write_provenance"]]
                assert len(accepted) == saved["raw_counts"]["accepted_writes"]
                assert trajectory.hexdigest() == next(item["sha256"] for item in seal["artifacts"] if item["kind"] == "trajectory")
                exposures += len(trace); case_count += 1; writes_count += len(accepted)
                details.append({"kind": kind, "case": case, "episode_uid": episode, "frames": len(trace),
                    "actual_accepted_write_frame_UID_count": len(accepted), "runtime_seal_sha256": sha256(OUT / prefix / "runtime" / case / (episode + ".json")),
                    "stored_offline_outcome_labels_relabeling_NOT_claimed": True})
        census[sequence] = {"valid_original_clicks": len(events),
                           "failed_initializations_retained": len(init["inputs"]) - len(events), "actual_cases": details}
        print({"raw_memory_sequence": sequence, "valid_original_clicks": len(events),
               "actual_nonempty_case_count_so_far": case_count}, flush=True)
    assert case_count == frontier["actual_case_clicks"] == 1062
    assert source == {name: sha256(ROOT / name) for name in CODE}
    assert all(sha256(path) == digest for path, digest in refs.items())
    report = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "source_sha256": source,
        "frontier_snapshot": pointer, "refs_sha256": refs, "all24_raw_case_census": census,
        "actual_case_clicks_NOT_independent_events": case_count,
        "actual_case_frame_exposures_NOT_independent_sample_size": exposures,
        "actual_accepted_write_exposures_NOT_independent_events": writes_count,
        "raw_current_write_UIDs_bank_causality_full_C0_outputs_trajectory_and_pinned9_CSV_reverified": True,
        "old_tensor_subset_AA_NEVER_complete_tracker_attribute_AA": True,
        "offline_label_outcomes_preserved_NOT_independently_reclassified_from_GT": True,
        "raw_GT_frame_annotations_encoder_tracker_rollout_training_or_new_point": False,
        "initialization_offline_identity_truth_read_by_original_checked_result": True,
        "scientific_goal_complete": False, "next_stage_authorized": False}
    write_json(relative, report)
    print({"memory_raw_readback": relative, "all24_complete": True, "case_clicks_NOT_independent": case_count,
           "write_exposures_NOT_independent": writes_count, "scientific_goal_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
