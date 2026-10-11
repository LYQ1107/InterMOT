"""Independently recompute original all24 M-A labels from raw offline TRAIN GT.

Uses the unchanged original strict one-to-one IoU truth rule, not an alternative
matcher. No runtime, model, decision, threshold, authority or stored label edit.
Whole original runtimes are reverified before opening frame GT annotations.
"""
import argparse
from collections import Counter
import math
import torch
from scripts.n72r21r2_common import (
    ROOT, OUT, TRAIN, ASSETS, GOAL, read_json, write_json, sha256, utcnow,
    storage, preregistration, development_sequence, append_log,
)
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome

CODE = ("scripts/n72r21r2_memory_gt_readback_v1.py", "scripts/n72r21r2_baseline.py",
    "scripts/n72r21r2_events.py", "scripts/n72r20r1_build_base_score_tape.py",
    "sam3_intermot/one_click/datasets.py", "sam3_intermot/evaluation/causal_identity_events.py")


def offline_truth(sequence, expected_sha256, *, runtimes_verified):
    if runtimes_verified is not True:
        raise ValueError("All original runtime artifacts must be verified before raw GT")
    development_sequence(sequence)
    path = TRAIN / sequence / "gt/gt.txt"
    if sha256(path) != expected_sha256:
        raise ValueError("Original initialization GT checksum changed")
    return dancetrack_annotations(TRAIN / sequence)


def recompute_case(trace, matching, visible, event, target, kind, *, runtimes_verified):
    if runtimes_verified is not True:
        raise ValueError("Unverified runtime cannot enter offline labels")
    if kind not in ("fixed", "fresh_risk"):
        raise ValueError("Only the original fixed and fresh-risk M-A cases")
    grid = list(range(event["frames"]))
    if [row["frame"] for row in trace] != grid or set(matching) != set(grid) or set(visible) != set(grid):
        raise ValueError("Original complete frame grid required; no missing-truth absence imputation")
    counts, writes, drift = Counter(), [], 0.
    for row in trace:
        frame = row["frame"]
        if row["runtime_gt_read"] is not False or row["runtime_future_gt_used"] is not False or row["extra_clicks"] != 0:
            raise ValueError("Offline labels cannot validate a GT-using or extra-click runtime")
        accepted = row["joint_identity_memory_write"]
        if type(accepted) is not bool:
            raise ValueError("Explicit actual current committed write flag required")
        if frame <= event["frame"]:
            if accepted: raise ValueError("No memory writes at or before the human anchor")
            continue
        mapping = matching[frame]
        positive, present = target in mapping.values(), target in visible[frame]
        if positive and not present:
            raise ValueError("Strict positive candidate without same-frame target GT is inconsistent")
        competitive = positive and any(identity is not None and identity != target for identity in mapping.values())
        committed = candidate_identity_outcome(row["target_uid"], mapping, target)
        claim = candidate_identity_outcome(row["identity_claim_UID_before_global_feedback"], mapping, target)
        rank = candidate_identity_outcome(row["identity_rank1_candidate_UID"], mapping, target)
        counts["positive_available_frames"] += positive
        counts["physically_visible_frames"] += present
        counts["competitive_available_frames"] += competitive
        counts["rank1_correct_when_positive_available"] += positive and rank == "TARGET"
        counts["competitive_rank1_correct"] += competitive and rank == "TARGET"
        counts["identity_correct_claim_available"] += positive and claim == "TARGET"
        counts["identity_correct_claim_visible"] += present and claim == "TARGET"
        counts["identity_claim_" + claim] += 1
        counts["C0_committed_" + committed] += 1
        counts["correct_C0_committed_observations"] += committed == "TARGET"
        counts["false_presence_no_positive_candidate"] += not positive and claim != "NONE"
        counts["accepted_writes"] += accepted
        if accepted:
            if row["joint_memory_write_candidate_uid"] != row["target_uid"] or committed == "NONE":
                raise ValueError("Actual write must be the current committed non-NONE UID")
            counts["write_" + committed] += 1
            writes.append({"frame": frame, "candidate_uid": row["target_uid"], "outcome": committed})
        elif row["joint_memory_write_candidate_uid"] is not None:
            raise ValueError("Rejected write cannot carry a committed write UID")
        value = row["bank_drift_cosine_distance_to_anchor"]
        if not math.isfinite(value) or not 0 <= value <= 2.000001:
            raise ValueError("Actual finite cosine-distance bank drift required")
        drift = max(drift, value)
        if kind == "fixed":
            counts["physically_unavailable_frames"] += not present
            counts["false_presence_physically_unavailable"] += not present and claim != "NONE"
            counts["rollback_events"] += bool(row.get("committed_memory_diagnostic", {}).get("rollback"))
    return {"raw_counts": dict(counts), "accepted_write_provenance": writes, "max_bank_drift": drift}


def compare_labels(saved, recomputed):
    expected, actual = saved["raw_counts"], recomputed["raw_counts"]
    if any(expected.get(k, 0) != actual.get(k, 0) for k in set(expected) | set(actual)):
        raise ValueError("Original offline count differs from raw GT recomputation")
    if saved["accepted_write_provenance"] != recomputed["accepted_write_provenance"]:
        raise ValueError("Original write frame/UID/outcome differs from raw GT")
    if saved["max_bank_drift"] != recomputed["max_bank_drift"]:
        raise ValueError("Original post-anchor bank-drift denominator differs")


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Immutable narrow evidence suffix required")
    relative = "audit/MEMORY_ALL24_RAW_GT_LABEL_RECOMPUTATION_V1" + suffix + ".json"
    if (OUT / relative).exists(): raise FileExistsError("Preserve all earlier audit attempts")
    torch.set_num_threads(1)
    storage(16 << 20)
    source = {name: sha256(ROOT / name) for name in CODE}
    raw_path = OUT / "audit/MEMORY_ALL24_RAW_FRONTIER_READBACK_V1ACTUAL_ALL24.json"
    raw = read_json(raw_path)
    assert raw["goal"] == GOAL and raw["actual_case_clicks_NOT_independent_events"] == 1062
    assert raw["raw_current_write_UIDs_bank_causality_full_C0_outputs_trajectory_and_pinned9_CSV_reverified"]
    assert raw["source_sha256"] == {name: sha256(ROOT / name) for name in raw["source_sha256"]}
    refs = {str(raw_path): sha256(raw_path), **raw["refs_sha256"]}
    # This full actual-artifact check precedes any raw frame GT parse below.
    assert all(sha256(path) == digest for path, digest in refs.items())
    p = preregistration()
    sequences = p["split"]["fit"] + p["split"]["inner"]
    assert set(sequences) == set(raw["all24_raw_case_census"])
    census = {}
    cases = frames_total = writes_total = 0
    labels = Counter()

    def checked(path):
        path = path.resolve()
        if not path.is_relative_to(OUT.resolve()): raise ValueError("Only original stage-local sealed evidence")
        digest = sha256(path)
        if str(path) in refs: assert refs[str(path)] == digest
        refs[str(path)] = digest
        return read_json(path)

    for sequence in sequences:
        init = checked(OUT / "data/initialization" / (sequence + ".json"))
        truth = checked(OUT / "data/initialization_truth" / (sequence + ".json"))
        assert truth["runtime_never_reads_this_file"] and all(label["offline_only"] for label in truth["labels"])
        targets = {label["episode_uid"]: label["target_gt_identity"] for label in truth["labels"]}
        events = {event["episode_uid"]: event for event in init["inputs"] if not event["initialization_failure"]}
        assert set(targets) == {event["episode_uid"] for event in init["inputs"]}
        original_frames, index_sha = checked_frames(sequence)
        assert index_sha == init["candidate_index_sha256"]
        gt_path = TRAIN / sequence / "gt/gt.txt"
        gt = offline_truth(sequence, truth["GT_sha256"], runtimes_verified=True)
        refs[str(gt_path)] = truth["GT_sha256"]
        mapping = {int(payload["frame"]): strict_candidate_matching(rows, gt.get(int(payload["frame"]), []))
                   for payload, rows in original_frames}
        assert set(gt) <= set(mapping), "No raw-GT frame may be silently truncated"
        visible = {frame: {a["identity"] for a in gt.get(frame, [])} for frame in mapping}
        for event in init["inputs"]:
            target = targets[event["episode_uid"]]
            anchor = [a for a in gt.get(event["frame"], []) if a["identity"] == target]
            assert len(anchor) == 1 and list(anchor[0]["box"]) == event["box_xyxy"]
            positive = sorted(uid for uid, identity in mapping[event["frame"]].items() if identity == target)
            assert len(positive) <= 1
            assert event["clicked_candidate_uid"] == (positive[0] if positive else None)
            assert event["initialization_failure"] == (not positive)
        details = []
        for kind, prefix in (("fixed", "memory/M_A"), ("fresh_risk", "memory/risk_v1")):
            result = checked(OUT / prefix / "results" / (sequence + ".json"))
            for key, saved in sorted(result["identity_and_write_results"].items()):
                case, episode = key.split("/", 1)
                seal = checked(OUT / prefix / "runtime" / case / (episode + ".json"))
                assert seal["event"] == events[episode]
                path = artifact(seal, "trace")
                expected = next(item["sha256"] for item in seal["artifacts"] if item["kind"] == "trace")
                assert sha256(path) == expected
                refs[str(path)] = expected
                trace = read_zstd_jsonl(path)
                recomputed = recompute_case(trace, mapping, visible, events[episode], targets[episode], kind, runtimes_verified=True)
                compare_labels(saved, recomputed)
                frames_total += len(trace); cases += 1
                writes_total += len(recomputed["accepted_write_provenance"])
                labels.update(write["outcome"] for write in recomputed["accepted_write_provenance"])
                details.append({"kind": kind, "case": case, "episode_uid": episode, "frames": len(trace),
                    "raw_GT_all_original_counts_write_frame_UID_outcomes_drift_equal": True})
        assert len(details) == len(events) * 18
        census[sequence] = {"valid_original_clicks": len(events), "failed_initializations_retained": len(init["inputs"]) - len(events),
            "raw_GT_sha256": truth["GT_sha256"], "actual_case_recomputations": details,
            "all_failed_no_case_NOT_RUN": not events}
        print({"raw_GT_memory_sequence": sequence, "actual_cases_so_far": cases,
               "failed_initializations_retained": len(init["inputs"]) - len(events)}, flush=True)
    assert cases == 1062 and frames_total == raw["actual_case_frame_exposures_NOT_independent_sample_size"]
    assert writes_total == raw["actual_accepted_write_exposures_NOT_independent_events"]
    assert source == {name: sha256(ROOT / name) for name in CODE}
    assert all(sha256(path) == digest for path, digest in refs.items())
    report = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "source_sha256": source,
        "refs_sha256": refs, "all24_census": census,
        "actual_case_clicks_NOT_independent_events": cases, "actual_case_frame_exposures_NOT_independent_sample_size": frames_total,
        "actual_write_exposures_NOT_independent_events": writes_total, "actual_write_outcome_exposures_NOT_independent_events": dict(labels),
        "all_original_counts_and_write_outcomes_independently_recomputed_from_raw_GT": True,
        "same_unchanged_original_one_to_one_IoU_truth_rule_NOT_alternative_matching_validation": True,
        "GT_annotation_presence_NOT_occlusion_truth_or_cross_scene_person_identity": True,
        "raw_GT_read_ONLY_offline_after_actual_runtime_artifacts_verified": True,
        "stored_results_labels_runtime_encoder_threshold_or_association_modified": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "independent_event_risk_uncertainty_still_unproven": True,
        "full_original_M0_M11_goal_complete": False, "scientific_success": None, "next_stage_authorized": False}
    write_json(relative, report)
    append_log("M9_ALL24_RAW_GT_LABEL_INDEPENDENT_RECOMPUTATION", path=relative, cases=cases, writes=writes_total,
               goal_complete=False)
    print({"raw_GT_memory_readback": relative, "all24_complete": True, "actual_cases_NOT_independent": cases}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
