"""Readback historical/fresh baselines and actual FIT pilot, no new rollouts."""
import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import subprocess
import torch
from scripts.n72r21r2_common import (ROOT, OUT, ASSETS, HISTORY, OLD_CANDIDATES, GOAL, read_json,
    write_json, sha256, preregistration, storage, append_log)
from scripts.n72r21r2_reproduce import CASES, SEQUENCES
from scripts.n72r21r2_learned_pilot import inputs, CODE as PILOT_CODE, PROTOCOL as PILOT_PROTOCOL
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import parse_trackeval, trackeval_summary
from scripts.n72r21r2_events import artifact
from sam3_intermot.evaluation.baseline_delivery import historical_summary
from sam3_intermot.evaluation.learned_policy_evidence import METRICS
from sam3_intermot.evaluation.whole_video_density import full_video_summary
from sam3_intermot.evaluation.optimization_census import assess_fit

PREDECESSOR = OUT / "mot/BASELINE_PILOT_DELIVERY_PROTOCOL_V2.json"
PROTOCOL = OUT / "mot/BASELINE_PILOT_DELIVERY_PROTOCOL_V3.json"
PREFIX = "mot/baseline_delivery_v3"
CODE = tuple(dict.fromkeys((*PILOT_CODE, "scripts/n72r21r2_baseline_delivery_v1.py", "scripts/n72r21r2_baseline_delivery_v2.py", "scripts/n72r21r2_baseline_delivery_v3.py",
    "sam3_intermot/evaluation/baseline_delivery.py", "sam3_intermot/evaluation/learned_policy_evidence.py",
    "sam3_intermot/evaluation/whole_video_density.py", "scripts/n72r21r2_reproduce.py",
    "scripts/n72r20r3r2r3_pipeline.py", "scripts/n72r21r2_baseline.py",
    "sam3_intermot/evaluation/optimization_census.py")))


def historical_index_binding(old, init, actual_init_sha256):
    """Old source binds the actual initialization, which binds the index."""
    if old.get("initialization_manifest_sha256") != actual_init_sha256:
        raise ValueError("Historical initialization manifest binding differs")
    if old.get("GT_parsed_by_runtime") is not False:
        raise ValueError("Historical GT-free runtime proof missing")
    expected = init["candidate_index_sha256"]
    if old.get("candidate_index_sha256", expected) != expected:
        raise ValueError("Explicit historical index binding contradicts initialization")
    return expected


def pilot_no_intervention(row, event_frame):
    """Missing learned fields are legal ONLY before/on the sole click."""
    authority = row["authority"]
    if row["frame"] <= event_frame:
        expected = {"family": "shadow", "approved": False, "features": None, "reason": ["NO_INTERVENTION"]}
        if authority != expected or row["selected_action"] is not None:
            raise ValueError("Preclick authority is not the exact original no-intervention schema")
    elif authority.get("effective_assignment_change") is not False or authority.get("approved") is not False or not isinstance(row["selected_action"], dict) or row["selected_action"].get("family") != "KEEP":
        raise ValueError("Missing or nonzero postclick intervention evidence")
    if row["joint_identity_memory_write"] is not False:
        raise ValueError("Actual pilot must retain zero writes")
    return True


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Preserve original baseline-delivery protocol")
    p = preregistration(); storage(32 << 20)
    predecessor = read_json(PREDECESSOR)
    assert predecessor["frozen"] and all(sha256(ROOT / name) == digest for name, digest in predecessor["source_sha256"].items())
    failure = OUT / "mot/baseline_delivery_v2/FAILED_ATTEMPT_V1.json"
    assert read_json(failure)["actual_exit_code"] == 1
    assert not (OUT / "mot/BASELINE.json").exists() and not (OUT / "mot/PILOT.json").exists()
    assert set(SEQUENCES) <= set(p["split"]["historical_development"])
    pilot = read_json(PILOT_PROTOCOL); assert set(pilot["sequences"]) <= set(p["split"]["fit"])
    write_json("mot/BASELINE_PILOT_DELIVERY_PROTOCOL_V3.json", {"stage": "N72R21R2", "goal": GOAL,
        "frozen": True, "preserved_predecessor_protocol_sha256": sha256(PREDECESSOR),
        "preserved_actual_failed_attempt_sha256": sha256(failure),
        "repair": "Preserve V1/V2 failures and code. Historical manifest chain retained. Exact pre/on-click shadow schema has no learned effective_assignment_change field; require that exact schema then explicit false postclick field + KEEP. No permissive missing-field default. Rehash GT bytes solely for existing audit integrity; no GT content parsing, new labels or runtime guidance.",
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
        "historical_sequences": SEQUENCES, "historical_cases": [case for case, _, _ in CASES],
        "historical_seeds": [seed for _, seed, _ in CASES if seed is not None],
        "fresh_sequences": p["split"]["fit"] + p["split"]["inner"],
        "pilot_protocol_sha256": sha256(PILOT_PROTOCOL),
        "audit": "Actual eight historical replay cells across TWO already-exposed videos: entire original/new sealed artifact and code/weight/index/click SHA, per-frame legacy decision/state AA, original pinned CSV scalar/sequence equality. Fresh24 baselines use independently raw-recomputed all24 M4 KEEP/SHADOW audit. Actual six-model FIT-only pilot18 cells: every original click/seed and initial/selected fit lineage; full-global output/state/byte C0 AA, zero-action/write counters and actual pinned CSVs. No new runtime or GT content parsing; existing raw GT bytes are SHA-rehashed only for sealed-audit integrity.",
        "statistics": "Historical and fresh cohorts remain separate. Equal videos and all seeds/clicks inside each video, not legacy combined metrics, seed-count-as-video or best seed. Historical density not imputed. Pilot requires actual zero actions; nonzero actions need separate actual onset audit, never skipped.",
        "new_rollouts_optimizer_steps_or_threshold_changes": 0,
        "full_goal_or_confirmation_authorized": False, "CPU_threads": 1, "storage_floor_GiB": 60})


def run():
    torch.set_num_threads(1); storage(32 << 20)
    p = read_json(PROTOCOL)
    assert p["goal"] == GOAL and p["frozen"] and p["source_sha256"] == {n: sha256(ROOT / n) for n in CODE}
    assert p["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json") and p["pilot_protocol_sha256"] == sha256(PILOT_PROTOCOL)
    pinned = HISTORY / "third_party/MOTIP/TrackEval"
    assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
    refs = {}
    @lru_cache(maxsize=None)
    def hashed(path):
        path = Path(path); refs[str(path)] = sha256(path); return refs[str(path)]
    def checked(path): hashed(path); return read_json(path)
    def artifacts(value):
        for a in value["artifacts"]: assert hashed(a["path"]) == a["sha256"]
    @lru_cache(maxsize=None)
    def actual_fit(uid):
        fit_path = OUT / "training/event_authority" / (uid + ".json")
        fit = checked(fit_path)
        proof = assess_fit(fit, protocol_sha256=hashed(OUT / "training/EVENT_AUTHORITY_PROTOCOL_V1.json"),
            artifact_sha256=hashed, source_sha256=lambda path: hashed(ROOT / path))
        assert proof["verified_actual_optimization"]
        initial_path = Path(fit["checkpoint_path"]).parent / "initial.pt"
        assert hashed(initial_path) == fit["initial_sha256"]
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        selected = torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=False)["model"]
        assert set(initial) == set(selected)
        changed = {}
        for key, before in initial.items():
            after = selected[key]
            assert before.shape == after.shape and before.dtype == after.dtype
            assert torch.isfinite(before).all() and torch.isfinite(after).all()
            changed[key] = int(torch.count_nonzero(before != after))
        assert changed == fit["changed_weight_elements_by_tensor"] and sum(changed.values()) > 0
        return {**proof, "initial_path": str(initial_path), "initial_sha256": hashed(initial_path),
            "actual_selected_vs_initial_tensor_changes": changed, "fit_record_sha256": hashed(fit_path)}
    def metric_readback(receipt, saved, sequences):
        assert receipt["returncode"] == 0 and hashed(receipt["log_path"]) == receipt["log_sha256"]
        args = receipt["command"]; root = Path(args[args.index("--OUTPUT_FOLDER") + 1])
        for tracker, values in saved.items():
            for pattern in ("*.csv", "*_summary.txt"):
                for path in (root / tracker).rglob(pattern): hashed(path)
            actual = trackeval_summary(parse_trackeval(root, tracker, sequences))
            assert all(actual[k] == values[k] for k in METRICS)
            assert all(actual["per_sequence"][s][k] == values["per_sequence"][s][k] for s in sequences for k in METRICS)
    historical = checked(OUT / "audit/BASELINE_REPRODUCTION.json")
    assert historical["actual_replays"] == len(CASES) * len(SEQUENCES) == 8
    assert historical["sequences"] == p["historical_sequences"] and historical["conditions"] == p["historical_cases"]
    old_result = checked(HISTORY / "outputs/N72R21/mot_pilot/RESULT.json")
    state_frames = 0
    for case, _, _ in CASES:
        for sequence in SEQUENCES:
            seal = checked(OUT / "audit/reproduction_seals" / case / (sequence + ".json"))
            old_path = HISTORY / "outputs/N72R21/mot_pilot/runtime_seals" / case / (sequence + ".json")
            old = checked(old_path); assert hashed(old_path) == seal["old_seal_sha256"]
            assert seal["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_reproduce.py")
            assert all(hashed(ROOT / n) == hashed(HISTORY / n) == digest for n, digest in seal["inherited_runtime_SHA"].items())
            artifacts(seal); artifacts(old)
            init_path = HISTORY / "outputs/N72R21/mot_pilot/initialization" / (sequence + ".json")
            init = checked(init_path); assert hashed(init_path) == seal["initialization_sha256"] and hashed(init["anchor_path"]) == init["anchor_sha256"]
            assert init["candidate_index_sha256"] == seal["candidate_index_sha256"] == historical_index_binding(old, init, hashed(init_path))
            assert seal["inherited_runtime_SHA"] == old["code_sha256"] and seal["model_source"] == old["source_model"]
            candidate_root = OLD_CANDIDATES / "candidates" / sequence
            index = checked(candidate_root / "index.json")
            assert hashed(candidate_root / "index.json") == seal["candidate_index_sha256"]
            assert not index.get("runtime_gt_read") and not index.get("runtime_future_gt_used")
            for key, name in (("metadata", "metadata.jsonl.zst"), ("embeddings", "embeddings.f16")):
                assert hashed(candidate_root / name) == index[key + "_sha256"]
            if seal["model_source"]:
                model = seal["model_source"]
                assert hashed(model["path"]) == model["sha256"] and hashed(model["fit_record_path"]) == model["fit_record_sha256"]
            a, b = [read_zstd_jsonl(artifact(value, "trace")) for value in (seal, old)]
            assert len(a) == len(b) == seal["frames"]
            for f, (new, previous) in enumerate(zip(a, b, strict=True)):
                assert new["frame"] == previous["frame"] == f
                assert all(new[k] == previous[k] for k in ("outputs", "target_uid", "selected_action", "state_before", "state_after", "joint_identity_memory_write", "joint_memory_write_candidate_uid"))
                assert len(new["outputs"]) == len({o["public_id"] for o in new["outputs"]}) == len({o["candidate_uid"] for o in new["outputs"]})
                state_frames += 1
            assert next(x["sha256"] for x in seal["artifacts"] if x["kind"] == "trajectory") == next(x["sha256"] for x in old["artifacts"] if x["kind"] == "trajectory")
    metric_readback(historical["invocation"], historical["actual_nine_metrics"], SEQUENCES)
    for case in p["historical_cases"]:
        assert all(abs(historical["actual_nine_metrics"][case][k] - old_result["actual_TrackEval_metrics"][case][k]) < 1e-12 for k in METRICS)
    history_summary = historical_summary(historical["actual_nine_metrics"], SEQUENCES, p["historical_seeds"])
    fresh = {}
    for name in ("KEEP", "SHADOW"):
        report = checked(OUT / "simple" / (name + ".json"))
        assert report["goal"] == GOAL and not report["CONFIRM_VAL_TEST_SOT_accessed"]
        for path, digest in report["all24_source_audit_receipts"].items():
            sequence_report = checked(path); assert hashed(path) == digest
            assert sequence_report["source_whole_runtime_verified_before_raw_GT"]
            assert all(hashed(source) == expected for source, expected in sequence_report["source_sha256"].items())
        assert {Path(path).stem for path in report["all24_source_audit_receipts"]} == set(p["fresh_sequences"])
        fresh[name] = {role: value["actual_full_video"] for role, value in report["roles"].items()}
    baseline_refs = dict(refs)
    baseline_report = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"], "evidence_sha256": baseline_refs,
        "status": "COMPLETE_ACTUAL_HISTORICAL_TWO_VIDEO_REPRODUCTION_AND_FRESH24_BASELINE_NOT_TASK_CLOSURE",
        "historical": history_summary, "historical_replay_decision_state_AA_frame_exposures_NOT_independent": state_frames,
        "fresh_FIT_INNER": fresh, "historical_and_fresh_NOT_pooled": True,
        "new_runtime_rollouts_optimizer_steps_or_GT_content_parses": 0,
        "GT_integrity_only_byte_hash_paths": [path for path in baseline_refs if path.endswith("/gt/gt.txt")], "CONFIRM_VAL_TEST_SOT_accessed": False,
        "final_learned_MOT_G1_G2_or_full_goal_complete": False, "next_stage_authorized": False}
    pilot = checked(PILOT_PROTOCOL); families = {}
    valid_runs = failed_slots = AA_frames = cli_receipts = 0
    for family in pilot["families"]:
        rows = []; lineage = []
        for seed in pilot["seeds"]:
            for sequence in pilot["sequences"]:
                _, uid, fit = inputs(sequence, family, seed)
                fit_path = OUT / "training/event_authority" / (uid + ".json"); hashed(fit_path)
                fit_proof = actual_fit(uid)
                assert fit["nonzero_gradient_steps"] > 0 and sum(fit["changed_weight_elements_by_tensor"].values()) > 0
                assert hashed(fit["checkpoint_path"]) == fit["checkpoint_sha256"]
                runtime_path = OUT / "training/pilot_policy_runtime" / uid / (sequence + ".json")
                runtime = checked(runtime_path); result = checked(OUT / "training/pilot_policy_results" / uid / (sequence + ".json"))
                assert result["runtime_seal_sha256"] == hashed(runtime_path)
                assert runtime["checkpoint_sha256"] == result["checkpoint_sha256"] == fit["checkpoint_sha256"]
                assert runtime["protocol_sha256"] == result["protocol_sha256"] == sha256(PILOT_PROTOCOL)
                assert runtime["source_freeze"] == {n: sha256(ROOT / n) for n in PILOT_CODE} and runtime["actual_GT_file_guard"]
                assert result["effective_direct_decisions_NOT_independent_onsets"] == 0, "Nonzero pilot needs a separate own-onset audit"
                init = checked(OUT / "data/initialization" / (sequence + ".json")); valid = {e["episode_uid"]: e for e in init["inputs"] if not e["initialization_failure"]}
                assert hashed(init["anchor_path"]) == init["anchor_sha256"]
                failed_slots += len(init["inputs"]) - len(valid)
                assert {e["episode_uid"] for e in runtime["episodes"]} == set(result["per_episode_target_components"]) == set(valid)
                base = checked(OUT / "mot/baseline_results" / (sequence + ".json"))
                density = checked(OUT / "data/candidate_integrity" / (sequence + ".json"))
                assert runtime["candidate_index_sha256"] == density["index_sha256"] == init["candidate_index_sha256"]
                assert hashed(density["index_path"]) == density["index_sha256"]
                index = checked(density["index_path"])
                for key in ("metadata", "embeddings"):
                    assert hashed(index[key]) == density[key + "_sha256"]
                for episode in runtime["episodes"]:
                    event = valid[episode["episode_uid"]]; assert event == episode["event"]
                    assert episode["memory_writes"] == episode["effective_direct_decisions_NOT_independent_onsets"] == 0
                    original = checked(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json")); artifacts(episode); artifacts(original)
                    actual_trace, base_trace = [read_zstd_jsonl(artifact(v, "trace")) for v in (episode, original)]
                    assert len(actual_trace) == len(base_trace) == event["frames"]
                    for f, (a, b) in enumerate(zip(actual_trace, base_trace, strict=True)):
                        assert a["frame"] == b["frame"] == f
                        assert all(a[k] == b[k] for k in ("outputs", "target_uid", "state_before", "state_after"))
                        assert pilot_no_intervention(a, event["frame"])
                        assert len(a["outputs"]) == len({o["public_id"] for o in a["outputs"]}) == len({o["candidate_uid"] for o in a["outputs"]})
                        AA_frames += 1
                    assert next(a["sha256"] for a in episode["artifacts"] if a["kind"] == "trajectory") == next(a["sha256"] for a in original["artifacts"] if a["kind"] == "trajectory")
                    coverage = base["coverage"][event["episode_uid"]]; count = result["per_episode_target_components"][event["episode_uid"]]
                    expected = Counter(TARGET_frames=coverage["C0_correct_frames"], VERIFIED_OTHER_frames=coverage["C0_verified_OTHER_takeover_frames"],
                        UNKNOWN_frames=coverage["C0_UNKNOWN_frames"], NONE_frames=event["frames"] - event["frame"] - 1 - coverage["C0_correct_frames"] - coverage["C0_verified_OTHER_takeover_frames"] - coverage["C0_UNKNOWN_frames"], N01_frames=0, N10_frames=0)
                    assert Counter(count) == expected
                    name = episode["tracker_name"]; metrics = result["all_nine_metrics"][name]; c0 = base["actual_nine_metrics"]["CLICK_C0__click" + str(event["slot"])]
                    assert all(metrics[k] == c0[k] and result["paired_deltas_vs_C0"][name][k] == 0 for k in METRICS)
                    rows.append({"sequence": sequence, "seed": seed, "episode_uid": event["episode_uid"],
                        "density": density["whole_video_density"], "metrics": {k: metrics[k] for k in METRICS},
                        "deltas": result["paired_deltas_vs_C0"][name], "counts": {**count, "target_correct_frames": count.get("TARGET_frames", 0),
                            "positive_available_frames": coverage["strict_positive_available_frames"], "physically_visible_frames": coverage["visible_frames"]}})
                    valid_runs += 1
                metric_readback(result["receipt"], result["all_nine_metrics"], [sequence]); cli_receipts += 1
                lineage.append({"seed": seed, "sequence": sequence, **fit_proof})
        families[family] = {"actual_all3_seed_FIT_only_full_video": full_video_summary(rows, pilot["sequences"], expected_seeds=pilot["seeds"]),
            "fit_and_runtime_lineage": lineage, "effective_actions": 0, "memory_writes": 0, "independent_beneficial_root_upper_bound": 0,
            "zero_action_is_NOT_safe_correction_success": True, "INNER_or_confirmation_access": False}
    pilot_report = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"], "evidence_sha256": refs,
        "status": "COMPLETE_ACTUAL_ORIGINAL_SIX_MODEL_EIGHTEEN_CELL_FIT_PILOT_NOT_MAIN_CLOSURE",
        "actual_model_families": families, "original_fixed_point": pilot["point"], "legacy_hard_filter_NOT_current_MAIN_support_policy": True,
        "actual_nonempty_pinned_CLI_receipts": cli_receipts, "valid_click_seed_runs": valid_runs,
        "repeated_failed_initialization_slots_NOT_new_clicks": failed_slots,
        "semantic_state_output_AA_frames_NOT_baseline_tensor_proof": AA_frames,
        "six_real_optimizer_models_NOT_eighteen_models_or_independent_videos": True,
        "no_new_GT_content_parsing_runtime_rollouts_optimizer_steps_or_threshold_changes": True,
        "GT_integrity_only_byte_hash_paths": [path for path in refs if path.endswith("/gt/gt.txt")],
        "next_stage_authorized": False, "full_scientific_goal_complete": False}
    write_json("mot/BASELINE.json", baseline_report); write_json("mot/PILOT.json", pilot_report)
    write_json(PREFIX + "/COMPLETE.json", {"stage": "N72R21R2", "goal": GOAL,
        "protocol_sha256": sha256(PROTOCOL), "named_reports": {name: sha256(OUT / "mot" / (name + ".json")) for name in ("BASELINE", "PILOT")},
        "historical_videos": 2, "historical_replay_cells": 8, "fresh_development_videos": 24,
        "pilot_cells": cli_receipts, "pilot_valid_click_seed_runs": valid_runs,
        "historical_state_AA_frame_exposures": state_frames, "pilot_semantic_state_AA_frame_exposures": AA_frames,
        "no_relabeling_history_as_fresh_data": True, "full_goal_complete": False, "next_stage_authorized": False})
    append_log("ACTUAL_HISTORICAL_FRESH_BASELINE_AND_FIT_PILOT_NAMED_DELIVERY", historical_videos=2, historical_cells=8, fresh_videos=24, pilot_cells=cli_receipts)
    print({"actual_baseline_and_pilot_delivery_complete": True, "historical_two_video_replay_cells": 8,
        "pilot_cells": cli_receipts, "pilot_valid_click_seed_runs": valid_runs, "pilot_AA_frames": AA_frames,
        "history_AA_frames": state_frames, "full_goal_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args(); freeze() if args.action == "freeze" else run()
