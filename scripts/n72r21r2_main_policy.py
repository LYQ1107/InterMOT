"""All registered MAIN models: GT-free own-video replay, actual pinned MOT.

Two shared INNER operating points, three seeds, no confirmation authority.
Compact traces preserve every global decision and complete tracker seals.
Actual own-prefix one-shot audits are separate from adaptive policy effects.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r2_common import (ROOT, OUT, ASSETS, TRAIN, HISTORY, GOAL, read_json,
    write_json, sha256, storage, preregistration, development_sequence, append_log)
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_event_context_v2 import sealed_click
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_learned_pilot import CODE as PILOT_CODE
from scripts.n72r21r2_simple_onset_audit import compact_full, save_rows
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.authority_support_ablation import SupportAblationBridge, POLICIES
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.one_click.learned_onset_branches import (onset_pair,
    complete_learned_state_fingerprint, chronological_nonoverlapping_candidates)
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.learned_policy_evidence import (METRICS, compact_policy_result,
    assert_decision_preserved, density_from_counts, choose_inner_point)

PREFIX = "mot/main_policy_v1"
BASE = ASSETS / "main_policy_v1"
PROTOCOL = OUT / "mot/MAIN_POLICY_PROTOCOL_V1.json"
JOBS = OUT / "training/MAIN_EXECUTION_PROTOCOL_V1.json"
SUPPORT_PROTOCOL = OUT / "training/SUPPORT_ABLATION_PILOT_PROTOCOL_V1.json"
CODE = tuple(dict.fromkeys((*PILOT_CODE, "scripts/n72r21r2_main_policy.py",
    "scripts/n72r21r2_main_policy_driver.py", "sam3_intermot/evaluation/learned_policy_evidence.py",
    "sam3_intermot/one_click/learned_onset_branches.py", "sam3_intermot/one_click/authority_support_ablation.py",
    "scripts/n72r21r2_simple_onset_audit.py", "scripts/n72r21r2_baseline.py",
    "scripts/n72r21r2_event_context_v2.py", "sam3_intermot/one_click/causal_state_fingerprint.py")))
POINTS = {"P0_CLAIM80": dict(claim_min=.8, risk_max=.02, confirmation_delay=3,
            global_regret_max=.2, anchor_advantage_min=.1),
          "P1_CLAIM50": dict(claim_min=.5, risk_max=.02, confirmation_delay=3,
            global_regret_max=.2, anchor_advantage_min=.1)}


def freeze():
    p = preregistration()
    jobs = read_json(JOBS)["jobs"]
    assert len(jobs) == 42 and {j["seed"] for j in jobs} == set(p["hyperparameters"]["seeds"])
    assert all(v["claim_min"] in p["hyperparameters"]["claim_thresholds"] for v in POINTS.values())
    assert not list((OUT / "training/event_authority").glob("MAIN__*.json")), "Freeze before seeing MAIN weights/outcomes"
    sizing = read_json(OUT / "mot/MAIN_COMPACT_TRACE_SIZING_V1.json")
    assert sizing["all_actual_decisions_tensor_seals_and_keep_proposals_preserved"]
    smoke_result = read_json(OUT / "mot/MAIN_RUNTIME_EXISTING_PILOT_SMOKE_V1.json")
    assert smoke_result["actually_replayed_frames"] == 51 and smoke_result["decision_evidence_round_trip_preserved"]
    storage(6 << 30)
    write_json("mot/MAIN_POLICY_PROTOCOL_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "final_goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "job_protocol_sha256": sha256(JOBS),
        "source_sha256": {n: sha256(ROOT / n) for n in CODE}, "jobs": jobs, "seeds": p["hyperparameters"]["seeds"],
        "split": {k: p["split"][k] for k in ("fit", "inner")}, "points": POINTS,
        "authority_support_policy": POLICIES[-1], "support_pilot_protocol_sha256": sha256(SUPPORT_PROTOCOL),
        "support_change_rationale": "Actual FIT16 support audit found every104 component-beneficial direct frame vetoed by legacy global cost. Explicit matched36-cell frozen-weight hard-filter pilot precedes MAIN replay. No score clipping, no memory/model/candidate change, no GT action selection.",
        "sampling": "Every original frame global output/action/core before-after SHA retained. All current choices and own primary causal feature/past3 vectors at every16 postclick frame and every actually effective decision. Complete learned-history/pending SHA at actual-action after-states and episode end; not quadratic whole-history hashing every frame.",
        "evaluation": "Actual pinned full original-video nine metrics for all valid clicks; zero-action tracker bytes and tracker tensors match real C0. Separate all-effective own-prefix same-prestate one-shot H1/5/20/50/100 audits, never substitute for adaptive full-video metrics.",
        "selection": "Both points on all INNER8 for every42 fixed weights. Shared point per family/objective using all3 seeds and video/click macro: nonvacuity and harm screen, severe harm, risky decisions, HOTA, AssA, IDF1, deterministic name tie. If all vacuous/unsafe, select fixed diagnostic point only, no qualification. Then all FIT16 at that point. No best seed or confirmation tuning.",
        "G1": "Frames, contiguous intervals, spaced windows and seeds are NOT independent causal roots. All actual one-shot action onsets reported; independent-root proof and full-policy G0/G1/G2 qualification remain a separate mandatory closure audit.",
        "G2": p["gates"]["G2"], "density_rule_previously_preregistered": p["density"],
        "development_only_uncalibrated_checkpoint_allowed": True,
        "confirmation_VAL_TEST_SOT_authorized": False, "memory_policy": "P0_FROZEN_NO_WRITES",
        "CPU_workers": 1, "OMP_threads": 1, "reserve_GiB": 60,
        "trace_sizing_sha256": sha256(OUT / "mot/MAIN_COMPACT_TRACE_SIZING_V1.json"),
        "runtime_smoke_sha256": sha256(OUT / "mot/MAIN_RUNTIME_EXISTING_PILOT_SMOKE_V1.json"),
        "estimated_extra_assets_GiB_planning": 6.,
        "storage_estimate_not_unbounded_onset_storage_permission": "Actual fixed pilot extrapolates3.69GiB traces; allow planning6GiB including trajectories/evaluation/branches, check32MiB minimum per cell and60GiB floor. Unexpected nonzero-onset growth requires measured re-audit, no deleting unique evidence.",
        "no_extra_training_or_cartesian_support_search": True})


def sizing():
    """Actual frozen pilot source, compact round-trip, no new scientific run."""
    support = read_json(SUPPORT_PROTOCOL)
    # Fixed first seed / first legacy policy / first pilot sequence; not best-size selection.
    uid = "PILOT__LOGISTIC_RISK__L3_HARM_AWARE__seed" + str(support["seeds"][0]) + "__" + POLICIES[0]
    source = OUT / "training/support_ablation_v1/runtime" / uid / (support["sequences"][0] + ".json")
    seal = read_json(source)
    records, raw_bytes, compressed_bytes = [], 0, 0
    for e in seal["episodes"]:
        path = artifact(e, "trace")
        original = read_zstd_jsonl(path)
        compact = [compact_policy_result(r, sample=r["frame"] > e["event"]["frame"] and
                   (r["frame"] - e["event"]["frame"] - 1) % 16 == 0) for r in original]
        destination = BASE / "trace_sizing" / (e["episode_uid"] + ".jsonl.zst")
        save_rows(destination, compact)
        restored = read_zstd_jsonl(destination)
        assert len(restored) == len(original)
        for a, b in zip(original, restored, strict=True):
            assert_decision_preserved(a, b)
        raw_bytes += path.stat().st_size
        compressed_bytes += destination.stat().st_size
        records.append({"source": str(path), "source_sha256": sha256(path), "compact": str(destination),
                        "compact_sha256": sha256(destination), "source_bytes": path.stat().st_size,
                        "compact_bytes": destination.stat().st_size, "frames": len(original)})
    frames = sum(r["frames"] for r in records)
    expected_cells = 42 * (2 * 8 + 16)
    # Planning extrapolation, NOT a promise from one short source sample.
    approximate = compressed_bytes / max(frames, 1) * 2000 * 3 * expected_cells
    write_json("mot/MAIN_COMPACT_TRACE_SIZING_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "source_runtime_seal_sha256": sha256(source), "records": records,
        "source_compressed_bytes": raw_bytes, "compact_compressed_bytes": compressed_bytes,
        "all_actual_decisions_tensor_seals_and_keep_proposals_preserved": True,
        "expected_cells": expected_cells, "planning_2000_frames_three_clicks_bytes": approximate,
        "planning_not_worst_case_proof": True, "trace_count_and_disk_checked_during_execution": True,
        "resource": storage(3 << 30), "not_new_MOT_or_scientific_result": True})
    print({"actual_compact_sizing": records, "planning_GiB": approximate / (1 << 30)}, flush=True)


def inputs(sequence, family, objective, seed, point):
    development_sequence(sequence)
    p = read_json(PROTOCOL)
    assert p["goal"] == GOAL and p["source_sha256"] == {n: sha256(ROOT / n) for n in CODE}
    assert p["job_protocol_sha256"] == sha256(JOBS)
    assert {"family": family, "objective": objective, "seed": seed} in p["jobs"] and point in p["points"]
    model_uid = "MAIN__" + family + "__" + objective + "__seed" + str(seed)
    fit_path = OUT / "training/event_authority" / (model_uid + ".json")
    fit = read_json(fit_path)
    assert not fit["pilot_not_main_fit_or_independent_confirmation"]
    assert fit["nonzero_gradient_steps"] > 0 and sum(fit["changed_weight_elements_by_tensor"].values()) > 0
    assert sha256(fit["checkpoint_path"]) == fit["checkpoint_sha256"]
    assert fit["source_freeze"] == {n: sha256(ROOT / n) for n in fit["source_freeze"]}
    assert set(fit["manifest"]["actual_fit_videos"]).issubset(p["split"]["fit"])
    assert set(fit["manifest"]["actual_validation_videos"]).issubset(p["split"]["inner"])
    if sequence in p["split"]["fit"]:
        selection = read_json(OUT / PREFIX / "selections" / (family + "__" + objective + ".json"))
        assert selection["selected_point"] == point and selection["protocol_sha256"] == sha256(PROTOCOL)
    return p, model_uid + "__" + point, fit, fit_path


def smoke():
    """Fixed existing pilot first51 frames; no MAIN outcomes, labels or fitting."""
    support = read_json(SUPPORT_PROTOCOL)
    sequence, seed = support["sequences"][0], support["seeds"][0]
    uid = "PILOT__LOGISTIC_RISK__L3_HARM_AWARE__seed" + str(seed)
    fit = read_json(OUT / "training/event_authority" / (uid + ".json"))
    assert sha256(fit["checkpoint_path"]) == fit["checkpoint_sha256"]
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    event = init["inputs"][0]
    assert not event["initialization_failure"]
    anchor = np.array(np.load(init["anchor_path"], mmap_mode="r")[event["anchor_index"]], np.float32)
    frames, index_sha = checked_frames(sequence)
    bridge = build_bridge({"current_point": POINTS["P0_CLAIM80"], "authority_support_policy": POLICIES[-1]}, fit, event, anchor)
    c0 = MOTIdentityBridge(sealed_click(event, anchor), frames=len(frames))
    c0.configure_fps(event["fps"])
    direct, compared, tensor_equal = 0, 0, 0
    torch.set_num_threads(1)
    for payload, rows in frames[:51]:
        f = payload["frame"]
        before = full_tracker_fingerprint(bridge)
        if not direct:
            assert before == full_tracker_fingerprint(c0)
        result, base = bridge.step(f, rows), c0.step(f, rows)
        direct += bool(result["authority"].get("effective_assignment_change"))
        if not direct:
            assert result["outputs"] == base["outputs"] and full_tracker_fingerprint(bridge) == full_tracker_fingerprint(c0)
            tensor_equal += 1
        result.update(full_tracker_state_before_sha256=before, full_tracker_state_after_sha256=full_tracker_fingerprint(bridge))
        compact = compact_policy_result(result, primary_vector=bridge.event_feature_history.get(f), sample=f % 16 == 0)
        assert_decision_preserved(result, compact)
        compared += 1
    write_json("mot/MAIN_RUNTIME_EXISTING_PILOT_SMOKE_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "fixed_existing_pilot": uid, "sequence": sequence, "episode_uid": event["episode_uid"],
        "candidate_index_sha256": index_sha, "checkpoint_sha256": fit["checkpoint_sha256"],
        "actually_replayed_frames": compared, "zero_action_prefix_core_tensor_C0_AA_frames": tensor_equal,
        "effective_decisions": direct, "decision_evidence_round_trip_preserved": True,
        "GT_guard_actual": True, "not_MAIN_training_or_full_MOT_or_G1_G2_evidence": True})
    print({"MAIN_runtime_existing_pilot_smoke": compared, "tensor_equal_C0_prefix_frames": tensor_equal, "actions": direct}, flush=True)


def build_bridge(p, fit, event, anchor):
    actor, _ = make_actor(event, anchor)
    predictor = EventAuthorityPredictor(torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=False),
                                        development_diagnostic=True)
    bridge = SupportAblationBridge(sealed_click(event, anchor), actor, predictor=predictor,
        point=p["current_point"], frames=event["frames"], support_policy=p["authority_support_policy"])
    bridge.configure_fps(event["fps"])
    return bridge


def runtime(sequence, family, objective, seed, point):
    torch.set_num_threads(1)
    p, uid, fit, fit_path = inputs(sequence, family, objective, seed, point)
    init_path = OUT / "data/initialization" / (sequence + ".json")
    init = read_json(init_path)
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    density = density_from_counts(len(rows) for _, rows in frames)
    receipts = []
    storage(32 << 20)
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        anchor = np.array(anchors[event["anchor_index"]], np.float32)
        bridge = build_bridge({**p, "current_point": p["points"][point]}, fit, event, anchor)
        c0 = MOTIdentityBridge(sealed_click(event, anchor), frames=len(frames))
        c0.configure_fps(event["fps"])
        base_seal = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
        baseline = read_zstd_jsonl(artifact(base_seal, "trace"))
        name = uid + "__click" + str(event["slot"])
        trace_path = BASE / "traces" / uid / (event["episode_uid"] + ".jsonl.zst")
        tracker_path = BASE / "trackers" / name / "data" / (sequence + ".txt")
        for path in (trace_path, tracker_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError("Preserve full-video partial/completed MAIN attempts")
        direct, writes, began = 0, 0, time.monotonic()
        first, zero_prefix_tensor_frames = None, 0
        with trace_path.open("xb") as handle, tracker_path.open("x") as tracker:
            proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
            try:
                for payload, rows in frames:
                    frame = int(payload["frame"])
                    before, c0_before = full_tracker_fingerprint(bridge), full_tracker_fingerprint(c0)
                    if direct == 0:
                        assert before == c0_before
                    result = bridge.step(frame, rows)
                    base_result = c0.step(frame, rows)
                    for key in ("outputs", "target_uid", "state_before", "state_after"):
                        assert base_result[key] == baseline[frame][key]
                    changed = bool(result["authority"].get("effective_assignment_change"))
                    if changed and first is None:
                        first = frame
                    direct += changed
                    if direct == 0:
                        assert full_tracker_fingerprint(bridge) == full_tracker_fingerprint(c0)
                        assert result["outputs"] == base_result["outputs"]
                        zero_prefix_tensor_frames += 1
                    result.update(full_tracker_state_before_sha256=before,
                                  full_tracker_state_after_sha256=full_tracker_fingerprint(bridge))
                    if changed:
                        result["full_learned_state_after_sha256"] = complete_learned_state_fingerprint(bridge)
                    sampled = frame > event["frame"] and (frame - event["frame"] - 1) % 16 == 0
                    compact = compact_policy_result(result, primary_vector=bridge.event_feature_history.get(frame),
                        past_vectors=[bridge.event_feature_history.get(f, [0.] * 32) for f in range(frame - 3, frame)], sample=sampled)
                    assert_decision_preserved(result, compact)
                    writes += compact["identity_memory_write"]
                    proc.stdin.write((json.dumps(compact, sort_keys=True, allow_nan=False) + "\n").encode())
                    tracker.write(trajectory_text([result]))
                proc.stdin.close()
                assert proc.wait() == 0
            finally:
                if proc.poll() is None:
                    proc.terminate(); proc.wait()
        assert writes == 0 and not bridge.identity.bank
        if direct == 0:
            assert sha256(tracker_path) == sha256(artifact(base_seal, "trajectory"))
        receipts.append({"episode_uid": event["episode_uid"], "event": event, "tracker_name": name,
            "effective_direct_decisions_NOT_independent_onsets": direct, "first_effective_decision": first,
            "zero_action_prefix_tensor_C0_AA_frames": zero_prefix_tensor_frames, "memory_writes": writes,
            "complete_learned_final_state_sha256": complete_learned_state_fingerprint(bridge),
            "seconds": time.monotonic() - began, "artifacts": [{"kind": k, "path": str(path),
            "sha256": sha256(path), "bytes": path.stat().st_size} for k, path in (("trace", trace_path), ("trajectory", tracker_path))]})
        print({"actual_MAIN_full_video_episode": event["episode_uid"], "model_point": uid, "direct_decisions": direct}, flush=True)
    write_json(PREFIX + "/runtime/" + uid + "/" + sequence + ".json", {"stage": "N72R21R2", "goal": GOAL,
        "sequence": sequence, "family": family, "objective": objective, "seed": seed, "point": point,
        "experiment_uid": uid, "status": "COMPLETE_ACTUAL_OWN_POLICY_FULL_VIDEO_UNQUALIFIED_DEVELOPMENT",
        "fit_record_sha256": sha256(fit_path), "checkpoint_sha256": fit["checkpoint_sha256"],
        "protocol_sha256": sha256(PROTOCOL), "source_freeze": p["source_sha256"], "candidate_index_sha256": index_sha,
        "initialization_sha256": sha256(init_path), "density": density, "episodes": receipts,
        "initialization_failures_not_replaced": sum(e["initialization_failure"] for e in init["inputs"]),
        "actual_GT_file_guard": True, "sole_click_full_global_original_hard_negatives": True,
        "runtime_future_GT_read": False, "scientific_success": None, "confirmation_authorized": False})


def verified_runtime(sequence, family, objective, seed, point):
    p, uid, fit, _ = inputs(sequence, family, objective, seed, point)
    path = OUT / PREFIX / "runtime" / uid / (sequence + ".json")
    seal = read_json(path)
    assert seal["protocol_sha256"] == sha256(PROTOCOL) and seal["source_freeze"] == p["source_sha256"]
    assert seal["checkpoint_sha256"] == fit["checkpoint_sha256"]
    assert all(sha256(a["path"]) == a["sha256"] for e in seal["episodes"] for a in e["artifacts"])
    return p, uid, fit, path, seal


def onset_runtime(sequence, family, objective, seed, point):
    torch.set_num_threads(1)
    p, uid, fit, source_path, seal = verified_runtime(sequence, family, objective, seed, point)
    entries, frames = [], None
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    for e in seal["episodes"]:
        observed = read_zstd_jsonl(artifact(e, "trace"))
        positions = [r["frame"] for r in observed if r["authority"]["effective_assignment_change"]]
        assert len(positions) == e["effective_direct_decisions_NOT_independent_onsets"]
        entry = {"episode_uid": e["episode_uid"], "event": e["event"], "effective_decision_frames": positions,
                 "status": "ACTUAL_OWN_PREFIX_PAIRS" if positions else "NOT_RUN_NO_EFFECTIVE_ACTION",
                 "branches": []}
        entries.append(entry)
        if not positions:
            continue
        if frames is None:
            frames, _ = checked_frames(sequence)
        bridge = build_bridge({**p, "current_point": p["points"][point]}, fit, e["event"],
                              np.array(anchors[e["event"]["anchor_index"]], np.float32))
        spaced = chronological_nonoverlapping_candidates(positions)
        for payload, candidates in frames:
            frame = int(payload["frame"])
            assert full_tracker_fingerprint(bridge) == observed[frame]["full_tracker_state_before_sha256"]
            if frame in positions:
                before = complete_learned_state_fingerprint(bridge)
                pairs = onset_pair(bridge, frames, frame, observed[frame], compact_full)
                artifacts = {}
                for name, rows in pairs.items():
                    path = BASE / "onsets" / uid / e["episode_uid"] / ("frame" + str(frame)) / (name + ".jsonl.zst")
                    save_rows(path, rows)
                    artifacts[name] = {"path": str(path), "sha256": sha256(path), "frames": len(rows)}
                entry["branches"].append({"frame": frame, "complete_learned_prestate_sha256": before,
                    "public_axis": sorted(bridge.tracker.states), "artifacts": artifacts,
                    "nonoverlapping_window_not_independent_root": frame in spaced})
                assert complete_learned_state_fingerprint(bridge) == before
            actual = bridge.step(frame, candidates)
            assert actual["outputs"] == observed[frame]["outputs"] and actual["selected_action"] == observed[frame]["selected_action"]
            assert full_tracker_fingerprint(bridge) == observed[frame]["full_tracker_state_after_sha256"]
        assert complete_learned_state_fingerprint(bridge) == e["complete_learned_final_state_sha256"]
    write_json(PREFIX + "/onset_runtime/" + uid + "/" + sequence + ".json", {"stage": "N72R21R2", "goal": GOAL,
        "sequence": sequence, "experiment_uid": uid, "protocol_sha256": sha256(PROTOCOL),
        "runtime_seal_sha256": sha256(source_path), "entries": entries, "GT_guard_actual": True,
        "complete_controller_clone_and_actual_own_prefix_AA": True, "not_G1_independence_proof": True})


def evaluate(sequence, family, objective, seed, point):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome
    from sam3_intermot.evaluation.safe_intervention_events import assignment_map
    from sam3_intermot.evaluation.event_causal_labels import label_actual_branch
    p, uid, fit, seal_path, seal = verified_runtime(sequence, family, objective, seed, point)
    onset_path = OUT / PREFIX / "onset_runtime" / uid / (sequence + ".json")
    onsets = read_json(onset_path)
    assert onsets["protocol_sha256"] == sha256(PROTOCOL) and onsets["runtime_seal_sha256"] == sha256(seal_path)
    assert all(sha256(a["path"]) == a["sha256"] for e in onsets["entries"] for b in e["branches"] for a in b["artifacts"].values())
    # Entire adaptive cell AND independent branch artifacts are sealed before truth.
    frames, _ = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {r["frame"]: strict_candidate_matching(rows, gt.get(r["frame"], [])) for r, rows in frames}
    truth = read_json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    components, onset_labels = {}, []
    summary = Counter()
    for e, onset in zip(seal["episodes"], onsets["entries"], strict=True):
        assert e["episode_uid"] == onset["episode_uid"]
        target = targets[e["episode_uid"]]
        trace = read_zstd_jsonl(artifact(e, "trace"))
        c0 = read_zstd_jsonl(artifact(read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (e["episode_uid"] + ".json")), "trace"))
        count, own_origins, c0_origins, rows_by_onset = Counter(), {}, {}, {}
        branches = {b["frame"]: b for b in onset["branches"]}
        for a, b in zip(trace, c0, strict=True):
            f = a["frame"]
            assert f == b["frame"]
            if f in branches:
                branch = branches[f]
                arms = {n: read_zstd_jsonl(Path(ref["path"])) for n, ref in branch["artifacts"].items()}
                active = {public: identity for public, identity in own_origins.items() if public in branch["public_axis"]}
                visible = {k: any(r["identity"] == target for r in gt.get(k, [])) for k in range(f, min(f + 101, len(frames)))}
                labels = label_actual_branch(arms["ACTUAL_CURRENT_THEN_KEEP"], arms["OWN_KEEP"], matched, target, active, visible)
                future = labels["future"]["H100"]
                observed_risk = any(r["N10"] or r["non_target_damage"] or r["verified_OTHER_writes"] or r["UNKNOWN_writes"] for r in labels["raw_frame_components"])
                summary["effective_decisions"] += 1
                summary["complete_H100_decisions"] += future["complete"]
                summary["incomplete_H100_decisions"] += not future["complete"]
                summary["beneficial_complete_H100_decisions"] += bool(future["benefit_label"] and not observed_risk)
                summary["risky_decisions"] += observed_risk
                summary["severe_non_target_harm_decisions"] += bool(future["severe_non_target_harm_public_ids"])
                rows_by_onset[f] = {"frame": f, "labels": labels,
                    "spaced_window_NOT_independent_root": branch["nonoverlapping_window_not_independent_root"]}
            for public, candidate in assignment_map(a).items():
                identity = matched[f][candidate]
                if identity is not None:
                    own_origins.setdefault(public, identity)
            actual_map, base_map = assignment_map(a), assignment_map(b)
            for public, candidate in base_map.items():
                identity = matched[f][candidate]
                if identity is not None:
                    c0_origins.setdefault(public, identity)
            if f <= e["event"]["frame"]:
                continue
            current = candidate_identity_outcome(a["target_uid"], matched[f], target)
            base = candidate_identity_outcome(b["target_uid"], matched[f], target)
            count["N01_frames"] += current == "TARGET" and base != "TARGET"
            count["N10_frames"] += base == "TARGET" and current != "TARGET"
            count[current + "_frames"] += 1
            count["C0_" + base + "_frames"] += 1
            count["positive_available_frames"] += target in matched[f].values()
            count["physically_visible_frames"] += any(r["identity"] == target for r in gt.get(f, []))
            for public, identity in c0_origins.items():
                if identity == target or public == a["target_public_id"]:
                    continue
                base_correct = matched[f].get(base_map.get(public)) == identity
                actual_correct = matched[f].get(actual_map.get(public)) == identity
                count["non_target_origin_proxy_damage_public_frames"] += base_correct and not actual_correct
                count["non_target_origin_proxy_benefit_public_frames"] += not base_correct and actual_correct
        components[e["episode_uid"]] = dict(count)
        onset_labels.append({"episode_uid": e["episode_uid"], "onsets": list(rows_by_onset.values()),
                             "prestate_public_origin_proxy_NOT_global_IDF1": True})
    evaluation = BASE / "trackeval" / uid / sequence
    evaluation.mkdir(parents=True, exist_ok=False)
    seqmap = evaluation / "seqmap.txt"
    with seqmap.open("x") as handle:
        handle.write("name\n" + sequence + "\n")
    names = [e["tracker_name"] for e in seal["episodes"]]
    metrics, receipt = {}, {"status": "NOT_RUN_NO_VALID_INITIALIZATION"}
    if names:
        pinned = HISTORY / "third_party/MOTIP/TrackEval"
        assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
        command = _trackeval_command(BASE / "trackers", evaluation, names, seqmap, gt_split="train", gt_folder=TRAIN)
        command[2] = str(pinned / "scripts/run_mot_challenge.py")
        log = evaluation / "trackeval.log"
        with log.open("x") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        receipt = {"command": command, "returncode": process.returncode, "log_path": str(log), "log_sha256": sha256(log)}
        if process.returncode:
            raise RuntimeError("Preserve failed actual MAIN TrackEval")
        metrics = {n: trackeval_summary(parse_trackeval(evaluation, n, [sequence])) for n in names}
        assert all(all(v[k] is not None for k in METRICS) for v in metrics.values())
    baseline = read_json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
    deltas = {n: {k: values[k] - baseline["CLICK_C0__click" + n.rsplit("__click", 1)[1]][k] for k in METRICS} for n, values in metrics.items()}
    for k in ("effective_decisions", "complete_H100_decisions", "incomplete_H100_decisions", "beneficial_complete_H100_decisions", "risky_decisions", "severe_non_target_harm_decisions"):
        summary.setdefault(k, 0)
    write_json(PREFIX + "/results/" + uid + "/" + sequence + ".json", {"stage": "N72R21R2", "goal": GOAL,
        "sequence": sequence, "family": family, "objective": objective, "seed": seed, "point": point,
        "experiment_uid": uid, "protocol_sha256": sha256(PROTOCOL), "checkpoint_sha256": fit["checkpoint_sha256"],
        "runtime_seal_sha256": sha256(seal_path), "onset_runtime_seal_sha256": sha256(onset_path),
        "all_nine_metrics": metrics, "paired_deltas_vs_C0": deltas, "per_episode_target_components": components,
        "actual_pinned_full_original_video_TrackEval": receipt, "valid_initializations": len(names),
        "initialization_failures_not_replaced": seal["initialization_failures_not_replaced"], "density": seal["density"],
        "effective_direct_decisions_NOT_independent_onsets": summary["effective_decisions"],
        "own_onset_summary": dict(summary), "own_prefix_one_shot_labels": onset_labels,
        "adaptive_non_target_damage_is_C0_public_origin_proxy_not_global_IDF1": True,
        "independent_G1_roots_proven": False, "zero_actions_not_safety_PASS": True,
        "scientific_success": None, "confirmation_authorized": False})
    append_log("M5_M6_MAIN_ACTUAL_FULL_VIDEO_DEVELOPMENT_COMPLETE", model=uid, sequence=sequence, effective=summary["effective_decisions"])


def select(family, objective):
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in CODE}
    cells, refs = {}, []
    for point in p["points"]:
        cells[point] = []
        for seed in p["seeds"]:
            for s in p["split"]["inner"]:
                uid = "MAIN__" + family + "__" + objective + "__seed" + str(seed) + "__" + point
                path = OUT / PREFIX / "results" / uid / (s + ".json")
                cell = read_json(path)
                assert cell["protocol_sha256"] == sha256(PROTOCOL)
                cells[point].append(cell)
                refs.append({"path": str(path), "sha256": sha256(path)})
    chosen = choose_inner_point(cells, p["seeds"], p["split"]["inner"])
    write_json(PREFIX + "/selections/" + family + "__" + objective + ".json", {"stage": "N72R21R2", "goal": GOAL,
        "protocol_sha256": sha256(PROTOCOL), "family": family, "objective": objective,
        "actual_INNER_result_refs": refs, **chosen})
    print({"actual_INNER_shared_selection": family + "__" + objective, "point": chosen["selected_point"], "status": chosen["status"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("sizing", "smoke", "freeze", "runtime", "onsets", "evaluate", "select"))
    parser.add_argument("--sequence")
    parser.add_argument("--family")
    parser.add_argument("--objective")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--point")
    args = parser.parse_args()
    if args.action == "sizing":
        sizing()
    elif args.action == "smoke":
        with runtime_file_guard():
            smoke()
    elif args.action == "freeze":
        freeze()
    elif args.action == "select":
        select(args.family, args.objective)
    elif args.action == "evaluate":
        evaluate(args.sequence, args.family, args.objective, args.seed, args.point)
    else:
        with runtime_file_guard():
            worker = runtime if args.action == "runtime" else onset_runtime
            worker(args.sequence, args.family, args.objective, args.seed, args.point)
