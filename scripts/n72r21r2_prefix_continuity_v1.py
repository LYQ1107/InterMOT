"""POSTHOC METRIC-SCOPE DIAGNOSTIC, NOT POLICY SELECTION OR FULL MOT.

Frozen structural late events; all original FIT16/INNER8; original prefixes
and entire same-prestate full-global arms. Actual pinned API AND CLI in both
future-only H100 and prefix-inclusive domains. No optimizer or runtime replay.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import (ROOT, OUT, ASSETS, TRAIN, HISTORY, GOAL,
    read_json, write_json, sha256, preregistration, storage, append_log)
from scripts.n72r21r2_event_delivery_v1 import runtime_inputs, reference
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_trajectory_utility import evaluator
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.evaluation.pinned_event_trajectory import (PinnedEventEvaluator,
    METRIC_NAMES, PINNED_COMMIT)
from sam3_intermot.evaluation.prefix_continuity import (
    structural_late_selection, assemble_prefix, paired_sign_counts)

PROTOCOL = OUT / "events/PREFIX_CONTINUITY_DIAGNOSTIC_PROTOCOL_V1.json"
PREFIX = "events/prefix_continuity_v1"
BASE = ASSETS / "prefix_continuity_v1"
CODE = ("scripts/n72r21r2_prefix_continuity_v1.py",
    "sam3_intermot/evaluation/prefix_continuity.py",
    "sam3_intermot/evaluation/pinned_event_trajectory.py",
    "scripts/n72r21r2_event_delivery_v1.py", "scripts/n72r21r2_trajectory_utility.py",
    "scripts/n72r20r4_run_causal_tracker.py", "scripts/n72r20r3r2r3_pipeline.py",
    "scripts/n72r20r3r2r3_trackeval_entry.py")


def freeze():
    p = preregistration()
    choices = {}
    # Only original initialization/registered plans: no GT, labels or utility.
    for sequence in p["split"]["fit"] + p["split"]["inner"]:
        init_path = OUT / "data/initialization" / (sequence + ".json")
        plan_path = OUT / "events/counterfactual_plans" / (sequence + ".json")
        init, plan = read_json(init_path), read_json(plan_path)
        assert plan["sampling_frozen_before_branch_effects"] and not plan["runtime_GT_or_future_label_fields"]
        choices[sequence] = {**structural_late_selection(init["inputs"], plan["episodes"]),
            "initialization_sha256": sha256(init_path), "plan_sha256": sha256(plan_path)}
    write_json("events/PREFIX_CONTINUITY_DIAGNOSTIC_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "choices": choices,
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
        "source_sha256": {n: sha256(ROOT / n) for n in CODE}, "TrackEval_commit": PINNED_COMMIT,
        "scope": "POSTHOC METRIC-SCOPE DIAGNOSTIC, NOT POLICY SELECTION. Earliest original valid sole click; latest registered complete H100 position for that click; every actual sealed branch at this position, all24 original FIT16/INNER8 with failed clicks/all-failed videos retained. Selection never reads GT, labels, branch effects, metric values or model scores.",
        "domains": {"future_H100": "original t+1..t+100, length100",
                    "current_and_future_H100": "original t..t+100, length101; API only for current-frame boundary sensitivity",
                    "prefix_through_H100": "unchanged full-global original C0_SHADOW source0..t-1 + entire actual brancht..t+100, lengtht+101"},
        "validation": "Actual frozen pinned API and actual CLI all9 metrics per original arm in BOTH future_H100 and prefix_through_H100. No target-only splice, ID permutation, candidate or geometry edits, oracle matching, detection reuse across arms or cross-event sum.",
        "numeric_API_CLI_absolute_tolerance": 1e-10,
        "synthetic_smoke_is_method_validation_not_scientific_data": True,
        "legacy_source_tensor_evidence_not_upgraded": True,
        "unchanged_L5_labels_42_models_thresholds_gates_and_policy_selection": True,
        "not_full_video_adaptive_policy_or_independent_roots_or_G1_G2": True,
        "CPU_workers": 1, "reserve_GiB": 60, "estimated_extra_assets_GiB_planning": 1.,
        "confirmation_VAL_TEST_SOT_authorized": False, "next_stage_authorized": False})


def checked():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in CODE}
    assert p["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    q = preregistration()
    assert set(p["choices"]) == set(q["split"]["fit"] + q["split"]["inner"])
    return p


def cli(root, sequence, gt_file, event, trajectories, *, first_frame, length):
    """Exclusive generated evaluation fixtures; retain originals/failures."""
    root.mkdir(parents=True, exist_ok=False)
    gt_root = root / "GT" / sequence
    (gt_root / "gt").mkdir(parents=True)
    with (gt_root / "gt/gt.txt").open("x") as handle:
        for line in gt_file.read_text().splitlines():
            fields = line.split(",")
            if fields[0].strip() and first_frame + 1 <= int(fields[0]) <= first_frame + length:
                fields[0] = str(int(fields[0]) - first_frame)
                handle.write(",".join(fields) + "\n")
    with (gt_root / "seqinfo.ini").open("x") as handle:
        handle.write(f'[Sequence]\nname={sequence}\nimDir=img1\nframeRate={event["fps"]}\nseqLength={length}\nimWidth={event["width"]}\nimHeight={event["height"]}\nimExt=.jpg\n')
    for name, rows in trajectories.items():
        assert len(rows) == length and [r["frame"] for r in rows] == list(range(first_frame, first_frame + length))
        path = root / "trackers" / name / "data" / (sequence + ".txt")
        path.parent.mkdir(parents=True)
        with path.open("x") as handle:
            handle.write(trajectory_text([{**r, "frame": r["frame"] - first_frame} for r in rows]))
    seqmap = root / "seqmap.txt"
    with seqmap.open("x") as handle:
        handle.write("name\n" + sequence + "\n")
    pinned = HISTORY / "third_party/MOTIP/TrackEval"
    actual = subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip()
    assert actual == PINNED_COMMIT
    evaluation = root / "evaluation"; evaluation.mkdir()
    command = _trackeval_command(root / "trackers", evaluation, list(trajectories), seqmap, gt_split="train", gt_folder=root / "GT")
    command[2] = str(pinned / "scripts/run_mot_challenge.py")
    log = evaluation / "trackeval.log"
    began = time.monotonic()
    with log.open("x") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, check=False)
    receipt = {"command": command, "actual_exit_code": completed.returncode,
               "log_path": str(log), "log_sha256": sha256(log), "seconds": time.monotonic() - began,
               "TrackEval_commit": actual, "first_original_frame": first_frame, "original_length": length}
    write_json(PREFIX + "/invocations/" + sequence + "__" + root.name + ".json", receipt)
    if completed.returncode:
        raise RuntimeError("Preserve failed actual prefix-continuity CLI")
    parsed = {name: trackeval_summary(parse_trackeval(evaluation, name, [sequence])) for name in trajectories}
    metrics = {name: {k: row[k] for k in METRIC_NAMES} for name, row in parsed.items()}
    assert all(all(v is not None for v in row.values()) for row in metrics.values())
    receipt["generated_artifacts"] = [{"path": str(path), "sha256": sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file()]
    return metrics, receipt


def compare(api, actual_cli, tolerance):
    error = max(abs(api[n][k] - actual_cli[n][k]) for n in api for k in METRIC_NAMES)
    if error > tolerance:
        raise AssertionError("Actual pinned API/CLI metric disagreement")
    return error


def smoke():
    p = checked(); storage(16 << 20)
    root = BASE / "synthetic_method_validation"; root.mkdir(parents=True, exist_ok=False)
    sequence = "SYNTHETIC_METRIC_SCOPE_NOT_DANCETRACK"
    gt = root / "original_GT" / sequence / "gt/gt.txt"; gt.parent.mkdir(parents=True)
    with gt.open("x") as handle:
        for f in range(1, 6):
            for identity, x in ((1, 0), (2, 100)):
                handle.write(f"{f},{identity},{x},0,10,10,1,1,1\n")
    source = [{"frame": f, "state_before": str(f), "target_public_id": 1,
               "outputs": [{"public_id": identity, "candidate_uid": f"{f}:{identity}",
                            "box_xyxy": [x, 0, x + 10, 10], "confidence": 1.}
                           for identity, x in ((1, 0), (2, 100))]} for f in range(5)]
    arm = [{**r, "outputs": [{**o, "public_id": 3 - o["public_id"]} for o in r["outputs"]]} for r in source[3:]]
    prefix = {"KEEP": source, "ID_SWAP": assemble_prefix(source, arm, frame=3, length=2)}
    event = {"fps": 25, "width": 200, "height": 100}
    actual_prefix, prefix_receipt = cli(root / "prefix", sequence, gt, event, prefix, first_frame=0, length=5)
    ev = PinnedEventEvaluator(HISTORY / "third_party/MOTIP/TrackEval", gt_root=root / "prefix/GT",
        tracker_root=root / "prefix/trackers", reference_tracker="KEEP", sequence=sequence, frames=5)
    api_prefix = {n: ev.evaluate(rows, first_frame=0, length=5) for n, rows in prefix.items()}
    future = {n: rows[3:] for n, rows in prefix.items()}
    actual_future, future_receipt = cli(root / "future", sequence, gt, event, future, first_frame=3, length=2)
    api_future = {n: ev.evaluate(rows, first_frame=3, length=2) for n, rows in future.items()}
    errors = [compare(api_prefix, actual_prefix, p["numeric_API_CLI_absolute_tolerance"]),
              compare(api_future, actual_future, p["numeric_API_CLI_absolute_tolerance"])]
    assert actual_future["ID_SWAP"]["HOTA"] == actual_future["ID_SWAP"]["AssA"] == actual_future["ID_SWAP"]["IDF1"] == 1.
    assert actual_prefix["ID_SWAP"]["HOTA"] < 1. and actual_prefix["ID_SWAP"]["AssA"] < 1.
    assert actual_prefix["ID_SWAP"]["IDSW"] == 2 and actual_future["ID_SWAP"]["IDSW"] == 0
    write_json("tests/PREFIX_METRIC_CONTINUITY_SMOKE_V1.json", {
        "stage": "N72R21R2", "goal": GOAL, "status": "COMPLETE_ACTUAL_PINNED_API_CLI_SYNTHETIC_METHOD_VALIDATION",
        "protocol_sha256": sha256(PROTOCOL), "future": actual_future, "prefix": actual_prefix,
        "actual_CLI_receipts": [prefix_receipt, future_receipt], "API_CLI_max_error": max(errors),
        "not_scientific_dataset_or_success_or_training": True, "next_stage_authorized": False})
    print({"actual_method_validation_API_CLI_max_error": max(errors),
           "future_swap": actual_future["ID_SWAP"], "prefix_swap": actual_prefix["ID_SWAP"]}, flush=True)


def run_sequence(sequence, p):
    choice = p["choices"][sequence]
    init_path = OUT / "data/initialization" / (sequence + ".json")
    plan_path = OUT / "events/counterfactual_plans" / (sequence + ".json")
    assert sha256(init_path) == choice["initialization_sha256"] and sha256(plan_path) == choice["plan_sha256"]
    init, plan = read_json(init_path), read_json(plan_path)
    assert all(choice[k] == v for k, v in structural_late_selection(init["inputs"], plan["episodes"]).items())
    cache = {}
    audit, _, _, seals, grouped, contexts = runtime_inputs(sequence, cache)
    result = {"stage": "N72R21R2", "goal": GOAL, "sequence": sequence,
        "role": "FIT" if sequence in preregistration()["split"]["fit"] else "INNER",
        "protocol_sha256": sha256(PROTOCOL), "choice": choice,
        "status": choice["status"], "arms": [], "not_full_adaptive_policy_or_independent_roots": True}
    if choice["selection"] is None:
        result["verified_original_runtime_source_refs"] = list(cache.values())
        return result
    selected = choice["selection"]; uid, frame = selected["episode_uid"], selected["frame"]
    event_uid = uid + "__f" + str(frame)
    seal_path, seal = seals[event_uid]
    records = {r["branch"]: r for r in grouped[event_uid]}
    event = next(e for e in init["inputs"] if e["episode_uid"] == uid)
    source_path = OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (uid + ".json")
    source_seal = read_json(source_path); reference(source_path, cache=cache)
    trace = artifact(source_seal, "trace"); ref = next(a for a in source_seal["artifacts"] if a["kind"] == "trace")
    reference(trace, ref["sha256"], cache)
    source = read_zstd_jsonl(trace)
    assert len(source) == event["frames"]
    assert all(not r["runtime_gt_read"] and not r["runtime_future_gt_used"] and r["extra_clicks"] == 0 for r in source)
    context = contexts[records["KEEP"]["runtime_features"]["current_context_path"]]
    assert context["reconstructed_prefix_sha256"] == ref["sha256"]
    arms = {a["branch"]: read_zstd_jsonl(Path(a["path"])) for a in seal["artifacts"]}
    assert all(all(a[k] == b[k] for k in ("outputs", "target_uid", "state_before", "state_after"))
               for a, b in zip(arms["KEEP"], source[frame:frame+101], strict=True))
    prefixes = {name: assemble_prefix(source, arm, frame=frame, length=101) for name, arm in arms.items()}
    for arm in arms.values():
        if "full_tracker_state_before_sha256" in arm[0]:
            assert arm[0]["full_tracker_state_before_sha256"] == context["tensor_inclusive_prestate_sha256"]
    # Actual raw GT opens only AFTER whole original runtime hashes plus prefix
    # and every selected global arm have been checked. No label chooses arms.
    reference(TRAIN / sequence / "gt/gt.txt", audit["GT_sha256"], cache)
    ev = evaluator(sequence)
    api_window = {n: ev.evaluate(rows[1:], first_frame=frame+1, length=100) for n, rows in arms.items()}
    api_current = {n: ev.evaluate(rows, first_frame=frame, length=101) for n, rows in arms.items()}
    api_prefix = {n: ev.evaluate(rows, first_frame=0, length=frame+101) for n, rows in prefixes.items()}
    actual_window, window_receipt = cli(BASE / sequence / "future_H100", sequence, TRAIN / sequence / "gt/gt.txt", event,
        {n: rows[1:] for n, rows in arms.items()}, first_frame=frame+1, length=100)
    actual_prefix, prefix_receipt = cli(BASE / sequence / "prefix_through_H100", sequence, TRAIN / sequence / "gt/gt.txt", event,
        prefixes, first_frame=0, length=frame+101)
    errors = [compare(api_window, actual_window, p["numeric_API_CLI_absolute_tolerance"]),
              compare(api_prefix, actual_prefix, p["numeric_API_CLI_absolute_tolerance"])]
    counter = Counter()
    for a in seal["artifacts"]:
        name = a["branch"]; labels = records[name]["offline_supervision_labels"]
        assert labels["future"]["H100"]["complete"]
        deltas = {domain: {k: values[name][k] - values["KEEP"][k] for k in METRIC_NAMES}
                  for domain, values in (("future_H100", actual_window), ("current_and_future_H100", api_current),
                                         ("prefix_through_H100", actual_prefix))}
        counts = paired_sign_counts(deltas["future_H100"], deltas["prefix_through_H100"])
        counter.update(counts)
        result["arms"].append({"branch": name, "raw_branch_sha256": a["sha256"],
            "original_action": a["current_action"], "duplicate_configuration_of": labels["same_executed_action_configuration_as"],
            "effective_action_frames_not_independent_roots": labels["effective_direct_action_frames"],
            "safe_positive_H100_components": bool(labels["future"]["H100"]["benefit_label"] and not any(labels["current_t"][k] for k in ("N10", "non_target_damage", "verified_OTHER_writes", "UNKNOWN_writes"))),
            "actual_metrics": {"future_H100": actual_window[name], "current_and_future_H100_API": api_current[name],
                               "prefix_through_H100": actual_prefix[name]},
            "paired_deltas_vs_own_KEEP": deltas, "descriptive_sign_counts": counts})
    result.update(status="COMPLETE_ACTUAL_LATE_SAME_PRESTATE_PREFIX_AND_FUTURE_API_CLI_DIAGNOSTIC",
        selected_CF_seal_sha256=sha256(seal_path), prefix_source_sha256=ref["sha256"],
        source_kind="ORIGINAL_ACTUAL_C0_SHADOW_P0_NOT_CURRENT_MAIN_CONTROLLER",
        selected_current_full_tensor_before_available=all("full_tracker_state_before_sha256" in a[0] for a in arms.values()),
        legacy_semantic_prefix_join_not_upgraded_to_new_full_tensor_replay=True,
        verified_original_runtime_source_refs=list(cache.values()), actual_CLI_receipts=[window_receipt, prefix_receipt],
        API_CLI_max_error=max(errors), correlated_arm_sign_counts=dict(counter),
        GT_sha256=sha256(TRAIN / sequence / "gt/gt.txt"), current_frame_boundary_API_only=True)
    return result


def run():
    p = checked(); storage(1 << 30)
    smoke_proof = read_json(OUT / "tests/PREFIX_METRIC_CONTINUITY_SMOKE_V1.json")
    assert smoke_proof["protocol_sha256"] == sha256(PROTOCOL) and smoke_proof["API_CLI_max_error"] <= p["numeric_API_CLI_absolute_tolerance"]
    for receipt in smoke_proof["actual_CLI_receipts"]:
        assert receipt["actual_exit_code"] == 0
        assert all(sha256(a["path"]) == a["sha256"] for a in receipt["generated_artifacts"])
    marker = PREFIX + "/driver.json"
    state = {"stage": "N72R21R2", "goal": GOAL, "pid": os.getpid(), "status": "ACTIVE_ACTUAL_FROZEN_ALL24_PREFIX_METRIC_DIAGNOSTIC",
             "protocol_sha256": sha256(PROTOCOL), "completed": [], "active": None, "scientific_success": None}
    write_json(marker, state)
    rows = []
    q = preregistration()
    for sequence in q["split"]["fit"] + q["split"]["inner"]:
        state["active"] = {"sequence": sequence, "pid": os.getpid()}; write_json(marker, state, mutable=True)
        storage(64 << 20)
        result = run_sequence(sequence, p)
        path = write_json(PREFIX + "/sequences/" + sequence + ".json", result)
        rows.append({"sequence": sequence, "path": str(path), "sha256": sha256(path)})
        state["completed"].append(sequence); state["active"] = None; write_json(marker, state, mutable=True)
        print({"actual_prefix_diagnostic_sequence": sequence, "status": result["status"],
               "arms": len(result["arms"]), "API_CLI_error": result.get("API_CLI_max_error")}, flush=True)
    counts = {}
    for role in ("FIT", "INNER"):
        counter, effective = Counter(), Counter()
        for r in rows:
            seq = read_json(r["path"])
            if seq["role"] != role:
                continue
            counter["registered_videos"] += 1
            counter["actual_nonempty_selected_videos"] += bool(seq["arms"])
            counter["failed_initialization_slots_retained"] += seq["choice"]["failed_initialization_slots_retained"]
            for arm in seq["arms"]:
                counter.update(arm["descriptive_sign_counts"])
                if arm["duplicate_configuration_of"] is None and arm["effective_action_frames_not_independent_roots"]:
                    effective.update(arm["descriptive_sign_counts"])
        counts[role] = {"all_arms_not_roots": dict(counter), "distinct_effective_one_shot_arms_not_roots": dict(effective)}
    write_json("events/PREFIX_CONTINUITY_DIAGNOSTIC_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "status": "COMPLETE_ACTUAL_ALL24_STRUCTURAL_LATE_PREFIX_METRIC_SCOPE_DIAGNOSTIC",
        "protocol_sha256": sha256(PROTOCOL), "actual_sequence_reports": rows, "role_counts": counts,
        "POSTHOC_DIAGNOSTIC_NOT_POLICY_SELECTION": True, "no_L5_model_gate_or_policy_change": True,
        "not_full_adaptive_policy_or_independent_roots_or_final_scientific_decision": True, "next_stage_authorized": False})
    state.update(status="COMPLETE_ACTUAL_ALL24_DIAGNOSTIC_NOT_SCIENTIFIC_CLOSURE", active=None)
    write_json(marker, state, mutable=True)
    append_log("ACTUAL_LATE_PREFIX_METRIC_DIAGNOSTIC_COMPLETE", sequences=24, role_counts=counts)
    print({"actual_all24_prefix_metric_diagnostic": counts}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "smoke", "run"))
    {"freeze": freeze, "smoke": smoke, "run": run}[parser.parse_args().action]()
