"""All registered MAIN/state-source own-MOT groups; no partial-grid success."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, TRAIN, GOAL, read_json, write_json, sha256, preregistration, append_log
from sam3_intermot.evaluation.development_closure import summarize_complete_group, development_gates
from sam3_intermot.evaluation.learned_policy_evidence import METRICS, choose_inner_point

PROTOCOL = OUT / "mot/DEVELOPMENT_CLOSURE_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_development_closure.py", "sam3_intermot/evaluation/development_closure.py",
        "sam3_intermot/evaluation/whole_video_density.py", "sam3_intermot/evaluation/learned_policy_evidence.py")


def studies():
    main = read_json(OUT / "mot/MAIN_POLICY_PROTOCOL_V1.json")
    state = read_json(OUT / "on_policy/STATE_POLICY_PROTOCOL_V1.json")
    common = {"seeds": main["seeds"], "split": main["split"], "points": main["points"]}
    assert all(state[k] == common[k] for k in common)
    return [dict(study="MAIN", prefix="mot/main_policy_v1", protocol="mot/MAIN_POLICY_PROTOCOL_V1.json",
                 groups=list(dict.fromkeys(j["family"] + "__" + j["objective"] for j in main["jobs"])),
                 fit_directory="training/event_authority", fit_prefix="MAIN__", **common),
            dict(study="STATE", prefix="mot/state_policy_v1", protocol="on_policy/STATE_POLICY_PROTOCOL_V1.json",
                 groups=[mode + "__" + state["objective"] for mode in state["modes"]],
                 fit_directory="on_policy/state_source_fits_v1", fit_prefix="", **common)]


def freeze():
    p = preregistration()
    assert not list((OUT / "mot/main_policy_v1/results").rglob("*.json"))
    assert not list((OUT / "mot/state_policy_v1/results").rglob("*.json"))
    specs = studies()
    assert sum(len(s["groups"]) for s in specs) == 17
    write_json("mot/DEVELOPMENT_CLOSURE_PROTOCOL_V1.json", {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "gates_unchanged": p["gates"],
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "studies": specs,
        "inherited_protocol_sha256": {s["protocol"]: sha256(OUT / s["protocol"]) for s in specs},
        "full_grid": "Actual all3-seed/all8-INNER results at BOTH original points, rechecked shared original selection, and selected-point all16-FIT results. Failed initializations are retained, not zero metrics; no best successful subset.",
        "G1": "Only logically valid negative bounds: ALL effective decisions upper-bound independent beneficial roots; action-video union upper-bounds support; seed-averaged frame N01>N10 and no observed severe harm. Enough proxy rows never prove independent causal roots. No new per-seed minimum or post-effect gate.",
        "G2": "Original unchanged numerical gates, actual nine full-video metrics and paired video CI separately. Zero C0 IDSW permits no increase; UNKNOWN not verified-other takeover.",
        "source_checks": "Current frozen code/weights/actual gradient changes/fit receipt; runtime and own-onset/source/pinned-log/artifact SHA; exact original density/click census; all raw onset labels and counter consistency; zero-action complete tensor/trajectory C0 equality.",
        "conditional_branches": "This summary never authorizes actual-model on-policy, staged training, M-B or confirmation. Separate whole-task/root/G0-G4 audit remains mandatory.",
        "partial": "Immutable input-hash snapshots retain missing cells and real failures; no metrics averaged from a ready subset.",
        "new_optimization_threshold_or_rollout_choices": 0, "CONFIRM_VAL_TEST_SOT_authorized": False})
    print({"frozen_all17_group_closure": sha256(PROTOCOL)}, flush=True)


def snapshot():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    assert p["gates_unchanged"] == preregistration()["gates"] and p["studies"] == studies()
    refs, groups = {}, {}
    @lru_cache(maxsize=None)
    def checked_sha(path):
        path = Path(path)
        value = sha256(path); refs[str(path)] = value
        return value
    def checked_json(path):
        checked_sha(path); return read_json(path)
    def artifact(a):
        assert checked_sha(a["path"]) == a["sha256"]
    for name, expected in p["inherited_protocol_sha256"].items():
        assert checked_sha(OUT / name) == expected
        old = checked_json(OUT / name)
        assert all(checked_sha(ROOT / source) == digest for source, digest in old["source_sha256"].items())
    def checked_cell(spec, group, seed, point, sequence):
        family, objective = group.split("__", 1)
        uid = spec["study"] + "__" + group + "__seed" + str(seed) + "__" + point
        prefix = OUT / spec["prefix"]
        cell = checked_json(prefix / "results" / uid / (sequence + ".json"))
        runtime_path = prefix / "runtime" / uid / (sequence + ".json")
        onset_path = prefix / "onset_runtime" / uid / (sequence + ".json")
        runtime, onsets = checked_json(runtime_path), checked_json(onset_path)
        own_protocol = checked_json(OUT / spec["protocol"])
        fit_name = spec["fit_prefix"] + (group if spec["study"] == "MAIN" else family) + "__seed" + str(seed) + ".json"
        fit_path = OUT / spec["fit_directory"] / fit_name
        fit = checked_json(fit_path)
        assert cell["experiment_uid"] == runtime["experiment_uid"] == uid
        assert all(cell[k] == expected for k, expected in dict(sequence=sequence, seed=seed, family=family, objective=objective, point=point).items())
        assert cell["protocol_sha256"] == runtime["protocol_sha256"] == onsets["protocol_sha256"] == checked_sha(OUT / spec["protocol"])
        assert cell["runtime_seal_sha256"] == onsets["runtime_seal_sha256"] == checked_sha(runtime_path)
        assert cell["onset_runtime_seal_sha256"] == checked_sha(onset_path)
        assert runtime["source_freeze"] == own_protocol["source_sha256"]
        assert runtime["fit_record_sha256"] == checked_sha(fit_path)
        assert cell["checkpoint_sha256"] == runtime["checkpoint_sha256"] == fit["checkpoint_sha256"] == checked_sha(fit["checkpoint_path"])
        assert fit["nonzero_gradient_steps"] > 0
        changed = fit.get("changed_weight_elements", sum(fit.get("changed_weight_elements_by_tensor", {}).values()))
        assert changed > 0 and fit["seed"] == seed
        fit_sources = fit["source_sha256"] if spec["study"] == "STATE" else fit["source_freeze"]
        assert all(checked_sha(ROOT / name) == digest for name, digest in fit_sources.items())
        if spec["study"] == "STATE":
            assert cell["actual_architecture_family"] == "SMALL_MLP" and not cell["actual_model_generated_on_policy_training"]
            assert fit["mode"] == family and fit["family"] == "SMALL_MLP" and not fit["actual_model_generated_on_policy"]
        else:
            assert fit["family"] == family and fit["objective"] == objective
            assert not fit["pilot_not_main_fit_or_independent_confirmation"]
        assert runtime["actual_GT_file_guard"] and runtime["sole_click_full_global_original_hard_negatives"] and not runtime["runtime_future_GT_read"]
        init_path = OUT / "data/initialization" / (sequence + ".json")
        init = checked_json(init_path)
        assert runtime["initialization_sha256"] == checked_sha(init_path)
        integrity = checked_json(OUT / "data/candidate_integrity" / (sequence + ".json"))
        assert runtime["candidate_index_sha256"] == integrity["index_sha256"] == init["candidate_index_sha256"]
        assert checked_sha(integrity["index_path"]) == integrity["index_sha256"]
        assert runtime["density"]["whole_video_density"] == integrity["whole_video_density"]
        valid = {e["episode_uid"]: e for e in init["inputs"] if not e["initialization_failure"]}
        assert {e["episode_uid"] for e in runtime["episodes"]} == {e["episode_uid"] for e in onsets["entries"]} == set(valid)
        label_rows = {e["episode_uid"]: e["onsets"] for e in cell["own_prefix_one_shot_labels"]}
        for e, onset in zip(runtime["episodes"], onsets["entries"], strict=True):
            assert e["episode_uid"] == onset["episode_uid"] and e["event"] == valid[e["episode_uid"]] and e["memory_writes"] == 0
            assert {r["frame"] for r in label_rows[e["episode_uid"]]} == set(onset["effective_decision_frames"]) == {r["frame"] for r in onset["branches"]}
            for a in e["artifacts"]: artifact(a)
            for b in onset["branches"]:
                assert set(b["artifacts"]) == {"OWN_KEEP", "ACTUAL_CURRENT_THEN_KEEP"} and b["complete_learned_prestate_sha256"]
                for a in b["artifacts"].values(): artifact(a)
            if e["effective_direct_decisions_NOT_independent_onsets"] == 0:
                c0 = checked_json(OUT / "mot/baseline_runtime/CLICK_C0" / (e["episode_uid"] + ".json"))
                old = next(a for a in c0["artifacts"] if a["kind"] == "trajectory")
                new = next(a for a in e["artifacts"] if a["kind"] == "trajectory")
                artifact(old); assert new["sha256"] == old["sha256"]
                assert e["zero_action_prefix_tensor_C0_AA_frames"] == valid[e["episode_uid"]]["frames"]
                metric = cell["all_nine_metrics"][e["tracker_name"]]
                baseline = checked_json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
                old_metrics = baseline["CLICK_C0__click" + str(valid[e["episode_uid"]]["slot"])]
                assert all(metric[k] == old_metrics[k] for k in METRICS)
        receipt = cell["actual_pinned_full_original_video_TrackEval"]
        if valid:
            assert receipt["returncode"] == 0 and checked_sha(receipt["log_path"]) == receipt["log_sha256"]
        else:
            assert receipt["status"] == "NOT_RUN_NO_VALID_INITIALIZATION"
        return cell
    for spec in p["studies"]:
        driver_path = OUT / spec["prefix"] / "driver.json"
        raw = driver_path.read_bytes()
        refs[str(driver_path)] = hashlib.sha256(raw).hexdigest()
        driver = json.loads(raw)
        for group in spec["groups"]:
            key = spec["study"] + "/" + group
            select_path = OUT / spec["prefix"] / "selections" / (group + ".json")
            if not select_path.exists():
                reason = driver.get("group_status", {}).get(group)
                terminal_failure = reason is not None and reason.startswith(("NOT_RUN", "NOT_SELECTED"))
                failures = [r for r in driver.get("failed_retained", []) if (spec["study"] + "__" + group + "__") in r["key"] or r["key"] == group]
                for r in failures:
                    if r.get("log"): checked_sha(r["log"])
                groups[key] = {"status": "NOT_RUN_COMPLETE_MOT_RETAINED_ATTEMPT_FAILURE_REQUIRES_SEPARATE_DIAGNOSIS" if terminal_failure else "PENDING_ACTUAL_COMPLETE_INNER_AND_SHARED_SELECTION",
                    "actual_group_driver_status": reason, "retained_attempt_failures": failures, "scientific_success": None}
                continue
            selection = checked_json(select_path)
            assert selection["protocol_sha256"] == checked_sha(OUT / spec["protocol"])
            point = selection["selected_point"]
            assert point in spec["points"]
            required = [(seed, q, s) for seed in spec["seeds"] for q in spec["points"] for s in spec["split"]["inner"]]
            required += [(seed, point, s) for seed in spec["seeds"] for s in spec["split"]["fit"]]
            missing = [dict(seed=seed, point=q, sequence=s) for seed, q, s in required if not (OUT / spec["prefix"] / "results" /
                (spec["study"] + "__" + group + "__seed" + str(seed) + "__" + q) / (s + ".json")).exists()]
            if missing:
                groups[key] = {"status": "PENDING_ALL_SEEDS_ALL_VIDEOS_NO_READY_SUBSET_SUMMARY", "missing_cells": missing, "scientific_success": None}
                continue
            cells = {(seed, q, s): checked_cell(spec, group, seed, q, s) for seed, q, s in required}
            inner = {q: [cells[seed, q, s] for seed in spec["seeds"] for s in spec["split"]["inner"]] for q in spec["points"]}
            recalculated = choose_inner_point(inner, spec["seeds"], spec["split"]["inner"])
            assert all(selection[k] == value for k, value in recalculated.items())
            groups[key] = {"status": "COMPLETE_ACTUAL_SELECTED_POINT_ALL24_DEVELOPMENT_NOT_WHOLE_STAGE", "selected_point": point, "selection": recalculated, "roles": {}}
            for role, names in spec["split"].items():
                baselines = {s: checked_json(OUT / "mot/baseline_results" / (s + ".json")) for s in names}
                initializations = {s: checked_json(OUT / "data/initialization" / (s + ".json")) for s in names}
                densities = {s: checked_json(OUT / "data/candidate_integrity" / (s + ".json"))["whole_video_density"] for s in names}
                summary = summarize_complete_group([cells[seed, point, s] for seed in spec["seeds"] for s in names], baselines,
                    initializations, densities, seeds=spec["seeds"], sequences=names)
                groups[key]["roles"][role] = {"full_MOT_and_density": summary, "original_development_gates": development_gates(summary, p["gates_unchanged"])}
    bundle = {"protocol_sha256": checked_sha(PROTOCOL), "input_source_sha256": refs, "all17_group_census": groups}
    digest = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    relative = "mot/development_closure_v1/snapshots/" + digest + ".json"
    complete = sum(g["status"].startswith("COMPLETE_ACTUAL") for g in groups.values())
    write_json(relative, {"stage": "N72R21R2", "goal": GOAL, **bundle,
        "status": "ALL17_GROUP_METRICS_COMPLETE_SEPARATE_FULL_TASK_CLOSURE_REQUIRED" if complete == 17 else "PARTIAL_NO_SUBSET_AS_COMPLETE_RESEARCH",
        "actual_complete_groups": complete, "registered_groups": 17, "scientific_success": None,
        "whole_goal_complete": False, "CONFIRM_VAL_TEST_SOT_accessed": False, "next_association_stage_authorized": False})
    write_json("mot/development_closure_v1/latest.json", {"path": str(OUT / relative), "sha256": sha256(OUT / relative), "actual_complete_groups": complete, "registered_groups": 17}, mutable=True)
    append_log("ACTUAL_ALL17_GROUP_DEVELOPMENT_CLOSURE_SNAPSHOT", complete=complete, pending=17-complete, snapshot=digest)
    print({"actual_complete_groups": complete, "registered_groups": 17, "whole_task_complete": False, "snapshot": digest}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "snapshot"))
    args = parser.parse_args(); freeze() if args.action == "freeze" else snapshot()
