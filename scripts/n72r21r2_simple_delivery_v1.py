"""Reverify all24 fixed full-MOT controls; no new policy or runtime replay."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import torch
from scripts.n72r21r2_common import (ROOT, OUT, ASSETS, TRAIN, HISTORY, GOAL, read_json,
    write_json, sha256, storage, preregistration, development_sequence, append_log)
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r21r2_simple import SOURCE, METRICS
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import parse_trackeval, trackeval_summary
from sam3_intermot.evaluation.simple_policy_delivery import POLICIES, onset_counts, complete_role
from sam3_intermot.evaluation.safe_intervention_events import assignment_map
from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome
from sam3_intermot.evaluation.event_causal_labels import label_actual_branch

PREFIX = "simple/delivery_audit_v1"
PROTOCOL = OUT / "simple/SIMPLE_DELIVERY_AUDIT_PROTOCOL_V1.json"
CODE = tuple(dict.fromkeys((*SOURCE, "scripts/n72r21r2_simple_delivery_v1.py", "sam3_intermot/evaluation/simple_policy_delivery.py",
        "scripts/n72r21r2_simple_onset_audit.py", "scripts/n72r20r3r2r3_pipeline.py",
        "sam3_intermot/evaluation/development_closure.py", "sam3_intermot/evaluation/whole_video_density.py",
        "sam3_intermot/evaluation/event_delivery_evidence.py", "sam3_intermot/evaluation/event_causal_labels.py")))


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("Preserve the frozen delivery audit")
    p = preregistration(); storage(64 << 20)
    assert list(POLICIES.values()) == read_json(OUT / "simple/SIMPLE_PROTOCOL_V1.json")["cases"]
    assert all(not (OUT / "simple" / (name + ".json")).exists() for name in POLICIES)
    write_json("simple/SIMPLE_DELIVERY_AUDIT_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
        "split": {k: p["split"][k] for k in ("fit", "inner")}, "gates_unchanged": p["gates"],
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "original_protocol_sha256": {name: sha256(OUT / name) for name in
            ("simple/SIMPLE_PROTOCOL_V1.json", "simple/ONSET_AUDIT_PROTOCOL_V1.json")},
        "named_policy_mapping": POLICIES,
        "DELAYED_CONFIRMATION": "Exactly existing P6: three consecutive causal proposals and P5 protections; not P8/P9, a new delayed action, retrospective smoothing, or a new rollout",
        "audit": "Entire original sequence runtime/artifact/code/CLI log hashes BEFORE new raw GT parse. Recompute postclick target outcomes for all actual policies and every own-prefix one-shot label from its actual branches; require same-tensor prestate and recorded current output/state. Re-read actual pinned CSVs. Zero-action semantic output/trajectory C0 AA is not baseline tensor proof.",
        "statistics": "All original FIT16/INNER8, all valid clicks and all failed slots. Fixed policies have no optimization seeds. Equal videos, clicks inside video, 2000 paired-video bootstrap730104. Nonoverlap/action/frame counts not independent causal roots.",
        "new_runtime_rollouts_optimizer_steps_thresholds": 0,
        "CPU_threads": 1, "storage_floor_GiB": 60,
        "CONFIRM_VAL_TEST_SOT_allowed": False, "next_stage_authorized": False})


def audit_sequence(sequence):
    development_sequence(sequence)
    refs = {}
    @lru_cache(maxsize=None)
    def hashed(path):
        path = Path(path).resolve()
        if not any(path.is_relative_to(root.resolve()) for root in (ROOT, ASSETS, TRAIN)):
            raise ValueError("Audit reference escaped existing stage data")
        refs[str(path)] = sha256(path)
        return refs[str(path)]
    def checked(relative):
        path = OUT / relative; hashed(path); return read_json(path)
    def artifacts(seal):
        for a in seal["artifacts"]:
            assert hashed(a["path"]) == a["sha256"]
    init_path = OUT / "data/initialization" / (sequence + ".json")
    init = checked(init_path); valid = {e["episode_uid"]: e for e in init["inputs"] if not e["initialization_failure"]}
    assert hashed(init["anchor_path"]) == init["anchor_sha256"]
    integrity_path = OUT / "data/candidate_integrity" / (sequence + ".json")
    integrity = checked(integrity_path)
    assert hashed(integrity["index_path"]) == integrity["index_sha256"] == init["candidate_index_sha256"]
    baseline = checked("mot/baseline_results/" + sequence + ".json")
    result = checked("simple/results/" + sequence + ".json")
    assert baseline["density_integrity_sha256"] == hashed(integrity_path)
    assert result["protocol_sha256"] == sha256(OUT / "simple/SIMPLE_PROTOCOL_V1.json")
    assert result["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_simple.py")
    runtime = checked("simple/runtime_sequences/" + sequence + ".json")
    assert runtime["GT_file_guard_actual"] and runtime["source_freeze"] == {name: sha256(ROOT / name) for name in SOURCE}
    onsets = checked("simple/onset_runtime/" + sequence + ".json")
    labels = checked("simple/onset_audit/" + sequence + ".json")
    assert labels["runtime_seal_sha256"] == hashed(OUT / "simple/onset_runtime" / (sequence + ".json"))
    assert onsets["actual_GT_file_guard"] and onsets["protocol_sha256"] == sha256(OUT / "simple/ONSET_AUDIT_PROTOCOL_V1.json")
    assert labels["source_freeze"] == onsets["source_freeze"] == {n: sha256(ROOT / n) for n in onsets["source_freeze"]}
    seals = {}
    for case in POLICIES.values():
        folder, original = ("mot/baseline_runtime", "CLICK_C0" if case == "P0_KEEP" else "ACIB_SHADOW_P0") if case in ("P0_KEEP", "P1_SHADOW") else ("simple/runtime", case)
        for uid, event in valid.items():
            seal = checked(folder + "/" + original + "/" + uid + ".json"); artifacts(seal)
            assert seal["event"] == event and seal["candidate_index_sha256"] == integrity["index_sha256"]
            assert seal["all_original_frames"] == event["frames"] and seal["one_click_only"]
            if case in ("P0_KEEP", "P1_SHADOW"):
                assert not seal["GT_runtime_read"] and seal["full_global_unique_ownership"]
                assert seal["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_baseline.py")
            else:
                assert seal["source_freeze"] == runtime["source_freeze"] and seal["actual_forbidden_GT_file_guard"] and seal["own_policy_future_state_only"]
            seals[case, uid] = seal
    keyed_runtime = {(e["case"], e["episode_uid"]): e for e in onsets["entries"]}
    keyed_labels = {(e["case"], e["episode_uid"]): e for e in labels["entries"]}
    expected = {(case, uid) for case in list(POLICIES.values())[2:] for uid in valid}
    assert set(keyed_runtime) == set(keyed_labels) == expected
    assert len(keyed_runtime) == len(onsets["entries"]) and len(keyed_labels) == len(labels["entries"])
    for key, item in keyed_runtime.items():
        assert item["event"] == valid[key[1]]
        assert item["trace_sha256"] == hashed(item["trace_path"])
        assert item["source_runtime_seal_sha256"] == hashed(OUT / "simple/runtime" / key[0] / (key[1] + ".json"))
        assert set(item["all_effective_decision_frames"]) == {b["frame"] for b in item["branches"]}
        assert len(item["branches"]) == len(keyed_labels[key]["one_shot_diagnostics"])
        for branch in item["branches"]:
            assert set(branch["artifacts"]) == {"OWN_KEEP", "ACTUAL_CURRENT_THEN_KEEP"}
            for a in branch["artifacts"].values(): assert hashed(a["path"]) == a["sha256"]
        if item["branches"]: assert item["tensor_inclusive_own_prefix_and_action_AA"]
    for receipt, saved in ((baseline["invocation"], baseline["actual_nine_metrics"]),
                           (result["receipt"], result["all_nine_metrics"])):
        if valid:
            assert receipt["returncode"] == 0 and hashed(receipt["log_path"]) == receipt["log_sha256"]
            command = receipt["command"]
            evaluation = Path(command[command.index("--OUTPUT_FOLDER") + 1])
            for tracker, values in saved.items():
                for pattern in ("*.csv", "*_summary.txt"):
                    for path in (evaluation / tracker).rglob(pattern): hashed(path)
                parsed = trackeval_summary(parse_trackeval(evaluation, tracker, [sequence]))
                assert all(parsed[k] == values[k] for k in METRICS)
        else:
            assert receipt["status"] == "NOT_RUN_NO_VALID_INITIALIZATION" and saved == {}
    assert set(result["per_episode_raw_target_components"]) == {case + "/" + uid for case, uid in expected}
    assert set(result["all_nine_metrics"]) == {case + "__click" + str(valid[uid]["slot"]) for case, uid in expected}
    assert set(baseline["actual_nine_metrics"]) == {case + "__click" + str(e["slot"]) for case in ("CLICK_C0", "ACIB_SHADOW_P0") for e in valid.values()}
    # Every full-policy and onset artifact is verified before new raw GT opens.
    frames, index_sha = checked_frames(sequence); assert index_sha == integrity["index_sha256"]
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    truth = checked("data/initialization_truth/" + sequence + ".json")
    assert hashed(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    targets = {e["episode_uid"]: e["target_gt_identity"] for e in truth["labels"]}
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {int(p["frame"]): strict_candidate_matching(rows, gt.get(int(p["frame"]), [])) for p, rows in frames}
    components, raw_onsets, aa_frames = {case: {} for case in POLICIES.values()}, 0, 0
    for uid, event in valid.items():
        base = read_zstd_jsonl(artifact(seals["P0_KEEP", uid], "trace"))
        for case in POLICIES.values():
            observed = base if case == "P0_KEEP" else read_zstd_jsonl(artifact(seals[case, uid], "trace"))
            assert len(observed) == len(base) == event["frames"]
            counter, source_counter = Counter(), Counter()
            for f, (a, b) in enumerate(zip(observed, base, strict=True)):
                assert a["frame"] == b["frame"] == f and a["current_candidate_index_sha256"] == index_sha
                am = assignment_map(a)
                assert len(am) == len(a["outputs"]) == len(set(am.values())) and set(am.values()) == set(matched[f])
                if case in ("P0_KEEP", "P1_SHADOW") or seals[case, uid].get("actual_effective_direct_decisions_NOT_independent_events") == 0:
                    assert all(a[k] == b[k] for k in ("outputs", "target_uid", "state_before", "state_after")); aa_frames += 1
                if f <= event["frame"]: continue
                outcome = candidate_identity_outcome(a["target_uid"], matched[f], targets[uid]); keep = candidate_identity_outcome(b["target_uid"], matched[f], targets[uid])
                effective = bool(a.get("authority", {}).get("effective_assignment_change"))
                source_counter.update(target_correct_frames=int(outcome == "TARGET"),
                    N01_frames_vs_full_C0=int(outcome == "TARGET" and keep != "TARGET"),
                    N10_frames_vs_full_C0=int(keep == "TARGET" and outcome != "TARGET"),
                    verified_OTHER_takeover_frames=int(outcome == "VERIFIED_OTHER"),
                    UNKNOWN_frames=int(outcome == "UNKNOWN"), NONE_frames=int(outcome == "NONE"),
                    effective_direct_decisions_NOT_independent_events=int(effective))
                counter.update(TARGET_frames=int(outcome == "TARGET"), C0_TARGET_frames=int(keep == "TARGET"),
                    N01_frames=int(outcome == "TARGET" and keep != "TARGET"), N10_frames=int(keep == "TARGET" and outcome != "TARGET"),
                    VERIFIED_OTHER_frames=int(outcome == "VERIFIED_OTHER"), C0_VERIFIED_OTHER_frames=int(keep == "VERIFIED_OTHER"),
                    UNKNOWN_frames=int(outcome == "UNKNOWN"), NONE_frames=int(outcome == "NONE"),
                    positive_available_frames=int(targets[uid] in matched[f].values()),
                    physically_visible_frames=int(any(g["identity"] == targets[uid] for g in gt.get(f, []))))
            coverage = baseline["coverage"][uid]
            assert counter["C0_TARGET_frames"] == coverage["C0_correct_frames"] and counter["C0_VERIFIED_OTHER_frames"] == coverage["C0_verified_OTHER_takeover_frames"]
            assert counter["positive_available_frames"] == coverage["strict_positive_available_frames"] and counter["physically_visible_frames"] == coverage["visible_frames"]
            if case not in ("P0_KEEP", "P1_SHADOW"):
                assert source_counter == Counter(result["per_episode_raw_target_components"][case + "/" + uid])
                entry, stored = keyed_runtime[case, uid], keyed_labels[case, uid]
                assert source_counter["effective_direct_decisions_NOT_independent_events"] == stored["effective_decisions"]
                origins, cursor = {}, 0
                for branch, diagnostic in zip(entry["branches"], stored["one_shot_diagnostics"], strict=True):
                    start = branch["frame"]; assert diagnostic["frame"] == start and diagnostic["branch_seals"] == branch["artifacts"]
                    while cursor < start:
                        for public, candidate in assignment_map(observed[cursor]).items():
                            identity = matched[cursor][candidate]
                            if identity is not None: origins.setdefault(public, identity)
                        cursor += 1
                    active = {p: i for p, i in origins.items() if p in branch["public_axis"]}
                    own, actual = [read_zstd_jsonl(Path(branch["artifacts"][k]["path"])) for k in ("OWN_KEEP", "ACTUAL_CURRENT_THEN_KEEP")]
                    assert own[0]["full_tracker_state_before_sha256"] == actual[0]["full_tracker_state_before_sha256"] == branch["prestate_sha256"] == observed[start]["full_tracker_state_before_sha256"]
                    assert actual[0]["outputs"] == observed[start]["outputs"] and actual[0]["full_tracker_state_after_sha256"] == observed[start]["full_tracker_state_after_sha256"]
                    assert assignment_map(actual[0]) != assignment_map(own[0])
                    visible = {f: any(g["identity"] == targets[uid] for g in gt.get(f, [])) for f in range(start, start + len(own))}
                    calculated = label_actual_branch(actual, own, matched, targets[uid], active, visible)
                    assert json.loads(json.dumps(calculated)) == diagnostic["labels"], "Independent raw own-prefix labels differ"
                    raw_onsets += 1
                onset_counts([stored])
            if case in ("P0_KEEP", "P1_SHADOW") or not source_counter["effective_direct_decisions_NOT_independent_events"]:
                assert next(a["sha256"] for a in seals[case, uid]["artifacts"] if a["kind"] == "trajectory") == next(a["sha256"] for a in seals["P0_KEEP", uid]["artifacts"] if a["kind"] == "trajectory")
            components[case][uid] = dict(counter)
    cells = {}
    for case in POLICIES.values():
        metrics, deltas = {}, {}
        for uid, event in valid.items():
            actual_name = ("CLICK_C0" if case == "P0_KEEP" else "ACIB_SHADOW_P0" if case == "P1_SHADOW" else case) + "__click" + str(event["slot"])
            values = baseline["actual_nine_metrics"][actual_name] if case in ("P0_KEEP", "P1_SHADOW") else result["all_nine_metrics"][actual_name]
            name = case + "__click" + str(event["slot"]); base = baseline["actual_nine_metrics"]["CLICK_C0__click" + str(event["slot"])]
            metrics[name] = values; deltas[name] = {k: values[k] - base[k] for k in METRICS}
            if case not in ("P0_KEEP", "P1_SHADOW"): assert deltas[name] == result["paired_deltas_vs_same_input_full_C0"][actual_name]
        entries = [keyed_labels[case, uid] for uid in valid] if case not in ("P0_KEEP", "P1_SHADOW") else [
            {"episode_uid": uid, "effective_decisions": 0, "nonvacuity": False, "one_shot_diagnostics": []} for uid in valid]
        counts = onset_counts(entries)
        cells[case] = {"sequence": sequence, "seed": "FIXED", "experiment_uid": case,
            "density": {"whole_video_density": integrity["whole_video_density"]}, "valid_initializations": len(valid),
            "all_nine_metrics": metrics, "paired_deltas_vs_C0": deltas, "per_episode_target_components": components[case],
            "own_onset_summary": counts, "effective_direct_decisions_NOT_independent_onsets": counts["effective_decisions"],
            "own_prefix_one_shot_labels": [{"episode_uid": e["episode_uid"], "onsets": e["one_shot_diagnostics"]} for e in entries]}
    report = {"stage": "N72R21R2", "goal": GOAL, "sequence": sequence, "protocol_sha256": sha256(PROTOCOL),
        "status": "COMPLETE_ORIGINAL_ALL10_FIXED_POLICIES_RAW_AUDIT_NOT_FULL_TASK_CLOSURE",
        "cells": cells, "source_sha256": refs, "raw_own_prefix_one_shot_labels_exact_recomputed": raw_onsets,
        "semantic_C0_output_AA_frames_NOT_baseline_tensor_proof": aa_frames,
        "source_whole_runtime_verified_before_raw_GT": True,
        "failed_initialization_slots_retained": len(init["inputs"]) - len(valid), "next_stage_authorized": False}
    write_json(PREFIX + "/sequences/" + sequence + ".json", report)
    print({"actual_M4_original_sequence_audited": sequence, "valid_clicks": len(valid), "raw_own_onsets": raw_onsets}, flush=True)


def run():
    torch.set_num_threads(1); storage(64 << 20)
    assert subprocess.check_output(["git", "-C", str(HISTORY / "third_party/MOTIP/TrackEval"), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
    p = read_json(PROTOCOL)
    assert p["goal"] == GOAL and p["frozen"] and p["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["named_policy_mapping"] == POLICIES and p["gates_unchanged"] == preregistration()["gates"]
    assert all(sha256(OUT / name) == digest for name, digest in p["original_protocol_sha256"].items())
    sequences = p["split"]["fit"] + p["split"]["inner"]
    for sequence in sequences:
        path = OUT / PREFIX / "sequences" / (sequence + ".json")
        if not path.exists(): audit_sequence(sequence)
        else: assert read_json(path)["protocol_sha256"] == sha256(PROTOCOL)
    reports = {s: read_json(OUT / PREFIX / "sequences" / (s + ".json")) for s in sequences}
    for report in reports.values():
        assert all(sha256(path) == digest for path, digest in report["source_sha256"].items())
    baselines = {s: read_json(OUT / "mot/baseline_results" / (s + ".json")) for s in sequences}
    initializations = {s: read_json(OUT / "data/initialization" / (s + ".json")) for s in sequences}
    densities = {s: read_json(OUT / "data/candidate_integrity" / (s + ".json"))["whole_video_density"] for s in sequences}
    sources = {str(OUT / PREFIX / "sequences" / (s + ".json")): sha256(OUT / PREFIX / "sequences" / (s + ".json")) for s in sequences}
    for name, case in POLICIES.items():
        roles = {role: complete_role([reports[s]["cells"][case] for s in selected], baselines, initializations,
                                    densities, selected, p["gates_unchanged"]) for role, selected in p["split"].items()}
        write_json("simple/" + name + ".json", {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
            "status": "COMPLETE_ACTUAL_ALL24_ORIGINAL_FIXED_MECHANISM_NOT_FULL_TASK_SCIENTIFIC_CLOSURE",
            "named_mechanism": name, "actual_frozen_case": case, "protocol_sha256": sha256(PROTOCOL),
            "source_sha256": p["source_sha256"], "all24_source_audit_receipts": sources, "roles": roles,
            "fixed_policy_has_no_optimization_seeds": True, "independent_event_precision_recall": None,
            "nonoverlap_is_NOT_independence": True, "full_pipeline_or_head_latency_NOT_inferred_from_offline_audit_time": True,
            "DELAYED_CONFIRMATION_semantics": p["DELAYED_CONFIRMATION"] if name == "DELAYED_CONFIRMATION" else None,
            "CONFIRM_VAL_TEST_SOT_accessed": False, "scientific_goal_complete": False, "next_stage_authorized": False})
    write_json(PREFIX + "/COMPLETE.json", {"stage": "N72R21R2", "goal": GOAL, "protocol_sha256": sha256(PROTOCOL),
        "all24_original_videos": sequences, "named_policy_reports": {name: sha256(OUT / "simple" / (name + ".json")) for name in POLICIES},
        "actual_valid_sole_clicks": sum(len(r["cells"]["P0_KEEP"]["per_episode_target_components"]) for r in reports.values()),
        "failed_initialization_slots_retained": sum(r["failed_initialization_slots_retained"] for r in reports.values()),
        "raw_own_prefix_one_shot_labels_exact_recomputed": sum(r["raw_own_prefix_one_shot_labels_exact_recomputed"] for r in reports.values()),
        "independent_causal_root_counts_NOT_promoted": True, "original_workers_or_policy_code_changed": False,
        "new_optimizer_steps_or_runtime_rollouts": 0, "next_stage_authorized": False})
    append_log("M4_ALL24_ACTUAL_SIMPLE_NAMED_REPORTS_RAW_AUDIT_COMPLETE", reports=len(POLICIES), sequences=24)
    print({"actual_all24_fixed_policy_delivery_complete": True, "named_reports": len(POLICIES), "full_goal_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args(); freeze() if args.action == "freeze" else run()
