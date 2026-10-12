"""Derived P0 native-counter support, not a recorded feature or policy repair.

Requires a complete original STATE point and the all24 actual source-counter
validation. The exact counter is absent from the frozen32 model input axis.
No new training, rollout, threshold, intervention or raw GT read is performed.
"""
import argparse
from pathlib import Path
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, storage
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts import n72r21r2_state_feature_support_v1 as support
from scripts import n72r21r2_train_joint_state as source
from scripts import n72r21r2_native_counter_audit_v1 as recurrence
from scripts.n72r21r2_state_inner_readback_v1 import checked_cell
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.native_streak_reconstruction import observed_native_before_after

SELF = "scripts/n72r21r2_native_deployment_support_v1.py"
PROTOCOL = OUT / "on_policy/NATIVE_DEPLOYMENT_SUPPORT_PROTOCOL_V1.json"
VALIDATION = OUT / recurrence.PREFIX / "ALL24_SOURCE_RECORDED_COUNTER_VALIDATION.json"
PREFIX = "on_policy/native_deployment_support_v1"
CODE = tuple(dict.fromkeys((SELF, *recurrence.CODE, *support.CODE)))


def integer_values(values):
    result = list(values)
    if any(type(v) is not int or v < 0 for v in result):
        raise ValueError("Exact nonnegative observation counts, never binary proxies")
    return result


def fit_support(rows):
    values, events = [], {}
    for row in rows:
        if row["role"] != "FIT":
            raise ValueError("Only original FIT optimizer support")
        value = integer_values([row["observables"]["observed_native_streak"]])[0]
        key = (row["sequence"], row["episode_uid"], row["event_uid"], row["frame"])
        if key in events and events[key] != value:
            raise ValueError("Same source probe cannot have inconsistent counters")
        values.append(value); events[key] = value
    if not values:
        raise ValueError("No fitted-source observations")
    return values, list(events.values())


def distribution(values, reference=None):
    values = integer_values(values)
    result = {"count_correlated": len(values), "quantiles_0_10_25_50_75_90_100": recurrence.quantiles(values)}
    if reference is not None:
        reference = integer_values(reference)
        if not reference:
            raise ValueError("No invented empty reference support")
        lo, hi = min(reference), max(reference)
        result.update({"FIT_min_max": [lo, hi], "below_FIT_min": sum(v < lo for v in values),
                       "above_FIT_max": sum(v > hi for v in values)})
    return result


def freeze():
    assert not list((OUT / PREFIX).glob("*.json"))
    p = read_json(support.PROTOCOL)
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in support.CODE}
    validation = read_json(VALIDATION)
    assert validation["all5588_recorded_source_counter_values_exactly_matched"]
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "design": "RETROSPECTIVE_DERIVED_STATE_COUNTER_SUPPORT_NOT_DEPLOYED_FEATURE_OR_CORRECTION",
        "source_sha256": {n: sha256(ROOT / n) for n in CODE},
        "source_counter_validation_sha256": sha256(VALIDATION), "state_support_protocol_sha256": sha256(support.PROTOCOL),
        "modes": list(support.MODES), "points": list(p["points"]), "seeds": p["seeds"], "split": p["split"],
        "whole_point": "All3 original seeds and INNER8 seals/onsets/actual pinned CSVs and completed original support report before FIT source is opened; retain failed initialization slots.",
        "units": "FIT exact optimizer rows plus source probes deduplicated across branch rows; own one before-frame count per original sample, never multiplied by candidate/branch choices.",
        "deployment": "Immutable full original P0 trace and current native axes, zero effective actions and zero writes; same recurrence validated against all5588 source probe records. DERIVED, not independently recorded deployment counter.",
        "not_in_frozen32_model_axis": True, "not_independent_events_or_joint_OOD_or_failure_cause": True,
        "bank_vector_drift_NOT_measured": True, "MAIN_exact_counter_optimizer_feature_NOT_RECORDED": True,
        "no_raw_GT_read_or_new_fits_rollouts_features_policy_changes": True,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "scientific_goal_complete": False, "next_stage_authorized": False})


def run(mode, point):
    ev = Evidence(); p = ev.json(PROTOCOL); ev.code(p["source_sha256"])
    assert p["frozen"] and p["goal"] == GOAL and mode in p["modes"] and point in p["points"]
    assert ev.digest(VALIDATION) == p["source_counter_validation_sha256"]
    assert ev.digest(support.PROTOCOL) == p["state_support_protocol_sha256"]
    recipe = ev.json(support.PROTOCOL)
    relative = PREFIX + "/" + mode + "__" + point + ".json"
    if (OUT / relative).exists():
        raise FileExistsError("Preserve actual derived audits")
    required = support.required_point_paths(mode, point, recipe)
    missing = [str(path) for path in required if not path.exists()]
    report_path = OUT / support.PREFIX / (mode + "__" + point + ".json")
    if missing or not report_path.exists():
        raise RuntimeError("Complete original point and support replay required before FIT reads")
    original_report = ev.json(report_path)
    assert original_report["mode"] == mode and original_report["point"] == point
    assert original_report["protocol_sha256"] == ev.digest(support.PROTOCOL)
    for seed in p["seeds"]:
        for seq in p["split"]["inner"]:
            checked_cell(mode, seed, point, seq, ev)
    storage(8 << 20)
    fit, _, manifest = source.load_records(mode)
    reference, unique_reference = fit_support(fit)
    axes = {}
    with runtime_file_guard():
        for seq in p["split"]["inner"]:
            frames, _ = checked_frames(seq)
            axes[seq] = {payload["frame"]: {r["candidate_uid"]: (r.get("native_scope"), int(r["native_tid"])) for r in rows}
                         for payload, rows in frames}
    results = {}
    for seed in p["seeds"]:
        uid = "STATE__" + mode + "__" + source.OBJECTIVE + "__seed" + str(seed) + "__" + point
        sampled, all_postclick, by_video = [], [], {}
        failed, valid = 0, 0
        for seq in p["split"]["inner"]:
            seal = ev.json(OUT / support.state.PREFIX / "runtime" / uid / (seq + ".json"))
            failed += seal["initialization_failures_not_replaced"]
            local = []
            for episode in seal["episodes"]:
                assert episode["memory_writes"] == 0 and episode["effective_direct_decisions_NOT_independent_onsets"] == 0
                valid += 1
                ref = next(a for a in episode["artifacts"] if a["kind"] == "trace"); ev.artifact(ref)
                with runtime_file_guard():
                    trace = read_zstd_jsonl(Path(ref["path"]))
                    assert len(trace) == episode["event"]["frames"]
                    before, _ = observed_native_before_after(trace, axes[seq], episode["event"]["frame"])
                    for row in trace:
                        assert not row["authority"]["effective_assignment_change"]
                        if row["frame"] > episode["event"]["frame"]:
                            all_postclick.append(before[row["frame"]])
                        if support.sampled_choices(row, episode["event"]) is not None:
                            local.append(before[row["frame"]])
                assert ev.digest(ref["path"]) == ref["sha256"]
            sampled += local; by_video[seq] = distribution(local, reference)
        assert len(sampled) == original_report["per_seed"][str(seed)]["actual_counts"]["sampled_current_frames_correlated"]
        results[str(seed)] = {"sampled_before_frame_counter": distribution(sampled, reference),
            "all_postclick_before_frame_counter": distribution(all_postclick, reference), "per_video_sampled": by_video,
            "valid_clicks": valid, "failed_initialization_slots_retained": failed,
            "deployment_counter_DERIVED_NOT_directly_recorded": True, "not_active_memory_write_counter": True}
        print({"derived_STATE_native_counter_complete": mode, "point": point, "seed": seed, "sample_frames": len(sampled)}, flush=True)
    for ref in manifest["receipts"]:
        seal_path = OUT / "on_policy/joint_state_v1/supervision" / (ref["sequence"] + ".json")
        assert ev.digest(seal_path) == ref["source_receipt_sha256"]; ev.artifact(ev.json(seal_path)["artifact"])
    write_json(relative, {"stage": "N72R21R2", "goal": GOAL, "mode": mode, "point": point,
        "status": "COMPLETE_DERIVED_COUNTER_SUPPORT_NOT_FULL_M7_OR_POLICY_REPAIR", "protocol_sha256": ev.digest(PROTOCOL),
        "FIT_optimizer_row_counter": distribution(reference), "FIT_unique_source_probe_counter_NOT_independent_roots": distribution(unique_reference),
        "per_seed": results, "evidence_sha256": ev.refs, "no_new_models_or_original_source_changes": True,
        "bank_vector_drift_NOT_measured": True, "scientific_goal_complete": False, "next_stage_authorized": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--mode", choices=support.MODES); parser.add_argument("--point")
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run(args.mode, args.point)
