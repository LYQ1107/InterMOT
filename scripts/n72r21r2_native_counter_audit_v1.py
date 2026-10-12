"""Recover derived native observations from sealed no-write P0 trajectories.

First validate the exact frozen recurrence against all directly recorded
controlled-source probe counters. Deployment quantities remain DERIVED,
not retroactively recorded features or independent correction/safety trials.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, preregistration, storage
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.native_streak_reconstruction import observed_native_before_after, controlled_prefix_probe_values


SELF = "scripts/n72r21r2_native_counter_audit_v1.py"
CODE = (SELF, "sam3_intermot/evaluation/native_streak_reconstruction.py",
        "sam3_intermot/association/opportunity_tracker.py", "sam3_intermot/association/opportunity_memory.py",
        "scripts/n72r21r2_baseline.py", "scripts/n72r21r2_joint_state_curriculum.py")
CANONICAL = OUT / "on_policy/ALL24_CONTROLLED_SOURCE_ACTUAL_CANONICAL_READBACK_V1.json"
PROTOCOL = OUT / "on_policy/NATIVE_COUNTER_DERIVED_AUDIT_PROTOCOL_V1.json"
PREFIX = "on_policy/native_counter_derived_audit_v1"


def freeze():
    assert not list((OUT / PREFIX).glob("*.json"))
    split = preregistration()["split"]
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "design": "READ_ONLY_DERIVED_RECURRENCE_AUDIT_NOT_DIRECTLY_RECORDED_M7_FEATURE_OR_MODEL_REPAIR",
        "source_sha256": {n: sha256(ROOT / n) for n in CODE}, "canonical_source_readback_sha256": sha256(CANONICAL),
        "split": {k: split[k] for k in ("fit", "inner")},
        "recurrence": "P0 postclick only; committed UID -> immutable current native_scope/int native_tid key; consecutive same key increments else1; NONE resets; event/preclick counter remains0. P0 write-native_streak remains inactive0, distinct from observed_native_streak.",
        "source_validation": "All24 actual recorded controlled-prefix probe counters, baseline prefix followed only by that SOURCE's own prefix; before-probe values, not after-current or baseline borrowed future.",
        "deployment": "Only complete zero-action/no-write MAIN/STATE original point reports; compare source mode's actually recorded FIT native observables to recurrence-derived own before-frame sampled values. Whole point all3 seeds/INNER8 required; no best seed/video subset.",
        "no_new_features_weights_thresholds_fits_rollouts_or_GT_access": True,
        "bank_vector_drift_NOT_recovered_by_native_counter": True,
        "direct_recorded_deployment_native_streak_claimed": False, "scientific_goal_complete": False,
        "next_association_stage_authorized": False, "CONFIRM_VAL_TEST_SOT_accessed": False})


def quantiles(values):
    return np.quantile(values, [0., .1, .25, .5, .75, .9, 1.]).tolist() if values else None


def run():
    ev = Evidence(); p = ev.json(PROTOCOL); ev.code(p["source_sha256"])
    assert p["frozen"] and p["goal"] == GOAL and p["canonical_source_readback_sha256"] == ev.digest(CANONICAL)
    canonical = ev.json(CANONICAL); assert canonical["actual_verified_sequences"] == 24
    for name, value in canonical["evidence_sha256"].items(): assert ev.digest(name) == value
    storage(32 << 20)
    all_sources, axes_by_video = [], {}
    for sequence in p["split"]["fit"] + p["split"]["inner"]:
        with runtime_file_guard():
            frames, _ = checked_frames(sequence)
            native_axes = {payload["frame"]: {r["candidate_uid"]: (r.get("native_scope"), int(r["native_tid"])) for r in rows} for payload, rows in frames}
        axes_by_video[sequence] = native_axes
        runtime = ev.json(OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json"))
        probes_verified, distinct = 0, set()
        baseline_values = []
        for episode_ref in runtime["episodes"]:
            ev.artifact(episode_ref); seal = ev.json(episode_ref["path"]); ev.artifact(seal["artifact"])
            event = seal["event"]
            baseline_seal = ev.json(OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (event["episode_uid"] + ".json"))
            ref = next(a for a in baseline_seal["artifacts"] if a["kind"] == "trace"); ev.artifact(ref)
            with runtime_file_guard():
                baseline = read_zstd_jsonl(Path(ref["path"])); records = read_zstd_jsonl(Path(seal["artifact"]["path"]))
                before, _ = observed_native_before_after(baseline, native_axes, event["frame"])
                baseline_values += [v for f, v in before.items() if f > event["frame"]]
                for parent in [r for r in records if r["record_type"] == "SOURCE_PREFIX"]:
                    probes = [r for r in records if r["record_type"] == "NESTED_ARM" and r["source_case"] == parent["source_case"] and r["source_frame"] == parent["source_frame"]]
                    derived = controlled_prefix_probe_values(baseline, parent, native_axes, event["frame"], probes)
                    probes_verified += len(probes)
                    distinct.update((event["episode_uid"], parent["source_case"], parent["source_frame"], f) for f in derived)
        row = {"sequence": sequence, "role": "FIT" if sequence in p["split"]["fit"] else "INNER",
            "actually_recorded_source_arm_probe_values_matched": probes_verified,
            "distinct_actual_source_probe_counters_matched_NOT_independent_events": len(distinct),
            "C0_derived_preframe_values": len(baseline_values), "C0_derived_quantiles": quantiles(baseline_values),
            "initialization_failure_not_replaced": True}
        all_sources.append(row)
        print({"actual_source_native_counter_recurrence_validated": sequence, "source_probe_rows": probes_verified}, flush=True)
    assert sum(r["actually_recorded_source_arm_probe_values_matched"] for r in all_sources) == 5588
    write_json(PREFIX + "/ALL24_SOURCE_RECORDED_COUNTER_VALIDATION.json", {"stage": "N72R21R2", "goal": GOAL,
        "protocol_sha256": ev.digest(PROTOCOL), "all24_source_rows": all_sources, "evidence_sha256": ev.refs,
        "all5588_recorded_source_counter_values_exactly_matched": True,
        "deploy_quantities_remain_derived_NOT_directly_recorded": True, "scientific_goal_complete": False})
    print({"all24_recorded_source_native_counter_validation_complete": True, "derived_deployment_audit_requires_complete_point": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    freeze() if parser.parse_args().action == "freeze" else run()
