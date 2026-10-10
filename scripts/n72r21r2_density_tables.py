"""M10 actual frozen-density / full-video baseline and simple-policy tables.

No new online rollout, GT read, policy selection or confirmation exposure.
Only original candidate-count strata frozen before effects are reused.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, GOAL, preregistration, read_json, write_json, sha256, append_log
from scripts.n72r21r2_events import artifact
from sam3_intermot.evaluation.learned_policy_evidence import METRICS, density_from_counts
from sam3_intermot.evaluation.whole_video_density import full_video_summary

CODE = ("scripts/n72r21r2_density_tables.py", "sam3_intermot/evaluation/whole_video_density.py")


def checked_density(sequence):
    path = OUT / "data/candidate_integrity" / (sequence + ".json")
    seal = read_json(path)
    assert seal["sequence"] == sequence and seal["density_frozen_before_any_policy_effects"]
    assert seal["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    counts = seal["valid_geometry_count_per_frame"]
    calculated = density_from_counts(counts)
    assert calculated["whole_video_density"] == seal["whole_video_density"]
    assert len(counts) == seal["frames"]
    assert {k: n for k, n in calculated["counts"].items() if n} == seal["density_frame_counts"]
    index = read_json(seal["index_path"])
    assert sha256(seal["index_path"]) == seal["index_sha256"]
    for kind in ("metadata", "embeddings"):
        assert sha256(index[kind]) == seal[kind + "_sha256"]
    return {"sequence": sequence, **calculated, "frozen_integrity_path": str(path),
        "frozen_integrity_sha256": sha256(path), "candidate_index_sha256": seal["index_sha256"],
        "candidate_count_quantiles": np.quantile(counts, [.1, .25, .5, .75, .9]).tolist(),
        "candidate_density_NOT_GT_people_count": True,
        "group_rule_was_frozen_before_policy_effects_not_chosen_by_these_tables": True}


def run():
    p = preregistration()
    cases = read_json(OUT / "simple/SIMPLE_PROTOCOL_V1.json")["cases"][2:]
    sequences = p["split"]["fit"] + p["split"]["inner"]
    density = {s: checked_density(s) for s in sequences}
    rows, refs = {"CLICK_C0": []}, []
    for case in cases:
        rows[case] = []
    for s in sequences:
        baseline_path = OUT / "mot/baseline_results" / (s + ".json")
        simple_path = OUT / "simple/results" / (s + ".json")
        b, simple = read_json(baseline_path), read_json(simple_path)
        init_path = OUT / "data/initialization" / (s + ".json")
        init = read_json(init_path)
        assert b["candidate_index_sha256"] == density[s]["candidate_index_sha256"]
        assert b["density_integrity_sha256"] == density[s]["frozen_integrity_sha256"]
        events = {e["slot"]: e for e in init["inputs"] if not e["initialization_failure"]}
        expected = {case + "__click" + str(slot) for case in cases for slot in events}
        assert set(simple["all_nine_metrics"]) == expected
        assert simple["protocol_sha256"] == sha256(OUT / "simple/SIMPLE_PROTOCOL_V1.json")
        assert sha256(ROOT / "scripts/n72r21r2_simple.py") == simple["source_sha256"]
        if events:
            assert simple["receipt"]["returncode"] == 0 and b["invocation"]["returncode"] == 0
            for r in (simple["receipt"], b["invocation"]):
                assert sha256(r["log_path"]) == r["log_sha256"]
        refs.extend({"path": str(path), "sha256": sha256(path)} for path in (baseline_path, simple_path, init_path))
        for slot, event in events.items():
            name = "CLICK_C0__click" + str(slot)
            metrics = {k: b["actual_nine_metrics"][name][k] for k in METRICS}
            coverage = b["coverage"][event["episode_uid"]]
            exposure = {"positive_available_frames": coverage["strict_positive_available_frames"],
                        "physically_visible_frames": coverage["visible_frames"]}
            rows["CLICK_C0"].append({"sequence": s, "seed": "FIXED", "episode_uid": event["episode_uid"],
                "density": density[s]["whole_video_density"], "metrics": metrics,
                "deltas": {k: 0. for k in METRICS}, "counts": {**exposure,
                    "target_correct_frames": coverage["C0_correct_frames"],
                    "verified_OTHER_takeover_frames": coverage["C0_verified_OTHER_takeover_frames"],
                    "UNKNOWN_frames": coverage["C0_UNKNOWN_frames"],
                    "N01_frames_vs_full_C0": 0, "N10_frames_vs_full_C0": 0,
                    "effective_direct_decisions_NOT_independent_events": 0}})
            base_seal = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
            for case in cases:
                name = case + "__click" + str(slot)
                values = {k: simple["all_nine_metrics"][name][k] for k in METRICS}
                deltas = {k: values[k] - metrics[k] for k in METRICS}
                assert deltas == simple["paired_deltas_vs_same_input_full_C0"][name]
                components = simple["per_episode_raw_target_components"][case + "/" + event["episode_uid"]]
                if not components["effective_direct_decisions_NOT_independent_events"]:
                    policy_seal = read_json(OUT / "simple/runtime" / case / (event["episode_uid"] + ".json"))
                    assert sha256(artifact(policy_seal, "trajectory")) == sha256(artifact(base_seal, "trajectory"))
                rows[case].append({"sequence": s, "seed": "FIXED", "episode_uid": event["episode_uid"],
                    "density": density[s]["whole_video_density"], "metrics": values, "deltas": deltas, "counts": {**components, **exposure}})
    table = {role: {case: full_video_summary([r for r in examples if r["sequence"] in p["split"][role]], p["split"][role])
                   for case, examples in rows.items()} for role in ("fit", "inner")}
    census = [{"sequence": s, "role": role, "status": "ORIGINAL_FROZEN_COUNT_STRATUM_REVERIFIED", **density[s]}
              if s in density else {"sequence": s, "role": role,
                  "status": "UNOPENED_CONFIRMATION_NO_CANDIDATE_EFFECT_TABLE" if role == "confirmation" else "HISTORICAL_NOT_FRESH_DENSITY_POPULATION"}
              for role, names in p["split"].items() for s in names]
    write_json("data/VIDEO_DENSITY.json", {"stage": "N72R21R2", "goal": GOAL,
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "rule": p["density"],
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "census40": census,
        "actual_fresh24_density_counts": dict(Counter(d["whole_video_density"] for d in density.values())),
        "candidate_counts_NOT_GT_people_density": True, "all24_original_candidate_payload_SHA_reverified": True,
        "CONFIRM_truth_or_effects_accessed": False, "density_rule_not_changed_by_model_outcomes": True})
    write_json("mot/DENSITY_BASELINE_SIMPLE_FULL_VIDEO_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "status": "COMPLETE_ACTUAL_ALL24_FIXED_DENSITY_BASELINE_SIMPLE_DESCRIPTIVE_TABLES_NOT_MAIN_CLOSURE",
        "density_manifest_sha256": sha256(OUT / "data/VIDEO_DENSITY.json"),
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "source_receipts": refs,
        "tables": table, "all_nine_metrics_from_actual_pinned_full_video_results": True,
        "zero_action_policy_full_trajectory_bytes_C0_AA_actual": True,
        "singleton_sparse_video_does_not_prove_generalization": True,
        "masked_density_TrackEval": {"status": "NOT_RUN", "reason": "These are original full-video metrics stratified by the preregistered whole-video counts; no masked trajectories or selected easy frames are substituted."},
        "MAIN_and_state_source_policy_density_tables": "PENDING_ACTUAL_FULL_VIDEO_RESULTS",
        "scientific_success": None, "confirmation_authorized": False})
    append_log("M10_ACTUAL_FROZEN_DENSITY_FULL_VIDEO_TABLES", density_sha256=sha256(OUT / "data/VIDEO_DENSITY.json"), sparse_support=1, confirmation_authorized=False)
    print({"fresh24_original_density": dict(Counter(d["whole_video_density"] for d in density.values())),
           "actual_full_video_policy_tables": len(rows), "MAIN_closure": False}, flush=True)


if __name__ == "__main__":
    run()
