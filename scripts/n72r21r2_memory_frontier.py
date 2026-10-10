"""Immutable partial/final M-A safety frontier from sealed actual full videos.

This offline report never changes gates, thresholds, memory or association.
Missing videos remain in the census, and partial reports cannot qualify G4.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from scripts.n72r21r2_common import ROOT, OUT, read_json, write_json, preregistration, sha256, append_log
from scripts.n72r21r2_zero_authority_memory import METRICS
from sam3_intermot.evaluation.memory_safety_frontier import checked_counts, cluster_summary, paired_recognition

PROTOCOL = OUT / "memory/SAFETY_FRONTIER_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_memory_frontier.py", "sam3_intermot/evaluation/memory_safety_frontier.py")


def freeze():
    p = preregistration()
    write_json("memory/SAFETY_FRONTIER_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": p["goal"], "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "split": p["split"],
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "frozen": True,
        "scope": "All24 fresh FIT16/INNER8 actual zero-authority full-memory videos, six fixed policies and all fresh risk-writer heads/points. Never subset selection or external confirmation access.",
        "exposure": "Actual accepted writes; verified-other + UNKNOWN contamination. Retention denominator all actual correct C0 committed observations, including those rejected by writer.",
        "aggregation": "Sum clicks inside each video/seed, compute rates, average seeds inside video; equal videos primary. Report pooled exposure rates separately. Seeds/clicks/frames do not increase video cluster count.",
        "uncertainty": "2000 percentile whole-video resamples seed730104. Additionally video-local GT-identity cluster description, NOT cross-scene person-disjoint proof. Independent causal write-event roots are unavailable until separately verified; do NOT substitute bursts/error intervals.",
        "zero_error_uncertainty": "Degenerate empirical bootstrap [0,0] is NOT population2% proof. Separately report conditional binomial95% interval for ANY-bad-write clusters, not per-write risk. Zero writes have undefined risk.",
        "paired_recognition": "Same C0 opportunities/clicks and actual frozen-bank reference; available/competitive Rank1 and visible/available claim recall, paired video macro delta/95% CI. HOTA equality is isolation only.",
        "bootstrap_repetitions": 2000, "bootstrap_seed": 730104,
        "G4_point_conditions": {"nonzero_actual_writes": True, "wrong_plus_UNKNOWN_max": .02, "correct_retention_min": .6},
        "frontier_points": "Only actually executed registered fixed/head operating points, never interpolated write coverage or retuned thresholds",
        "partial_reports": "Immutable input-hash snapshots include all24 census and explicit missing videos; no G4/main-stage authorization",
        "final_requires_all24_fixed_and_all24_fresh_risk_results": True,
        "M_B_automatic": False, "association_stage_authorized": False})


def case_group(case):
    if "__seed" not in case:
        return case, "FIXED"
    family, rest = case.split("__seed", 1)
    seed, point = rest.split("__", 1)
    return family + "__" + point, int(seed)


def checked_result(sequence, kind):
    prefix = "memory/M_A" if kind == "fixed" else "memory/risk_v1"
    path = OUT / prefix / "results" / (sequence + ".json")
    if not path.exists():
        return None
    result = read_json(path)
    if result["sequence"] != sequence or not result["association_authority_zero"]:
        raise ValueError("Frontier requires exact zero-authority development source")
    protocol = OUT / "memory" / ("ZERO_AUTHORITY_PROTOCOL_V1.json" if kind == "fixed" else "RISK_WRITE_PROTOCOL_V1.json")
    assert result["protocol_sha256"] == sha256(protocol)
    code_hashes = result["source_freeze"] if kind == "fixed" else result["source_sha256"]
    assert all(sha256(ROOT / name) == expected for name, expected in code_hashes.items())
    runtime_path = OUT / prefix / "runtime_sequences" / (sequence + ".json")
    runtime = read_json(runtime_path)
    key = "runtime_sequence_sha256" if kind == "fixed" else "whole_video_runtime_receipt_sha256"
    assert result[key] == sha256(runtime_path)
    init_path = OUT / "data/initialization" / (sequence + ".json")
    truth_path = OUT / "data/initialization_truth" / (sequence + ".json")
    init, truth = read_json(init_path), read_json(truth_path)
    events = {e["episode_uid"]: e for e in init["inputs"] if not e["initialization_failure"]}
    targets = {e["episode_uid"]: e["target_gt_identity"] for e in truth["labels"]}
    cases = read_json(protocol)["cases"] if kind == "fixed" else runtime["all6_heads_all_seeds_both_points"]
    expected = {case + "/" + episode for case in cases for episode in events}
    if set(result["identity_and_write_results"]) != expected:
        raise ValueError("Do not summarize a subset of valid clicks/heads/points")
    if bool(events) != bool(result["all_nine_equal_C0"]):
        raise ValueError("Actual nonempty full-metric AA receipt missing")
    baseline_path = OUT / "mot/baseline_results" / (sequence + ".json")
    baseline = read_json(baseline_path)["actual_nine_metrics"]
    expected_trackers = {case + "__click" + str(e["slot"]) for case in cases for e in events.values()}
    assert set(result["all_nine_actual_metrics"]) == expected_trackers
    for name, metrics in result["all_nine_actual_metrics"].items():
        base = baseline["CLICK_C0__click" + name.rsplit("__click", 1)[1]]
        assert all(metrics[k] == base[k] for k in METRICS)
    rows, refs = [], [(path, sha256(path)), (runtime_path, sha256(runtime_path)),
                     (init_path, sha256(init_path)), (truth_path, sha256(truth_path)),
                     (baseline_path, sha256(baseline_path)), (protocol, sha256(protocol))]
    for key, value in sorted(result["identity_and_write_results"].items()):
        case, episode = key.split("/", 1)
        seal_path = OUT / prefix / "runtime" / case / (episode + ".json")
        seal = read_json(seal_path)
        assert seal["protocol_sha256"] == sha256(protocol)
        assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
        counts = checked_counts(value["raw_counts"])
        writes = value["accepted_write_provenance"]
        if len(writes) != counts["accepted_writes"] or len({w["frame"] for w in writes}) != len(writes):
            raise ValueError("Accepted writes need actual unique original-frame provenance")
        origins = Counter(w["outcome"] for w in writes)
        assert set(origins) <= {"TARGET", "VERIFIED_OTHER", "UNKNOWN"}
        assert all(origins.get(k, 0) == counts["write_" + k] for k in ("TARGET", "VERIFIED_OTHER", "UNKNOWN"))
        expected_writes = seal["actual_writes"] if kind == "fixed" else seal["counts"].get("actual_accepted_writes", 0)
        assert expected_writes == counts["accepted_writes"]
        family, seed = case_group(case)
        rows.append({"sequence": sequence, "role": events[episode]["role"], "episode_uid": episode,
                     "target_gt_identity": targets[episode], "case": case, "family": family, "seed": seed,
                     "raw_counts": counts, "max_bank_drift": value["max_bank_drift"],
                     "actual_trajectory_bytes_and_all9_C0_AA": True, "case_runtime_sha256": sha256(seal_path)})
        refs.append((seal_path, sha256(seal_path)))
    return rows, refs, len(events), sum(bool(e["initialization_failure"]) for e in init["inputs"])


def snapshot(final=False):
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    sequences = p["split"]["fit"] + p["split"]["inner"]
    rows, refs, census = [], {}, {}
    for sequence in sequences:
        census[sequence] = {"role": "FIT" if sequence in p["split"]["fit"] else "INNER"}
        for kind in ("fixed", "fresh_risk"):
            checked = checked_result(sequence, kind)
            if checked is None:
                census[sequence][kind] = {"status": "MISSING_NOT_SUBSTITUTED"}
                continue
            examples, evidence, valid, failed = checked
            rows.extend(examples)
            refs.update((str(path), digest) for path, digest in evidence)
            census[sequence][kind] = {"status": "COMPLETE_SEALED_ACTUAL_ALL_CASES", "valid_sole_clicks": valid,
                                      "failed_initializations_retained": failed, "actual_case_clicks": len(examples)}
    complete = all(census[s][kind]["status"] == "COMPLETE_SEALED_ACTUAL_ALL_CASES" for s in sequences for kind in ("fixed", "fresh_risk"))
    if final and not complete:
        raise ValueError("Final memory frontier requires ALL24 fixed + ALL24 fresh risk results")
    summaries = {}
    for role, role_sequences in (("FIT", p["split"]["fit"]), ("INNER", p["split"]["inner"])):
        grouped = defaultdict(list)
        for row in rows:
            if row["sequence"] in role_sequences:
                grouped[row["family"]].append(row)
        summaries[role] = {}
        for family, examples in sorted(grouped.items()):
            reference_sequences = {e["sequence"] for e in examples}
            reference = [r for r in grouped["FROZEN"] if r["sequence"] in reference_sequences]
            summaries[role][family] = {
                "ready_sequences_in_frozen_order": [s for s in role_sequences if s in reference_sequences],
                "all_role_videos_required_and_complete": all(s in reference_sequences or census[s]["fixed"].get("valid_sole_clicks") == 0 for s in role_sequences),
                "video": cluster_summary(examples, "video", repetitions=p["bootstrap_repetitions"], seed=p["bootstrap_seed"]),
                "video_local_identity": cluster_summary(examples, "video_local_identity", repetitions=p["bootstrap_repetitions"], seed=p["bootstrap_seed"]),
                "paired_frozen_recognition": paired_recognition(examples, reference, repetitions=p["bootstrap_repetitions"], seed=p["bootstrap_seed"]),
                "actual_max_bank_drift_over_click_seed_trajectories": max(r["max_bank_drift"] for r in examples),
                "full_global_MOT_all9_delta": {k: 0. for k in METRICS},
                "constant_MOT_is_ISOLATION_NOT_memory_effect": True,
                "point_is_unqualified_diagnostic": "UNQUALIFIED" in family,
                "scientific_memory_safety_PASS": False,
                "G4_final_assessment": "PENDING_FULL_CENSUS_AND_INDEPENDENT_UNCERTAINTY_AUDIT"}
    bundle = {"protocol_sha256": sha256(PROTOCOL), "evidence_sha256": refs, "census": census}
    digest = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    relative = "memory/frontier_v1/snapshots/" + digest + ".json"
    result = {"stage": "N72R21R2", "goal": p["goal"], "goal_file": p["goal_file"], **bundle,
              "snapshot_input_bundle_sha256": digest, "all24_fixed_and_risk_complete": complete,
              "status": "COMPLETE_ACTUAL_FRONTIER_DESCRIPTIVE_UNCERTAINTY_UNRESOLVED" if complete else "PARTIAL_ACTUAL_FRONTIER_NOT_STAGE_QUALIFICATION",
              "counts": {kind: sum(census[s][kind]["status"] == "COMPLETE_SEALED_ACTUAL_ALL_CASES" for s in sequences) for kind in ("fixed", "fresh_risk")},
              "role_summaries": summaries, "actual_case_clicks": len(rows),
              "independent_causal_event_risk_CI_NOT_fabricated": True, "scientific_success": None,
              "M_B_authorized": False, "next_association_stage_authorized": False}
    write_json(relative, result)
    write_json("memory/frontier_v1/latest.json", {"path": str(OUT / relative), "sha256": sha256(OUT / relative),
               "counts": result["counts"], "status": result["status"]}, mutable=True)
    append_log("M9_ACTUAL_CLUSTERED_MEMORY_FRONTIER", bundle=digest, counts=result["counts"], partial=not complete)
    print({"memory_frontier": str(OUT / relative), "counts": result["counts"], "partial": not complete}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "snapshot", "final"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else snapshot(final=args.action == "final")
