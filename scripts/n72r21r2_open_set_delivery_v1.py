"""Deliver actual M8 current-claim evidence, keeping MOT takeover incomplete.

All three learned seeds are averaged inside each original video before CIs.
Original INNER calibration grids are reported, never retuned on exact-CF truth.
"""
import argparse
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, utcnow, preregistration, update_status
from sam3_intermot.evaluation.fixed_identity_opportunities import video_interval

COMPARISON = OUT / "availability/fixed_control_cf_v1/ALL24_EXACT_FRAME_COMPARISON.json"
RESULT = "availability/ALL24_CURRENT_IDENTITY_DELIVERY_V1.json"


def family_video_intervals(strategies, models, seeds, split):
    families = {}
    for family in sorted({m["family"] for m in models}):
        members = [m for m in models if m["family"] == family]
        if len(members) != len(seeds) or {m["seed"] for m in members} != set(seeds):
            raise ValueError("Every original family seed required; no best or ready subset")
        roles = {}
        for role in ("fit", "inner"):
            values, supported = {}, {}
            for sequence in split[role]:
                rows = [strategies[m["model"]][role]["per_sequence"][sequence]["rates"] for m in members]
                if any(set(r) != set(rows[0]) for r in rows):
                    raise ValueError("All model/seed rate schemas must match")
                values[sequence], supported[sequence] = {}, {}
                for metric in rows[0]:
                    observed = [r[metric] for r in rows if r[metric] is not None]
                    values[sequence][metric] = float(np.mean(observed)) if observed else None
                    supported[sequence][metric] = len(observed)
            keys = list(next(iter(values.values()))) if values else []
            roles[role] = {"all_three_seeds_retained": list(seeds),
                "seed_averaged_per_video_rates": values, "defined_seed_counts_per_video": supported,
                "video_cluster_CI95": {k: video_interval([v[k] for v in values.values()]) for k in keys},
                "seeds_frames_and_arms_NOT_extra_independent_video_clusters": True}
        families[family] = roles
    return families


def original_grid_summary(record):
    grid = record["all_INNER_operating_points"]
    feasible = [r for r in grid if r["result"]["micro"]["verified_OTHER_row_FPR"] is not None
                and r["result"]["micro"]["verified_OTHER_row_FPR"] <= .02]
    return {"selection_unchanged": record["selection"],
        "original_INNER_grid_points": len(grid),
        "Recall_at_verified_OTHER_row_FPR2_original_measured_GRID_ONLY": max(
            (r["result"]["micro"]["target_recall_given_available"] or 0. for r in feasible), default=None),
        "NOT_physical_absence_FPR_or_accepted_claim_contamination_bound": True,
        "original_grid_current_risk_coverage": [{"selection": r["selection"],
            "counts": r["result"]["counts"], "micro": r["result"]["micro"],
            "original_nonvacuous": r["nonvacuous"], "original_empirical_current_safe": r["empirical_current_safe"]} for r in grid],
        "exact_CF_truth_never_selects_new_point": True}


def run():
    if (OUT / RESULT).exists():
        raise FileExistsError("Keep actual report; version later completion")
    comparison = read_json(COMPARISON)
    assert comparison["goal"] == GOAL and not comparison["association_authority"]
    assert len(comparison["original_all24_sequences_retained"]) == 24
    refs = {str(COMPARISON): sha256(COMPARISON)}
    for path, digest in comparison["evidence_sha256"].items():
        assert sha256(path) == digest
        refs[path] = digest
    prereg = preregistration()
    fitting = read_json(OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json")
    assert fitting["split"] == prereg["split"]
    exact_protocol = OUT / "availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json"
    models = read_json(exact_protocol)["models"]
    refs[str(exact_protocol)] = sha256(exact_protocol)
    families = family_video_intervals(comparison["strategies"], models, fitting["seeds"], fitting["split"])
    grids = {}
    for model in models:
        path = OUT / "availability/current_axis_fits" / (model["model"] + ".json")
        record = read_json(path)
        assert record["protocol_sha256"] == sha256(OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json")
        assert sha256(record["checkpoint_path"]) == record["checkpoint_sha256"] == model["checkpoint_sha256"]
        refs[str(path)] = sha256(path)
        refs[record["checkpoint_path"]] = record["checkpoint_sha256"]
        grids[model["model"]] = original_grid_summary(record)
    simple_path = OUT / "availability/SIMPLE_CURRENT_AXIS_CALIBRATION_V1.json"
    simple = read_json(simple_path)
    refs[str(simple_path)] = sha256(simple_path)
    for name, record in simple["controls"].items():
        grids[name] = original_grid_summary(record)
    result = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
        "status": "COMPLETE_ALL24_FIXED_CURRENT_IDENTITY_DIAGNOSTIC_NOT_FULL_OPEN_SET_MOT_CLOSURE",
        "source_sha256": {"scripts/n72r21r2_open_set_delivery_v1.py": sha256(__file__)},
        "comparison": {"path": str(COMPARISON), "sha256": sha256(COMPARISON)},
        "all_three_seed_family_video_CIs": families, "original_current_calibration_grids": grids,
        "frame_opportunities_NOT_independent_or_executed_correction_events": True,
        "deployed_MOT_wrong_takeover_or_improvement": None,
        "deployed_MOT_evaluation_status": "PENDING_ALL17_ORIGINAL_OWN_POLICY_GROUP_DISPOSITIONS_AND_EFFECTS",
        "current_identity_claim_authorizes_association": False,
        "UNKNOWN_never_verified_OTHER": True, "physically_absent_vs_no_positive_separate": True,
        "scientific_goal_complete": False, "next_stage_authorized": False, "evidence_sha256": refs}
    report = write_json(RESULT, result)
    common = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
        "status": "PARTIAL_COMPLETE_CURRENT_IDENTITY_DIAGNOSTICS_DEFERRED_FULL_MOT_EFFECTS_NOT_SCIENTIFIC_CLOSURE",
        "actual_all24_current_diagnostic": {"path": str(report), "sha256": sha256(report)},
        "exact_frame_comparison": result["comparison"], "deployed_MOT_wrong_takeover": None,
        "deployed_MOT_wrong_takeover_status": result["deployed_MOT_evaluation_status"],
        "association_authority": False, "scientific_goal_complete": False, "next_association_stage_authorized": False}
    for name, purpose in (("OPEN_SET", "Original calibrated controls/verifiers and exact-frame missed safe repairs; physical absence, no valid positive, UNKNOWN and NONE separate"),
                          ("WRONG_TAKEOVER", "Actual wrong current identity claims only; never recast as executed global-MOT takeover"),
                          ("RISK_COVERAGE", "Original INNER grid current risk/coverage and Recall@verified-OTHER-row-FPR2; no exact-CF retuning or population2% guarantee")):
        write_json("availability/" + name + ".json", {**common, "scope": purpose})
    update_status(latest_named_open_set_current_diagnostic={"path": str(report), "sha256": sha256(report)},
                  all_current_working_changes_published=False)
    print({"delivered_current_diagnostic_reports": 3, "learned_families_all3_seeds": len(families), "MOT_scope_pending": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    run()
