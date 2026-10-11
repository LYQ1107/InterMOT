"""Bind six M9 named reports to the complete original M-A/raw frontier.

No runtime, labels, models, operating points, gates or association changes.
Failure of executed necessary point conditions is not a population estimate
or a claim that all authority families/full M0-M11 research have closed.
"""
import argparse
import math
from scripts.n72r21r2_common import (
    ROOT, OUT, GOAL, read_json, write_json, sha256, utcnow, storage, append_log,
)

FAMILIES = ("FROZEN", "MEAN", "DELAYED", "MULTICUE", "PENDING_TRUSTED", "ROLLBACK",
    "LOGISTIC__SELECTED_CURRENT_RISK_ONLY", "MLP__SELECTED_CURRENT_RISK_ONLY",
    "LOGISTIC__FIXED_P50_UNQUALIFIED_DIAGNOSTIC", "MLP__FIXED_P50_UNQUALIFIED_DIAGNOSTIC")
REPORTS = {"FROZEN": ("FROZEN",), "DELAYED": ("DELAYED",),
    "TRUSTED_BANK": ("MULTICUE", "PENDING_TRUSTED"), "RISK_WRITE": FAMILIES[6:],
    "ROLLBACK": ("ROLLBACK",), "SAFETY_FRONTIER": FAMILIES}


def point_disposition(summary, conditions):
    """Check existing seed-averaged exposures; zero writes have undefined risk."""
    counts = summary["exposure_counts_seed_averaged_NOT_independent_sample_size"]
    names = ("accepted_writes", "write_TARGET", "write_VERIFIED_OTHER", "write_UNKNOWN",
             "correct_C0_committed_observations")
    values = {name: counts[name] for name in names}
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Original exposure counts must be finite nonnegative numbers")
    accepted, correct, other, unknown, opportunities = (values[name] for name in names)
    if not math.isclose(accepted, correct + other + unknown, abs_tol=1e-8) or correct > opportunities + 1e-8:
        raise ValueError("Keep UNKNOWN in accepted exposure and all C0 positives in retention")
    risk = (other + unknown) / accepted if accepted else None
    retention = correct / opportunities if opportunities else None
    observed = summary["pooled_exposure_rates"]
    for name, value in (("wrong_or_UNKNOWN_write_rate", risk), ("correct_write_retention", retention)):
        if (value is None) != (observed[name] is None) or value is not None and not math.isclose(value, observed[name], abs_tol=1e-12):
            raise ValueError("Stored rates must match original seed-averaged denominators")
    qualified = bool(accepted > 0 and risk <= conditions["wrong_plus_UNKNOWN_max"]
                     and retention is not None and retention >= conditions["correct_retention_min"])
    if summary["empirical_pooled_G4_point_conditions_only"] is not qualified:
        raise ValueError("Do not change the original necessary point-gate result")
    reasons = []
    if accepted == 0: reasons.append("ZERO_WRITES_RISK_UNDEFINED_NOT_SAFETY")
    elif risk > conditions["wrong_plus_UNKNOWN_max"]: reasons.append("VERIFIED_OTHER_PLUS_UNKNOWN_EXCEEDS_FROZEN_LIMIT")
    if retention is None or retention < conditions["correct_retention_min"]:
        reasons.append("CORRECT_WRITE_RETENTION_BELOW_FROZEN_LIMIT_OR_UNDEFINED")
    return {"necessary_empirical_G4_point_conditions": qualified,
            "disposition": "NECESSARY_POINT_PASS_UNCERTAINTY_NOT_QUALIFIED" if qualified else "FAIL_EXECUTED_NECESSARY_M_A_POINT_CONDITIONS",
            "reasons": reasons, "wrong_plus_UNKNOWN_exposure_rate": risk, "correct_write_retention": retention,
            "population_two_percent_safety_established": False, "independent_causal_event_risk_CI": None,
            "M_B_authorized": False}


def run(raw_receipt):
    storage(16 << 20)
    raw_path = (OUT / raw_receipt).resolve()
    if not raw_path.is_relative_to((OUT / "audit").resolve()):
        raise ValueError("Only stage-local immutable independent raw audit evidence")
    raw = read_json(raw_path)
    assert raw["goal"] == GOAL and raw["actual_case_clicks_NOT_independent_events"] == 1062
    assert raw["raw_current_write_UIDs_bank_causality_full_C0_outputs_trajectory_and_pinned9_CSV_reverified"]
    assert raw["offline_label_outcomes_preserved_NOT_independently_reclassified_from_GT"]
    pointer = raw["frontier_snapshot"]
    assert sha256(pointer["path"]) == pointer["sha256"]
    frontier = read_json(pointer["path"])
    assert frontier["goal"] == GOAL and frontier["all24_fixed_and_risk_complete"]
    assert frontier["counts"] == {"fixed": 24, "fresh_risk": 24}
    assert frontier["actual_case_clicks"] == raw["actual_case_clicks_NOT_independent_events"]
    assert set(raw["all24_raw_case_census"]) == set(frontier["census"]) and len(frontier["census"]) == 24
    protocol_path = OUT / "memory/SAFETY_FRONTIER_PROTOCOL_V1.json"
    assert frontier["protocol_sha256"] == sha256(protocol_path)
    protocol = read_json(protocol_path)
    assert protocol["frozen"] and protocol["goal"] == GOAL
    conditions = protocol["G4_point_conditions"]
    refs = {**frontier["evidence_sha256"], **raw["refs_sha256"],
            str(raw_path): sha256(raw_path), str(protocol_path): sha256(protocol_path)}
    assert all(sha256(path) == digest for path, digest in refs.items())
    assert raw["source_sha256"] == {name: sha256(ROOT / name) for name in raw["source_sha256"]}
    roles = {}
    for role in ("FIT", "INNER"):
        rows = frontier["role_summaries"][role]
        assert set(rows) == set(FAMILIES)
        roles[role] = {}
        for family in FAMILIES:
            row = rows[family]
            assert row["all_role_videos_required_and_complete"]
            assert row["video"]["cluster_count"] == (15 if role == "FIT" else 7)
            assert row["full_global_MOT_all9_delta"] and all(value == 0 for value in row["full_global_MOT_all9_delta"].values())
            roles[role][family] = {"original_actual_summary": row,
                "independently_recomputed_necessary_point_disposition": point_disposition(row["video"], conditions)}
    any_point = any(row["independently_recomputed_necessary_point_disposition"]["necessary_empirical_G4_point_conditions"]
                    for rows in roles.values() for row in rows.values())
    common = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "utc": utcnow(),
        "source_sha256": {"scripts/n72r21r2_memory_delivery_v1.py": sha256(ROOT / "scripts/n72r21r2_memory_delivery_v1.py")},
        "status": "COMPLETE_EXECUTED_M_A_POINT_CENSUS_NOT_FULL_M0_M11_OR_POPULATION_SAFETY_CLOSURE",
        "source_frontier": pointer, "independent_raw_readback": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "refs_sha256": refs, "frozen_G4_necessary_conditions": conditions, "all24_sequence_census": frontier["census"],
        "all_failed_initializations_NOT_RUN_NEVER_imputed_or_replaced": True,
        "actual_case_clicks_NOT_independent_sample_size": 1062,
        "offline_write_outcomes_preserved_NOT_independent_GT_relabeling": True,
        "GT_identity_clusters_video_local_NOT_person_disjoint": True,
        "legacy_tensor_AA_NOT_complete_tracker_attribute_AA": True,
        "global_MOT_equality_ISOLATION_NOT_memory_benefit": True,
        "any_executed_point_meets_necessary_empirical_G4_conditions": any_point,
        "G4_population_uncertainty_and_independent_write_roots_unresolved": True,
        "M_B_disposition": "NOT_AUTHORIZED_UNCERTAINTY_AND_AUTHORITY_QUALIFICATION_PENDING" if any_point else
                            "NOT_RUN_NO_EXECUTED_M_A_POINT_MEETS_NECESSARY_MEMORY_GATE",
        "full_authority_group_closure_still_pending": True, "M_B_authorized": False,
        "new_runtime_GT_parse_fit_threshold_or_association_change": False,
        "scientific_goal_complete": False, "scientific_success": False, "next_stage_authorized": False}
    # Preflight all six destinations. Preserve partial/older evidence, never replace it.
    destinations = {name: OUT / "memory" / (name + ".json") for name in REPORTS}
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError("Named memory reports already exist; inspect/version without overwriting")
    assert all(sha256(path) == digest for path, digest in refs.items())
    delivered = {}
    for name, families in REPORTS.items():
        report = {**common, "named_report": name, "included_executed_families": families,
                  "role_summaries": {role: {family: rows[family] for family in families} for role, rows in roles.items()}}
        path = write_json("memory/" + name + ".json", report)
        delivered[name] = {"path": str(path), "sha256": sha256(path)}
    append_log("M9_ALL24_NAMED_MEMORY_ACTUAL_DELIVERY", reports=delivered, any_necessary_point_pass=any_point,
               M_B_authorized=False, scientific_goal_complete=False)
    print({"named_memory_reports": delivered, "any_necessary_point_pass": any_point,
           "M_B_disposition": common["M_B_disposition"], "scientific_goal_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-receipt", required=True)
    run(parser.parse_args().raw_receipt)
