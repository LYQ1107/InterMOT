"""Diagnose frozen learned hand-filter support on real sealed CF opportunities.

This is not optimizer input, operating-point selection, an oracle policy, an
actual model prediction or a deployment evaluation. No runtime source is edited.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, read_json, write_json, sha256, preregistration, storage, append_log
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r2_exact_cf_verifier import direct_repair_opportunities
from sam3_intermot.evaluation.authority_support_audit import hard_filter_violations, safe_direct_source_row
from sam3_intermot.one_click.event_authority_runtime import qualified_current_candidate
from sam3_intermot.one_click.event_authority_learning import encode_runtime_input

PROTOCOL = "training/AUTHORITY_SUPPORT_AUDIT_PROTOCOL_V1.json"
PILOT = "training/PILOT_POLICY_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_authority_support_audit.py", "sam3_intermot/evaluation/authority_support_audit.py",
        "sam3_intermot/one_click/event_authority_runtime.py", "sam3_intermot/one_click/event_authority_learning.py",
        "scripts/n72r21r2_exact_cf_verifier.py", "sam3_intermot/one_click/intervention_features.py")
IDEAL = {"beneficial": 1., "harmful": 0., "value": 1.}


def freeze():
    write_json(PROTOCOL, {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "unchanged_registered_pilot_point": read_json(OUT / PILOT)["point"],
        "registered_point_protocol_sha256": sha256(OUT / PILOT), "split": preregistration()["split"],
        "source_sha256": {p: sha256(ROOT / p) for p in CODE},
        "input": "Actual sealed C0 CF current causal features; original source arm labels joined only OFFLINE",
        "denominator": "Complete H100 current-effective TARGET N01=1 safe positive component, no target/other/write harm; unique episode/frame and distinct executed action configs separately. Delayed excluded, not independent onsets.",
        "observed_confirmation": "Exact source pending_confirmation_count; C0 source count0 is not own-policy confirmations",
        "confirmation_satisfied_ceiling": "HYPOTHETICAL HAND-FILTER UPPER BOUND ONLY: assume confirmation requirement met and beneficial1/harmful0/value1. No real model score or policy action, no oracle deployment claim.",
        "thresholds_not_retuned_from_results": True, "association_actions_executed": 0,
        "all24_census_required_partial_snapshots_explicit": True, "no_G1_G2_G4_or_generalization_qualification": True})


def audit_video(path, point):
    receipt = read_json(path)
    assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
    rows = read_zstd_jsonl(Path(receipt["artifact"]["path"]))
    direct, _ = direct_repair_opportunities(rows)
    safe = [r for r in rows if safe_direct_source_row(r)]
    assert {(r["episode_uid"], r["frame"]) for r in safe} == set(direct)
    reasons, observed, confirmations = Counter(), Counter(), Counter()
    groups, seen, arm_count, duplicates = defaultdict(list), set(), 0, 0
    for row in safe:
        signature = (row["episode_uid"], row["frame"], json.dumps(row["action"], sort_keys=True))
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature)
        assert row["current_features_captured_before_branch_future"] and row["future_truth_not_runtime_feature"]
        encode_runtime_input(row["runtime_features"], row["branch"])
        f = row["runtime_features"]["features"]
        count = f["pending_confirmation_count"]
        a = hard_filter_violations(f, point, confirmations=count)
        b = hard_filter_violations(f, point, confirmations=max(count, point["confirmation_delay"]))
        # Actual direct N01 and effective-current TARGET action guarantee the
        # source arm did execute a feasible changed assignment. Model score here
        # is deliberately ideal/constant, not ground-truth-derived prediction.
        assert (not a) == qualified_current_candidate(f, IDEAL, point, feasible=True, changed=True, confirmations=count)
        assert (not b) == qualified_current_candidate(f, IDEAL, point, feasible=True, changed=True,
                                                     confirmations=max(count, point["confirmation_delay"]))
        observed.update(a)
        reasons.update(b)
        confirmations[str(count)] += 1
        groups[row["episode_uid"], row["frame"]].append({"observed_pass": not a, "hypothetical_pass": not b,
            "hard_filter_vetoes_without_confirmation": b})
        arm_count += 1
    return {"label_receipt_sha256": sha256(path), "CF_runtime_seal_sha256": receipt.get("CF_runtime_seal_sha256"),
            "source_label_rows": len(rows), "safe_direct_event_frame_groups_NOT_independent": len(direct),
            "distinct_safe_executed_action_configs": arm_count, "duplicate_safe_action_config_rows_excluded": duplicates,
            "source_confirmation_counts_distinct_arms": dict(confirmations),
            "observed_source_confirmation_ideal_model_pass_event_frames": sum(any(a["observed_pass"] for a in g) for g in groups.values()),
            "hypothetical_confirmation_satisfied_ideal_model_pass_event_frames": sum(any(a["hypothetical_pass"] for a in g) for g in groups.values()),
            "hypothetical_confirmation_satisfied_all_arms_vetoed_event_frames": sum(not any(a["hypothetical_pass"] for a in g) for g in groups.values()),
            "overlapping_arm_veto_counts_not_additive": dict(reasons), "observed_overlapping_arm_veto_counts": dict(observed),
            "per_episode_frame_groups_NOT_independent": [{"episode_uid": e, "frame": f, "distinct_safe_arm_filters": a}
                                                       for (e, f), a in sorted(groups.items())]}


def snapshot():
    p = read_json(OUT / PROTOCOL)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["registered_point_protocol_sha256"] == sha256(OUT / PILOT)
    sequences = p["split"]["fit"] + p["split"]["inner"]
    inputs = [{"sequence": s, "path": str(path), "sha256": sha256(path)} for s in sequences
              if (path := OUT / "events/label_audit" / (s + ".json")).exists()]
    manifest = {"protocol_sha256": sha256(OUT / PROTOCOL), "inputs": inputs}
    uid = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    destination = "training/authority_support_v1/snapshots/" + uid + ".json"
    if (OUT / destination).exists():
        print({"authority_support_existing_immutable_snapshot": destination}, flush=True)
        return
    storage(16 << 20)
    results = {}
    for ref in inputs:
        assert sha256(ref["path"]) == ref["sha256"]
        results[ref["sequence"]] = audit_video(Path(ref["path"]), p["unchanged_registered_pilot_point"])
        print({"actual_authority_support_video": ref["sequence"],
               "safe_direct_event_frames": results[ref["sequence"]]["safe_direct_event_frame_groups_NOT_independent"]}, flush=True)
    totals = {}
    for role in ("fit", "inner"):
        selected = [v for s, v in results.items() if s in p["split"][role]]
        numeric = ("safe_direct_event_frame_groups_NOT_independent", "distinct_safe_executed_action_configs",
                   "duplicate_safe_action_config_rows_excluded", "observed_source_confirmation_ideal_model_pass_event_frames",
                   "hypothetical_confirmation_satisfied_ideal_model_pass_event_frames",
                   "hypothetical_confirmation_satisfied_all_arms_vetoed_event_frames")
        totals[role] = {"actual_videos": len(selected), **{k: sum(v[k] for v in selected) for k in numeric}}
        vetoes = Counter()
        for v in selected:
            vetoes.update(v["overlapping_arm_veto_counts_not_additive"])
        totals[role]["overlapping_arm_veto_counts_not_additive"] = dict(vetoes)
    write_json(destination, {"stage": "N72R21R2", **manifest, "results": results, "role_totals": totals,
        "all24_complete": len(results) == 24, "missing_required_sequences": [s for s in sequences if s not in results],
        "scientific_scope": "POSTHOC FROZEN-HAND-FILTER SUPPORT DIAGNOSTIC; no model success/failure attribution or gate change",
        "independent_onsets_not_measured": True, "association_actions_executed": 0, "scientific_success": None,
        "CONFIRM_VAL_TEST_SOT_unopened": True})
    write_json("training/authority_support_v1/latest.json", {"path": str(OUT / destination), "sha256": sha256(OUT / destination)}, mutable=True)
    append_log("FROZEN_AUTHORITY_HAND_FILTER_SUPPORT_DIAGNOSTIC", snapshot=destination, actual_videos=len(results), totals=totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "snapshot"))
    {"freeze": freeze, "snapshot": snapshot}[parser.parse_args().action]()
