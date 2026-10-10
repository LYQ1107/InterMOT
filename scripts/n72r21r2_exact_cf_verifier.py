"""Frozen current verifiers on exact CF frames, then offline missed repairs.

Counts are correlated event-frame opportunities, NOT independent beneficial
onsets or deployed MOT corrections. No nearest32-frame imputation, retuning,
best-seed selection, GT-scored UID injection or association authority.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, preregistration, development_sequence, storage, append_log
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r2_joint_state_curriculum import writer
from sam3_intermot.one_click.fresh_open_set_runtime import FreshOpenSetPredictor
from sam3_intermot.one_click.open_set_verifier import choose
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_exact_cf_verifier.py", "sam3_intermot/one_click/fresh_open_set_runtime.py",
        "scripts/n72r21r2_exact_cf_verifier_driver.py",
        "sam3_intermot/one_click/open_set_verifier.py", "sam3_intermot/one_click/runtime_file_guard.py")


def freeze():
    p = read_json(OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json")
    models = []
    for family in p["families"]:
        for seed in p["seeds"]:
            uid = family + "__seed" + str(seed)
            fit_path = OUT / "availability/current_axis_fits" / (uid + ".json")
            fit = read_json(fit_path)
            assert sha256(fit["checkpoint_path"]) == fit["checkpoint_sha256"]
            assert fit["nonzero_gradient_steps"] > 0 and not fit["association_authority"]
            models.append({"model": uid, "family": family, "seed": seed,
                           "checkpoint_path": fit["checkpoint_path"], "checkpoint_sha256": fit["checkpoint_sha256"],
                           "fit_record_sha256": sha256(fit_path)})
    assert len(models) == 12
    write_json("availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "models": models,
        "split": preregistration()["split"], "source_sha256": {p: sha256(ROOT / p) for p in CODE},
        "frozen_before_exact_CF_predictions_or_missed_repair_labels": True,
        "current_source": "Sealed real C0 replay at EVERY originally registered CF frame; all current UIDs+NONE",
        "operating_points": "Reuse each frozen selected INNER point, including CALIBRATION_ABSTAIN; no retuning or best-seed dropping",
        "direct_repair_denominator": "Unique click/frame with an actually effective current N01=1 branch, committed TARGET candidate, complete own future H100, future safe positive component, no current/future target/other/write harm. Delayed effects excluded from direct-current denominator; not independent onset count.",
        "future_improvement_denominator": "Separately retain safe target-improving H100 opportunities even if current N01=0; not direct-current repair",
        "UNKNOWN": "Independent uncertainty; never verified OTHER or correct identity",
        "full_MOT_or_future_association_authority": False, "no_nearest_frame_or_GT_candidate_substitution": True,
        "all24_required_before_final_summary": True, "CPU_threads": 1})


def direct_repair_opportunities(rows):
    direct, future = defaultdict(set), defaultdict(set)
    for row in rows:
        labels = row["offline_supervision_labels"]
        window = labels["future"]["H100"]
        if (row["branch"] in ("KEEP", "DELAYED_CHALLENGER") or not window["complete"]
                or window["benefit_label"] is not True or window["N01"] <= window["N10"]
                or labels["H100_any_harm_including_current_t_label"]):
            continue
        key = (row["episode_uid"], row["frame"])
        uid = row["action"].get("candidate_uid")
        if uid is not None and labels["current_t"]["outcome"] == "TARGET":
            future[key].add(uid)
            if (labels["current_t"]["N01"] == 1 and labels["current_t"]["outcome"] == "TARGET"
                    and row["frame"] in labels["effective_direct_action_frames"]):
                direct[key].add(uid)
    return direct, future


def score(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    protocol = read_json(PROTOCOL)
    assert protocol["source_sha256"] == {p: sha256(ROOT / p) for p in CODE}
    source_path = OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json")
    source = read_json(source_path)
    groups = []
    for ref in source["episodes"]:
        assert sha256(ref["path"]) == ref["sha256"]
        receipt = read_json(ref["path"])
        assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
        for row in read_zstd_jsonl(Path(receipt["artifact"]["path"])):
            if row["record_type"] == "EXACT_CF_CURRENT_AXIS":
                groups.append(row)
    assert len(groups) == source["counts"].get("exact_CF_current_axis_frames", 0)
    storage(32 << 20)
    destination = ASSETS / "exact_cf_verifier_v1/predictions" / (sequence + ".jsonl.zst")
    with runtime_file_guard(), writer(destination) as emit:
        for model in protocol["models"]:
            assert sha256(model["checkpoint_path"]) == model["checkpoint_sha256"]
            predictor = FreshOpenSetPredictor(model["checkpoint_path"])
            for group in groups:
                pred = predictor.predict_axis(group["axis"])
                chosen = choose(group["axis"], pred, predictor.selection)
                emit({"model": model["model"], "sequence": sequence, "episode_uid": group["episode_uid"],
                      "frame": group["frame"], "selected_current_claim": chosen,
                      "current_axis_predictions": [{"candidate_uid": r["candidate_uid"], **p} for r, p in zip(group["axis"], pred, strict=True)],
                      "association_authority": False, "runtime_GT_or_future_labels_used": False})
    write_json("availability/exact_cf_v1/predictions/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "protocol_sha256": sha256(PROTOCOL),
        "source_sequence_sha256": sha256(source_path), "source_sha256": protocol["source_sha256"],
        "actual_exact_frame_groups": len(groups), "actual_models": 12,
        "all_model_scores_sealed_before_offline_repair_join": True,
        "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size}})


def report(sequence):
    development_sequence(sequence)
    protocol = read_json(PROTOCOL)
    assert protocol["source_sha256"] == {p: sha256(ROOT / p) for p in CODE}
    scored_path = OUT / "availability/exact_cf_v1/predictions" / (sequence + ".json")
    scored = read_json(scored_path)
    assert scored["protocol_sha256"] == sha256(PROTOCOL) and sha256(scored["artifact"]["path"]) == scored["artifact"]["sha256"]
    label_path = OUT / "on_policy/joint_state_v1/supervision" / (sequence + ".json")
    labels = read_json(label_path)
    assert sha256(labels["artifact"]["path"]) == labels["artifact"]["sha256"]
    groups = {(r["episode_uid"], r["frame"]): r for r in read_zstd_jsonl(Path(labels["artifact"]["path"])) if r["record_type"] == "EXACT_CF_CURRENT_AXIS"}
    cf_path = OUT / "events/label_audit" / (sequence + ".json")
    cf_labels = read_json(cf_path)
    assert sha256(cf_labels["artifact"]["path"]) == cf_labels["artifact"]["sha256"]
    cf_rows = read_zstd_jsonl(Path(cf_labels["artifact"]["path"]))
    direct, future = direct_repair_opportunities(cf_rows)
    models = {r["model"]: Counter() for r in protocol["models"]}
    for row in read_zstd_jsonl(Path(scored["artifact"]["path"])):
        key = (row["episode_uid"], row["frame"])
        group = groups[key]
        claim = row["selected_current_claim"]
        assert [p["candidate_uid"] for p in row["current_axis_predictions"]] == [r["candidate_uid"] for r in group["axis"]]
        chosen = group["axis"][claim["axis_index"]]
        assert chosen["candidate_uid"] == claim["candidate_uid"]
        outcome = chosen["current_outcome"]
        count = models[row["model"]]
        count["exact_CF_event_frame_groups"] += 1
        count["positive_candidate_available"] += group["target_candidate_available_label"]
        count["accepted_identity_claims"] += claim["accepted"] and claim["candidate_uid"] is not None
        count["accepted_correct_TARGET"] += claim["accepted"] and outcome == "TARGET"
        count["accepted_verified_OTHER"] += claim["accepted"] and outcome == "VERIFIED_OTHER"
        count["accepted_UNKNOWN"] += claim["accepted"] and outcome == "UNKNOWN_UNMATCHED"
        count["accepted_incorrect_NONE"] += claim["accepted"] and outcome == "INCORRECT_NONE"
        count["accepted_correct_NONE"] += claim["accepted"] and outcome == "CORRECT_NONE"
        count["false_presence_no_positive_candidate"] += claim["accepted"] and claim["candidate_uid"] is not None and not group["target_candidate_available_label"]
        if key in direct:
            count["actual_safe_direct_repair_event_frames_NOT_independent_onsets"] += 1
            retained = claim["accepted"] and claim["candidate_uid"] in direct[key]
            count["safe_direct_repair_identity_claim_retained_NOT_executed_MOT_correction"] += retained
            count["safe_direct_repair_identity_claim_missed"] += not retained
        if key in future:
            count["actual_safe_future_target_improvement_event_frames_NOT_independent_onsets"] += 1
            count["safe_future_improvement_identity_claim_retained"] += claim["accepted"] and claim["candidate_uid"] in future[key]
    assert all(c["exact_CF_event_frame_groups"] == len(groups) for c in models.values())
    write_json("availability/exact_cf_v1/results/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "protocol_sha256": sha256(PROTOCOL), "models": {k: dict(v) for k, v in models.items()},
        "score_receipt_sha256": sha256(scored_path), "actual_CF_labels_sha256": sha256(cf_path), "exact_axis_labels_sha256": sha256(label_path),
        "direct_repair_event_frame_opportunities": len(direct), "future_target_improvement_event_frame_opportunities": len(future),
        "source_frames_not_nearest32_imputation": True, "association_actions_executed": 0,
        "correlated_event_frames_NOT_independent_onsets": True, "zero_claims_not_safety_PASS": True,
        "not_G1_G2_generalization_or_scientific_closure": True})
    append_log("M8_EXACT_CF_CURRENT_VERIFIER_MISSED_REPAIR_DIAGNOSTIC", sequence=sequence, direct_event_frames=len(direct), future_event_frames=len(future))
    print({"exact_CF_verifier": sequence, "groups": len(groups), "models": 12, "actual_direct_repair_event_frames": len(direct)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "score", "report"])
    parser.add_argument("--sequence")
    args = parser.parse_args()
    freeze() if args.action == "freeze" else {"score": score, "report": report}[args.action](args.sequence)
