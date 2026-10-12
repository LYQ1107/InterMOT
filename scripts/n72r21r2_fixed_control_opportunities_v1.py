"""Fixed-rule retrospective exact-CF reanalysis, not new rule selection.

Finish M8's missed-repair comparison for the five already calibrated simple
rules alongside all12 sealed learned verifiers and both pure controls.
No tracker action, optimization, threshold/grid search or new candidate.
"""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, utcnow, storage, preregistration, update_status
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r2_joint_state_curriculum import writer
from scripts.n72r21r2_exact_cf_verifier import direct_repair_opportunities
from sam3_intermot.one_click.open_set_verifier import simple_predictions, choose
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.fixed_identity_opportunities import check_runtime_axis, axis_uids, count_choice, rates, video_interval

PREFIX = "availability/fixed_control_cf_v1"
PROTOCOL = OUT / "availability/FIXED_CONTROL_CF_REANALYSIS_PROTOCOL_V1.json"
SIMPLE = OUT / "availability/SIMPLE_CURRENT_AXIS_CALIBRATION_V1.json"
CONTROLS = ("RAW_SCORE", "CALIBRATED_SCORE", "CANDIDATE_MARGIN", "BASE_AGREEMENT", "TEMPORAL_CONFIRMATION")
CODE = ("scripts/n72r21r2_fixed_control_opportunities_v1.py",
        "sam3_intermot/evaluation/fixed_identity_opportunities.py",
        "sam3_intermot/one_click/open_set_verifier.py",
        "sam3_intermot/one_click/runtime_file_guard.py",
        "scripts/n72r21r2_exact_cf_verifier.py")


def prepare():
    if PROTOCOL.exists():
        raise FileExistsError("Preserve the original reanalysis preparation")
    storage(64 << 20)
    prior = read_json(SIMPLE)
    assert set(prior["controls"]) == set(CONTROLS)
    assert prior["protocol_sha256"] == sha256(OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json")
    write_json(str(PROTOCOL.relative_to(OUT)), {
        "stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "frozen": True,
        "analysis_type": "RETROSPECTIVE_REANALYSIS_OF_PREVIOUSLY_FROZEN_RULES_NOT_NEW_PROSPECTIVE_SELECTION",
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "split": preregistration()["split"], "original_simple_calibration_sha256": sha256(SIMPLE),
        "original_joint_source_protocol_sha256": sha256(OUT / "on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json"),
        "controls": {name: prior["controls"][name]["selection"] for name in CONTROLS},
        "no_new_diagnostic_point_or_threshold_search": True,
        "scores": "Exact already registered C0 current axes; all UIDs+NONE. Seal all24 new scores before offline labels. Reuse original score and selector implementations.",
        "opportunities": "Original complete safe H100 current-effective N01 TARGET branches, plus separately safe future-target-improvement branches; correlated click/frame opportunities, not independent onsets.",
        "cluster_unit": "Video; all3 learned seeds kept, family intervals average seeds inside each video",
        "bootstrap_seed": 730104, "bootstrap_repetitions": 2000,
        "association_authority": False, "new_model_or_dataset": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False})


def protocol():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["original_simple_calibration_sha256"] == sha256(SIMPLE)
    assert p["split"] == preregistration()["split"]
    return p


def score(sequence):
    p = protocol()
    assert sequence in p["split"]["fit"] + p["split"]["inner"]
    relative = PREFIX + "/scores/" + sequence + ".json"
    if (OUT / relative).exists():
        old = read_json(OUT / relative)
        assert old["protocol_sha256"] == sha256(PROTOCOL)
        assert sha256(old["artifact"]["path"]) == old["artifact"]["sha256"]
        assert old["source_sequence_sha256"] == sha256(OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json"))
        return
    storage(32 << 20)
    source_path = OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json")
    destination = ASSETS / "fixed_control_cf_v1" / (sequence + ".jsonl.zst")
    seen = set()
    with runtime_file_guard(), writer(destination) as emit:
        source = read_json(source_path)
        assert source["sequence"] == sequence and source["protocol_sha256"] == p["original_joint_source_protocol_sha256"]
        assert all(sha256(ROOT / name) == h for name, h in source["source_sha256"].items())
        for ref in source["episodes"]:
            assert sha256(ref["path"]) == ref["sha256"]
            receipt = read_json(ref["path"])
            assert receipt["actual_runtime_GT_file_guard"] and receipt["source_and_nested_clone_tensor_actor_history_isolation"]
            assert receipt["protocol_sha256"] == source["protocol_sha256"]
            assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
            for group in read_zstd_jsonl(Path(receipt["artifact"]["path"])):
                if group["record_type"] != "EXACT_CF_CURRENT_AXIS":
                    continue
                key = group["episode_uid"], group["frame"]
                assert key not in seen
                seen.add(key)
                check_runtime_axis(group["axis"])
                choices = {}
                for name, selection in p["controls"].items():
                    pred = simple_predictions(group["axis"], selection["simple_family"], selection["temperature"])
                    choices[name] = choose(group["axis"], pred, selection)
                emit({"sequence": sequence, "episode_uid": key[0], "frame": key[1],
                      "axis_uids": axis_uids(group["axis"]), "choices": choices,
                      "tensor_inclusive_C0_prestate_sha256": group["tensor_inclusive_C0_prestate_sha256"],
                      "association_authority": False, "runtime_GT_or_future_features_used": False})
    assert len(seen) == source["counts"].get("exact_CF_current_axis_frames", 0)
    write_json(relative, {"stage": "N72R21R2", "sequence": sequence, "protocol_sha256": sha256(PROTOCOL),
        "source_sequence_sha256": sha256(source_path), "current_groups": len(seen),
        "actual_GT_file_guard": True, "scores_sealed_before_offline_label_join": True,
        "empty_axes_NOT_identity_success": not seen,
        "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size}})
    print({"fixed_simple_exact_CF_scored": sequence, "groups": len(seen)}, flush=True)


def score_all():
    p = protocol()
    marker = PREFIX + "/owner.json"
    if (OUT / marker).exists():
        raise FileExistsError("Never duplicate a running or partial score owner")
    state = {"pid": os.getpid(), "active": None, "completed": [], "status": "ACTIVE_FIXED_RULE_REANALYSIS", "protocol_sha256": sha256(PROTOCOL)}
    write_json(marker, state)
    for sequence in p["split"]["fit"] + p["split"]["inner"]:
        state["active"] = sequence
        write_json(marker, state, mutable=True)
        score(sequence)
        state["completed"].append(sequence)
    state.update(active=None, status="COMPLETE_ALL24_FIXED_RULE_SCORE_SEALS_NOT_SCIENTIFIC_CLOSURE")
    write_json(marker, state, mutable=True)


def report():
    p = protocol()
    if (OUT / PREFIX / "ALL24_EXACT_FRAME_COMPARISON.json").exists():
        raise FileExistsError("Preserve completed reanalysis")
    sequences = p["split"]["fit"] + p["split"]["inner"]
    refs = {str(PROTOCOL): sha256(PROTOCOL), str(SIMPLE): sha256(SIMPLE)}
    def load(path):
        value = read_json(path)
        refs[str(path)] = sha256(path)
        if "artifact" in value:
            ref = value["artifact"]
            assert sha256(ref["path"]) == ref["sha256"]
            refs[ref["path"]] = ref["sha256"]
        return value
    score_seals = {s: load(OUT / PREFIX / "scores" / (s + ".json")) for s in sequences}
    assert all(r["scores_sealed_before_offline_label_join"] and r["protocol_sha256"] == sha256(PROTOCOL) for r in score_seals.values())
    learned_p = load(OUT / "availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json")
    pure_p = load(OUT / "availability/PURE_MARGIN_PROTOCOL_V1.json")
    pure_selected = load(OUT / "availability/PURE_MARGIN_SELECTED_POINTS_V1.json")
    assert all(sha256(ROOT / f) == h for f, h in learned_p["source_sha256"].items())
    assert all(sha256(ROOT / f) == h for f, h in pure_p["source_sha256"].items())
    strategies = list(CONTROLS) + [m["model"] for m in learned_p["models"]]
    strategies += [c + "__" + point for c in pure_p["controls"] for point in ("selected", "diagnostic")]
    counts = {name: {s: Counter() for s in sequences} for name in strategies}
    def join(sequence, truth, rows, name, choice_fn, uid_fn, direct, future):
        seen = set()
        for row in rows:
            key = row["episode_uid"], row["frame"]
            if key in seen or key not in truth:
                raise ValueError("No duplicate, missing or nearest-frame claim join")
            seen.add(key)
            counts[name][sequence].update(count_choice(truth[key], uid_fn(row), choice_fn(row), direct, future))
        assert seen == set(truth)
    for sequence in sequences:
        labels = load(OUT / "on_policy/joint_state_v1/supervision" / (sequence + ".json"))
        assert labels["protocol_sha256"] == p["original_joint_source_protocol_sha256"]
        groups = [r for r in read_zstd_jsonl(Path(labels["artifact"]["path"])) if r["record_type"] == "EXACT_CF_CURRENT_AXIS"]
        truth = {(g["episode_uid"], g["frame"]): g for g in groups}
        assert len(truth) == len(groups)
        cf = load(OUT / "events/label_audit" / (sequence + ".json"))
        direct, future = direct_repair_opportunities(read_zstd_jsonl(Path(cf["artifact"]["path"])))
        assert set(direct) <= set(truth) and set(future) <= set(truth)
        rows = read_zstd_jsonl(Path(score_seals[sequence]["artifact"]["path"]))
        assert all(r["tensor_inclusive_C0_prestate_sha256"] == truth[r["episode_uid"], r["frame"]]["tensor_inclusive_C0_prestate_sha256"] for r in rows)
        for name in CONTROLS:
            join(sequence, truth, rows, name, lambda r: r["choices"][name], lambda r: r["axis_uids"], direct, future)
        learned = load(OUT / "availability/exact_cf_v1/predictions" / (sequence + ".json"))
        assert learned["protocol_sha256"] == sha256(OUT / "availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json")
        assert learned["source_sequence_sha256"] == score_seals[sequence]["source_sequence_sha256"]
        predictions = read_zstd_jsonl(Path(learned["artifact"]["path"]))
        for model in learned_p["models"]:
            chosen = [r for r in predictions if r["model"] == model["model"]]
            join(sequence, truth, chosen, model["model"], lambda r: r["selected_current_claim"], lambda r: [a["candidate_uid"] for a in r["current_axis_predictions"]], direct, future)
        pure = load(OUT / "availability/pure_margin_v1/CF_predictions" / (sequence + ".json"))
        assert pure["protocol_sha256"] == sha256(OUT / "availability/PURE_MARGIN_PROTOCOL_V1.json")
        assert pure["selected_points_sha256"] == sha256(OUT / "availability/PURE_MARGIN_SELECTED_POINTS_V1.json")
        assert pure["source_sequence_sha256"] == score_seals[sequence]["source_sequence_sha256"]
        pure_rows = read_zstd_jsonl(Path(pure["artifact"]["path"]))
        for control in pure_p["controls"]:
            for point in ("selected", "diagnostic"):
                join(sequence, truth, pure_rows, control + "__" + point, lambda r: r["choices"][control][point], lambda r: r["axis_uids"], direct, future)
    results = {}
    for name, per_seq in counts.items():
        roles = {}
        for role in ("fit", "inner"):
            total = Counter()
            for sequence in p["split"][role]:
                total.update(per_seq[sequence])
            per_video = {s: {"counts": dict(per_seq[s]), "rates": rates(per_seq[s])} for s in p["split"][role]}
            roles[role] = {"counts": dict(total), "micro_rates": rates(total), "per_sequence": per_video,
                "video_macro_CI95": {k: video_interval([v["rates"][k] for v in per_video.values()]) for k in rates(total)}}
        results[name] = roles
    result = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "protocol_sha256": sha256(PROTOCOL),
        "status": "COMPLETE_ALL24_FIXED_RULE_CURRENT_IDENTITY_MISSED_REPAIR_DIAGNOSTIC_NOT_MOT_CLOSURE",
        "analysis_type": p["analysis_type"], "strategies": results, "evidence_sha256": refs,
        "original_all24_sequences_retained": sequences, "UNKNOWN_separate_from_VERIFIED_OTHER": True,
        "physical_absence_separate_from_no_valid_positive_candidate": True,
        "frames_NOT_independent_events_or_executed_corrections": True,
        "global_MOT_wrong_takeover_or_improvement": None, "association_authority": False,
        "scientific_goal_complete": False, "CONFIRM_VAL_TEST_SOT_accessed": False}
    path = write_json(PREFIX + "/ALL24_EXACT_FRAME_COMPARISON.json", result)
    update_status(latest_all24_fixed_identity_missed_opportunities={"path": str(path), "sha256": sha256(path)},
                  all_current_working_changes_published=False)
    print({"all24_compared_strategies": len(strategies), "result": str(path)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "score_all", "report"))
    {"prepare": prepare, "score_all": score_all, "report": report}[parser.parse_args().action]()
