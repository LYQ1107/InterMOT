"""Read all original STATE INNER cells, including actual pinned CSVs.

No replay, fit, selection write, policy patch or confirmation access. A complete
INNER audit is not a complete96-cell group or a deployed on-policy experiment.
"""
import argparse
from scripts.n72r21r2_common import ROOT, OUT, GOAL, write_json, preregistration
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts import n72r21r2_state_policy as state
from sam3_intermot.evaluation.learned_policy_evidence import METRICS, choose_inner_point


SELF = "scripts/n72r21r2_state_inner_readback_v1.py"


def validate_state_binding(fit, runtime, result, mode, seed, objective):
    """The replay family is a source mode, not the actual MLP architecture."""
    if type(seed) is not int or mode not in ("TREATMENT_STATE", "MIXED_STATE"):
        raise ValueError("Only actual trained state-source modes and exact seeds")
    if fit["mode"] != mode or fit["family"] != "SMALL_MLP":
        raise ValueError("Actual mode and architecture must remain separate")
    if fit["status"] != "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE":
        raise ValueError("A prefit failure cannot substitute for an actual model")
    if fit["actual_model_generated_on_policy"] or fit["association_authority"]:
        raise ValueError("Controlled sources cannot acquire on-policy authority")
    if fit["nonzero_gradient_steps"] <= 0 or fit["changed_weight_elements"] <= 0:
        raise ValueError("An unchanged or unoptimized model cannot supply evidence")
    for record in (fit, runtime, result):
        if record["seed"] != seed or record["objective"] != objective:
            raise ValueError("Wrong seed or objective")
    for record in (runtime, result):
        if record["family"] != mode or record["actual_architecture_family"] != "SMALL_MLP":
            raise ValueError("Wrong replay mode or actual architecture")
        if not record["controlled_training_state_source_contrast"] or record["actual_model_generated_on_policy_training"]:
            raise ValueError("Controlled state replay is not actual-model on-policy training")


def checked_cell(mode, seed, point, sequence, ev):
    p = ev.json(state.PROTOCOL)
    assert mode in p["modes"] and seed in p["seeds"] and point in p["points"]
    assert sequence in p["split"]["inner"], "Only the full original INNER grid"
    ev.code(p["source_sha256"])
    objective = p["objective"]
    uid = "STATE__" + mode + "__" + objective + "__seed" + str(seed) + "__" + point
    base = OUT / state.PREFIX
    runtime_path = base / "runtime" / uid / (sequence + ".json")
    onset_path = base / "onset_runtime" / uid / (sequence + ".json")
    result = ev.json(base / "results" / uid / (sequence + ".json"))
    runtime, onsets = ev.json(runtime_path), ev.json(onset_path)
    fit_path = OUT / "on_policy/state_source_fits_v1" / (mode + "__seed" + str(seed) + ".json")
    fit = ev.json(fit_path)
    validate_state_binding(fit, runtime, result, mode, seed, objective)
    ev.code(fit["source_sha256"])
    assert fit["source_sha256"] == ev.json(state.FIT_PROTOCOL)["source_sha256"]
    assert fit["protocol_sha256"] == p["state_fit_protocol_sha256"] == ev.digest(state.FIT_PROTOCOL)
    assert p["inherited_MAIN_replay_protocol_sha256"] == ev.digest(OUT / "mot/MAIN_POLICY_PROTOCOL_V1.json")
    assert runtime["fit_record_sha256"] == ev.digest(fit_path)
    assert all(r["protocol_sha256"] == ev.digest(state.PROTOCOL) for r in (runtime, onsets, result))
    assert runtime["source_freeze"] == p["source_sha256"]
    # Original onset seals bind the point through experiment_uid, not a
    # separate point field. Do not impose a retrospective seal schema.
    assert all(r["experiment_uid"] == uid and r["sequence"] == sequence for r in (runtime, onsets, result))
    assert runtime["point"] == result["point"] == point
    assert result["runtime_seal_sha256"] == onsets["runtime_seal_sha256"] == ev.digest(runtime_path)
    assert result["onset_runtime_seal_sha256"] == ev.digest(onset_path)
    assert result["checkpoint_sha256"] == runtime["checkpoint_sha256"] == fit["checkpoint_sha256"] == ev.digest(fit["checkpoint_path"])
    assert runtime["actual_GT_file_guard"] and not runtime["runtime_future_GT_read"]
    assert runtime["sole_click_full_global_original_hard_negatives"] and onsets["GT_guard_actual"]
    init_path = OUT / "data/initialization" / (sequence + ".json")
    init = ev.json(init_path)
    assert runtime["initialization_sha256"] == ev.digest(init_path)
    ev.artifact({"path": init["anchor_path"], "sha256": init["anchor_sha256"]})
    integrity = ev.json(OUT / "data/candidate_integrity" / (sequence + ".json"))
    assert runtime["candidate_index_sha256"] == init["candidate_index_sha256"] == integrity["index_sha256"] == ev.digest(integrity["index_path"])
    assert runtime["density"]["whole_video_density"] == integrity["whole_video_density"]
    valid = {e["episode_uid"]: e for e in init["inputs"] if not e["initialization_failure"]}
    assert {e["episode_uid"] for e in runtime["episodes"]} == {e["episode_uid"] for e in onsets["entries"]} == set(valid)
    assert result["valid_initializations"] == len(valid)
    assert result["initialization_failures_not_replaced"] == sum(bool(e["initialization_failure"]) for e in init["inputs"])
    labels = {e["episode_uid"]: e["onsets"] for e in result["own_prefix_one_shot_labels"]}
    assert set(labels) == set(valid)
    baseline = ev.json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
    for episode, onset in zip(runtime["episodes"], onsets["entries"], strict=True):
        key = episode["episode_uid"]
        assert key == onset["episode_uid"] and episode["event"] == valid[key] and episode["memory_writes"] == 0
        assert {r["frame"] for r in labels[key]} == set(onset["effective_decision_frames"]) == {r["frame"] for r in onset["branches"]}
        for ref in episode["artifacts"]:
            ev.artifact(ref)
        for branch in onset["branches"]:
            assert set(branch["artifacts"]) == {"OWN_KEEP", "ACTUAL_CURRENT_THEN_KEEP"}
            assert branch["complete_learned_prestate_sha256"]
            for ref in branch["artifacts"].values():
                ev.artifact(ref)
        if episode["effective_direct_decisions_NOT_independent_onsets"] == 0:
            c0 = ev.json(OUT / "mot/baseline_runtime/CLICK_C0" / (key + ".json"))
            old = next(a for a in c0["artifacts"] if a["kind"] == "trajectory")
            new = next(a for a in episode["artifacts"] if a["kind"] == "trajectory")
            ev.artifact(old)
            assert new["sha256"] == old["sha256"]
            assert episode["zero_action_prefix_tensor_C0_AA_frames"] == valid[key]["frames"]
            old_metrics = baseline["CLICK_C0__click" + str(valid[key]["slot"])]
            assert all(result["all_nine_metrics"][episode["tracker_name"]][k] == old_metrics[k] for k in METRICS)
    ev.pinned(result["actual_pinned_full_original_video_TrackEval"], result["all_nine_metrics"],
              [e["tracker_name"] for e in runtime["episodes"]], sequence)
    return result


def verify_selection(cells, selected, protocol):
    for point in protocol["points"]:
        actual = [(r["seed"], r["sequence"]) for r in cells[point]]
        expected = {(seed, sequence) for seed in protocol["seeds"] for sequence in protocol["split"]["inner"]}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("No duplicate, best-seed or ready-video subset")
    calculated = choose_inner_point(cells, protocol["seeds"], protocol["split"]["inner"])
    if any(selected[k] != value for k, value in calculated.items()):
        raise ValueError("Original all-seed shared selection did not reproduce")
    return calculated


def run(mode):
    ev = Evidence()
    p = ev.json(state.PROTOCOL)
    assert p["split"]["inner"] == preregistration()["split"]["inner"]
    selection_path = OUT / state.PREFIX / "selections" / (mode + "__" + p["objective"] + ".json")
    selected = ev.json(selection_path)
    assert selected["protocol_sha256"] == ev.digest(state.PROTOCOL)
    cells = {point: [checked_cell(mode, seed, point, sequence, ev) for seed in p["seeds"]
                    for sequence in p["split"]["inner"]] for point in p["points"]}
    expected_refs = []
    for point in p["points"]:
        for cell in cells[point]:
            path = OUT / state.PREFIX / "results" / cell["experiment_uid"] / (cell["sequence"] + ".json")
            expected_refs.append({"path": str(path), "sha256": ev.digest(path)})
    assert selected["actual_INNER_result_refs"] == expected_refs
    calculated = verify_selection(cells, selected, p)
    negative = all(r["effective_decisions"] == 0 for r in calculated["candidates"])
    ev.digest(ROOT / SELF)
    relative = "mot/STATE_" + mode + "_ALL48_INNER_READBACK_V1.json"
    report = {"stage": "N72R21R2", "goal": GOAL, "mode": mode,
        "actual_architecture_family": "SMALL_MLP", "checked_cells": len(expected_refs),
        "status": "COMPLETE_ORIGINAL_INNER_GRID_NOT_FULL96_GROUP_OR_M7_CLOSURE",
        "selection": calculated, "evidence_sha256": ev.refs,
        "all_original_seeds_points_videos_retained": True,
        "all_zero_actions_proved_C0_trajectory_tensor_and_nine_metric_equality": negative,
        "G1_necessary_screen": "FAIL_ZERO_EFFECTIVE_ACTIONS" if negative else "SEPARATE_INDEPENDENT_ROOT_AUDIT_REQUIRED",
        "G2_necessary_screen": "FAIL_NO_POSITIVE_MOT_GAIN" if negative else "SEPARATE_FULL_GROUP_GATES_REQUIRED",
        "controlled_source_is_actual_model_generated_on_policy": False,
        "zero_action_safety_PASS": False, "selected_FIT16_still_required": True,
        "full96_group_complete": False, "scientific_goal_complete": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "next_association_stage_authorized": False}
    write_json(relative, report)
    print({"checked_STATE_INNER_cells": len(expected_refs), "mode": mode,
           "zero_actions": negative, "full_group_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("TREATMENT_STATE", "MIXED_STATE"))
    run(parser.parse_args().mode)
