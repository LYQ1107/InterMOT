"""Known-source error-unit supervision diagnostic; no fits or new rollouts.

Does not replace frozen L1 labels, implement natural/online root independence,
authorize G1, or put root IDs/GT into runtime features. All original controlled
cases remain dispositions; earliest attributable probe chosen before repairs.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, TRAIN, GOAL, read_json, write_json, sha256, preregistration, storage
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r2_baseline import checked_frames
from scripts import n72r21r2_joint_state_curriculum as original
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.one_click.event_authority_learning import encode_runtime_input
from sam3_intermot.evaluation.controlled_error_units import source_error_unit, attributable_probe, current_repair_label, outcome, stable_hash
from sam3_intermot.evaluation.safe_intervention_events import assignment_map

SELF = "scripts/n72r21r2_controlled_error_units_v1.py"
CODE = (SELF, "sam3_intermot/evaluation/controlled_error_units.py", "scripts/n72r21r2_joint_state_curriculum.py",
        "sam3_intermot/evaluation/causal_identity_events.py", "sam3_intermot/evaluation/event_causal_labels.py")
CANONICAL = OUT / "on_policy/ALL24_CONTROLLED_SOURCE_ACTUAL_CANONICAL_READBACK_V1.json"
PROTOCOL = OUT / "events/CONTROLLED_ERROR_UNIT_DIAGNOSTIC_PROTOCOL_V1.json"
PREFIX = "events/controlled_error_units_v1"


def freeze():
    assert not list((OUT / PREFIX).glob("*.json"))
    p = preregistration()
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "design": "RETROSPECTIVE_CONTROLLED_SINGLE_SOURCE_ERROR_UNIT_DIAGNOSTIC_NOT_NATURAL_EVENT_L1_REPLACEMENT",
        "source_sha256": {n: sha256(ROOT / n) for n in CODE}, "canonical_readback_sha256": sha256(CANONICAL),
        "source_protocol_sha256": sha256(original.PROTOCOL), "split": {k: p["split"][k] for k in ("fit", "inner")},
        "source_error_unit": "One actual effective non-KEEP source action, exact equal tensor prestate with paired KEEP, TARGET -> verified OTHER/NONE at that frame. Unknown/reference errors/missing tensor pair remain unproven dispositions.",
        "reference": "Prefer actual controlled KEEP prefix at same source frame; otherwise exact original source KEEP tape when its tensor prestate is available. Never impute/nearest-frame prestate.",
        "unit_aliases": "Same video/anonymous identity/source-action frame/full core prestate/action configuration is one controlled unit. Source-case names, later offsets and propagated frames do not create new units. Videos remain independent inference clusters; distinct controlled units are not natural/population-independent roots.",
        "probe_selection": "Earliest original probe after that source error where own contaminated KEEP is verified OTHER/NONE and unmodified C0 is TARGET. Source/case availability and current KEEP/C0 only; never repair scores/GT effects/future reward/best offset.",
        "repair_label": "All distinct original current repair arms at the selected probe; actual current TARGET and effective N01 define correction. UNKNOWN yields null, duplicate trajectory aliases excluded from training. Current harm and H100 effects separate, not opaque event reward.",
        "future_components": "Retain existing sealed current/future original components as separate diagnostic labels; no new threshold, feature, weight, rollout or optimization.",
        "required_before_GT": "All24 canonical runtime/label/artifact and code SHA rechecked before any raw truth in this diagnostic.",
        "runtime_features": "Exact original nested runtime_features, no root/source IDs or GT added to feature axis.",
        "independent_natural_L1_implemented": False, "actual_model_generated_on_policy": False,
        "no_train_or_optimization_authorized_by_diagnostic": True, "next_association_stage_authorized": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "scientific_goal_complete": False})


def verified_inputs():
    ev = Evidence(); p = ev.json(PROTOCOL); ev.code(p["source_sha256"])
    assert p["goal"] == GOAL and p["frozen"]
    assert p["canonical_readback_sha256"] == ev.digest(CANONICAL)
    canonical = ev.json(CANONICAL)
    assert canonical["actual_verified_sequences"] == 24
    assert canonical["required_registered_sequences"] == p["split"]["fit"] + p["split"]["inner"]
    for name, digest in canonical["evidence_sha256"].items():
        assert ev.digest(name) == digest
    assert p["source_protocol_sha256"] == ev.digest(original.PROTOCOL)
    ev.code(ev.json(original.PROTOCOL)["source_sha256"])
    assert p["split"] == {k: preregistration()["split"][k] for k in ("fit", "inner")}
    return ev, p


def sequence(sequence, ev, protocol):
    role = "FIT" if sequence in protocol["split"]["fit"] else "INNER"
    runtime = ev.json(OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json"))
    supervision = ev.json(OUT / "on_policy/joint_state_v1/supervision" / (sequence + ".json"))
    plan = ev.json(OUT / "on_policy/joint_state_v1/plans" / (sequence + ".json"))
    assert runtime["protocol_sha256"] == supervision["protocol_sha256"] == ev.digest(original.PROTOCOL)
    assert runtime["plan_sha256"] == ev.digest(OUT / "on_policy/joint_state_v1/plans" / (sequence + ".json"))
    ev.artifact(supervision["artifact"])
    old_labels = [r for r in read_zstd_jsonl(Path(supervision["artifact"]["path"])) if r["record_type"] == "NESTED_ARM"]
    labels = {(r["episode_uid"], r["source_case"], r["source_frame"], r["frame"], r["branch"]): r for r in old_labels}
    assert len(labels) == len(old_labels)
    # Full24 canonical seals were verified by caller before first raw GT.
    truth = ev.json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert ev.digest(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"] == supervision["GT_sha256"]
    frames, index_sha = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {payload["frame"]: strict_candidate_matching(rows, gt.get(payload["frame"], [])) for payload, rows in frames}
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    dispositions, groups, records, repair_inputs = [], {}, [], []
    for episode_ref in runtime["episodes"]:
        ev.artifact(episode_ref); seal = ev.json(episode_ref["path"]); ev.artifact(seal["artifact"])
        assert seal["candidate_index_sha256"] == index_sha and seal["actual_runtime_GT_file_guard"]
        assert seal["source_and_nested_clone_tensor_actor_history_isolation"] and not seal["actual_model_generated_on_policy"]
        episode = seal["event"]["episode_uid"]; target = targets[episode]
        anonymous = "anonymous__" + stable_hash({"sequence": sequence, "identity": target})
        source_plan = next(e for e in plan["episodes"] if e["episode_uid"] == episode)
        raw = read_zstd_jsonl(Path(seal["artifact"]["path"]))
        parents = {(r["source_case"], r["source_frame"]): r for r in raw if r["record_type"] == "SOURCE_PREFIX"}
        arms = defaultdict(list)
        for r in raw:
            if r["record_type"] == "NESTED_ARM": arms[r["source_case"], r["source_frame"], r["frame"]].append(r)
        base_seal = ev.json(OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (episode + ".json"))
        base_ref = next(a for a in base_seal["artifacts"] if a["kind"] == "trace"); ev.artifact(base_ref)
        baseline = read_zstd_jsonl(Path(base_ref["path"]))
        for case in source_plan["cases"]:
            if "source_frame" not in case:
                dispositions.append({"episode_uid": episode, "source_case": case["source_case"], "status": case["status"], "unit_uid": None})
                continue
            key = case["source_case"], case["source_frame"]
            parent = parents[key]
            keep_parent = parents.get(("KEEP", case["source_frame"]))
            if keep_parent is not None:
                reference = keep_parent["own_actual_source_prefix"]
            else:
                ev.artifact(case["source_KEEP_trace"])
                reference = read_zstd_jsonl(Path(case["source_KEEP_trace"]["path"]))
            unit = source_error_unit(parent, reference, matched, target, {"sequence": sequence, "anonymous_identity_scope": anonymous})
            unit.update(episode_uid=episode, role=role, runtime_artifact_sha256=seal["artifact"]["sha256"])
            probes = []
            for frame in sorted(f for c, s, f in arms if (c, s) == key):
                same = arms[key[0], key[1], frame]
                keep = next(a for a in same if a["branch"] == "KEEP")["actual_full_global_trajectory"][0]
                status = attributable_probe(unit, keep, baseline[frame], matched[frame], target, frame)
                probes.append({"frame": frame, "status": status})
            unit["all_original_probe_dispositions"] = probes
            selected = next((r["frame"] for r in probes if r["status"] == "KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE"), None)
            unit["selected_probe_before_repair_effects"] = selected
            dispositions.append(unit)
            if unit["unit_uid"] is None:
                continue
            group = groups.setdefault(unit["unit_uid"], {"unit_key": unit["unit_key"], "source_case_aliases": [], "probes": []})
            assert group["unit_key"] == unit["unit_key"]
            group["source_case_aliases"].append({"episode_uid": episode, "source_case": case["source_case"]})
            if selected is None:
                continue
            group["probes"].append({"episode_uid": episode, "source_case": key[0], "frame": selected})
            repair_inputs.append((unit, episode, target, key, selected, arms[key[0], key[1], selected]))
    # Resolve all aliases BEFORE looking at any repair effect. One earliest
    # eligible probe per causal unit, not one per source-case name/offset.
    for group in groups.values():
        group["selected_probe_before_all_repair_effects"] = min((r["frame"] for r in group["probes"]), default=None)
    for unit, episode, target, key, selected, same_arms in repair_inputs:
        if selected != groups[unit["unit_uid"]]["selected_probe_before_all_repair_effects"]:
            continue
        for arm in same_arms:
            row = labels[episode, key[0], key[1], selected, arm["branch"]]
            assert row["runtime_features"] == arm["runtime_features"]
            encode_runtime_input(row["runtime_features"], arm["branch"])
            current = arm["actual_full_global_trajectory"][0]
            keep = next(a for a in same_arms if a["branch"] == "KEEP")["actual_full_global_trajectory"][0]
            actual_outcome = outcome(current, matched[selected], target, unit["target_public_id"])
            effective = assignment_map(current) != assignment_map(keep)
            diagnostic = current_repair_label("KNOWN_SOURCE_ERROR_PRESENT_AT_PROBE", row["offline_supervision_labels"], actual_outcome, effective,
                duplicate=row["offline_supervision_labels"]["same_executed_action_configuration_as"] is not None)
            records.append({"sequence": sequence, "role": role, "episode_uid": episode, "source_case": key[0],
                "source_frame": key[1], "frame": selected, "branch": arm["branch"], "unit_uid": unit["unit_uid"],
                "runtime_features": row["runtime_features"], "action": row["action"],
                "offline_supervision_labels": {"controlled_event_correction": diagnostic,
                    "original_current": row["offline_supervision_labels"]["current_t"],
                    "original_future": row["offline_supervision_labels"]["future"],
                    "natural_G1_root_or_deployment_success_proved": False},
                "original_runtime_artifact_sha256": row["runtime_artifact_sha256"]})
    # Source aliases cannot create extra training arms for the same unit/probe.
    unique = {}; duplicates = 0
    for record in records:
        key = record["unit_uid"], record["frame"], stable_hash(record["action"])
        if key in unique:
            old, new = unique[key]["offline_supervision_labels"], record["offline_supervision_labels"]
            assert old["original_current"] == new["original_current"] and old["original_future"] == new["original_future"]
            if old["controlled_event_correction"] is None and new["controlled_event_correction"] is not None:
                unique[key] = record
            elif old["controlled_event_correction"] is not None and new["controlled_event_correction"] is not None:
                assert old["controlled_event_correction"] == new["controlled_event_correction"]
            duplicates += 1
        else: unique[key] = record
    result = {"stage": "N72R21R2", "goal": GOAL, "sequence": sequence, "role": role, "protocol_sha256": ev.digest(PROTOCOL),
        "status": "CONTROLLED_CAUSAL_ERROR_UNIT_DIAGNOSTIC_NOT_NATURAL_L1_REPLACEMENT",
        "source_case_dispositions": dispositions, "controlled_units": groups, "diagnostic_rows": list(unique.values()),
        "merged_unit_action_alias_rows": duplicates, "original_labels_or_models_overwritten": False,
        "independent_natural_L1_implemented": False, "new_optimization_or_runtime": False}
    path = write_json(PREFIX + "/" + sequence + ".json", result)
    count = Counter(d["status"] for d in dispositions)
    positives = sum(r["offline_supervision_labels"]["controlled_event_correction"]["current_error_unit_corrected"] for r in unique.values()
                    if r["offline_supervision_labels"]["controlled_event_correction"] is not None)
    print({"actual_controlled_error_unit_sequence": sequence, "controlled_units": len(groups), "current_repair_arms_not_independent_events": positives}, flush=True)
    return {"sequence": sequence, "role": role, "path": str(path), "sha256": sha256(path),
            "controlled_units_NOT_natural_independent_roots": len(groups), "selected_probe_arms": len(unique),
            "current_repair_arms_NOT_independent_roots": positives, "source_dispositions": dict(count)}


def run():
    ev, p = verified_inputs(); storage(64 << 20)
    rows = [sequence(seq, ev, p) for seq in p["split"]["fit"] + p["split"]["inner"]]
    write_json(PREFIX + "/ALL24_ACTUAL_DIAGNOSTIC.json", {"stage": "N72R21R2", "goal": GOAL,
        "protocol_sha256": ev.digest(PROTOCOL), "sequences": rows, "evidence_sha256": ev.refs,
        "all24_original_sources_retained": True, "population_inference_clusters": "VIDEO_NOT_SOURCE_UNITS_PROBES_OR_ARMS",
        "independent_natural_L1_implemented": False, "actual_model_generated_on_policy": False,
        "new_fits_rollouts_or_original_label_edits": False, "full_G1_G2_or_scientific_goal_complete": False,
        "next_association_stage_authorized": False, "CONFIRM_VAL_TEST_SOT_accessed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    freeze() if parser.parse_args().action == "freeze" else run()
