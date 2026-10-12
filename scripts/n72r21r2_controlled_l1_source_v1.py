"""One fixed matched-source L1 target contrast; no old label overwrite."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, preregistration, storage
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts import n72r21r2_train_event_authority as original
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.event_authority_learning import encode_runtime_input, correlated_sequence_weights, fit_normalizer
from sam3_intermot.evaluation.controlled_unit_objective import CONDITIONS, paired_targets

SELF = "scripts/n72r21r2_controlled_l1_source_v1.py"
PROTOCOL = OUT / "objectives/CONTROLLED_UNIT_L1_MATCHED_CONTRAST_PROTOCOL_V1.json"
DATA = OUT / "objectives/controlled_unit_l1_contrast_v1/SOURCE.json"
INDEX = OUT / "events/controlled_error_units_v1/ALL24_ACTUAL_DIAGNOSTIC.json"
CANONICAL = OUT / "on_policy/ALL24_CONTROLLED_SOURCE_ACTUAL_CANONICAL_READBACK_V1.json"
CODE = (SELF, "scripts/n72r21r2_train_controlled_l1_v1.py",
    "sam3_intermot/evaluation/controlled_unit_objective.py",
    "sam3_intermot/one_click/event_authority_runtime.py", "sam3_intermot/one_click/intervention_features.py",
    "sam3_intermot/one_click/runtime_file_guard.py", *original.CODE)


def freeze():
    assert not DATA.exists()
    p = read_json(original.PROTOCOL)
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {n: sha256(ROOT / n) for n in CODE}, "original_fit_protocol_sha256": sha256(original.PROTOCOL),
        "controlled_unit_index_sha256": sha256(INDEX), "canonical_source_sha256": sha256(CANONICAL),
        "split": preregistration()["split"], "seeds": p["seeds"], "training": p["training"],
        "family": "SMALL_MLP", "objective_loader_identity": "L1_EVENT_CORRECTION", "conditions": list(CONDITIONS),
        "harm_weight": p["loss_harm_weights"]["L1_EVENT_CORRECTION"],
        "source": "All24 sealed controlled-unit earliest-probe distinct arms plus original source_case KEEP current-TARGET background, original duplicate/incomplete H100 excluded identically from both conditions. UNKNOWN/null is not negative. No future-outcome probe selection.",
        "weights": "Original equal video/click/group/event/distinct-arm hierarchy; controlled unit is one group/event, background retains original correlation group/event. Not population-independent roots.",
        "contrast": "Only benefit target changes: known current unit corrected vs original H100 benefit. Exactly same encoded inputs/past3/KEEP, row order, FIT-only mean/scale, weights, risk/value targets, architecture, optimizer settings and same-seed initialization.",
        "not_natural_L1_or_actual_model_generated_on_policy_or_G1": True,
        "new_bounded_fit_count": 6, "no_new_architecture_threshold_search_candidate_memory_matcher_or_association_change": True,
        "epoch_selection": "Whole original INNER source only; no optimizer/normalizer on INNER and no best seed choice. Teacher source, not own full-MOT qualification.",
        "MOT_rollout_or_next_stage_authorized_by_this_fit": False, "scientific_goal_complete": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False})


def run():
    ev = Evidence(); p = ev.json(PROTOCOL); ev.code(p["source_sha256"])
    assert p["frozen"] and p["goal"] == GOAL and p["split"] == preregistration()["split"]
    assert ev.digest(original.PROTOCOL) == p["original_fit_protocol_sha256"]
    assert ev.digest(INDEX) == p["controlled_unit_index_sha256"] and ev.digest(CANONICAL) == p["canonical_source_sha256"]
    index = ev.json(INDEX); canonical = ev.json(CANONICAL)
    assert len(index["sequences"]) == canonical["actual_verified_sequences"] == 24
    # Full source/code/artifact binding before opening the new target records.
    for report in (index, canonical):
        for path, digest in report["evidence_sha256"].items(): assert ev.digest(path) == digest
    required = p["split"]["fit"] + p["split"]["inner"]
    assert [r["sequence"] for r in index["sequences"]] == required
    for ref in index["sequences"]: ev.artifact(ref)
    storage(32 << 20)
    rows, excluded = [], Counter()
    for ref in index["sequences"]:
        seq = ref["sequence"]; role = "FIT" if seq in p["split"]["fit"] else "INNER"
        diagnostic = ev.json(ref["path"])
        seal = ev.json(OUT / "on_policy/joint_state_v1/supervision" / (seq + ".json")); ev.artifact(seal["artifact"])
        original_rows = [r for r in read_zstd_jsonl(Path(seal["artifact"]["path"])) if r["record_type"] == "NESTED_ARM"]
        lookup = {(r["episode_uid"], r["source_case"], r["source_frame"], r["frame"], r["branch"]): r for r in original_rows}
        assert len(lookup) == len(original_rows)
        for row in diagnostic["diagnostic_rows"]:
            assert row["sequence"] == seq and row["role"] == role
            unit = diagnostic["controlled_units"][row["unit_uid"]]
            assert row["frame"] == unit["selected_probe_before_all_repair_effects"]
            labels = row["offline_supervision_labels"]
            target = paired_targets(labels["original_current"], labels["original_future"], labels["controlled_event_correction"])
            if target is None: excluded[role + "/controlled_null_or_incomplete"] += 1; continue
            origin = (row["episode_uid"], row["source_case"], row["source_frame"], row["frame"])
            actual = lookup[origin + (row["branch"],)]
            assert actual["runtime_features"] == row["runtime_features"]
            keep_runtime = lookup[origin + ("KEEP",)]["runtime_features"]
            rows.append(pack(row, keep_runtime, target, "CONTROLLED_CURRENT_ERROR", row["unit_uid"], row["unit_uid"] + "__" + actual["event_uid"]))
        for row in original_rows:
            if row["source_case"] != "KEEP": continue
            assert row["sequence"] == seq and row["role"] == role
            labels = row["offline_supervision_labels"]
            if labels["same_executed_action_configuration_as"] is not None:
                excluded[role + "/background_duplicate"] += 1; continue
            if labels["current_t"]["KEEP_outcome"] != "TARGET":
                excluded[role + "/background_not_verified_current_correct"] += 1; continue
            target = paired_targets(labels["current_t"], labels["future"], None, correct_background=True)
            if target is None: excluded[role + "/background_incomplete"] += 1; continue
            origin = (row["episode_uid"], row["source_case"], row["source_frame"], row["frame"])
            rows.append(pack(row, lookup[origin + ("KEEP",)]["runtime_features"], target, "CURRENT_CORRECT_BACKGROUND", "BACKGROUND__" + row["correlation_group_not_proven_causal_origin"], row["event_uid"]))
        print({"matched_L1_source_sequence": seq, "rows_so_far_correlated": len(rows)}, flush=True)
    keys = [(r["sequence"], r["episode_uid"], r["event_uid"], r["branch"]) for r in rows]
    assert len(set(keys)) == len(rows), "No duplicate unit-action aliases"
    fit = [r for r in rows if r["role"] == "FIT"]; inner = [r for r in rows if r["role"] == "INNER"]
    assert fit and inner
    weights, iw = correlated_sequence_weights(fit), correlated_sequence_weights(inner)
    mean, scale = fit_normalizer(np.asarray([r["x"] for r in fit], np.float32), weights)
    summary = {}
    for role, values in [("FIT", fit), ("INNER", inner)]:
        summary[role] = {"rows_correlated": len(values), "source_counts": dict(Counter(r["source_kind"] for r in values)),
            "conditions": {c: {"benefit": sum(r["targets"][c]["benefit"] for r in values), "risk": sum(r["targets"][c]["risk"] for r in values)} for c in CONDITIONS},
            "benefit_labels_differ": sum(r["targets"][CONDITIONS[0]]["benefit"] != r["targets"][CONDITIONS[1]]["benefit"] for r in values)}
    assert all(summary["FIT"]["conditions"][c]["benefit"] > 0 and summary["FIT"]["conditions"][c]["risk"] > 0 for c in CONDITIONS)
    write_json(str(DATA.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "protocol_sha256": ev.digest(PROTOCOL),
        "rows": rows, "FIT_weights": weights.tolist(), "INNER_weights": iw.tolist(), "FIT_mean": mean.tolist(), "FIT_scale": scale.tolist(),
        "summary": summary, "excluded_not_negative": dict(excluded), "evidence_sha256": ev.refs,
        "independent_natural_error_roots_or_G1_proved": False, "no_original_source_models_labels_changed": True,
        "scientific_goal_complete": False, "next_association_stage_authorized": False})
    print({"matched_L1_source_complete": True, "summary": summary, "scientific_goal_complete": False}, flush=True)


def pack(row, keep_runtime, targets, kind, group, event):
    x, past = encode_runtime_input(row["runtime_features"], row["branch"])
    keep, keep_past = encode_runtime_input(keep_runtime, "KEEP")
    assert np.array_equal(past, keep_past), "Actual paired source history, not a hypothetical KEEP"
    return {k: row[k] for k in ("sequence", "episode_uid", "branch", "frame", "role")} | {
        "event_uid": event, "correlation_group_not_proven_causal_origin": group, "source_kind": kind,
        "x": x.tolist(), "past": past.tolist(), "keep": keep.tolist(), "targets": targets,
        "runtime_features": row["runtime_features"], "keep_runtime_features": keep_runtime}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    freeze() if parser.parse_args().action == "freeze" else run()
