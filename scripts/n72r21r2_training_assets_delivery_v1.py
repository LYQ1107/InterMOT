"""Required training-data / checkpoint deliverables from actual frozen fits.

Reconstruct source arrays and FIT-only normalization, hash every existing
registered epoch, and retain baseline no-fit / pilot / passive-head roles.
No optimizer, calibration, rollout, GT parser or deployment authority.
"""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, preregistration, storage
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts import n72r21r2_main_policy as main_policy
from scripts import n72r21r2_train_event_authority as main
from scripts import n72r21r2_train_joint_state as state
from scripts import n72r21r2_train_open_set as openset
from scripts import n72r21r2_train_memory_risk as risk
from scripts.n72r21r2_objective_delivery_v1 import summarize, input_signature
from sam3_intermot.evaluation.optimization_census import assess_fit, changed_elements
from sam3_intermot.one_click.event_authority_learning import correlated_sequence_weights, fit_normalizer
from sam3_intermot.one_click.fresh_open_set_learning import flatten_groups

SELF = "scripts/n72r21r2_training_assets_delivery_v1.py"
PROTOCOL = OUT / "training/TRAINING_ASSETS_DELIVERY_PROTOCOL_V1.json"
CODE = tuple(dict.fromkeys((SELF, "scripts/n72r21r2_objective_delivery_v1.py",
    "sam3_intermot/evaluation/optimization_census.py", *main.CODE, *state.CODE, *openset.CODE, *risk.CODE)))


def checkpoint_specs(record):
    if not record.get("checkpoint_path") or not record.get("checkpoint_sha256"):
        raise ValueError("Only actual optimized fits have checkpoint inventory")
    epochs = record["epochs"]
    if not epochs or any(type(e["epoch"]) is not int for e in epochs) or [e["epoch"] for e in epochs] != list(range(1, len(epochs) + 1)):
        raise ValueError("Complete original contiguous epoch history required")
    selected_epoch = record.get("selected_epoch", record.get("best_epoch"))
    if type(selected_epoch) is not int or not 1 <= selected_epoch <= len(epochs):
        raise ValueError("Original selected epoch must exist")
    entries = [(record["checkpoint_path"], record["checkpoint_sha256"], "SELECTED")]
    entries += [(e["checkpoint_path"], e["checkpoint_sha256"], "EPOCH") for e in epochs]
    initial = Path(record["checkpoint_path"]).parent / "initial.pt"
    entries.append((str(initial), record.get("initial_sha256"), "INITIAL"))
    by_path = {}
    for path, digest, role in entries:
        key = str(Path(path).resolve())
        if key in by_path and by_path[key]["expected_sha256"] != digest:
            raise ValueError("Conflicting expected byte hashes for one physical checkpoint")
        if key not in by_path:
            by_path[key] = {"expected_sha256": digest, "roles": []}
        by_path[key]["roles"].append(role)
    return by_path


def tensor_changes(saved, initial, record):
    if saved["seed"] != record["seed"] or saved["family"] != record["family"] or saved["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("Original selected schema/model/seed/protocol identity required")
    if saved["epoch"] != record.get("selected_epoch", record.get("best_epoch")):
        raise ValueError("Selected epoch differs")
    if set(saved["model"]) != set(initial):
        raise ValueError("Initial/selected tensor axes differ")
    change = 0
    for name, value in saved["model"].items():
        before = initial[name]
        if value.shape != before.shape or value.dtype != before.dtype or not torch.isfinite(value).all() or not torch.isfinite(before).all():
            raise ValueError("Exact finite initial/selected tensor axes required")
        change += int(torch.count_nonzero(value != before))
    if change <= 0 or change != changed_elements(record):
        raise ValueError("Actual tensor-change count does not match fit receipt")
    return change


def numeric_digest(array):
    a = np.asarray(array)
    if a.dtype not in (np.dtype("float32"), np.dtype("int64")) or not np.isfinite(a).all():
        raise ValueError("Explicit finite original float32/int64 axes required")
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def freeze():
    assert not (OUT / "training/TRAINING_DATA.json").exists() and not (OUT / "checkpoints/SHA256_MANIFEST.json").exists()
    refs = [main_policy.PROTOCOL, main.PROTOCOL, state.PROTOCOL, openset.PROTOCOL, risk.PROTOCOL,
            OUT / "audit/CHECKPOINT_LINEAGE.json", OUT / "objectives/ALL7_ACTUAL_MLP_SUPERVISION_READBACK_V1.json"]
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "design": "READ_ONLY_ACTUAL_REQUIRED_TRAINING_SOURCE_AND_CHECKPOINT_DELIVERY_NOT_NEW_FITS_OR_SCIENTIFIC_CLOSURE",
        "source_sha256": {n: sha256(ROOT / n) for n in CODE}, "references": {str(p): sha256(p) for p in refs},
        "expected_actual_fits": {"MAIN": 42, "PILOT_INTERNAL_FIT": 6, "STATE": 6, "PASSIVE_OPEN_SET": 12, "PASSIVE_MEMORY_RISK": 6},
        "expected_no_fit_state_dispositions": 3, "checkpoint_inventory": "All selected/initial/epoch files in exact fit receipts; equality with current stage training *.pt physical-file census. Three inherited foundation/identity weights separately verified.",
        "initial_unsealed_bytes": "Current tensor/byte verification is explicit; missing original initial_sha256 is never invented as an original byte seal.",
        "split": preregistration()["split"], "no_optimizer_calibration_rollout_or_raw_GT_parser": True,
        "scientific_goal_complete": False, "next_association_stage_authorized": False})


def run():
    storage(16 << 20); torch.set_num_threads(1)
    ev = Evidence(); p = ev.json(PROTOCOL); ev.code(p["source_sha256"])
    assert p["frozen"] and p["goal"] == GOAL and p["split"] == preregistration()["split"]
    for path, digest in p["references"].items(): assert ev.digest(path) == digest
    for path, digest in ev.json(OUT / "objectives/ALL7_ACTUAL_MLP_SUPERVISION_READBACK_V1.json")["evidence_sha256"].items():
        assert ev.digest(path) == digest
    policies = ev.json(main_policy.PROTOCOL); seeds = policies["seeds"]
    specifications = []
    for j in policies["jobs"]:
        uid = "MAIN__" + j["family"] + "__" + j["objective"] + "__seed" + str(j["seed"])
        specifications.append(("MAIN", OUT / "training/event_authority" / (uid + ".json"), main.PROTOCOL, j["objective"]))
    for family in ev.json(main.PROTOCOL)["pilot"]["families"]:
        for seed in seeds:
            uid = "PILOT__" + family + "__L3_HARM_AWARE__seed" + str(seed)
            specifications.append(("PILOT_INTERNAL_FIT", OUT / "training/event_authority" / (uid + ".json"), main.PROTOCOL, "L3_HARM_AWARE"))
    for mode in state.MODES:
        for seed in seeds:
            specifications.append(("STATE", OUT / "on_policy/state_source_fits_v1" / (mode + "__seed" + str(seed) + ".json"), state.PROTOCOL, mode))
    for study, folder, protocol in [("PASSIVE_OPEN_SET", "availability/current_axis_fits", openset.PROTOCOL), ("PASSIVE_MEMORY_RISK", "memory/risk_v1/fits", risk.PROTOCOL)]:
        for family in ev.json(protocol)["families"]:
            for seed in seeds:
                specifications.append((study, OUT / folder / (family + "__seed" + str(seed) + ".json"), protocol, study))
    assert len(specifications) == 75
    sources, normalizers, weights, fits, no_fit = {}, {}, {}, [], []
    open_data, memory_data = None, None
    for study, path, protocol, source_key in specifications:
        record = ev.json(path)
        assert type(record["seed"]) is int and record["seed"] in seeds and path.stem.endswith("__seed" + str(record["seed"]))
        if study == "STATE":
            assert path.stem == record["mode"] + "__seed" + str(record["seed"]) and record["mode"] == source_key
        elif study in ("MAIN", "PILOT_INTERNAL_FIT"):
            prefix = "MAIN__" if study == "MAIN" else "PILOT__"
            assert path.stem == prefix + record["family"] + "__" + record["objective"] + "__seed" + str(record["seed"])
            assert record["objective"] == source_key and record["pilot_not_main_fit_or_independent_confirmation"] == (study == "PILOT_INTERNAL_FIT")
        else:
            assert path.stem == record["family"] + "__seed" + str(record["seed"])
        verdict = assess_fit(record, protocol_sha256=ev.digest(protocol), artifact_sha256=ev.digest, source_sha256=lambda n: ev.digest(ROOT / n))
        if verdict["classification"] == "MEASURED_NO_OPTIMIZATION":
            assert study == "STATE" and source_key == "BASELINE_STATE"
            no_fit.append({"fit_record_path": str(path), "fit_record_sha256": ev.digest(path), "FIT_rows": record["FIT_rows"], "FIT_target_counts": record["FIT_target_counts"], "optimizer_steps": 0, "checkpoint": None})
        else:
            assert verdict["verified_actual_optimization"]
        key = study + "/" + source_key
        if key not in sources:
            if study in ("MAIN", "PILOT_INTERNAL_FIT", "STATE"):
                if study == "STATE":
                    fit, inner, manifest = state.load_records(source_key)
                    assert {r["role"] for r in fit} == {"FIT"} and {r["role"] for r in inner} == {"INNER"}
                else:
                    fit, inner, manifest = main.load_records(source_key, pilot=study == "PILOT_INTERNAL_FIT")
                    assert {r["optimization_role"] for r in fit} == {"FIT"}
                    expected = "PILOT_INTERNAL_FIT_VALIDATION" if study == "PILOT_INTERNAL_FIT" else "INNER"
                    assert {r["optimization_role"] for r in inner} == {expected}
                mean, scale = fit_normalizer(np.stack([r["x"] for r in fit]), correlated_sequence_weights(fit))
                sources[key] = {"FIT": summarize(fit), "INTERNAL_SELECTION": summarize(inner), "manifest": manifest,
                    "ordered_FIT_runtime_input_digest": hashlib.sha256("".join(input_signature(r) for r in fit).encode()).hexdigest(),
                    "ordered_selection_runtime_input_digest": hashlib.sha256("".join(input_signature(r) for r in inner).encode()).hexdigest(),
                    "selection_is_FIT_internal_NOT_INNER": study == "PILOT_INTERNAL_FIT", "independent_roots_proved": False}
            else:
                if study == "PASSIVE_OPEN_SET":
                    if open_data is None: open_data = openset.data()
                    original_p, fit, inner, receipts = open_data
                    raw, classes, available, w, _, _ = flatten_groups(fit)
                    iraw, iclasses, iavailable, _, _, _ = flatten_groups(inner)
                else:
                    if memory_data is None: memory_data = risk.data()
                    original_p, fit, inner, receipts = memory_data
                    raw, iraw = [np.asarray([r["runtime_input_vector"] for r in rows], np.float32) for rows in (fit, inner)]
                    classes, iclasses = [np.asarray([r["class"] for r in rows], np.int64) for rows in (fit, inner)]
                    available = iavailable = None; w = risk.teacher_weights(fit)
                assert original_p["split"] == p["split"]
                assert {r["sequence"] for r in fit} <= set(p["split"]["fit"]) and {r["sequence"] for r in inner} <= set(p["split"]["inner"])
                mean = (raw * w[:, None]).sum(0)
                scale = np.maximum(np.sqrt(((raw - mean) ** 2 * w[:, None]).sum(0)), .05)
                sources[key] = {"FIT_rows": len(raw), "INNER_rows": len(iraw), "FIT_class_counts": dict(Counter(map(int, classes))),
                    "INNER_class_counts": dict(Counter(map(int, iclasses))), "FIT_current_input_digest": numeric_digest(raw),
                    "INNER_current_input_digest": numeric_digest(iraw), "FIT_current_class_digest": numeric_digest(classes),
                    "INNER_current_class_digest": numeric_digest(iclasses), "source_receipts": receipts,
                    "UNKNOWN_separate_NOT_verified_negative": True, "passive_current_target_NOT_future_intervention_or_G4": True}
                if available is not None:
                    sources[key].update(FIT_availability_digest=numeric_digest(available), INNER_availability_digest=numeric_digest(iavailable))
            normalizers[key] = mean, scale
        if verdict["classification"] == "MEASURED_NO_OPTIMIZATION": continue
        for checkpoint, spec in checkpoint_specs(record).items():
            cp = Path(checkpoint); assert cp.is_relative_to(ASSETS.resolve() / "training")
            actual = ev.digest(cp)
            if spec["expected_sha256"] is not None: assert actual == spec["expected_sha256"]
            weights[checkpoint] = {"sha256": actual, "bytes": cp.stat().st_size, "original_receipt_SHA_present": spec["expected_sha256"] is not None}
        saved = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=True)
        initial_path = Path(record["checkpoint_path"]).parent / "initial.pt"
        changes = tensor_changes(saved, torch.load(initial_path, map_location="cpu", weights_only=True), record)
        mean, scale = normalizers[key]
        assert np.array_equal(np.asarray(saved["FIT_mean"], np.float32), mean)
        assert np.array_equal(np.asarray(saved.get("FIT_scale", saved.get("FIT_std")), np.float32), scale)
        support_row = sources[key]
        if study in ("MAIN", "PILOT_INTERNAL_FIT", "STATE"):
            assert record["manifest"] == support_row["manifest"] and record["FIT_rows"] == support_row["FIT"]["rows"]
            assert record.get("validation_rows", record.get("COMMON_INNER_rows")) == support_row["INTERNAL_SELECTION"]["rows"]
        else:
            assert record["FIT_rows"] == support_row["FIT_rows"] and record["INNER_rows"] == support_row["INNER_rows"]
            assert {str(k):v for k,v in record["FIT_class_counts"].items()} == {str(k):v for k,v in support_row["FIT_class_counts"].items()}
        fits.append({"study": study, "source": key, "family": record["family"], "seed": record["seed"],
            "fit_record_path": str(path), "fit_record_sha256": ev.digest(path), "selected_path": record["checkpoint_path"],
            "selected_sha256": record["checkpoint_sha256"], "selected_schema": saved["schema"], "tensor_changed_elements": changes,
            "original_initial_SHA_present": "initial_sha256" in record, "FIT_only_normalizer_exact": True,
            "no_deployment_or_future_safety_authority_inferred": True})
    assert dict(Counter(r["study"] for r in fits)) == p["expected_actual_fits"] and len(no_fit) == 3
    actual_files = {str(f.resolve()) for f in (ASSETS / "training").rglob("*.pt")}
    assert set(weights) == actual_files, "Every current training weight must have an actual fit receipt; no silently omitted partial/unregistered file"
    lineage = ev.json(OUT / "audit/CHECKPOINT_LINEAGE.json"); inherited = dict(lineage["weights"])
    inherited["common_frozen_identity_model"] = lineage["common_frozen_identity_model"]
    common_model = lineage["common_frozen_identity_model"]
    exposed = set(common_model["fit_sequences"] + [common_model["inner_sequence"], common_model["outer_sequence"]])
    assert not exposed.intersection(p["split"]["fit"] + p["split"]["inner"] + p["split"]["confirmation"])
    for row in inherited.values():
        assert ev.digest(row["path"]) == row["sha256"]
        if "fit_record_path" in row: assert ev.digest(row["fit_record_path"]) == row["fit_record_sha256"]
    common = {"stage": "N72R21R2", "goal": GOAL, "protocol_sha256": ev.digest(PROTOCOL),
        "split": p["split"], "scientific_goal_complete": False, "CONFIRM_VAL_TEST_SOT_accessed": False,
        "no_new_fits_scores_calibration_rollouts_or_original_models_changed": True, "next_association_stage_authorized": False}
    data = {**common, "status": "COMPLETE_ACTUAL_REGISTERED_SOURCE_ARRAYS_SPLIT_NORMALIZER_READBACK_WITH_LABEL_LIMITATIONS",
        "datasets": sources, "actual_optimized_fits": fits, "baseline_no_optimizer_dispositions": no_fit,
        "L1_independent_causal_event_target_implemented": False, "L1_L3_original_support_targets_identical_risk_loss_weights_differ": True,
        "L2_one_mean_three_horizons_NOT_three_separate_horizon_fits": True, "evidence_sha256": ev.refs}
    data_path = write_json("training/TRAINING_DATA.json", data)
    write_json("checkpoints/SHA256_MANIFEST.json", {**common,
        "status": "COMPLETE_ACTUAL_CURRENT_REGISTERED_72_FITS_ALL_EPOCH_SELECTED_INITIAL_BYTES_AND_THREE_INHERITED_WEIGHTS",
        "actual_optimized_fit_count": len(fits), "all_stage_training_pt_file_count": len(weights),
        "registered_weight_files": weights, "inherited_weights": inherited, "fit_records": fits,
        "no_optimizer_dispositions_not_weights": no_fit,
        "initial_bytes_without_original_seal_not_claimed_retroactively_sealed": True,
        "future_or_unregistered_fits_NOT_covered": True, "foundation_new_loader_smoke_or_vendor_SHA_independently_claimed": False,
        "training_data": {"path": str(data_path), "sha256": ev.digest(data_path)}, "evidence_sha256": ev.refs})
    print({"actual_training_data_delivery": True, "actual_optimized_fits": len(fits), "no_fit": len(no_fit), "all_training_weight_files": len(weights), "inherited_weight_count": len(inherited), "scientific_goal_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    freeze() if parser.parse_args().action == "freeze" else run()
