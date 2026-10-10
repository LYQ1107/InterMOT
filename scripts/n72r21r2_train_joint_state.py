"""Paired baseline/treatment/mixed FIT-only state-source contrast.

All models use the same SMALL_MLP/L3 optimization and the same complete
mixed INNER set for epoch selection. Controlled one-shot treatment replay
is not model-generated on-policy training or deployment qualification.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, preregistration, storage, append_log
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r2_train_event_authority import make_tensors
from sam3_intermot.one_click.event_authority_learning import encode_runtime_input, objective_target, correlated_sequence_weights, fit_normalizer
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead, independent_head_loss, runtime_predictions
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES

PROTOCOL = OUT / "on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json"
MODES = ("BASELINE_STATE", "TREATMENT_STATE", "MIXED_STATE")
CODE = ("scripts/n72r21r2_train_joint_state.py", "scripts/n72r21r2_joint_state_fit_driver.py",
        "scripts/n72r21r2_train_event_authority.py", "sam3_intermot/one_click/event_authority_learning.py",
        "sam3_intermot/one_click/event_authority_models.py", "sam3_intermot/one_click/matched_event_observer.py")
OBJECTIVE = "L3_HARM_AWARE"


def freeze():
    base = read_json(OUT / "training/EVENT_AUTHORITY_PROTOCOL_V1.json")
    write_json("on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "split": preregistration()["split"],
        "frozen_before_state_source_fits": True, "modes": list(MODES), "family": "SMALL_MLP", "objective": OBJECTIVE,
        "seeds": base["seeds"], "training": base["training"], "harm_weight": base["loss_harm_weights"][OBJECTIVE],
        "source_protocol_sha256": sha256(OUT / "on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json"),
        "source_sha256": {p: sha256(ROOT / p) for p in CODE}, "original_event_fit_protocol_sha256": sha256(OUT / "training/EVENT_AUTHORITY_PROTOCOL_V1.json"),
        "required_source_receipts": "All16 FIT+8 INNER joint-state whole-video runtime and supervision; no ready-subset training or failed-click replacement",
        "treatment_rows": "Source case !=KEEP and an actual effective source action strictly BEFORE this probe frame. Includes recovery of already changed states, excludes not-yet-delayed/vacuous source phases; never selects future reward.",
        "common_INNER_selection": "Identical union of baseline+past-actually-treated states for ALL three training modes; original hierarchical video/click/correlation-group/event/arm weighted loss",
        "normalizer_and_optimizer": "Each mode's FIT rows only; INNER never normalizes or updates gradients",
        "duplicates": "Drop same executed full trajectory duplicates; retain own KEEP reference",
        "supervision_failure": "If a mode has no FIT benefit/harm diversity, record measured failure without meaningless optimization",
        "staged_policy_curriculum_or_true_model_on_policy": False,
        "association_authority_or_confirmation_automatic": False, "CPU_workers": 1})


def past_actual_treatment(row):
    return row["source_case"] != "KEEP" and any(f < row["frame"] for f in row["actual_source_effective_action_frames"])


def load_records(mode):
    if mode not in MODES:
        raise ValueError("Unregistered state-source contrast")
    p = read_json(PROTOCOL)
    split = preregistration()["split"]
    required = split["fit"] + split["inner"]
    records, receipts, exclusions = [], [], Counter()
    for sequence in required:
        path = OUT / "on_policy/joint_state_v1/supervision" / (sequence + ".json")
        receipt = read_json(path)
        assert receipt["protocol_sha256"] == p["source_protocol_sha256"]
        assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
        assert receipt["whole_video_runtime_verified_before_GT"]
        rows = [r for r in read_zstd_jsonl(Path(receipt["artifact"]["path"])) if r["record_type"] == "NESTED_ARM"]
        keeps = {r["event_uid"]: r for r in rows if r["branch"] == "KEEP"}
        fit = sequence in split["fit"]
        for row in rows:
            baseline, treated = row["source_case"] == "KEEP", past_actual_treatment(row)
            if not baseline and not treated:
                exclusions["no_past_effective_treatment_phase"] += 1
                continue
            if fit and (mode == "BASELINE_STATE" and not baseline or mode == "TREATMENT_STATE" and not treated):
                exclusions["not_this_FIT_state_source"] += 1
                continue
            labels = row["offline_supervision_labels"]
            if labels["same_executed_action_configuration_as"] is not None:
                exclusions["duplicate_actual_trajectory"] += 1
                continue
            target = objective_target(labels, OBJECTIVE)
            if target is None:
                exclusions["incomplete_future"] += 1
                continue
            x, past = encode_runtime_input(row["runtime_features"], row["branch"], past_steps=p["training"]["past_steps"])
            keep, _ = encode_runtime_input(keeps[row["event_uid"]]["runtime_features"], "KEEP", past_steps=p["training"]["past_steps"])
            assert row["sequence"] == sequence and row["role"] == ("FIT" if fit else "INNER")
            records.append({k: row[k] for k in ("sequence", "episode_uid", "event_uid", "branch", "frame", "correlation_group_not_proven_causal_origin")} | {
                "x": x, "past": past, "keep": keep, "target": target, "role": "FIT" if fit else "INNER",
                "state_source": "BASELINE_STATE" if baseline else "TREATMENT_STATE",
                "runtime_features": row["runtime_features"], "observables": row["causal_state_distribution_observables"]})
        receipts.append({"sequence": sequence, "source_receipt_sha256": sha256(path), "source_artifact_sha256": receipt["artifact"]["sha256"], "raw_nested_rows": len(rows)})
    fit, inner = [r for r in records if r["role"] == "FIT"], [r for r in records if r["role"] == "INNER"]
    return fit, inner, {"required_all24_videos": required, "receipts": receipts, "exclusions": dict(exclusions),
                       "actual_FIT_videos": sorted({r["sequence"] for r in fit}), "actual_INNER_videos": sorted({r["sequence"] for r in inner}),
                       "COMMON_INNER_state_source_rows": dict(Counter(r["state_source"] for r in inner)),
                       "actual_model_generated_on_policy": False}


def run(mode, seed):
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {x: sha256(ROOT / x) for x in CODE} and seed in p["seeds"] and mode in MODES
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    uid = mode + "__seed" + str(seed)
    relative = "on_policy/state_source_fits_v1/" + uid + ".json"
    destination = ASSETS / "training/state_source_fits_v1" / uid
    if destination.exists() or (OUT / relative).exists():
        raise FileExistsError("Retain actual/partial state-source fit")
    fit, inner, manifest = load_records(mode)
    counts = {"benefit": sum(r["target"]["benefit"] for r in fit), "risk": sum(r["target"]["risk"] for r in fit)}
    if not fit or not inner or not all(v > 0 for v in counts.values()):
        write_json(relative, {"stage": "N72R21R2", "status": "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY", "mode": mode,
                   "seed": seed, "FIT_rows": len(fit), "INNER_rows": len(inner), "FIT_target_counts": counts, "manifest": manifest,
                   "optimizer_steps": 0, "protocol_sha256": sha256(PROTOCOL), "scientific_success": False})
        return
    storage(32 << 20)
    destination.mkdir(parents=True)
    weights, iw = correlated_sequence_weights(fit), correlated_sequence_weights(inner)
    mean, scale = fit_normalizer(np.stack([r["x"] for r in fit]), weights)
    x, past, keep, target = make_tensors(fit, mean, scale)
    ix, ipast, ikeep, itarget = make_tensors(inner, mean, scale)
    settings = p["training"]
    model = EventAuthorityHead(p["family"], hidden=settings["hidden"])
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, destination / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    w, inner_w = torch.from_numpy(weights.astype(np.float32)), torch.from_numpy(iw.astype(np.float32))
    best, selected_epoch, best_state, stale, steps, nonzero, max_gradient = float("inf"), None, None, 0, 0, 0, 0.
    history, began = [], time.monotonic()
    with (destination / "epochs.jsonl").open("x") as log:
        for epoch in range(1, settings["epochs_max"] + 1):
            if time.monotonic() - began > settings["max_seconds_per_fit"]:
                raise RuntimeError("Retain bounded fit partial; no implicit refit")
            model.train()
            epoch_loss = 0.
            for ids in torch.randperm(len(fit)).split(settings["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                output = model(x[ids], past[ids], keep_current=keep[ids])
                contribution = (independent_head_loss(output, target[ids], harm_weight=p["harm_weight"]) * w[ids]).sum()
                (contribution * len(fit) / len(ids)).backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
                if not np.isfinite(gradient):
                    raise RuntimeError("Nonfinite actual state-source gradient")
                steps += 1
                nonzero += gradient > 0
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
                epoch_loss += float(contribution.detach())
            model.eval()
            with torch.inference_mode():
                output = model(ix, ipast, keep_current=ikeep)
                validation_loss = float((independent_head_loss(output, itarget, harm_weight=p["harm_weight"]) * inner_w).sum())
            checkpoint = destination / ("epoch" + str(epoch) + ".pt")
            torch.save({"schema": "N72R21R2_EVENT_AUTHORITY_V1", "family": p["family"], "objective": OBJECTIVE,
                        "seed": seed, "epoch": epoch, "hidden": settings["hidden"], "model": model.state_dict(),
                        "FIT_mean": mean.tolist(), "FIT_scale": scale.tolist(), "feature_names": list(FEATURE_NAMES),
                        "training_state_source": mode, "protocol_sha256": sha256(PROTOCOL),
                        "authority_status": "TRAINED_UNCALIBRATED_NOT_DEPLOYABLE", "actual_model_generated_on_policy": False}, checkpoint)
            record = {"epoch": epoch, "FIT_correlated_macro_loss": epoch_loss, "COMMON_MIXED_INNER_loss": validation_loss,
                      "optimizer_steps": steps, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
            history.append(record)
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            if validation_loss < best:
                best, selected_epoch, stale = validation_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    changed = sum(int(torch.count_nonzero(initial[k] != best_state[k])) for k in initial)
    assert nonzero > 0 and changed > 0
    model.load_state_dict(best_state, strict=True)
    with torch.inference_mode():
        model.eval()
        predictions = runtime_predictions(model(ix, ipast, keep_current=ikeep))
    output_path = destination / "COMMON_INNER_predictions.jsonl"
    with output_path.open("x") as handle:
        for row, pred in zip(inner, predictions, strict=True):
            handle.write(json.dumps({k: row[k] for k in ("sequence", "episode_uid", "event_uid", "branch", "state_source")} | {
                "prediction": pred, "offline_target": row["target"]}, sort_keys=True, allow_nan=False) + "\n")
    distributions = {}
    for key in ("trusted_gap", "native_same", "prototype_anchor_agreement", "base_KEEP_margin", "joint_entropy", "pending_confirmation_count", "actor_bank_size"):
        column = FEATURE_NAMES.index(key)
        raw_fit = np.stack([r["x"][:32] for r in fit])
        raw_inner = np.stack([r["x"][:32] for r in inner])
        distributions[key] = {"FIT_scaled_feature_quantiles": np.quantile(raw_fit[:, column], [.1, .5, .9]).tolist(),
                              "COMMON_INNER_scaled_feature_quantiles": np.quantile(raw_inner[:, column], [.1, .5, .9]).tolist(),
                              "not_own_deployment_distribution": True}
    for key in ("observed_native_streak", "actor_bank_drift"):
        distributions[key] = {"FIT_actual_observable_quantiles": np.quantile([r["observables"][key] for r in fit], [.1, .5, .9]).tolist(),
                              "COMMON_INNER_actual_observable_quantiles": np.quantile([r["observables"][key] for r in inner], [.1, .5, .9]).tolist(),
                              "empty_P0_bank_not_learned_memory_drift": key == "actor_bank_drift",
                              "not_own_deployment_distribution": True}
    chosen = destination / ("epoch" + str(selected_epoch) + ".pt")
    restored = EventAuthorityPredictor(torch.load(chosen, map_location="cpu", weights_only=True), development_diagnostic=True)
    keeps = {r["event_uid"]: r["runtime_features"] for r in inner if r["branch"] == "KEEP"}
    for row, expected in zip(inner, predictions, strict=True):
        actual = restored.predict(row["runtime_features"], row["branch"], keeps[row["event_uid"]])
        assert all(abs(actual[k] - expected[k]) <= 1.e-6 for k in ("beneficial", "harmful", "value"))
    write_json(relative, {"stage": "N72R21R2", "experiment_uid": uid, "mode": mode, "seed": seed, "family": p["family"], "objective": OBJECTIVE,
        "status": "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE", "manifest": manifest,
        "FIT_rows": len(fit), "COMMON_INNER_rows": len(inner), "FIT_target_counts": counts,
        "epochs": history, "selected_epoch": selected_epoch, "optimizer_steps": steps, "nonzero_gradient_steps": nonzero,
        "maximum_gradient_norm": max_gradient, "changed_weight_elements": changed,
        "checkpoint_path": str(chosen), "checkpoint_sha256": sha256(chosen), "initial_sha256": sha256(destination / "initial.pt"),
        "COMMON_INNER_predictions_path": str(output_path), "COMMON_INNER_predictions_sha256": sha256(output_path),
        "distribution_audit": distributions, "source_sha256": p["source_sha256"], "protocol_sha256": sha256(PROTOCOL),
        "selected_strict_loader_ALL_COMMON_INNER_score_AA": True,
        "parameter_count": sum(t.numel() for t in model.parameters()), "seconds": time.monotonic() - began,
        "actual_model_generated_on_policy": False, "association_authority": False, "scientific_success": None})
    append_log("M7_ACTUAL_PAIRED_STATE_SOURCE_FIT", mode=mode, seed=seed, steps=steps, nonzero=nonzero)
    print({"state_source_fit": uid, "steps": steps, "changed_elements": changed, "authority": "UNCALIBRATED"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    freeze() if args.freeze else run(args.mode, args.seed)
