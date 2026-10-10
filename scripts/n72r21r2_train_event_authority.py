"""Bounded actual fresh-event fits, FIT-only optimizer and normalization.

The pilot is explicitly held-out FIT development, never an INNER/CONFIRM
result. Main fits cannot silently run on an incomplete 24-video corpus.
Checkpoints remain uncalibrated and have no deployment authority.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, preregistration, append_log
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.event_authority_learning import (OBJECTIVES, encode_runtime_input, objective_target,
    correlated_sequence_weights, fit_normalizer, normalize_inputs)
from sam3_intermot.one_click.event_authority_models import FAMILIES, EventAuthorityHead, independent_head_loss, runtime_predictions
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES

PROTOCOL = OUT / "training/EVENT_AUTHORITY_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_train_event_authority.py", "sam3_intermot/one_click/event_authority_learning.py",
        "sam3_intermot/one_click/event_authority_models.py")


def load_records(objective, *, pilot):
    prereg, protocol = preregistration(), read_json(PROTOCOL)
    fits = protocol["pilot"]["fit"] if pilot else prereg["split"]["fit"]
    inners = protocol["pilot"]["internal_validation"] if pilot else prereg["split"]["inner"]
    required = fits + inners
    accepted, excluded, receipts = [], Counter(), []
    for sequence in required:
        # Precondition is structural completeness, never pilot effects or best scenes.
        for directory in ("data/candidate_integrity", "mot/baseline_results", "events/current_context_sequences", "events/label_audit", "simple/results"):
            if not (OUT / directory / (sequence + ".json")).exists():
                raise RuntimeError("Required fresh corpus/control is not complete: " + directory + "/" + sequence)
        audit_path = OUT / "events/label_audit" / (sequence + ".json")
        audit = read_json(audit_path)
        assert sha256(audit["artifact"]["path"]) == audit["artifact"]["sha256"]
        assert audit["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_label_counterfactual.py")
        utilities = {}
        utility_audit_path = None
        if objective == "L5_ACTUAL_TRAJECTORY_UTILITY":
            utility_audit_path = OUT / "events/trajectory_utility_audit" / (sequence + ".json")
            utility_audit = read_json(utility_audit_path)
            assert sha256(utility_audit["artifact"]["path"]) == utility_audit["artifact"]["sha256"]
            utilities = {(r["event_uid"], r["branch"]): r for r in read_zstd_jsonl(Path(utility_audit["artifact"]["path"]))}
        rows = read_zstd_jsonl(Path(audit["artifact"]["path"]))
        expected_role = "FIT" if sequence in prereg["split"]["fit"] else "INNER"
        assert all(r["sequence"] == sequence and r["role"] == expected_role for r in rows)
        contexts = {}
        keeps = {r["event_uid"]: r for r in rows if r["branch"] == "KEEP"}
        for row in rows:
            if row["state_source"] != protocol["source_distribution"].split(";")[0]:
                raise ValueError("State-source shift needs an explicit new protocol, not a silent C0 fit")
            runtime = row["runtime_features"]
            context = runtime["current_context_path"]
            if context not in contexts:
                contexts[context] = sha256(context)
            assert contexts[context] == runtime["current_context_sha256"]
            labels = row["offline_supervision_labels"]
            if labels["same_executed_action_configuration_as"] is not None:
                excluded["duplicate_executed_action_configuration"] += 1
                continue
            utility = utilities.get((row["event_uid"], row["branch"]))
            if utility is not None:
                assert utility["actual_branch_sha256"] == row["actual_branch_sha256"]
            target = objective_target(labels, objective, actual_trajectory_utility=utility)
            if target is None:
                excluded["incomplete_or_missing_required_future_reward"] += 1
                continue
            current, past = encode_runtime_input(runtime, row["branch"], past_steps=protocol["training"]["past_steps"])
            keep, _ = encode_runtime_input(keeps[row["event_uid"]]["runtime_features"], "KEEP", past_steps=protocol["training"]["past_steps"])
            accepted.append({k: row[k] for k in ("sequence", "episode_uid", "event_uid", "correlation_group_not_proven_causal_origin", "branch", "frame")} | {
                "x": current, "past": past, "keep": keep, "target": target,
                "optimization_role": "FIT" if sequence in fits else "PILOT_INTERNAL_FIT_VALIDATION" if pilot else "INNER"})
        receipts.append({"sequence": sequence, "original_split_role": expected_role,
                         "label_audit_sha256": sha256(audit_path), "labels_sha256": audit["artifact"]["sha256"],
                         "utility_audit_sha256": sha256(utility_audit_path) if utility_audit_path else None,
                         "original_rows": len(rows), "initialization_failures_not_replaced": True})
    fit = [r for r in accepted if r["optimization_role"] == "FIT"]
    inner = [r for r in accepted if r["optimization_role"] != "FIT"]
    if not fit or not inner or not any(r["target"]["benefit"] for r in fit) or not any(r["target"]["risk"] for r in fit):
        raise RuntimeError("No usable actual benefit/harm diversity; no meaningless optimization")
    return fit, inner, {"required_sequences": required, "actual_fit_videos": sorted({r["sequence"] for r in fit}),
                        "actual_validation_videos": sorted({r["sequence"] for r in inner}), "receipts": receipts, "excluded": dict(excluded),
                        "pilot_not_main_fit_or_independent_confirmation": pilot}


def make_tensors(rows, mean, scale):
    current, past = normalize_inputs(np.stack([r["x"] for r in rows]), np.stack([r["past"] for r in rows]), mean, scale)
    keep, _ = normalize_inputs(np.stack([r["keep"] for r in rows]), np.stack([r["past"] for r in rows]), mean, scale)
    target = np.asarray([[r["target"][k] for k in ("benefit", "risk", "value")] for r in rows], dtype=np.float32)
    return tuple(torch.from_numpy(a) for a in (current, past, keep, target))


def same_event_pairs(rows, weights):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["event_uid"]].append(i)
    pairs, pair_weights = [], []
    for ids in groups.values():
        ordered = sorted(ids, key=lambda i: (rows[i]["target"]["value"], i))
        if rows[ordered[-1]]["target"]["value"] > rows[ordered[0]]["target"]["value"]:
            pairs.append((ordered[-1], ordered[0]))
            pair_weights.append(sum(weights[i] for i in ids))
    return pairs, np.asarray(pair_weights, dtype=np.float32)


def run(family, objective, seed, *, pilot=False):
    protocol = read_json(PROTOCOL)
    assert protocol["frozen"] and family in FAMILIES and objective in OBJECTIVES and seed in protocol["seeds"]
    if pilot:
        assert family in protocol["pilot"]["families"] and objective == protocol["pilot"]["objective"]
    elif objective != protocol["primary_family_objective"]:
        assert family == "SMALL_MLP", "Loss comparisons require the exact same architecture"
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    settings = protocol["training"]
    uid = ("PILOT__" if pilot else "MAIN__") + family + "__" + objective + "__seed" + str(seed)
    relative = "training/event_authority/" + uid + ".json"
    destination = ASSETS / "training/event_authority_v1" / uid
    if destination.exists() or (OUT / relative).exists():
        raise FileExistsError("Preserve completed/partial fits; never refit an existing experiment UID")
    fit, inner, manifest = load_records(objective, pilot=pilot)
    storage(32 << 20)
    destination.mkdir(parents=True)
    weights, iw = correlated_sequence_weights(fit), correlated_sequence_weights(inner)
    mean, scale = fit_normalizer(np.stack([r["x"] for r in fit]), weights)
    x, past, keep, target = make_tensors(fit, mean, scale)
    ix, ipast, ikeep, itarget = make_tensors(inner, mean, scale)
    model = EventAuthorityHead(family, hidden=settings["hidden"])
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, destination / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    w, inner_weights = torch.from_numpy(weights.astype(np.float32)), torch.from_numpy(iw.astype(np.float32))
    pairs, pair_weights = same_event_pairs(fit, weights)
    harm_weight = protocol["loss_harm_weights"][objective]
    history, best, best_state, best_epoch, stale, steps, nonzero, max_gradient = [], float("inf"), None, None, 0, 0, 0, 0.
    began = time.monotonic()
    with (destination / "epochs.jsonl").open("x") as log:
        for epoch in range(1, settings["epochs_max"] + 1):
            if time.monotonic() - began > settings["max_seconds_per_fit"]:
                raise RuntimeError("Bounded fit exceeded its actual time limit; preserve partial evidence")
            model.train()
            permutation = torch.randperm(len(fit))
            epoch_loss = 0.
            for ids in permutation.split(settings["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                out = model(x[ids], past[ids], keep_current=keep[ids])
                losses = independent_head_loss(out, target[ids], harm_weight=harm_weight)
                contribution = (losses * w[ids]).sum()
                loss = contribution * len(fit) / len(ids)
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
                if not np.isfinite(gradient):
                    raise RuntimeError("Non-finite actual fit gradient")
                nonzero += gradient > 0
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
                steps += 1
                epoch_loss += float(contribution.detach())
            pair_loss = 0.
            if family == "PAIRWISE_RANKER" and pairs:
                optimizer.zero_grad(set_to_none=True)
                output = model(x, past, keep_current=keep)
                plus, minus = torch.tensor(pairs).T
                pw = torch.from_numpy(pair_weights)
                pl = (F.softplus(-(output[plus, 2] - output[minus, 2])) * pw).sum() / pw.sum()
                pl.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
                if not np.isfinite(gradient):
                    raise RuntimeError("Non-finite same-prestate pair gradient")
                nonzero += gradient > 0
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
                steps += 1
                pair_loss = float(pl.detach())
            model.eval()
            with torch.inference_mode():
                output = model(ix, ipast, keep_current=ikeep)
                validation_loss = float((independent_head_loss(output, itarget, harm_weight=harm_weight) * inner_weights).sum())
                risk_brier = float(((output[:, 1].sigmoid() - itarget[:, 1]) ** 2 * inner_weights).sum())
            checkpoint = destination / ("epoch" + str(epoch) + ".pt")
            torch.save({"schema": "N72R21R2_EVENT_AUTHORITY_V1", "model": model.state_dict(), "family": family,
                        "objective": objective, "seed": seed, "epoch": epoch, "hidden": settings["hidden"],
                        "FIT_mean": mean.tolist(), "FIT_scale": scale.tolist(), "feature_names": list(FEATURE_NAMES),
                        "protocol_sha256": sha256(PROTOCOL), "source_freeze": {p: sha256(ROOT / p) for p in CODE},
                        "pilot": pilot, "authority_status": "TRAINED_UNCALIBRATED_NOT_DEPLOYABLE"}, checkpoint)
            item = {"epoch": epoch, "optimizer_steps": steps, "FIT_correlated_macro_loss": epoch_loss,
                    "actual_same_prestate_pair_loss": pair_loss, "validation_correlated_video_macro_loss": validation_loss,
                    "validation_risk_Brier": risk_brier, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
            history.append(item)
            log.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            if validation_loss < best:
                best, best_epoch, stale = validation_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best_state, strict=True)
    changed = {k: int(torch.count_nonzero(best_state[k] != initial[k])) for k in initial}
    assert nonzero > 0 and sum(changed.values()) > 0
    chosen = destination / ("epoch" + str(best_epoch) + ".pt")
    with torch.inference_mode():
        model.eval()
        output = model(ix, ipast, keep_current=ikeep)
        latency_started = time.perf_counter()
        for _ in range(100):
            model(ix[:1], ipast[:1], keep_current=ikeep[:1])
        latency_ms = (time.perf_counter() - latency_started) * 10.
    prediction_path = destination / "validation_predictions.jsonl"
    with prediction_path.open("x") as handle:
        for row, prediction in zip(inner, runtime_predictions(output), strict=True):
            handle.write(json.dumps({k: row[k] for k in ("sequence", "episode_uid", "event_uid", "branch")} | {
                "prediction": prediction, "offline_target": row["target"], "not_own_policy_outcome": True}, sort_keys=True, allow_nan=False) + "\n")
    write_json(relative, {"stage": "N72R21R2", "experiment_uid": uid, "family": family, "objective": objective,
        "seed": seed, "status": "COMPLETE_ACTUAL_OPTIMIZATION_UNCALIBRATED_NO_DEPLOYMENT_AUTHORITY",
        "pilot_not_main_fit_or_independent_confirmation": pilot, "manifest": manifest,
        "FIT_rows": len(fit), "validation_rows": len(inner), "same_prestate_pair_count": len(pairs),
        "FIT_target_counts": {"benefit": sum(r["target"]["benefit"] for r in fit), "risk": sum(r["target"]["risk"] for r in fit)},
        "best_epoch": best_epoch, "epochs": history, "optimizer_steps": steps, "nonzero_gradient_steps": nonzero,
        "maximum_gradient_norm": max_gradient, "changed_weight_elements_by_tensor": changed,
        "checkpoint_path": str(chosen), "checkpoint_sha256": sha256(chosen), "initial_sha256": sha256(destination / "initial.pt"),
        "validation_predictions_path": str(prediction_path), "validation_predictions_sha256": sha256(prediction_path),
        "source_freeze": {p: sha256(ROOT / p) for p in CODE}, "protocol_sha256": sha256(PROTOCOL),
        "FIT_normalizer_only": True, "optimizer_validation_or_confirmation_usage": False,
        "parameter_count": sum(p.numel() for p in model.parameters()), "CPU_single_event_latency_ms": latency_ms,
        "seconds": time.monotonic() - began, "not_global_MOT_improvement": True,
        "operating_point_frozen": False, "scientific_success": None, "association_deployment_authorized": False})
    append_log("M5_ACTUAL_FRESH_EVENT_AUTHORITY_FIT_COMPLETE", experiment_uid=uid, steps=steps, nonzero=nonzero, pilot=pilot)
    print({"actual_fit": uid, "steps": steps, "nonzero_gradient_steps": nonzero, "changed_elements": sum(changed.values()),
           "fit_videos": manifest["actual_fit_videos"], "validation_videos": manifest["actual_validation_videos"],
           "authority": "UNCALIBRATED_NOT_DEPLOYABLE"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(args.family, args.objective, args.seed, pilot=args.pilot)
