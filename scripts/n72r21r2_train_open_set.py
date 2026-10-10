"""Bounded fresh FIT-only current-axis fits; INNER-only calibration.

No old fit scenes are substituted for failed clicks. All24 input receipts
are required, including all-initialization-failure videos with zero rows.
These current identity models do not acquire MOT intervention authority.
"""
import argparse
from collections import Counter
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, preregistration, append_log
from scripts.n72r21r2_open_set_data import PROTOCOL as DATA_PROTOCOL
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.open_set_verifier import OpenSetVerifierHead, predictions, simple_predictions
from sam3_intermot.one_click.fresh_open_set_learning import HEAD_FAMILIES, SIMPLE_CONTROLS, flatten_groups, select_point, assess_claims

PROTOCOL = OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_train_open_set.py", "sam3_intermot/one_click/fresh_open_set_learning.py",
        "sam3_intermot/one_click/open_set_verifier.py", "sam3_intermot/one_click/intervention_features.py")
SIMPLE_MAP = {"RAW_SCORE": "E0_FROZEN", "CALIBRATED_SCORE": "E1_CALIBRATED", "CANDIDATE_MARGIN": "E2_RELATIVE_NONE",
              "BASE_AGREEMENT": "E4_BASE", "TEMPORAL_CONFIRMATION": "E5_TEMPORAL"}


def freeze():
    protocol = preregistration()
    write_json("availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": protocol["goal"], "formal_goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "frozen_before_actual_new_fits_or_claim_results": True, "data_protocol_sha256": sha256(DATA_PROTOCOL),
        "source_sha256": {p: sha256(ROOT / p) for p in CODE}, "split": protocol["split"],
        "families": list(HEAD_FAMILIES), "simple_controls": list(SIMPLE_CONTROLS), "seeds": [730101, 730102, 730103],
        "loss": "3-class current correctness CE plus current positive-candidate availability BCE. UNKNOWN independent class, not a verified identity negative.",
        "hard_negative_contrast": "Same MLP architecture; FIT verified-other anchor-cosine>=.7 CE weight5; UNKNOWN never enters verified-negative weighting",
        "weighting": "Equal videos with valid source, equal sole-click episodes, equal sampled frame groups, then equal current-axis rows. All failed clicks/videos remain in source manifest, never replaced or counted as correct.",
        "normalization": "Hierarchical weighted FIT-only feature mean/std, std floor.05",
        "epochs_max": 60, "patience": 8, "batch_size": 256, "learning_rate": .001, "weight_decay": .0001,
        "gradient_clip": 10., "max_seconds_per_fit": 180, "CPU_threads": 1,
        "temperatures": [.5, 1., 2.], "cutoffs": [.5, .8, .9, .95, .98], "margins": [0., .05, .15],
        "unknown_max": .02, "risk_max": .02, "min_identity_claims": 10, "min_identity_claim_videos": 3,
        "selection": "INNER weighted current-class NLL epoch/temperature; max correct TARGET claims at <=2% empirical decision risk including UNKNOWN and incorrect NONE, at least10 actual identity claims in3 videos; deterministic ties higher cutoff/margin. No point => CALIBRATION_ABSTAIN, not safety PASS.",
        "raw_reference": "RAW_SCORE fixed T1 cutoff0 margin0 unknown1; not tuned or claimed safe",
        "diagnostic_point": {"status": "UNCONSTRAINED_DIAGNOSTIC_NOT_SELECTED", "probability_min": .5, "margin_min": 0., "unknown_max": .5},
        "evaluation_scope": "Fixed every32 current-axis frame groups, not dense full-video claims or full-MOT outcomes",
        "risk_sensitive_selective_prediction": "Calibrated learned verifier with explicit UNKNOWN abstention and INNER nonvacuity/risk test, distinct from unfiltered argmax",
        "future_intervention_safety_or_MOT_authority": False, "CONFIRM_VAL_TEST_opened": False})


def data():
    p = read_json(PROTOCOL)
    assert p["data_protocol_sha256"] == sha256(DATA_PROTOCOL)
    assert p["source_sha256"] == {path: sha256(ROOT / path) for path in CODE}
    fit, inner, receipts = [], [], []
    for role in ("fit", "inner"):
        for sequence in p["split"][role]:
            path = OUT / "availability/current_axis_v1/supervision" / (sequence + ".json")
            if not path.exists():
                raise RuntimeError("Required full fresh current-axis dataset missing: " + sequence)
            receipt = read_json(path)
            assert receipt["all_registered_video_runtime_sealed_before_GT"] and receipt["UNKNOWN_separate_class_not_verified_negative"]
            assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
            groups = read_zstd_jsonl(Path(receipt["artifact"]["path"]))
            assert len(groups) == receipt["counts"].get("frame_groups", 0)
            assert all(g["sequence"] == sequence and g["role"] == role.upper() and g["UNKNOWN_not_verified_negative"] and g["current_GT_labels_not_features"] for g in groups)
            init = read_json(OUT / "data/initialization" / (sequence + ".json"))
            valid = sum(not e["initialization_failure"] for e in init["inputs"])
            assert groups or valid == 0, "A readable valid episode cannot silently vanish from fresh data"
            (fit if role == "fit" else inner).extend(groups)
            receipts.append({"sequence": sequence, "split": role, "supervision_manifest_sha256": sha256(path),
                             "data_sha256": receipt["artifact"]["sha256"], "groups": len(groups),
                             "valid_clicks": valid, "initialization_failures_not_replaced": len(init["inputs"]) - valid})
    assert fit and inner
    return p, fit, inner, receipts


def probabilities(model, x, spans, temperature):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
    return [predictions(logits[start:stop], temperature) for start, stop in spans], logits


def run(family, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    p, fit, inner, receipts = data()
    assert family in p["families"] and seed in p["seeds"]
    uid = family + "__seed" + str(seed)
    relative = "availability/current_axis_fits/" + uid + ".json"
    destination = ASSETS / "training/fresh_open_set_v1" / uid
    if destination.exists() or (OUT / relative).exists():
        raise FileExistsError("Preserve completed or partial actual fresh verifier fit")
    storage(64 << 20)
    raw, classes, available, weights, hard, spans = flatten_groups(fit)
    iraw, iclasses, iavailable, iweights, _, ispans = flatten_groups(inner)
    mean = (raw * weights[:, None]).sum(0)
    scale = np.maximum(np.sqrt(((raw - mean) ** 2 * weights[:, None]).sum(0)), .05)
    x, ix = (torch.from_numpy((v - mean) / scale) for v in (raw, iraw))
    c, ic = torch.from_numpy(classes), torch.from_numpy(iclasses)
    a, ia = torch.from_numpy(available), torch.from_numpy(iavailable)
    w, iw = torch.from_numpy(weights * len(weights)), torch.from_numpy(iweights * len(iweights))
    class_mass = np.asarray([weights[classes == k].sum() for k in range(3)], np.float32)
    class_weight = torch.from_numpy(1. / np.maximum(class_mass, 1e-4) / 3)
    model = OpenSetVerifierHead("MLP" if family == "MLP_HARD_NEGATIVE" else family)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    hard_weight = torch.from_numpy(hard if family == "MLP_HARD_NEGATIVE" else np.ones_like(hard))
    destination.mkdir(parents=True)
    torch.save(initial, destination / "initial.pt")
    def loss(logits, labels, avail):
        return F.cross_entropy(logits[:, :3], labels, weight=class_weight, reduction="none") + F.binary_cross_entropy_with_logits(logits[:, 3], avail, reduction="none")
    began, best_loss, best, best_epoch, stale = time.monotonic(), float("inf"), None, None, 0
    steps, nonzero, max_gradient, epochs = 0, 0, 0., []
    log_path = destination / "epochs.jsonl"
    with log_path.open("x") as log:
        for epoch in range(1, p["epochs_max"] + 1):
            if time.monotonic() - began > p["max_seconds_per_fit"]:
                raise RuntimeError("Bounded fresh fit exceeded wall time; retain partial actual fit")
            model.train()
            fit_loss = []
            for ids in torch.randperm(len(x)).split(p["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                value = (loss(model(x[ids]), c[ids], a[ids]) * w[ids] * hard_weight[ids]).mean()
                if not torch.isfinite(value): raise RuntimeError("Nonfinite fresh current-axis loss")
                value.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), p["gradient_clip"]))
                if not np.isfinite(gradient): raise RuntimeError("Nonfinite fresh current-axis gradient")
                nonzero += gradient > 0
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
                steps += 1
                fit_loss.append(float(value.detach()))
            model.eval()
            with torch.inference_mode():
                validation = float((loss(model(ix), ic, ia) * iw).mean())
            if not np.isfinite(validation): raise RuntimeError("Nonfinite fresh INNER class loss")
            checkpoint = destination / ("epoch" + str(epoch) + ".pt")
            torch.save({"schema": "N72R21R2_CURRENT_AXIS_OPEN_SET_V1", "model": model.state_dict(), "family": family,
                        "head_family": "MLP" if family == "MLP_HARD_NEGATIVE" else family, "feature_names": list(FEATURE_NAMES),
                        "FIT_mean": mean.tolist(), "FIT_std": scale.tolist(), "seed": seed, "epoch": epoch,
                        "selection": {"status": "UNCALIBRATED_EPOCH_NOT_DEPLOYABLE"}, "protocol_sha256": sha256(PROTOCOL)}, checkpoint)
            row = {"epoch": epoch, "FIT_loss": float(np.mean(fit_loss)), "INNER_loss": validation, "optimizer_steps": steps,
                   "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
            epochs.append(row)
            log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"); log.flush()
            if validation < best_loss:
                best_loss, best_epoch, stale = validation, epoch, 0
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= p["patience"]: break
    model.load_state_dict(best, strict=True)
    _, logits = probabilities(model, ix, ispans, 1.)
    temperatures = [(float((F.cross_entropy(logits[:, :3] / t, ic, reduction="none") * iw).mean()), t) for t in p["temperatures"]]
    temperature = min(temperatures)[1]
    predicted, _ = probabilities(model, ix, ispans, temperature)
    selection, points = select_point(inner, predicted, p)
    selection = {**selection, "temperature": temperature}
    selected_result = assess_claims(inner, predicted, selection)
    diagnostic = assess_claims(inner, predicted, p["diagnostic_point"])
    checkpoint = destination / "selected_current_identity_only.pt"
    torch.save({"schema": "N72R21R2_CURRENT_AXIS_OPEN_SET_V1", "model": best, "family": family,
                "head_family": "MLP" if family == "MLP_HARD_NEGATIVE" else family, "feature_names": list(FEATURE_NAMES),
                "FIT_mean": mean.tolist(), "FIT_std": scale.tolist(), "seed": seed, "epoch": best_epoch,
                "selection": selection, "protocol_sha256": sha256(PROTOCOL), "association_authority": False}, checkpoint)
    changed = {k: int(torch.count_nonzero(v != initial[k])) for k, v in best.items()}
    assert nonzero and sum(changed.values())
    with torch.inference_mode():
        candidate_truth = (ic == 0).float()
        calibrated_correct = torch.softmax(logits[:, :3] / temperature, dim=1)[:, 0]
        brier = float(((calibrated_correct - candidate_truth) ** 2 * iw).mean())
        avail_brier = float(((torch.sigmoid(logits[:, 3] / temperature) - ia) ** 2 * iw).mean())
    write_json(relative, {"stage": "N72R21R2", "status": "COMPLETE_ACTUAL_FRESH_CURRENT_AXIS_FIT_AND_INNER_CALIBRATION",
        "family": family, "seed": seed, "all24_source_receipts": receipts, "FIT_videos_with_rows": sorted({r["sequence"] for r in fit}),
        "INNER_videos_with_rows": sorted({r["sequence"] for r in inner}), "FIT_rows": len(x), "INNER_rows": len(ix),
        "FIT_class_counts": dict(Counter(classes.tolist())), "INNER_class_counts": dict(Counter(iclasses.tolist())),
        "optimizer_steps": steps, "nonzero_gradient_steps": nonzero, "maximum_gradient_norm": max_gradient,
        "changed_weight_elements": changed, "parameter_count": sum(v.numel() for v in model.parameters()),
        "epochs": epochs, "selected_epoch": best_epoch, "seconds": time.monotonic() - began,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "epoch_log_sha256": sha256(log_path),
        "source_sha256": p["source_sha256"], "protocol_sha256": sha256(PROTOCOL), "temperature_NLL": temperatures,
        "selection": selection, "all_INNER_operating_points": points, "selected_INNER_current_claims": selected_result,
        "unconstrained_diagnostic_NOT_selected": diagnostic, "INNER_weighted_correctness_Brier": brier,
        "INNER_weighted_candidate_availability_Brier": avail_brier, "FIT_only_normalization_and_optimizer": True,
        "UNKNOWN_not_verified_identity_negative": True, "dense_full_video_results": False,
        "association_authority": False, "scientific_success": None})
    append_log("M8_ACTUAL_FRESH_OPEN_SET_FIT_COMPLETE", family=family, seed=seed, steps=steps, selection=selection["status"])
    print({"actual_fresh_open_set_fit": uid, "steps": steps, "selection": selection["status"], "claims": selected_result["counts"].get("accepted_identity_claims", 0)}, flush=True)


def calibrate_simple():
    p, _, inner, receipts = data()
    result = {}
    for family in p["simple_controls"]:
        temperature_options = [1.] if family == "RAW_SCORE" else p["temperatures"]
        options = []
        for t in temperature_options:
            predicted = [simple_predictions(g["axis"], SIMPLE_MAP[family], t) for g in inner]
            nll = float(np.mean([-np.log(max(sum(s["correct"] for a, s in zip(g["axis"], pred, strict=True) if a["class"] == 0), 1e-12)) for g, pred in zip(inner, predicted, strict=True)]))
            options.append((nll, t, predicted))
        _, temperature, predicted = min(options, key=lambda a: (a[0], a[1]))
        settings = {**p, "margins": [.05, .15]} if family == "CANDIDATE_MARGIN" else p
        selection, points = select_point(inner, predicted, settings, SIMPLE_MAP[family])
        if family == "RAW_SCORE":
            selection = {"status": "FROZEN_UNCALIBRATED_NOT_SAFE", "probability_min": 0., "margin_min": 0., "unknown_max": 1., "simple_family": SIMPLE_MAP[family]}
        selection = {**selection, "temperature": temperature}
        result[family] = {"selection": selection, "all_INNER_operating_points": points, "selected_INNER_current_claims": assess_claims(inner, predicted, selection),
                          "temperature_NLL": [(n, t) for n, t, _ in options], "not_learned_future_safety": True}
    write_json("availability/SIMPLE_CURRENT_AXIS_CALIBRATION_V1.json", {"stage": "N72R21R2", "protocol_sha256": sha256(PROTOCOL),
               "source_receipts": receipts, "controls": result, "full_global_MOT_authority": False, "scientific_success": None})
    print({"actual_fresh_simple_open_set_calibration": len(result)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--family", choices=HEAD_FAMILIES)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    freeze() if args.freeze else calibrate_simple() if args.simple else run(args.family, args.seed)
