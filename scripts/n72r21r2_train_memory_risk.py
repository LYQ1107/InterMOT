"""All24-source, FIT-only fresh same-bank current write-risk learning."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, preregistration, append_log
from scripts.n72r21r2_memory_risk_data import PROTOCOL, CODE
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.fresh_memory_risk import INPUT_NAMES, MemoryRiskHead, FreshMemoryRiskPredictor


def teacher_weights(rows):
    episodes, cases, count = defaultdict(set), defaultdict(set), Counter()
    for row in rows:
        s, e, c = row["sequence"], row["episode_uid"], row["source_case"]
        episodes[s].add(e)
        cases[s, e].add(c)
        count[s, e, c] += 1
    w = np.asarray([1. / (len(episodes) * len(episodes[r["sequence"]]) * len(cases[r["sequence"], r["episode_uid"]])
                         * count[r["sequence"], r["episode_uid"], r["source_case"]]) for r in rows], np.float32)
    if not rows or not np.isclose(w.sum(), 1.):
        raise ValueError("Actual video/click/teacher-state weights required")
    return w


def data():
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {x: sha256(ROOT / x) for x in CODE}
    fit, inner, receipts = [], [], []
    for role in ("fit", "inner"):
        for sequence in p["split"][role]:
            path = OUT / "memory/risk_v1/supervision" / (sequence + ".json")
            receipt = read_json(path)
            assert receipt["protocol_sha256"] == sha256(PROTOCOL) and receipt["all_teacher_video_runtime_verified_before_current_labels"]
            assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
            rows = read_zstd_jsonl(Path(receipt["artifact"]["path"]))
            assert all(r["sequence"] == sequence and r["role"] == role.upper() and len(r["runtime_input_vector"]) == len(INPUT_NAMES) for r in rows)
            assert all(r["verified_identity_negative"] == (r["current_committed_crop_outcome"] == "VERIFIED_OTHER")
                       and (r["class"] == 2) == (r["current_committed_crop_outcome"] == "UNKNOWN") for r in rows)
            (fit if role == "fit" else inner).extend(rows)
            receipts.append({"sequence": sequence, "split": role, "receipt_sha256": sha256(path),
                             "data_sha256": receipt["artifact"]["sha256"], "actual_rows": len(rows),
                             "failed_initializations_or_NONE_commits_not_replaced": True})
    assert len(receipts) == 24 and fit and inner
    return p, fit, inner, receipts


def assess(rows, probabilities, point, weights):
    classes = np.asarray([r["class"] for r in rows])
    accepted = (probabilities[:, 0] >= point["probability_min"]) & (probabilities[:, 2] <= point["unknown_max"])
    if point["status"] == "CALIBRATION_ABSTAIN":
        accepted[:] = False
    mass, correct_mass = float(weights[accepted].sum()), float(weights[classes == 0].sum())
    videos = sorted({r["sequence"] for r, use in zip(rows, accepted, strict=True) if use})
    return {"sampled_accepted_write_opportunities": int(accepted.sum()), "accepted_in_videos": videos,
            "sampled_correct_accepted": int(np.sum(accepted & (classes == 0))),
            "sampled_verified_OTHER_accepted": int(np.sum(accepted & (classes == 1))),
            "sampled_UNKNOWN_accepted": int(np.sum(accepted & (classes == 2))),
            "weighted_wrong_plus_UNKNOWN_risk": float(weights[accepted & (classes != 0)].sum()) / mass if mass else None,
            "weighted_correct_write_retention": float(weights[accepted & (classes == 0)].sum()) / correct_mass if correct_mass else None,
            "source_policy_replicas_and_frames_NOT_independent_units": True,
            "sampled_teacher_state_opportunities_NOT_own_policy_writes_or_G4": True}


def run(family, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    p, fit, inner, receipts = data()
    assert family in p["families"] and seed in p["seeds"]
    uid = family + "__seed" + str(seed)
    relative = "memory/risk_v1/fits/" + uid + ".json"
    destination = ASSETS / "training/memory_current_risk_v1" / uid
    if destination.exists() or (OUT / relative).exists():
        raise FileExistsError("Preserve actual/partial fresh bank-risk fit")
    raw, iraw = [np.asarray([r["runtime_input_vector"] for r in rows], np.float32) for rows in (fit, inner)]
    classes, iclasses = [np.asarray([r["class"] for r in rows], np.int64) for rows in (fit, inner)]
    if not np.any(classes == 0) or not np.any(classes != 0):
        write_json(relative, {"stage": "N72R21R2", "status": "FAIL_ACTUAL_CURRENT_WRITE_SUPERVISION_DIVERSITY",
                   "family": family, "seed": seed, "source_receipts": receipts, "optimizer_steps": 0, "scientific_success": False})
        return
    weights, iw = teacher_weights(fit), teacher_weights(inner)
    mean = (raw * weights[:, None]).sum(0)
    scale = np.maximum(np.sqrt(((raw - mean) ** 2 * weights[:, None]).sum(0)), .05)
    x, ix = [torch.from_numpy((v - mean) / scale) for v in (raw, iraw)]
    c, ic = torch.from_numpy(classes), torch.from_numpy(iclasses)
    w, inner_w = torch.from_numpy(weights * len(weights)), torch.from_numpy(iw)
    class_mass = np.asarray([weights[classes == k].sum() for k in range(3)], np.float32)
    class_weight = torch.from_numpy(1. / np.maximum(class_mass, 1.e-4) / 3.)
    model = MemoryRiskHead(family)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    settings = p["training"]
    storage(32 << 20)
    destination.mkdir(parents=True)
    torch.save(initial, destination / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    best, best_state, best_epoch, stale, steps, nonzero, maximum_gradient = float("inf"), None, None, 0, 0, 0, 0.
    history, began = [], time.monotonic()
    with (destination / "epochs.jsonl").open("x") as log:
        for epoch in range(1, settings["epochs_max"] + 1):
            if time.monotonic() - began > settings["max_seconds_per_fit"]:
                raise RuntimeError("Bounded bank-risk fit timed out; retain partial")
            model.train()
            losses = []
            for ids in torch.randperm(len(x)).split(settings["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = (F.cross_entropy(model(x[ids]), c[ids], weight=class_weight, reduction="none") * w[ids]).mean()
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
                if not np.isfinite(gradient):
                    raise RuntimeError("Nonfinite actual bank-risk gradient")
                nonzero += gradient > 0
                maximum_gradient = max(maximum_gradient, gradient)
                steps += 1
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                validation = float((F.cross_entropy(model(ix), ic, weight=class_weight, reduction="none") * inner_w).sum())
            checkpoint = destination / ("epoch" + str(epoch) + ".pt")
            torch.save({"schema": "N72R21R2_COMMITTED_WRITE_CURRENT_RISK_V1", "family": family, "model": model.state_dict(),
                        "input_names": list(INPUT_NAMES), "FIT_mean": mean.tolist(), "FIT_scale": scale.tolist(),
                        "seed": seed, "epoch": epoch, "association_authority": False,
                        "selection": {"status": "UNCALIBRATED_NOT_DEPLOYABLE"}, "protocol_sha256": sha256(PROTOCOL)}, checkpoint)
            row = {"epoch": epoch, "FIT_mean_loss": float(np.mean(losses)), "COMMON_INNER_teacher_loss": validation,
                   "optimizer_steps": steps, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
            history.append(row)
            log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            if validation < best:
                best, best_epoch, stale = validation, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        logits = model(ix)
    temperatures = [{"temperature": t, "COMMON_INNER_weighted_class_NLL": float((F.cross_entropy(logits / t, ic, reduction="none") * inner_w).sum())}
                    for t in p["temperature_grid"]]
    temperature = min(temperatures, key=lambda r: (r["COMMON_INNER_weighted_class_NLL"], r["temperature"]))["temperature"]
    probabilities = torch.softmax(logits / temperature, dim=1).cpu().numpy()
    points = []
    for cutoff in p["probability_grid"]:
        point = dict(status="INNER_SELECTED_CURRENT_WRITE_ONLY", probability_min=cutoff, unknown_max=p["unknown_max"], temperature=temperature)
        measured = assess(inner, probabilities, point, iw)
        accepted, risk, retention = measured["sampled_accepted_write_opportunities"], measured["weighted_wrong_plus_UNKNOWN_risk"], measured["weighted_correct_write_retention"]
        good = accepted >= 50 and len(measured["accepted_in_videos"]) >= 3 and risk is not None and risk <= .02 and retention is not None and retention >= .6
        points.append({"point": point, "measured": measured, "sampled_teacher_empirical_point_qualified_NOT_G4": good})
    good = [r for r in points if r["sampled_teacher_empirical_point_qualified_NOT_G4"]]
    selection = (min(good, key=lambda r: (-r["measured"]["weighted_correct_write_retention"], r["measured"]["weighted_wrong_plus_UNKNOWN_risk"], -r["point"]["probability_min"]))["point"]
                 if good else dict(status="CALIBRATION_ABSTAIN", probability_min=1.01, unknown_max=p["unknown_max"], temperature=temperature))
    selected = destination / "selected_current_write_only.pt"
    torch.save({"schema": "N72R21R2_COMMITTED_WRITE_CURRENT_RISK_V1", "family": family, "model": best_state,
                "input_names": list(INPUT_NAMES), "FIT_mean": mean.tolist(), "FIT_scale": scale.tolist(), "seed": seed,
                "epoch": best_epoch, "selection": selection, "association_authority": False,
                "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"]}, selected)
    restored = FreshMemoryRiskPredictor(selected)
    for vector, expected in zip(iraw, probabilities, strict=True):
        actual = restored.predict(vector)
        assert np.allclose([actual["beneficial"], actual["verified_OTHER"], actual["unknown"]], expected, atol=1.e-6, rtol=0.)
    changed = sum(int(torch.count_nonzero(initial[k] != best_state[k])) for k in initial)
    assert nonzero > 0 and changed > 0
    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(100):
            model(ix[:1])
        latency_ms = (time.perf_counter() - start) * 10.
    write_json(relative, {"stage": "N72R21R2", "status": "COMPLETE_ACTUAL_SAME_BANK_CURRENT_WRITE_RISK_FIT", "family": family, "seed": seed,
        "source_receipts": receipts, "FIT_videos": sorted({r["sequence"] for r in fit}), "INNER_videos": sorted({r["sequence"] for r in inner}),
        "FIT_rows": len(fit), "INNER_rows": len(inner), "FIT_class_counts": dict(Counter(classes.tolist())), "INNER_class_counts": dict(Counter(iclasses.tolist())),
        "optimizer_steps": steps, "nonzero_gradient_steps": nonzero, "maximum_gradient_norm": maximum_gradient,
        "changed_weight_elements": changed, "epochs": history, "selected_epoch": best_epoch, "parameter_count": sum(t.numel() for t in model.parameters()),
        "checkpoint_path": str(selected), "checkpoint_sha256": sha256(selected), "initial_sha256": sha256(destination / "initial.pt"),
        "selection": selection, "temperature_NLL": temperatures, "all_INNER_sampled_teacher_points": points,
        "selected_INNER_sampled_teacher_result": assess(inner, probabilities, selection, iw),
        "fixed_diagnostic_NOT_selected": assess(inner, probabilities, {**p["fixed_diagnostic"], "temperature": temperature}, iw),
        "strict_loader_ALL_INNER_scores_AA": True, "CPU_head_only_latency_ms": latency_ms,
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"], "seconds": time.monotonic() - began,
        "source_bank_policy_to_own_policy_shift_still_requires_actual_audit": True,
        "source_frames_and_teacher_policies_NOT_independent_units": True, "association_authority": False,
        "future_safety_or_own_policy_G4_claimed": False, "scientific_success": None})
    append_log("M9_ACTUAL_SAME_BANK_CURRENT_WRITE_RISK_FIT", family=family, seed=seed, steps=steps, nonzero=nonzero)
    print({"actual_memory_risk_fit": uid, "steps": steps, "selected_status": selection["status"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["LOGISTIC", "MLP"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.family, args.seed)
