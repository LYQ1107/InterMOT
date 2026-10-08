#!/usr/bin/env python3
"""M7 real online counterfactual decision supervision, never sealed base tape."""
from __future__ import annotations
from dataclasses import replace

from scripts.n72r20r4_common import *
from scripts.n72r20r4_run_causal_tracker import make_bank
from scripts.n72r20r4_adapter_integration import fold_split, strict_ensemble
from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.identity_authority import AuthorityConfig, AuthorityController, FEATURE_NAMES
from scripts.n72r20r3_common import gt_by_frame, iou


def target_truth(event: dict, gt: dict) -> int:
    frame = int(event["event_frame"])
    values = [(iou(event["target_box_xyxy"], box), identity) for identity,box in gt.get(frame, [])]
    overlap, identity = max(values, default=(0, -1))
    if overlap < 0.5:
        raise ValueError("offline clicked target GT could not be identified")
    return int(identity)


def matched_identity(box: list, truth: list) -> int | None:
    overlap, identity = max(((iou(box, gt_box), identity) for identity, gt_box in truth), default=(0, None))
    return int(identity) if overlap >= 0.5 else None


def correct_public(decision: dict, frame: int, gt: dict, identities: dict[int,int | None]) -> dict[int,bool]:
    return {int(r["public_id"]): identities.get(int(r["public_id"])) is not None and matched_identity(r["box_xyxy"], gt.get(frame, [])) == identities.get(int(r["public_id"])) for r in decision["outputs"]}


def mine_sequence(sequence: str, adapter, *, fit_config: AuthorityConfig | None = None) -> tuple[list[dict], dict]:
    frames = load_frames(sequence)
    event = events()[sequence]
    gt = gt_by_frame(DATASET / "train" / sequence / "gt/gt.txt")
    target_gt = target_truth(event, gt)
    tracker = CausalIdentityTracker(config=AuthorityConfig(mode="off"), event=event, adapter=adapter, bank=make_bank())
    identities = {}
    scores = []
    raw_scores = []
    legacy_scores = []
    scales = []
    examples = []
    quality = []
    for payload, rows in frames:
        frame = int(payload["frame"])
        sampled = frame > int(event["event_frame"]) and frame % 5 == 0 and bool(rows)
        before = tracker.clone() if sampled else None
        baseline = tracker.step(rows, frame)
        target = tracker.target_public
        for r in baseline["outputs"]:
            p = int(r["public_id"])
            if p not in identities:
                identities[p] = matched_identity(r["box_xyxy"], gt.get(frame, []))
        if target is not None:
            identities[target] = target_gt
        if not sampled:
            continue
        scores.extend(baseline["identity_scores"])
        raw_scores.extend(float(np.dot(np.asarray(event["human_anchor"]), r["feature"])) for r in rows)
        matrix = baseline["base_matrix"]
        if target in baseline["states_before_commit_axis"]:
            column = baseline["states_before_commit_axis"].index(target)
            legacy_scores.extend((1/(1+np.exp(-(matrix[:,column]-2)/1.5))).tolist())
        if matrix.size:
            scales.append(float(np.quantile(matrix,0.75)-np.quantile(matrix,0.25)))
        features = np.asarray(baseline["authority_features"], dtype=np.float32)
        quality.append(float(features[4]))
        if fit_config is None:
            continue
        before.config = replace(fit_config, mode="fixed", strength=0.5, memory="P0")
        challenger = before.step(rows, frame)
        bc = correct_public(baseline,frame,gt,identities)
        tc = correct_public(challenger,frame,gt,identities)
        current_gain = int(tc.get(target,False))-int(bc.get(target,False))
        other_damage = sum(bool(v) and not tc.get(p,False) for p,v in bc.items() if p != target)
        next_gain = 0
        next_damage = 0
        if frame+1 < len(frames):
            next_rows = frames[frame+1][1]
            base_branch = tracker.clone()
            rb = base_branch.step(next_rows,frame+1)
            rt = before.step(next_rows,frame+1)
            bnext = correct_public(rb,frame+1,gt,identities)
            tnext = correct_public(rt,frame+1,gt,identities)
            next_gain = int(tnext.get(target,False))-int(bnext.get(target,False))
            next_damage = sum(bool(v) and not tnext.get(p,False) for p,v in bnext.items() if p != target)
        changed = challenger["assignments"] != baseline["assignments"]
        wrong_override = bc.get(target,False) and not tc.get(target,False)
        wrong_write_risk = changed and not tc.get(target,False)
        value = current_gain + next_gain - 2*(other_damage+next_damage) - int(wrong_write_risk)
        beneficial = bool(changed and value > 0 and not wrong_override)
        examples.append({"sequence": sequence, "frame": frame, "features": features.tolist(), "beneficial_intervention": beneficial, "weight": 3.0 if wrong_override else 2.0 if (other_damage+next_damage) else 1.0, "counterfactual_assignment_changed": changed, "target_current_gain": current_gain, "target_next_gain": next_gain, "non_target_damage": other_damage+next_damage, "wrong_write_risk_label": wrong_write_risk, "base_assignment": baseline["assignments"], "challenger_assignment": challenger["assignments"], "causal_tracker_state_hash": baseline["state_before"], "next_state_supervision_only": True, "GT_input_to_controller": False})
    summary = {"sequence": sequence, "scores": scores, "raw_scores": raw_scores, "legacy_scores": legacy_scores, "base_scales": scales, "qualities": quality, "rows": len(examples), "beneficial": sum(r["beneficial_intervention"] for r in examples), "runtime_future_gt_used": False}
    return examples, summary


def train_fold(heldout: str, *, deployment_adapter=None) -> dict:
    deployment = heldout == "ALL_DEV_FOR_VAL"
    fit, validation = (list(SEQUENCES), "NO_VAL_SUPERVISION") if deployment else fold_split(heldout)
    manifest_path = OUT / "association/training" / f"{heldout}.json"
    if manifest_path.exists():
        result = read_json(manifest_path)
        for r in result["checkpoints"]:
            if sha256(Path(r["path"])) != r["sha256"]:
                raise ValueError("authority checkpoint resume mismatch")
        return result
    if read_json(OUT / "causal_tracker/A_A_EQUIVALENCE.json")["status"] != "PASS_CAUSAL_BASELINE_EQUIVALENCE":
        raise RuntimeError("cannot train association authority before causal A/A")
    adapter = deployment_adapter if deployment else strict_ensemble(heldout)
    if adapter is None:
        raise ValueError("deployment requires its frozen training-side Adapter")
    calibration = []
    for s in fit:
        _, summary = mine_sequence(s, adapter)
        calibration.append(summary)
        print(json.dumps({"fold":heldout,"phase":"live_calibration","sequence":s}),flush=True)
    scores = np.concatenate([r["scores"] for r in calibration])
    scales = np.concatenate([r["base_scales"] for r in calibration])
    qualities = np.concatenate([r["qualities"] for r in calibration])
    config = AuthorityConfig(identity_mean=float(scores.mean()), identity_std=max(0.05,float(scores.std())), base_scale=max(0.25,float(np.median(scales))), write_score=float(np.quantile(scores,0.5)), write_quality=float(np.quantile(qualities,0.25)))
    source_calibrations = {}
    for name,key in (("raw","raw_scores"),("legacy","legacy_scores")):
        values = np.concatenate([r[key] for r in calibration])
        source_calibrations[name] = {"identity_mean": float(values.mean()), "identity_std": max(0.05,float(values.std()))}
    examples = []
    for s in fit:
        rows, _ = mine_sequence(s, adapter, fit_config=config)
        examples.extend(rows)
        print(json.dumps({"fold":heldout,"phase":"causal_counterfactual_training","sequence":s,"examples":len(rows)}),flush=True)
    if not examples:
        raise RuntimeError("empty real causal authority supervision")
    margins = np.array([r["features"][1] for r in examples])
    config = replace(config, write_margin=max(0.0,float(np.quantile(margins,0.5))))
    write_zstd(ASSETS / "training" / f"{heldout}__decisions.jsonl.zst", examples)
    x = torch.tensor([r["features"] for r in examples], dtype=torch.float32)
    y = torch.tensor([r["beneficial_intervention"] for r in examples], dtype=torch.float32)
    weights = torch.tensor([r["weight"] for r in examples], dtype=torch.float32)
    checkpoints = []
    for mode in ("scalar", "logistic", "mlp", "structured"):
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = AuthorityController(mode)
            optimizer = torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.0001)
            generator = torch.Generator().manual_seed(seed)
            history = []
            for epoch in range(30):
                order = torch.randperm(len(x), generator=generator)
                losses = []
                for start in range(0,len(x),512):
                    ix = order[start:start+512]
                    probability = model(x[ix]).clamp(1e-6,1-1e-6)
                    loss = (torch.nn.functional.binary_cross_entropy(probability,y[ix],reduction="none")*weights[ix]).mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
            path = ASSETS / "models" / f"authority__{heldout}__{mode}__seed{seed}.pt"
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"stage": STAGE, "mode": mode, "feature_names": FEATURE_NAMES, "state_dict": model.state_dict(), "seed": seed, "fit_sequences": fit, "heldout_sequence": heldout, "inner_validation_sequence": validation, "config": config.to_dict()}, path)
            checkpoints.append({"path": path, "sha256": sha256(path), "mode": mode, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "loss_history": history})
    result = {"stage": STAGE, "heldout_sequence": heldout, "fit_sequences": fit, "inner_validation_sequence": validation, "calibration": config.to_dict(), "source_calibrations": source_calibrations, "examples": len(examples), "beneficial_examples": int(y.sum()), "training_data": ASSETS / "training" / f"{heldout}__decisions.jsonl.zst", "training_data_sha256": sha256(ASSETS / "training" / f"{heldout}__decisions.jsonl.zst"), "checkpoints": checkpoints, "real_causal_rollout_supervision": True, "sealed_base_score_tape_used": False, "outer_GT_used_for_training": False, "adapter_lineage": adapter.manifest}
    write_json(manifest_path,result)
    return plain(result)


def load_controller(heldout: str, mode: str, seed: int) -> AuthorityController:
    cp = torch.load(ASSETS / "models" / f"authority__{heldout}__{mode}__seed{seed}.pt",map_location="cpu",weights_only=False)
    fit = list(SEQUENCES) if heldout == "ALL_DEV_FOR_VAL" else fold_split(heldout)[0]
    if cp["fit_sequences"] != fit:
        raise ValueError("authority fold training lineage mismatch")
    model = AuthorityController(mode)
    model.load_state_dict(cp["state_dict"],strict=True)
    return model.eval()
