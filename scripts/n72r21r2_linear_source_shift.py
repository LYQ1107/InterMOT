"""Frozen read-only source/own-policy linear attribution, no model repair.

GT never changes scores. The pilot FIT observations define actual training
support; current features and stored predictions come from sealed GT-free
raw-anchor legacy rollouts. No counterfactual logit is deployed or optimized.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, read_json, write_json, sha256, append_log
from scripts.n72r21r2_train_event_authority import load_records
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector
from sam3_intermot.one_click.event_authority_learning import BRANCH_NAMES, normalize_inputs
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead, runtime_predictions
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.linear_source_shift import constant_feature_decomposition, score_only_pass

PROTOCOL = OUT / "training/LINEAR_SOURCE_SHIFT_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_linear_source_shift.py", "sam3_intermot/evaluation/linear_source_shift.py",
        "scripts/n72r21r2_train_event_authority.py", "sam3_intermot/one_click/intervention_features.py",
        "sam3_intermot/one_click/event_authority_learning.py", "sam3_intermot/one_click/event_authority_models.py")


def freeze():
    p = read_json(OUT / "training/SUPPORT_ABLATION_PILOT_PROTOCOL_V1.json")
    write_json("training/LINEAR_SOURCE_SHIFT_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "support_ablation_protocol_sha256": sha256(OUT / "training/SUPPORT_ABLATION_PILOT_PROTOCOL_V1.json"),
        "sequences": p["sequences"], "seed": 730101, "family": "LOGISTIC_RISK", "policy": "LEGACY_RAW_ANCHOR",
        "point": p["point"], "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "score_reproduction_absolute_tolerance": .0001, "linear_logit_absolute_tolerance": .0001,
        "FIT_support": "Exact actual L3 pilot optimizer rows; first32 causal input columns, exact max-minus-min0 defines constant support. No parameter refit.",
        "own_support": "All current choices at every actual frame from all three original raw-anchor legacy FIT pilot videos, not successful frame/UID selection",
        "counterfactual": "Remove only algebraic FIT-constant-feature logit contributions OFFLINE to isolate source/own covariate shift. Never overwrite feature/model/policy or infer resulting future behavior.",
        "not_model_repair_or_qualification_or_architecture_failure": True, "GT_cannot_change_scores": True})


def run():
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    torch.set_num_threads(1)
    fit, _, manifest = load_records("L3_HARM_AWARE", pilot=True)
    raw = np.stack([r["x"] for r in fit])
    uid = "PILOT__LOGISTIC_RISK__L3_HARM_AWARE__seed" + str(p["seed"])
    fit_path = OUT / "training/event_authority" / (uid + ".json")
    f = read_json(fit_path)
    assert sha256(f["checkpoint_path"]) == f["checkpoint_sha256"]
    saved = torch.load(f["checkpoint_path"], map_location="cpu", weights_only=False)
    model = EventAuthorityHead(saved["family"], hidden=saved["hidden"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean, scale = np.array(saved["FIT_mean"], np.float32), np.array(saved["FIT_scale"], np.float32)
    records, seals = [], []
    for sequence in p["sequences"]:
        path = OUT / "training/support_ablation_v1/runtime" / (uid + "__" + p["policy"]) / (sequence + ".json")
        seal = read_json(path)
        assert seal["same_raw_sealed_click_all_variants"] and seal["actual_GT_file_guard"]
        assert seal["checkpoint_sha256"] == f["checkpoint_sha256"]
        seals.append({"sequence": sequence, "path": str(path), "sha256": sha256(path)})
        for e in seal["episodes"]:
            a = next(a for a in e["artifacts"] if a["kind"] == "trace")
            assert sha256(a["path"]) == a["sha256"]
            with runtime_file_guard():
                for row in read_zstd_jsonl(Path(a["path"])):
                    for c in row.get("authority", {}).get("all_current_choices", []):
                        if set(c["features"]) != set(FEATURE_NAMES):
                            raise ValueError("No truth/future dimensions in actual current features")
                        x = np.zeros(40, np.float32)
                        x[:32] = feature_vector(c["features"])
                        x[32 + BRANCH_NAMES.index(c["branch"])] = 1
                        records.append({"sequence": sequence, "branch": c["branch"], "x": x, "prediction": c["prediction"]})
    x, past = normalize_inputs(np.stack([r["x"] for r in records]), np.zeros((len(records), 3, 32), np.float32), mean, scale)
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(x), torch.from_numpy(past))
        pred = runtime_predictions(torch_logits)
    score_error = max(abs(actual[k] - row["prediction"][k]) for actual, row in zip(pred, records, strict=True) for k in actual)
    assert score_error <= p["score_reproduction_absolute_tolerance"]
    weight, bias = model.net.weight.detach().numpy(), model.net.bias.detach().numpy()
    const, logits, contribution, residual = constant_feature_decomposition(raw, x, weight, bias)
    logit_error = float(np.max(np.abs(logits - torch_logits.numpy())))
    assert logit_error <= p["linear_logit_absolute_tolerance"]
    initial = torch.load(Path(f["checkpoint_path"]).parent / "initial.pt", map_location="cpu", weights_only=True)
    actual_pass, hypothetical_pass = score_only_pass(logits, p["point"]), score_only_pass(residual, p["point"])
    per_video = {}
    for sequence in p["sequences"]:
        ids = [i for i, r in enumerate(records) if r["sequence"] == sequence and r["branch"] != "KEEP"]
        per_video[sequence] = {"non_KEEP_current_choices_NOT_independent": len(ids),
            "actual_benefit_risk_value_score_only_pass": int(actual_pass[ids].sum()),
            "hypothetical_without_constant_feature_contribution_score_only_pass": int(hypothetical_pass[ids].sum())}
    partial = OUT / "training/SUPPORT_LINEAR_SOURCE_SHIFT_DIAGNOSTIC_V1.json.tmp"
    result = {"stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "protocol_sha256": sha256(PROTOCOL),
        "model": uid, "fit_record_sha256": sha256(fit_path), "runtime_seals": seals,
        "source_FIT_rows": len(fit), "source_FIT_videos": manifest["actual_fit_videos"], "per_video_counts": per_video,
        "measured_score_absolute_max_error": float(score_error), "measured_logit_absolute_max_error": logit_error,
        "reproduction_is_numeric_tolerance_NOT_bitwise_AA": True,
        "source_constant_causal_features": [{"name": FEATURE_NAMES[i], "scaled_source_value": float(raw[0, i]),
            "normalizer_mean": float(mean[i]), "normalizer_scale": float(scale[i]),
            "max_absolute_own_normalized_value": float(np.abs(x[:, i]).max()),
            "max_absolute_logit_contribution": np.abs(x[:, i, None] * weight[:, i]).max(0).tolist(),
            "initial_to_best_max_abs_coefficient_change": float(torch.abs(model.net.weight[:, i] - initial["net.weight"][:, i]).max())} for i in const],
        "previous_failed_read_only_probe": {"session": 38864, "exit_code": 1, "cause": "numpy int64 count not JSON serializable",
            "partial_path": str(partial), "partial_sha256": sha256(partial) if partial.exists() else None, "not_overwritten": True},
        "counterfactual_score_pass_not_action_or_future_safety": True, "model_features_or_runtime_modified": False,
        "not_architecture_failure_or_scientific_closure": True, "next_stage_authorized": False}
    write_json("training/SUPPORT_LINEAR_SOURCE_SHIFT_DIAGNOSTIC_V2.json", result)
    append_log("READ_ONLY_LINEAR_SOURCE_OWN_SHIFT_DIAGNOSTIC", source_const_features=[FEATURE_NAMES[i] for i in const], per_video=per_video)
    print({"source_const_features": result["source_constant_causal_features"], "per_video": per_video,
           "score_abs_error": score_error, "logit_abs_error": logit_error}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
