"""Six bounded paired teacher-source fits; no MOT authority or on-policy claim."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, storage
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts.n72r21r2_train_event_authority import make_tensors
from scripts.n72r21r2_controlled_l1_source_v1 import PROTOCOL, DATA
from sam3_intermot.evaluation.controlled_unit_objective import CONDITIONS
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead, independent_head_loss, runtime_predictions
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard


def rows_for_condition(data, condition, role):
    if condition not in CONDITIONS or role not in ("FIT", "INNER"):
        raise ValueError("Only the frozen paired conditions and split roles")
    return [{**row, **{k: np.asarray(row[k], np.float32) for k in ("x", "past", "keep")},
             "target": row["targets"][condition]} for row in data["rows"] if row["role"] == role]


def run(condition, seed):
    ev = Evidence(); p = ev.json(PROTOCOL); ev.code(p["source_sha256"])
    assert p["frozen"] and p["goal"] == GOAL and condition in p["conditions"] and type(seed) is int and seed in p["seeds"]
    d = ev.json(DATA); assert d["protocol_sha256"] == ev.digest(PROTOCOL)
    storage(32 << 20); torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    uid = condition + "__seed" + str(seed)
    relative = "objectives/controlled_unit_l1_contrast_v1/fits/" + uid + ".json"
    directory = ASSETS / "training/controlled_unit_l1_contrast_v1" / uid
    if directory.exists() or (OUT / relative).exists():
        raise FileExistsError("No implicit refit or partial-weight overwrite")
    fit, inner = [rows_for_condition(d, condition, role) for role in ("FIT", "INNER")]
    mean, scale = np.asarray(d["FIT_mean"], np.float32), np.asarray(d["FIT_scale"], np.float32)
    x, past, keep, target = make_tensors(fit, mean, scale)
    ix, ipast, ikeep, itarget = make_tensors(inner, mean, scale)
    weights, iw = torch.tensor(d["FIT_weights"], dtype=torch.float32), torch.tensor(d["INNER_weights"], dtype=torch.float32)
    s = p["training"]; model = EventAuthorityHead(p["family"], hidden=s["hidden"])
    directory.mkdir(parents=True)
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}; torch.save(initial, directory / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
    history, best, stale, steps, nonzero, began = [], float("inf"), 0, 0, 0, time.monotonic()
    best_state, selected = None, None
    for epoch in range(1, s["epochs_max"] + 1):
        if time.monotonic() - began > s["max_seconds_per_fit"]:
            raise RuntimeError("Actual original fit time bound; keep partial files, never auto-refit")
        model.train(); train_loss = 0.
        for ids in torch.randperm(len(fit)).split(s["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            out = model(x[ids], past[ids], keep_current=keep[ids])
            contribution = (independent_head_loss(out, target[ids], harm_weight=p["harm_weight"]) * weights[ids]).sum()
            (contribution * len(fit) / len(ids)).backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), s["gradient_clip"]))
            if not np.isfinite(gradient): raise RuntimeError("Nonfinite actual gradient")
            steps += 1; nonzero += gradient > 0; optimizer.step(); train_loss += float(contribution.detach())
        model.eval()
        with torch.inference_mode():
            validation = float((independent_head_loss(model(ix, ipast, keep_current=ikeep), itarget, harm_weight=p["harm_weight"]) * iw).sum())
        checkpoint = directory / ("epoch" + str(epoch) + ".pt")
        torch.save({"schema":"N72R21R2_EVENT_AUTHORITY_V1", "family":p["family"],"objective":"L1_EVENT_CORRECTION",
                    "seed":seed,"epoch":epoch,"hidden":s["hidden"],"model":model.state_dict(),
                    "FIT_mean":mean.tolist(),"FIT_scale":scale.tolist(),"feature_names":list(FEATURE_NAMES),
                    "protocol_sha256":ev.digest(PROTOCOL),"source_freeze":p["source_sha256"],"pilot":False,
                    "authority_status":"TRAINED_UNCALIBRATED_NOT_DEPLOYABLE","controlled_unit_condition":condition},checkpoint)
        history.append({"epoch":epoch,"FIT_teacher_weighted_loss":train_loss,"INNER_teacher_weighted_loss":validation,
                        "optimizer_steps":steps,"checkpoint_path":str(checkpoint),"checkpoint_sha256":sha256(checkpoint)})
        if validation < best:
            best, selected, stale = validation, epoch, 0
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if stale >= s["patience"]: break
    assert best_state is not None and nonzero > 0
    changed = {k:int(torch.count_nonzero(initial[k] != best_state[k])) for k in initial}; assert sum(changed.values()) > 0
    model.load_state_dict(best_state, strict=True); model.eval()
    chosen = directory / ("epoch" + str(selected) + ".pt")
    predictor = EventAuthorityPredictor(torch.load(chosen,map_location="cpu",weights_only=True),development_diagnostic=True)
    with torch.inference_mode():
        predictions = runtime_predictions(model(ix,ipast,keep_current=ikeep))
    max_error = 0.
    with runtime_file_guard():
        for row, expected in zip(inner, predictions, strict=True):
            actual = predictor.predict(row["runtime_features"], row["branch"], row["keep_runtime_features"])
            max_error = max(max_error, max(abs(actual[k]-expected[k]) for k in expected))
    assert max_error <= .0001, "Loaded frozen runtime predictions must reproduce actual teacher head outputs"
    write_json(relative,{"stage":"N72R21R2","goal":GOAL,"status":"COMPLETE_ACTUAL_MATCHED_CONTROLLED_UNIT_TEACHER_FIT_NOT_MOT_AUTHORITY",
        "condition":condition,"family":p["family"],"objective":"L1_EVENT_CORRECTION","seed":seed,
        "protocol_sha256":ev.digest(PROTOCOL),"source_dataset_sha256":ev.digest(DATA),"source_sha256":p["source_sha256"],
        "FIT_rows":len(fit),"INNER_rows":len(inner),"epochs":history,"selected_epoch":selected,
        "optimizer_steps":steps,"nonzero_gradient_steps":nonzero,"changed_weight_elements_by_tensor":changed,
        "parameter_count":sum(v.numel() for v in model.parameters()),"initial_path":str(directory/"initial.pt"),"initial_sha256":sha256(directory/"initial.pt"),
        "checkpoint_path":str(chosen),"checkpoint_sha256":sha256(chosen),"selected_loader_instantiated":predictor is not None,
        "actual_GT_guarded_ALL_INNER_loader_prediction_max_error":max_error,
        "actual_INNER_teacher_predictions_NOT_MOT_outputs":predictions,"original_settings_no_expansion":s,
        "actual_model_generated_on_policy":False,"natural_independent_L1_G1_or_deployable_memory_proved":False,
        "scientific_goal_complete":False,"next_association_stage_authorized":False,"seconds":time.monotonic()-began})
    print({"actual_controlled_unit_L1_fit":uid,"parameters":sum(v.numel() for v in model.parameters()),"steps":steps,"nonzero":nonzero,"epoch":selected,"scientific_goal_complete":False},flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--condition",choices=CONDITIONS,required=True);parser.add_argument("--seed",type=int,required=True)
    a=parser.parse_args();run(a.condition,a.seed)
