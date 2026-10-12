"""M5/M6/M7 controller inventory: real weights/gradients, not MOT success.

Scope is MAIN42 plus nine controlled-state dispositions (six optimized).
Current-identity, memory-risk and pilot substudies remain separate schemas.
Preserve all seeds, objective-specific eligible support and baseline no-fit.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch

from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, utcnow, storage, preregistration, update_status
from scripts import n72r21r2_train_joint_state as state_study
from scripts import n72r21r2_model_cost_readback_v1 as main_cost
from scripts.n72r21r2_state_fit_storage_resume_v3 import valid_no_fit
from sam3_intermot.evaluation.optimization_census import assess_fit, census
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES

PROTOCOL = OUT / "training/CONTROLLER_DELIVERY_TIMING_PROTOCOL_V1.json"
RESULT = "training/MAIN_AND_STATE_CONTROLLER_INVENTORY_V1.json"
MAIN_COST = OUT / main_cost.RESULT
CODE = ("scripts/n72r21r2_training_delivery_v1.py", "scripts/n72r21r2_model_cost_readback_v1.py",
        "scripts/n72r21r2_train_joint_state.py", "sam3_intermot/evaluation/optimization_census.py",
        "sam3_intermot/one_click/event_authority_models.py")
REPORT_FAMILIES = {
    "SCALAR": "SCALAR", "LOGISTIC": "LOGISTIC_RISK", "MLP": "SMALL_MLP",
    "RANKER": "PAIRWISE_RANKER", "GLOBAL_RISK": "HARM_FIRST_DUAL_HEAD",
    "TEMPORAL": "CAUSAL_TEMPORAL", "VALUE_MODEL": "COUNTERFACTUAL_ACTION_VALUE",
    "DUAL_STATE": "DUAL_IDENTITY_STATE",
}


def all_seed_groups(rows, seeds):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["study"] + "/" + row["family"] + "__" + row["objective"]].append(row)
    for values in grouped.values():
        if len(values) != len(seeds) or {r["seed"] for r in values} != set(seeds):
            raise ValueError("All original seeds required; failed records are not substitute trained models")
        if any(r["parameter_count"] != values[0]["parameter_count"] or r["tensor_layout"] != values[0]["tensor_layout"] for r in values):
            raise ValueError("Registered same-family capacity cannot drift across seeds")
    return dict(grouped)


def state_capacity(saved, initial, record):
    if saved["schema"] != "N72R21R2_EVENT_AUTHORITY_V1" or saved["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Exact event-controller schema/feature axes required")
    for key in ("seed", "family", "objective"):
        if saved[key] != record[key]: raise ValueError("Wrong actual state model identity")
    if saved["training_state_source"] != record["mode"] or saved["epoch"] != record["selected_epoch"]:
        raise ValueError("Original selected state checkpoint required")
    if saved["protocol_sha256"] != record["protocol_sha256"] or saved["actual_model_generated_on_policy"]:
        raise ValueError("Controlled source cannot become model-generated on-policy")
    model = EventAuthorityHead(saved["family"], hidden=saved["hidden"])
    model.load_state_dict(saved["model"], strict=True)
    parameters = dict(model.named_parameters())
    count = sum(t.numel() for t in parameters.values())
    if type(record["parameter_count"]) is not int or count != record["parameter_count"]:
        raise ValueError("Actual architecture parameter count mismatch")
    current = model.state_dict()
    if set(initial) != set(current): raise ValueError("Initial/selected tensor axes differ")
    changed = 0
    for name, value in current.items():
        if value.shape != initial[name].shape or value.dtype != initial[name].dtype:
            raise ValueError("Initial/selected tensor shapes or dtypes differ")
        changed += int(torch.count_nonzero(value != initial[name]))
    if changed != record["changed_weight_elements"] or changed <= 0:
        raise ValueError("Missing actual changed selected weights")
    layout = {n: {"shape": list(t.shape), "dtype": str(t.dtype), "trainable": t.requires_grad} for n,t in parameters.items()}
    model.eval().requires_grad_(False)
    return model, {"parameter_count": count, "actual_changed_weight_elements": changed, "tensor_layout": layout}


def prepare():
    if PROTOCOL.exists(): raise FileExistsError("Preserve the original cost protocol")
    storage(16 << 20)
    state = read_json(state_study.PROTOCOL)
    refs = {str(MAIN_COST): sha256(MAIN_COST), str(state_study.PROTOCOL): sha256(state_study.PROTOCOL)}
    for mode in state["modes"]:
        for seed in state["seeds"]:
            path = OUT / "on_policy/state_source_fits_v1" / (mode + "__seed" + str(seed) + ".json")
            refs[str(path)] = sha256(path)
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
        "frozen_before_new_STATE_forward_timing": True, "source_sha256": {p:sha256(ROOT/p) for p in CODE},
        "references": refs, "split": preregistration()["split"], "CPU_threads": 1,
        "STATE_timing": "Original MAIN timing recipe: eval/inference mode, one full real COMMON_INNER forward before timer, then100 forwards of its first real branch row/past3/same-prestate KEEP. No optimizer or action. FIT-only saved normalization.",
        "timing_scope": "Controller forward only, excludes normalization, sigmoid/output conversion, feature extraction, global solver/tracker, SAM3/OSNet. Concurrent project workloads remain; no hardware-controlled family ranking or timing population CI.",
        "registered_controller_dispositions": 51, "actual_trained_controllers": 48,
        "baseline_no_optimizer_dispositions": 3, "new_training_selection_or_association_authority": False})


def run():
    if (OUT / RESULT).exists(): raise FileExistsError("Preserve completed inventory/timing")
    p = read_json(PROTOCOL)
    assert p["goal"] == GOAL and p["source_sha256"] == {n:sha256(ROOT/n) for n in CODE}
    assert all(sha256(path)==h for path,h in p["references"].items())
    storage(16 << 20); torch.set_num_threads(1)
    prereg = preregistration(); seeds=prereg["hyperparameters"]["seeds"]
    assert p["split"] == prereg["split"]
    refs={**p["references"], str(PROTOCOL):sha256(PROTOCOL)}
    def attest(path, expected=None):
        key=str(path)
        if key not in refs: refs[key]=sha256(path)
        if expected is not None: assert refs[key]==expected
        return refs[key]
    cost=read_json(MAIN_COST)
    for path,h in cost["evidence_sha256"].items(): attest(path,h)
    rows=[]; assessed=[]; no_fit=[]
    main_protocol=OUT/"training/EVENT_AUTHORITY_PROTOCOL_V1.json"
    attest(main_protocol)
    for values in cost["groups"].values():
        for row in values["actual_models"]:
            path=Path(row["fit_record"]["path"]); attest(path,row["fit_record"]["sha256"])
            record=read_json(path)
            verdict=assess_fit(record,protocol_sha256=attest(main_protocol),artifact_sha256=attest,source_sha256=lambda n:attest(ROOT/n))
            assert verdict["verified_actual_optimization"]
            initial_path=Path(record["checkpoint_path"]).parent/"initial.pt"
            loaded=torch.load(record["checkpoint_path"],map_location="cpu",weights_only=True)
            initial=torch.load(initial_path,map_location="cpu",weights_only=True)
            actual=main_cost.verified_capacity(loaded,initial,record)
            assert actual["parameter_count"]==row["parameter_count"]
            rows.append({**row, "study":"MAIN", "optimization":verdict,
                         "eligible_support_and_exclusions":record["manifest"],
                         "latency_scope":p["timing_scope"], "MOT_effects":None, "MOT_effects_status":"PENDING_ORIGINAL_COMPLETE_GROUP_AUDIT"})
            assessed.append(verdict)
    state_p=read_json(state_study.PROTOCOL)
    datasets={}
    for mode in state_p["modes"]:
        for seed in seeds:
            path=OUT/"on_policy/state_source_fits_v1"/(mode+"__seed"+str(seed)+".json")
            record=read_json(path); attest(path)
            verdict=assess_fit(record,protocol_sha256=attest(state_study.PROTOCOL),artifact_sha256=attest,source_sha256=lambda n:attest(ROOT/n))
            assessed.append(verdict)
            if mode=="BASELINE_STATE":
                assert valid_no_fit(record,seed,attest(state_study.PROTOCOL)) and verdict["classification"]=="MEASURED_NO_OPTIMIZATION"
                no_fit.append({"mode":mode,"seed":seed,"fit_record":{"path":str(path),"sha256":attest(path)},"optimization":verdict,"parameter_count":None,"latency_ms":None,"MOT_metrics":None})
                continue
            assert verdict["verified_actual_optimization"]
            checkpoint=torch.load(record["checkpoint_path"],map_location="cpu",weights_only=True)
            initial_path=Path(record["checkpoint_path"]).parent/"initial.pt"; attest(initial_path,record["initial_sha256"])
            model, capacity=state_capacity(checkpoint,torch.load(initial_path,map_location="cpu",weights_only=True),record)
            if mode not in datasets: datasets[mode]=state_study.load_records(mode)
            fit,inner,manifest=datasets[mode]
            assert manifest==record["manifest"] and len(fit)==record["FIT_rows"] and len(inner)==record["COMMON_INNER_rows"]
            x,past,keep,_=state_study.make_tensors(inner,np.asarray(checkpoint["FIT_mean"],np.float32),np.asarray(checkpoint["FIT_scale"],np.float32))
            with torch.inference_mode():
                model(x,past,keep_current=keep)
                began=time.perf_counter()
                for _ in range(100): model(x[:1],past[:1],keep_current=keep[:1])
                ms=(time.perf_counter()-began)*10.
            assert np.isfinite(ms) and ms>0
            row={"study":"STATE", "family":mode, "actual_architecture":"SMALL_MLP", "objective":record["objective"],"seed":seed,
                 "experiment_uid":record["experiment_uid"],**capacity, "new_CPU_single_branch_forward_latency_ms":ms,
                 "latency_scope":p["timing_scope"],"timing_actual_input":{k:inner[0][k] for k in ("sequence","episode_uid","event_uid","branch","frame","role")},
                 "FIT_rows":len(fit),"INNER_rows":len(inner),"FIT_target_counts":record["FIT_target_counts"],
                 "eligible_support_and_exclusions":manifest, "optimization":verdict,
                 "fit_record":{"path":str(path),"sha256":attest(path)},
                 "checkpoint":{"path":record["checkpoint_path"],"sha256":attest(record["checkpoint_path"],record["checkpoint_sha256"])},
                 "actual_model_generated_on_policy":False,"MOT_effects":None,"MOT_effects_status":"PENDING_ORIGINAL_COMPLETE_GROUP_AUDIT"}
            rows.append(row)
            print({"actual_STATE_head_cost":record["experiment_uid"],"parameters":capacity["parameter_count"],"forward_ms":ms},flush=True)
    groups=all_seed_groups(rows,seeds)
    inventory=census(assessed)
    assert inventory["attempt_records"]==51 and inventory["verified_actual_optimizer_models"]==48 and inventory["measured_no_optimization_records"]==3 and inventory["unverified_records"]==0
    relational={"status":"NOT_TRAINED_PENDING_ORIGINAL_MEASURED_RELATIONAL_NECESSITY_TRIGGER",
        "original_trigger":prereg["relational_trigger"],"parameter_count":None,"latency_ms":None,"MOT_metrics":None,
        "generic_model_failure_or_parameter_count_not_necessity_proof":True}
    report={"stage":"N72R21R2","goal":GOAL,"utc":utcnow(),"status":"COMPLETE_ACTUAL_MAIN_STATE_CONTROLLER_INVENTORY_COSTS_NOT_TABLE3_TABLE4_EFFECT_OR_WHOLE_GOAL_CLOSURE",
        "scope":"MAIN42 + original nine controlled STATE dispositions only; pilot/current-identity/memory-risk heads have their own separate schemas and reports",
        "registered_controller_dispositions":51,"actual_optimized_controller_models":48,"measured_no_optimizer_records":no_fit,
        "optimization_census":inventory,"all3_seed_groups":groups,"relational_conditional":relational,
        "timing_protocol_sha256":sha256(PROTOCOL),"timing_scope":p["timing_scope"],
        "all_seven_MLP_objectives_same_architecture":cost["all_seven_MLP_objectives_same_actual_tensor_layout_and_capacity"],
        "same_eligible_rows_across_different_objectives_claimed":False,"evidence_sha256":refs,
        "MOT_effect_generalization_or_deployment_success_implied":False,"scientific_goal_complete":False,"next_association_stage_authorized":False}
    destination=write_json(RESULT,report)
    reference={"path":str(destination),"sha256":sha256(destination)}
    write_json("training/MODEL_MANIFEST.json",{**report,"detailed_inventory":reference})
    for name,family in REPORT_FAMILIES.items():
        values={k:v for k,v in groups.items() if k.startswith("MAIN/"+family+"__")}
        write_json("training/"+name+".json",{"stage":"N72R21R2","goal":GOAL,"family":family,
            "status":"COMPLETE_ACTUAL_ORIGINAL_ALL3_SEED_FITS_COSTS_FULL_MOT_EFFECTS_PENDING",
            "controller_inventory":reference,"groups":values,"MOT_metrics":None,"scientific_goal_complete":False,
            "initializations_candidates_FIT_INNER_and_sources_unchanged":True,"objective_specific_support_not_hidden":True})
    write_json("training/RELATIONAL.json",{"stage":"N72R21R2","goal":GOAL,**relational,"controller_inventory":reference,"scientific_goal_complete":False})
    update_status(latest_actual_controller_inventory_costs=reference,actual_controller_trained_models=48,
                  all_current_working_changes_published=False)
    print({"actual_controller_models":48,"no_fit_dispositions":3,"all3_seed_groups":len(groups),"named_training_reports":10},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("prepare","run"));args=parser.parse_args()
    (prepare if args.action=="prepare" else run)()
