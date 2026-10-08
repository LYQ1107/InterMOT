#!/usr/bin/env python3
"""M4-M11 preregistered inner selection and batched outer causal TrackEval."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace

from scripts.n72r20r4_common import *
from scripts.n72r20r4_run_causal_tracker import run_rollout, export_run
from scripts.n72r20r4_adapter_integration import strict_ensemble, historical_ensemble, fold_split
from scripts.n72r20r4_train_authority import train_fold, load_controller
from scripts.n72r20r4_counterfactual import audit_pair
from scripts.n72r20r4_trackeval import evaluate, paired_metrics
from sam3_intermot.association.identity_authority import AuthorityConfig, ControllerEnsemble


def configurations(calibration: dict) -> dict[str, AuthorityConfig]:
    c = AuthorityConfig(**calibration)
    result={"G0":replace(c,mode="off",strength=0),"RAW_REID":replace(c,mode="fixed",source="raw",strength=0.5),"LEGACY_BASE_SCORE":replace(c,mode="fixed",source="legacy",strength=0.5),"ADAPTER_P0":replace(c,mode="fixed",strength=0.5),"ADAPTER_P1":replace(c,mode="fixed",strength=0.5,memory="P1"),"ADAPTER_P2":replace(c,mode="fixed",strength=0.5,memory="P2"),"SHUFFLED_STATE":replace(c,mode="fixed",source="shuffled",strength=0.5),"R4_CONFLICT_GUARD":replace(c,mode="fixed",strength=0.5,conflict_guard=True),"R6_PERSISTENCE":replace(c,mode="recovery",strength=0.5,conflict_guard=True,persistence_guard=True)}
    for strength in (0.1,0.25,0.5,1.0):
        result[f"G1_FIXED_{strength:g}"]=replace(c,mode="fixed",strength=strength)
    for mode,prefix in (("low_confidence","G2_LOW_CONFIDENCE"),("recovery","G3_RECOVERY")):
        result[prefix]=replace(c,mode=mode,strength=0.5)
    for mode in ("scalar","logistic","mlp","structured"):
        result[f"G4_{mode.upper()}"]=replace(c,mode=mode,strength=1)
    return result


def load_trace(group: str, name: str, sequence: str) -> list[dict]:
    return read_zstd_jsonl(ASSETS / group / "traces" / name / f"{sequence}.jsonl.zst")


def run_one(group: str,name: str,sequence: str,config: AuthorityConfig,adapter,controller=None,reference=None,frames=None,event=None) -> tuple[list,dict]:
    path=ASSETS/group/"manifests"/name/f"{sequence}.json"
    if path.exists():
        manifest=read_json(path)
        if manifest.get("configuration") != config.to_dict():
            raise ValueError(f"changed frozen configuration in completed run {path}")
        if sha256(Path(manifest["trajectory_path"]))!=manifest["trajectory_sha256"] or sha256(Path(manifest["trace_path"]))!=manifest["trace_sha256"]:
            raise ValueError("completed trajectory/trace SHA mismatch")
        expected=[] if adapter is None else adapter.manifest
        if manifest.get("model_manifest")!=expected:
            raise ValueError("changed completed adapter ensemble")
        return load_trace(group,name,sequence),manifest
    trace,profile=run_rollout(sequence,config=config,adapter=adapter,controller=controller,reference=reference,frames=frames,event=event)
    return trace,export_run(name,sequence,trace,profile,group=group)


def choose_inner(metrics: dict, configs: dict) -> dict:
    base=metrics["G0"]
    feasible=[name for name,m in metrics.items() if name in configs and m["DetA"]>=base["DetA"]-0.005]
    selected=max(feasible,key=lambda n:(metrics[n]["HOTA"],metrics[n]["AssA"],-metrics[n]["IDSW"],-(0 if n=="G0" else 1 if configs[n].mode in {"fixed","low_confidence","recovery"} else 2),n))
    return {"name":selected,"configuration":configs[selected].to_dict(),"metrics":metrics[selected],"baseline_metrics":base,"selection_split":"inner_validation_only","heldout_GT_used":False,"criterion":"HOTA with DetA>=baseline-0.005; then AssA, IDSW and complexity"}


def run_fold(heldout: str) -> dict:
    fit,validation=fold_split(heldout)
    training=train_fold(heldout)
    adapter=strict_ensemble(heldout)
    configs=configurations(training["calibration"])
    for name in ("RAW_REID","LEGACY_BASE_SCORE"):
        configs[name]=replace(configs[name],**training["source_calibrations"][configs[name].source])
    controller_ensembles={mode:ControllerEnsemble([load_controller(heldout,mode,seed) for seed in SEEDS]) for mode in ("scalar","logistic","mlp","structured")}
    group=f"inner/{heldout}"
    selection_path=OUT/"association/inner"/f"{heldout}.json"
    if selection_path.exists():
        selection=read_json(selection_path)
    else:
        frames=load_frames(validation)
        event=events()[validation]
        baseline,_=run_one(group,"G0",validation,configs["G0"],None,frames=frames,event=event)
        for name,config in configs.items():
            if name=="G0":continue
            controller=controller_ensembles.get(config.mode)
            needed_adapter=adapter if config.source in {"adapter","shuffled"} else None
            run_one(group,name,validation,config,needed_adapter,controller,baseline,frames,event)
        evaluation=evaluate(group,list(configs),[validation])
        selection={"heldout_sequence":heldout,"fit_sequences":fit,"inner_validation_sequence":validation,"selection":choose_inner(evaluation["metrics"],configs),"all_inner_metrics":evaluation["metrics"],"adapter_manifest":adapter.manifest,"frozen_before_heldout":True,"outer_tuning":False}
        write_json(selection_path,selection)
    # All outer configurations/weights are fixed before any heldout GT is
    # loaded by audit_pair or official TrackEval.
    names_to_run={
        "B1_CAUSAL_BASELINE":configs["G0"],
        "B2_ADAPTER_NO_WRITES":configs["ADAPTER_P0"],
        "B3_CONSENSUS_MEMORY":configs["ADAPTER_P1"],
        "MEMORY_P2_RELIABLE":configs["ADAPTER_P2"],
        "RAW_REID":configs["RAW_REID"],
        "LEGACY_BASE_SCORE":configs["LEGACY_BASE_SCORE"],
        "B8_SHUFFLED_STATE":configs["SHUFFLED_STATE"],
        "R2_LOW_CONFIDENCE":configs["G2_LOW_CONFIDENCE"],
        "R2_RECOVERY":configs["G3_RECOVERY"],
        "R4_CONFLICT_GUARD":configs["R4_CONFLICT_GUARD"],
        "R6_PERSISTENCE":configs["R6_PERSISTENCE"],
        "B5_SCALAR":configs["G4_SCALAR"],
        "LOGISTIC_GATE":configs["G4_LOGISTIC"],
        "B6_MLP":configs["G4_MLP"],
        "B7_STRUCTURED":configs["G4_STRUCTURED"],
        "SELECTED_TREATMENT":AuthorityConfig(**selection["selection"]["configuration"]),
    }
    fixed_names=[n for n in configs if n.startswith("G1_FIXED")]
    best_fixed=max(fixed_names,key=lambda n:selection["all_inner_metrics"][n]["HOTA"])
    names_to_run["B4_FIXED_INNER_SELECTED"]=configs[best_fixed]
    for n in fixed_names:names_to_run[n]=configs[n]
    for mode in ("scalar","logistic","mlp","structured"):
        for seed in SEEDS:names_to_run[f"SEED_{mode.upper()}_{seed}"]=configs[f"G4_{mode.upper()}"]
    write_json(OUT/"association/frozen_outer"/f"{heldout}.json",{"heldout":heldout,"configs":{n:c.to_dict() for n,c in names_to_run.items()},"inner_selection":selection["selection"],"fit_sequences":fit,"validation":validation,"frozen_before_heldout":True})
    frames=load_frames(heldout)
    event=events()[heldout]
    baseline,_=run_one("dev","B1_CAUSAL_BASELINE",heldout,names_to_run["B1_CAUSAL_BASELINE"],None,frames=frames,event=event)
    audits={}
    manifests={}
    for name,config in names_to_run.items():
        controller=controller_ensembles.get(config.mode)
        if name.startswith("SEED_"):
            controller=load_controller(heldout,config.mode,int(name.rsplit("_",1)[1]))
        needed_adapter=adapter if config.source in {"adapter","shuffled"} and config.mode!="off" else None
        trace,manifest=run_one("dev",name,heldout,config,needed_adapter,controller,baseline,frames,event)
        manifests[name]=manifest
        audits[name]=audit_pair(heldout,baseline,trace)
        write_zstd(ASSETS/"dev/audits"/name/f"{heldout}__interventions.jsonl.zst",audits[name]["interventions"])
        write_zstd(ASSETS/"dev/audits"/name/f"{heldout}__memory.jsonl.zst",audits[name]["memory_rows"])
        audits[name]={"summary":audits[name]["summary"]}
        print(json.dumps({"heldout":heldout,"variant":name,"changed_frames":audits[name]["summary"]["changed_frames"],"N01":audits[name]["summary"]["N01"],"N10":audits[name]["summary"]["N10"]}),flush=True)
    # The verified seven-sequence historic adapter is used only here, after
    # policy freeze, as an explicit model-transfer comparison on its heldout.
    historical=historical_ensemble(heldout)
    name="HISTORICAL_R3R2_ADAPTER7"
    trace,manifest=run_one("dev",name,heldout,configs["ADAPTER_P0"],historical,reference=baseline,frames=frames,event=event)
    audit=audit_pair(heldout,baseline,trace)
    audits[name]={"summary":audit["summary"]};manifests[name]=manifest
    write_zstd(ASSETS/"dev/audits"/name/f"{heldout}__interventions.jsonl.zst",audit["interventions"])
    write_zstd(ASSETS/"dev/audits"/name/f"{heldout}__memory.jsonl.zst",audit["memory_rows"])
    result={"heldout":heldout,"selection":selection,"manifests":manifests,"posthoc_audits":audits,"outer_tuning":False,"runtime_future_gt_used":False,"code_sha256":code_manifest()}
    write_json(OUT/"association/folds"/f"{heldout}.json",result)
    return result


def finalize_dev(folds: list[dict]) -> dict:
    if len(folds)!=8:raise ValueError("all eight folds required")
    names=list(folds[0]["manifests"])
    evaluation=evaluate("dev",names,list(SEQUENCES))
    metric=evaluation["metrics"]
    baseline=metric["B1_CAUSAL_BASELINE"];treatment=metric["SELECTED_TREATMENT"]
    paired=paired_metrics(baseline,treatment)
    delta=paired["combined_delta"]
    gate=delta["HOTA"]>=0.005 and delta["AssA"]>0 and delta["DetA"]>=-0.005
    for filename,value in (("BASELINE.json",baseline),("TREATMENT.json",treatment),("ABLATIONS.json",metric),("PER_SEQUENCE.json",{n:m["per_sequence"] for n,m in metric.items()}),("PAIRED_BOOTSTRAP.json",paired)):
        write_json(OUT/"dev_trackeval"/filename,value)
    write_json(OUT/"association/INNER_SELECTION.json",{"folds":[f["selection"] for f in folds],"all_architectures_selected_on_inner_only":True})
    summary={n:{"changed_frames":sum(f["posthoc_audits"][n]["summary"]["changed_frames"] for f in folds),"N01":sum(f["posthoc_audits"][n]["summary"]["N01"] for f in folds),"N10":sum(f["posthoc_audits"][n]["summary"]["N10"] for f in folds),"global_conflicts":sum(f["posthoc_audits"][n]["summary"]["global_conflicts"] for f in folds),"persistent_corrections_30":sum(f["posthoc_audits"][n]["summary"]["persistent_corrections_30"] for f in folds),"paired_HOTA":metric[n]["HOTA"]-baseline["HOTA"]} for n in names}
    write_json(OUT/"association/GLOBAL_INTERVENTION_SUMMARY.json",summary)
    for g,prefix in (("G0","B1"),("G1","G1_FIXED"),("G2","R2_LOW"),("G3","R2_RECOVERY"),("G4","B7")):
        write_json(OUT/f"association/{g}.json",{n:{"metrics":metric[n],"intervention":summary[n]} for n in names if n.startswith(prefix)})
    result={"status":"DEV_COMPLETE","baseline":baseline,"treatment":treatment,"deltas":delta,"paired":paired,"development_gate_pass":gate,"interventions":summary,"evaluation_provenance":evaluation["provenance"],"selected_names":[f["selection"]["selection"]["name"] for f in folds],"outer_GT_used_for_selection":False}
    write_json(OUT/"dev_trackeval/RESULT.json",result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--heldouts",nargs="+",default=list(SEQUENCES))
    parser.add_argument("--finalize-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    folds=[]
    for heldout in args.heldouts:
        path=OUT/"association/folds"/f"{heldout}.json"
        folds.append(read_json(path) if path.exists() else run_fold(heldout))
    if len(folds)==8:print(json.dumps(finalize_dev(folds),sort_keys=True))
