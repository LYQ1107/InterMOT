#!/usr/bin/env python3
"""M12: train-side policy closure followed by fixed all-25 VAL confirmation."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
from scripts.n72r20r4_common import *
from scripts import n72r20r3r2_representation as r3
from scripts.n72r20r4_adapter_integration import AdapterEnsemble
from scripts.n72r20r4_train_authority import train_fold, load_controller
from scripts.n72r20r4_run_causal_tracker import run_rollout, export_run
from scripts.n72r20r4_trackeval import evaluate, paired_metrics
from scripts.n72r20r4_counterfactual import audit_pair
from sam3_intermot.association.identity_authority import AuthorityConfig, ControllerEnsemble


def freeze() -> dict:
    path=OUT/"val/FROZEN_POLICY.json"
    if path.exists():
        policy=read_json(path)
        for r in policy["adapter_checkpoints"]:
            if sha256(Path(r["path"]))!=r["sha256"]:raise ValueError("frozen VAL checkpoint SHA mismatch")
        return policy
    dev=read_json(OUT/"dev_trackeval/RESULT.json")
    folds=[read_json(OUT/"association/inner"/f"{s}.json") for s in SEQUENCES]
    counts=Counter(f["selection"]["name"] for f in folds)
    chosen=max(counts,key=lambda name:(counts[name],name=="G0",-len(name),name))
    matching=[f["selection"]["configuration"] for f in folds if f["selection"]["name"]==chosen]
    config=AuthorityConfig(**matching[0])
    # Fits use only the eight train-development sequences and frozen
    # architecture. Neither checkpoint nor hyperparameters see VAL labels.
    checkpoint_records=[]
    dataset=None
    histories=read_json(OUT/"adapter/STRICT_FOLD_CHECKPOINTS.json")["completed_folds"]
    epochs=max(1,int(np.median([r["training"]["best_epoch"] for records in histories.values() for r in records])))
    for seed in SEEDS:
        model_path=ASSETS/"models"/f"adapter8__VAL__seed{seed}.pt"
        manifest_path=model_path.with_suffix(".json")
        if manifest_path.exists():
            record=read_json(manifest_path)
        else:
            check_storage(reserve_gib=0.03)
            dataset=r3.load_episode_dataset() if dataset is None else dataset
            model,training=r3._fit_final(dataset,dataset.episodes,seed,epochs,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            model_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"stage":STAGE,"architecture":"CrossSceneIdentityAdapter","feature_dim":512,"bottleneck_dim":128,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"trainable_parameters":265472,"seed":seed,"actual_training_sequences":list(SEQUENCES),"parameter_fit_sequences":list(SEQUENCES),"VAL_used_for_training":False},model_path)
            record={"path":model_path,"sha256":sha256(model_path),"seed":seed,"epochs":epochs,"fit_sequences":SEQUENCES,"training":training}
            write_json(manifest_path,record)
        if sha256(model_path)!=record["sha256"]:raise ValueError("adapter8 SHA mismatch")
        checkpoint_records.append(record)
    adapter=AdapterEnsemble([Path(r["path"]) for r in checkpoint_records],expected_shas=[r["sha256"] for r in checkpoint_records])
    if config.mode in {"scalar","logistic","mlp","structured"}:
        training=train_fold("ALL_DEV_FOR_VAL",deployment_adapter=adapter)
        calibrated=training["calibration"]
    else:
        # Calibrations/thresholds were already selected on training sides.
        # Their sequence-macro median is frozen before any VAL run.
        calibrated={key:float(np.median([f["selection"]["configuration"][key] for f in folds])) for key in ("identity_mean","identity_std","base_scale","write_score","write_margin","write_quality")}
        training=None
    config=replace(config,**{key:calibrated[key] for key in ("identity_mean","identity_std","base_scale","write_score","write_margin","write_quality")})
    policy={"stage":STAGE,"goal_file":"outputs/N72R20R4/FINAL_GOAL.json","status":"FROZEN_BEFORE_VAL","selected_name":chosen,"selection_counts":counts,"configuration":config.to_dict(),"adapter_checkpoints":checkpoint_records,"authority_training":training,"source_development_result_sha256":sha256(OUT/"dev_trackeval/RESULT.json"),"code_sha256":code_manifest(),"VAL_previously_accessed_in_history":True,"VAL_used_for_selection_or_training":False,"test_accessed":False,"runtime_future_gt_used":False,"simulated_click_protocol":"reuse frozen R3R2R3 first available candidate+GT event; anchor is permitted human observation only, no future labels for runtime scheduling"}
    write_json(path,policy)
    return plain(policy)


def run() -> dict:
    policy=freeze()
    config=AuthorityConfig(**policy["configuration"])
    records=policy["adapter_checkpoints"]
    adapter=AdapterEnsemble([Path(r["path"]) for r in records],expected_shas=[r["sha256"] for r in records])
    controller=ControllerEnsemble([load_controller("ALL_DEV_FOR_VAL",config.mode,seed) for seed in SEEDS]) if config.mode in {"scalar","logistic","mlp","structured"} else None
    event_map=events("val")
    sequences=sorted(event_map)
    if len(sequences)!=25:raise ValueError("VAL must contain all 25 sequences")
    summaries={};manifests={}
    for sequence in sequences:
        manifest_path=ASSETS/"val/completed"/f"{sequence}.json"
        if manifest_path.exists():
            r=read_json(manifest_path);summaries[sequence]=r["posthoc_summary"];manifests[sequence]=r["manifests"]
            continue
        check_storage(reserve_gib=0.03)
        frames=load_frames(sequence,"val")
        event=event_map[sequence]
        b,bp=run_rollout(sequence,config=AuthorityConfig(),split="val",frames=frames,event=event)
        t,tp=run_rollout(sequence,config=config,split="val",adapter=adapter if config.source in {"adapter","shuffled"} and config.mode!="off" else None,controller=controller,reference=b,frames=frames,event=event)
        bm=export_run("BASELINE_CAUSAL",sequence,b,bp,group="val")
        tm=export_run("TREATMENT_CAUSAL",sequence,t,tp,group="val")
        audit=audit_pair(sequence,b,t,split="val")
        write_zstd(ASSETS/"val/audits"/f"{sequence}__interventions.jsonl.zst",audit["interventions"])
        write_zstd(ASSETS/"val/audits"/f"{sequence}__memory.jsonl.zst",audit["memory_rows"])
        result={"manifests":{"baseline":bm,"treatment":tm},"posthoc_summary":audit["summary"],"policy_sha256":sha256(OUT/"val/FROZEN_POLICY.json"),"VAL_tuning":False}
        write_json(manifest_path,result)
        summaries[sequence]=audit["summary"];manifests[sequence]=result["manifests"]
        write_json(OUT/"val/PROGRESS.json",{"completed":len(summaries),"total":25,"sequences":sorted(summaries),"policy_sha256":sha256(OUT/"val/FROZEN_POLICY.json")})
        print(json.dumps({"VAL_sequence":sequence,"completed":len(summaries),"N01":audit["summary"]["N01"],"N10":audit["summary"]["N10"]}),flush=True)
    evaluated=evaluate("val",["BASELINE_CAUSAL","TREATMENT_CAUSAL"],sequences,split="val")
    b=evaluated["metrics"]["BASELINE_CAUSAL"];t=evaluated["metrics"]["TREATMENT_CAUSAL"]
    paired=paired_metrics(b,t)
    write_json(OUT/"val/BASELINE.json",b);write_json(OUT/"val/TREATMENT.json",t);write_json(OUT/"val/DELTA.json",paired)
    result={"status":"VAL_COMPLETE","all_25_sequences":True,"baseline":b,"treatment":t,"paired":paired,"target_summaries":summaries,"manifests":manifests,"evaluation_provenance":evaluated["provenance"],"VAL_tuning":False,"test_accessed":False,"runtime_future_gt_used":False}
    write_json(OUT/"val/RESULT.json",result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--freeze-only",action="store_true");args=parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(freeze() if args.freeze_only else run(),sort_keys=True))
