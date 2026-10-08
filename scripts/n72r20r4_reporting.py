"""Sealed-run delivery proofs and honest cached-feature timing summaries."""
from __future__ import annotations

from scripts.n72r20r4_common import *


def timing_summary(manifests: list[dict], baseline: list[dict]) -> dict:
    if [r["sequence"] for r in manifests] != [r["sequence"] for r in baseline]:
        raise ValueError("timing sequence axis mismatch")
    frames=sum(r["frames"] for r in manifests)
    seconds=sum(r["seconds"] for r in manifests)
    base_seconds=sum(r["seconds"] for r in baseline)
    if frames!=sum(r["frames"] for r in baseline) or seconds<=0:
        raise ValueError("invalid timing frame denominator")
    return {"frames":frames,"seconds":seconds,"cached_feature_fps":frames/seconds,
            "milliseconds_per_frame":1000*seconds/frames,
            "incremental_milliseconds_per_frame_vs_causal_baseline":1000*(seconds-base_seconds)/frames,
            "includes_stateless_candidate_projection":True,
            "excludes_disk_loading_MOT_export_TrackEval_backbones":True,
            "live_SAM3_OSNet_end_to_end_FPS":"NOT_RUN_REUSED_FROZEN_CANDIDATE_STREAM",
            "timing_noise_may_give_negative_increment":True}


def verify_manifest(record: dict) -> dict:
    for kind in ("trajectory","trace"):
        path=Path(record[f"{kind}_path"])
        if sha256(path)!=record[f"{kind}_sha256"]:
            raise RuntimeError(f"sealed {kind} SHA mismatch: {path}")
    if record.get("runtime_gt_read") is not False or record.get("runtime_future_gt_used") is not False:
        raise RuntimeError("manifest lacks runtime GT exclusion")
    return {k:record[k] for k in ("sequence","name","trajectory_path","trajectory_sha256","trajectory_rows","trace_path","trace_sha256","model_manifest","configuration","code_sha256")}


def evidence(folds: list[dict], val: dict) -> dict:
    if len(folds)!=8 or set(f["heldout"] for f in folds)!=set(SEQUENCES):
        raise ValueError("incomplete dev axis")
    if len(val["manifests"])!=25:
        raise ValueError("incomplete VAL axis")
    names=list(folds[0]["manifests"])
    if any(set(f["manifests"])!=set(names) for f in folds):
        raise ValueError("incomplete ablation axis")
    dev_baseline=[f["manifests"]["B1_CAUSAL_BASELINE"] for f in folds]
    verified={};timings={};pair_sha={}
    for name in names:
        records=[f["manifests"][name] for f in folds]
        verified[name]=[verify_manifest(r) for r in records]
        timings[name]=timing_summary(records,dev_baseline)
        pair_sha[name]={r["sequence"]:{"baseline":b["trajectory_sha256"],"treatment":r["trajectory_sha256"],"equal":r["trajectory_sha256"]==b["trajectory_sha256"]} for b,r in zip(dev_baseline,records)}
    val_axis=sorted(val["manifests"])
    val_baseline=[val["manifests"][s]["baseline"] for s in val_axis]
    val_treatment=[val["manifests"][s]["treatment"] for s in val_axis]
    val_verified={s:{name:verify_manifest(r) for name,r in val["manifests"][s].items()} for s in val_axis}
    timings["VAL_BASELINE"]=timing_summary(val_baseline,val_baseline)
    timings["VAL_TREATMENT"]=timing_summary(val_treatment,val_baseline)
    training={};checkpoint_proofs=[]
    for sequence in SEQUENCES:
        r=read_json(OUT/"association/training"/f"{sequence}.json")
        path=Path(r["training_data"])
        if sha256(path)!=r["training_data_sha256"]:raise RuntimeError("training decision SHA mismatch")
        rows=read_zstd_jsonl(path)
        actual_positive=sum(x["beneficial_intervention"] for x in rows)
        if len(rows)!=r["examples"] or actual_positive!=r["beneficial_examples"]:
            raise RuntimeError("authority label count mismatch")
        training[sequence]={"examples":len(rows),"beneficial_examples":actual_positive,
                            "changed_counterfactual_examples":sum(x["counterfactual_assignment_changed"] for x in rows),
                            "next_frame_supervision_examples":sum(x["next_state_supervision_only"] for x in rows),
                            "positive_label_collapse":actual_positive==0,
                            "training_data_path":str(path),"training_data_sha256":r["training_data_sha256"]}
        for cp in r["checkpoints"]:
            if sha256(Path(cp["path"]))!=cp["sha256"]:raise RuntimeError("authority checkpoint SHA mismatch")
            checkpoint_proofs.append({k:cp[k] for k in ("path","sha256","seed","mode","parameters")})
    for records in read_json(OUT/"adapter/STRICT_FOLD_CHECKPOINTS.json")["completed_folds"].values():
        for cp in records:
            if sha256(Path(cp["path"]))!=cp["sha256"]:raise RuntimeError("strict Adapter checkpoint SHA mismatch")
            checkpoint_proofs.append({k:cp[k] for k in ("path","sha256","seed","actual_training_sequences","parameters")})
    if sha256(MEMORY_CHECKPOINT)!=MEMORY_SHA:raise RuntimeError("frozen GRU changed")
    checkpoint_proofs.append({"path":str(MEMORY_CHECKPOINT),"sha256":MEMORY_SHA,"architecture":"frozen N72R18 GRU","new_training":False})
    for cp in read_json(OUT/"val/FROZEN_POLICY.json")["adapter_checkpoints"]:
        if sha256(Path(cp["path"]))!=cp["sha256"]:raise RuntimeError("VAL Adapter checkpoint SHA mismatch")
        checkpoint_proofs.append({k:cp[k] for k in ("path","sha256","seed","epochs","fit_sequences")})
    frozen=read_json(OUT/"causal_tracker/RUNTIME_CODE_FREEZE.json")["code_sha256"]
    active=code_manifest()
    causal_files=[p for p in frozen if p.startswith("sam3_intermot/association/") or p in {
        "scripts/n72r20r4_common.py","scripts/n72r20r4_run_causal_tracker.py",
        "scripts/n72r20r4_run_loso.py","scripts/n72r20r4_train_authority.py",
        "scripts/n72r20r4_adapter_integration.py","scripts/n72r20r4_counterfactual.py",
        "scripts/n72r20r4_trackeval.py"}]
    if any(active[p]!=frozen[p] for p in causal_files):
        raise RuntimeError("formal causal runtime changed after freeze")
    result={"status":"ALL_SEALED_CONTENT_SHA_VERIFIED","dev_run_count":sum(len(r) for r in verified.values()),
            "val_run_count":sum(len(r) for r in val_verified.values()),"dev_manifests":verified,"val_manifests":val_verified,
            "dev_baseline_treatment_trajectory_SHA":pair_sha,"model_SHA_proofs":checkpoint_proofs,
            "authority_supervision":training,"runtime_core_frozen_hashes_verified":{p:active[p] for p in causal_files},
            "diagnostic_reporting_storage_code_changes_not_policy_changes":True}
    write_json(OUT/"checkpoints/SEALED_EVIDENCE.json",result)
    write_json(OUT/"causal_tracker/LATENCY.json",timings)
    write_json(OUT/"association/AUTHORITY_SUPERVISION_AUDIT.json",training)
    return {"sealed_run_count":result["dev_run_count"]+result["val_run_count"],"timings":timings,"authority_supervision":training}
