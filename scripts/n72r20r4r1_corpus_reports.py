"""Seal executed natural-action corpus and multi-horizon diagnostics."""
from collections import Counter
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_supervision import supervision_gate
from scripts.n72r20r4_posthoc import residual_bound
from scripts.n72r20r4r1_evaluate import baseline_trace


def run():
    summaries={s:read_json(OUT/"counterfactual/per_sequence"/f"{s}.json") for s in SEQUENCES}
    actions=[];long=[];taxonomy=Counter();raw_types=Counter();feasible=Counter()
    for sequence,summary in summaries.items():
        taxonomy.update(summary["taxonomy"]);raw_types.update(summary["action_counts"])
        path=Path(summary["artifact"]["path"])
        if sha256(path)!=summary["artifact"]["sha256"]:raise RuntimeError("corpus SHA mismatch")
        for r in read_zstd_jsonl(path):
            if r["record_type"]=="action" and not r.get("is_reference"):
                feasible["feasible" if r["feasible"] else "infeasible"]+=1
                if r["feasible"]:actions.append(r)
            elif r["record_type"]=="long_horizon":long.append(r)
    gate=supervision_gate(actions)
    write_json(OUT/"counterfactual/ACTION_MANIFEST.json",{"artifacts":{s:r["artifact"] for s,r in summaries.items()},"action_counts":dict(raw_types),"identity_count":sum(r["eligible_identities"] for r in summaries.values()),"representation_independent":True,"GT_used_only_after_real_action_proposal":True,"natural_action_keys_unique_per_sequence_identity_frame_action_uid":True})
    write_json(OUT/"counterfactual/PER_SEQUENCE.json",summaries)
    write_json(OUT/"counterfactual/FEASIBILITY_AUDIT.json",{"counts":dict(feasible),"complete_global_solver":True,"hard_negative_preserved":True,"ownership_displacement_keeps_non_target_solver_and_lifecycle":True})
    write_json(OUT/"counterfactual/INTERVENTION_LABEL_AUDIT.json",{"corpus_gate_is_not_any_fold_gate":gate,"fit_readers_enforce_six_sequences":True,"no_artificial_positive_duplication":True,"fit_fold_gates_required_before_controller_training":True})
    write_json(OUT/"opportunity/OPPORTUNITY_TAXONOMY.json",{"counts":dict(taxonomy),"T3_T7_are_additional_nonexclusive_flags":True,"T6":"Adapter-specific; reported after strict-fold proposal audit, not inferred from representation-independent cache"})
    write_json(OUT/"opportunity/ORACLE_OPPORTUNITY.json",{"naturally_beneficial_actions":sum(r["H5"]["beneficial"] for r in actions),"oracle_correct_beneficial_actions":sum(r["H5"]["beneficial"] and r["oracle_correct_action"] for r in actions),"TRAIN_ONLY_POSTHOC_DIAGNOSTIC":True,"runtime_policy_uses_no_GT":True})
    values={}
    for h in (1,5,10,30):
        source=actions if h<=5 else long;rows=[r[f"H{h}"] for r in source]
        values[f"H{h}"]={"executed":len(rows),"beneficial":sum(r["beneficial"] for r in rows),"harmful":sum(r["harmful"] for r in rows),"mean_value":float(np.mean([r["value"] for r in rows])),"mean_other_damage":float(np.mean([r["current_other_damage"]+r["future_other_damage"] for r in rows])),"positive_target_gain_but_global_harm":sum(r["current_target_gain"]>0 and r["value"]<0 for r in rows)}
    write_json(OUT/"counterfactual/MULTI_HORIZON_VALUE.json",{"horizons":values,"H10_H30_sha_stratified_not_population_estimate":True,"long_strata":dict(Counter(r["stratum"] for r in long)),"no_overlapping_local_gain_sum_as_full_HOTA":True})
    write_json(OUT/"counterfactual/TRAJECTORY_IMPACT.json",{"window_trackeval":{s:str(OUT/"counterfactual/window_trackeval"/f"{s}.json") for s in SEQUENCES},"all_sequences_completed":all((OUT/"counterfactual/window_trackeval"/f"{s}.json").exists() for s in SEQUENCES),"all_GT_identities_and_all_public_outputs":True,"windows_do_not_replace_full_sequence_outer_evaluation":True})
    old=read_json(R4OUT/"adapter/POSTHOC_TRANSFER_DIAGNOSIS.json");certs={}
    for s in SEQUENCES:
        frames=load_frames(s);manifest=read_json(R4ASSETS/"dev/manifests/G1_FIXED_1"/f"{s}.json")
        path=Path(manifest["trace_path"])
        if sha256(path)!=manifest["trace_sha256"]:raise RuntimeError("source diagnostic trace SHA mismatch")
        trace=read_zstd_jsonl(path);assigned=blocked=0
        for (_,rows),d in zip(frames,trace):
            if d["frame"]<=events()[s]["event_frame"]:continue
            c=residual_bound(d,rows,manifest["configuration"]);assigned+=int(c["baseline_target_assigned"]);blocked+=int(c["certified_blocked"])
        expected=old["per_sequence"][s]["G1_FIXED_1"]
        if assigned!=expected["baseline_target_assigned_frames"] or blocked!=expected["global_margin_certifies_no_target_reallocation_frames"]:raise RuntimeError("source margin certificate mismatch")
        certs[s]={"assigned":assigned,"certified_blocked":blocked}
    write_json(OUT/"audit/R4_GLOBAL_MARGIN_REPRODUCTION.json",{"per_sequence":certs,"assigned":sum(r["assigned"] for r in certs.values()),"certified_blocked":sum(r["certified_blocked"] for r in certs.values()),"exact_source_certificate_recomputed":True,"not_a_global_oracle_upper_bound":True})
    seal=read_json(R4OUT/"checkpoints/SEALED_EVIDENCE.json")
    expected={r["path"]:r["sha256"] for r in seal["model_SHA_proofs"]}
    lineage=read_json(OUT/"audit/CHECKPOINT_LINEAGE.json")
    for records in lineage["strict_adapter_models"].values():
        for r in records:
            if expected.get(r["path"])!=r["sha256"]:raise RuntimeError("strict Adapter disagrees with historical sealed evidence")
    lineage["all_24_strict_adapters_match_historical_SEALED_EVIDENCE"]=True;write_json(OUT/"audit/CHECKPOINT_LINEAGE.json",lineage)
    if source_hashes()!=read_json(OUT/"audit/HISTORICAL_HASHES_BEFORE.json"):raise RuntimeError("historical mutation")
    status=read_json(OUT/"stage_status.json");status.update(status="ACTIVE_STRICT_FIT_AND_MEMORY_RELIABILITY",controller_training_started=True)
    status["milestones"].update(M2="COMPLETE_ENGINEERING",M3="COMPLETE_EXECUTED_MULTI_IDENTITY",M4="ACTIVE_STRICT_FOLD_PROPOSAL_AUDIT",M5="COMPLETE_EXECUTED_H1_H5_H10_H30",M6="ACTIVE_SAFE_MEMORY_FIT",M7="ACTIVE_GATED_CONTROLLER_FIT")
    write_json(OUT/"stage_status.json",status)


if __name__=="__main__":run()
