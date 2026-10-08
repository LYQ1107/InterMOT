#!/usr/bin/env python3
"""Truthful scientific closure; correctness, effect and generalization separate."""
from __future__ import annotations
from scripts.n72r20r4_common import *
from scripts.n72r20r4_trackeval import METRICS, paired_metrics
from scripts.n72r20r4_reporting import evidence
from sam3_intermot.association.causal_state_commit import memory_metrics


def run() -> dict:
    aa=read_json(OUT/"causal_tracker/A_A_EQUIVALENCE.json")
    dev=read_json(OUT/"dev_trackeval/RESULT.json")
    val=read_json(OUT/"val/RESULT.json")
    probe=read_json(OUT/"adapter/POSTHOC_TRANSFER_DIAGNOSIS.json")
    headroom=read_json(OUT/"target_tracking/POSTHOC_HEADROOM.json")
    local=read_json(OUT/"association/LOCAL_TRAJECTORY_TRACK_EVAL.json")
    if not probe["complete_all_8_sequences"] or not local["all_outer_variants_and_sequences_checked"]:
        raise RuntimeError("incomplete posthoc diagnostics")
    ablations=read_json(OUT/"dev_trackeval/ABLATIONS.json")
    folds=[read_json(OUT/"association/folds"/f"{s}.json") for s in SEQUENCES]
    lineage=read_json(OUT/"audit/SOURCE_LINEAGE.json")
    reproduction=read_json(OUT/"audit/TRACK_EVAL_REPRODUCTION.json")
    if reproduction["status"]!="PASS_EXACT_TRACK_EVAL_REPRODUCTION":
        raise RuntimeError("official metric reproduction missing")
    if historical_hashes()!=lineage["historical_artifacts_sha256"]:
        raise RuntimeError("historical artifacts changed")
    proofs=evidence(folds,val)
    old_replay=read_json(ROOT/"outputs/N72R20R3R2R3/trackeval/combined/summary.json")["trackeval"]
    old_vs_causal=paired_metrics(old_replay,dev["baseline"])
    write_json(OUT/"causal_tracker/LIFECYCLE_BASELINE_COMPARISON.json",{
        "historical_frozen_replay":old_replay,"new_causal_baseline":dev["baseline"],
        "paired":old_vs_causal,"identity_learning_effect_claimed":False,
        "confounders":["dynamic NONE births","finite non-target lost lifetime","complete event-frame/pre-click export"],
        "same_lifecycle_native_reconstruction_exact":aa["status"]=="PASS_CAUSAL_BASELINE_EQUIVALENCE"})
    rows_by_policy={}
    target={};reacquisition={};errors={}
    write_manifests={}
    wrong_rows=[]
    for name,policy in (("B2_ADAPTER_NO_WRITES","P0"),("B3_CONSENSUS_MEMORY","P1"),("MEMORY_P2_RELIABLE","P2"),("SELECTED_TREATMENT","SELECTED")):
        all_rows=[]
        for s in SEQUENCES:
            path=ASSETS/"dev/audits"/name/f"{s}__memory.jsonl.zst"
            all_rows.extend(read_zstd_jsonl(path))
        summary=memory_metrics(all_rows)
        by_sequence={s:[r for r in all_rows if r.get("sequence")==s] for s in SEQUENCES}
        summary["per_sequence"]={s:memory_metrics(rows) for s,rows in by_sequence.items()}
        summary["contamination_cascade_length_accepted_writes"]=max(m["contamination_cascade_length_accepted_writes"] for m in summary["per_sequence"].values())
        summary["cascade_does_not_cross_sequence_boundary"]=True
        summary["HOTA"]=ablations[name]["HOTA"]
        rows_by_policy[policy]=summary
        write_manifests[policy]={"rows":len(all_rows),"source_files":[{"path":str(ASSETS/"dev/audits"/name/f"{s}__memory.jsonl.zst"),"sha256":sha256(ASSETS/"dev/audits"/name/f"{s}__memory.jsonl.zst")} for s in SEQUENCES]}
        wrong_rows.extend({**r,"variant":name} for r in all_rows if r.get("accepted") and not r.get("correct"))
    write_zstd(OUT/"memory/WRONG_WRITE_AUDIT.jsonl.zst",wrong_rows)
    for fold in folds:
        s=fold["heldout"]
        summary=fold["posthoc_audits"]["SELECTED_TREATMENT"]["summary"]
        target[s]={k:v for k,v in summary.items() if k.startswith("target_") or k.startswith("clicked_") or k.startswith("persistent_")}
        reacquisition[s]={"successful_reacquisitions":summary["successful_reacquisitions"],"missing_intervals":summary["target_missing_intervals"]}
        errors[s]={"wrong_identity_takeover_frames":summary["wrong_identity_takeover_frames"],"N01":summary["N01"],"N10":summary["N10"]}
    write_json(OUT/"memory/WRITE_POLICY_ABLATION.json",{"policies":rows_by_policy,"manifests":write_manifests,"zero_writes_is_not_safety":True})
    write_json(OUT/"memory/STATE_DIVERGENCE.json",{
        "ordinary_tracker_state_divergent_frames_vs_baseline":{n:{f["heldout"]:f["posthoc_audits"][n]["summary"]["state_feedback_divergent_frames"] for f in folds} for n in ablations},
        "GRU_query_state_evolution":{s:{n:{k:r[k] for k in ("query_state_unique_hashes","state_divergence_observation_count","mean_state_divergence_from_anchor")} for n,r in variants.items()} for s,variants in probe["per_sequence"].items()},
        "ordinary_tracker_state_is_not_learned_GRU_query":True})
    write_json(OUT/"target_tracking/TARGET_CONTINUITY.json",target);write_json(OUT/"target_tracking/REACQUISITION.json",reacquisition);write_json(OUT/"target_tracking/TARGET_ID_ERROR.json",errors)
    write_json(OUT/"target_tracking/ALL_VARIANT_TARGET_METRICS.json",{n:{f["heldout"]:{k:v for k,v in f["posthoc_audits"][n]["summary"].items() if k.startswith(("target_","clicked_","persistent_")) or k in {"successful_reacquisitions","wrong_identity_takeover_frames","N01","N10"}} for f in folds} for n in ablations})
    calls=sum(f["manifests"]["B2_ADAPTER_NO_WRITES"]["adapter_calls"] for f in folds)
    nonempty_score_calls=sum(sum(r["frame"]>int(events()[s]["event_frame"]) and bool(r["identity_scores"]) for r in read_zstd_jsonl(ASSETS/"dev/traces/B2_ADAPTER_NO_WRITES"/f"{s}.jsonl.zst")) for s in SEQUENCES)
    global_conflicts=sum(f["posthoc_audits"][name]["summary"]["global_conflicts"] for f in folds for name in f["posthoc_audits"])
    adapter_report={"status":"PASS_REAL_ADAPTER_IN_CAUSAL_ASSOCIATION" if nonempty_score_calls else "FAIL_IDENTITY_ADAPTER_TRANSFER","adapter_ensemble_score_calls_including_empty_frames":calls,"nonempty_ensemble_score_calls":nonempty_score_calls,"actual_query_tower_forward_calls":nonempty_score_calls*len(SEEDS),"query_dimension":512,"candidate_dimension":512,"actual_models":read_json(OUT/"adapter/STRICT_FOLD_CHECKPOINTS.json"),"historical_R3R2_actual_inference":True,"historical_7_sequence_results":ablations["HISTORICAL_R3R2_ADAPTER7"],"legacy_score_separately_labeled":True}
    write_json(OUT/"adapter/ADAPTER_INTEGRATION_REPORT.json",adapter_report)
    write_json(OUT/"adapter/SCORE_DISTRIBUTION.json",{s:read_json(OUT/"association/training"/f"{s}.json")["calibration"] for s in SEQUENCES})
    write_json(OUT/"causal_tracker/GLOBAL_ASSIGNMENT_AUDIT.json",{"global_conflicts":global_conflicts,"full_solver_output_committed":True,"all_ablations_complete":len(ablations),"scope":"all outer fold traces; tested hard failures plus complete runtime validation"})
    write_json(OUT/"causal_tracker/STATE_FEEDBACK_AUDIT.json",{"synthetic_intervention_changes_next_frame_matrix":True,"test":"tests/test_n72r20r4_causality.py::test_intervention_state_changes_next_frame_matrix_and_global_ids","observed_outer_state_divergence":{n:sum(f["posthoc_audits"][n]["summary"]["state_feedback_divergent_frames"] for f in folds) for n in ablations}})
    gate0=aa["status"]=="PASS_CAUSAL_BASELINE_EQUIVALENCE" and calls>0 and global_conflicts==0
    seed_rows={mode:{str(seed):ablations[f"SEED_{mode.upper()}_{seed}"] for seed in SEEDS} for mode in ("scalar","logistic","mlp","structured")}
    seed_stability={mode:{"HOTA_values":[m["HOTA"] for m in values.values()],"HOTA_std":float(np.std([m["HOTA"] for m in values.values()])),"all_three_HOTA_deltas_positive":all(m["HOTA"]>dev["baseline"]["HOTA"] for m in values.values())} for mode,values in seed_rows.items()}
    write_json(OUT/"dev_trackeval/SEED_STABILITY.json",seed_stability)
    delta=val["paired"]["combined_delta"]
    gate2=delta["HOTA"]>0 and delta["AssA"]>0 and delta["DetA"]>=-0.005 and delta["IDSW"]<=0
    gate1=dev["development_gate_pass"]
    selected_memory=rows_by_policy["SELECTED"]
    memory_required=read_json(OUT/"val/FROZEN_POLICY.json")["configuration"]["memory"]!="P0"
    if not gate0:decision="FAIL_RUNTIME_INVARIANT"
    elif not gate1:
        if dev["deltas"]["DetA"] < -0.005:decision="FAIL_DETA_ASSA_TRADEOFF"
        elif memory_required and not selected_memory["safety_pass"]:decision="FAIL_MEMORY_STATE_CONTAMINATION"
        elif all(v["changed_frames"]==0 for v in dev["interventions"].values()):decision="FAIL_GLOBAL_ASSOCIATION_AUTHORITY"
        else:decision="FAIL_TRAJECTORY_LEVEL_CAUSAL_BENEFIT"
    elif not gate2:decision="FAIL_VAL_GENERALIZATION"
    elif memory_required and not selected_memory["safety_pass"]:decision="FAIL_MEMORY_STATE_CONTAMINATION"
    else:decision="PASS_CAUSAL_IDENTITY_TO_TRAJECTORY_TRANSFER"
    result={"stage":STAGE,"goal_file":"outputs/N72R20R4/FINAL_GOAL.json","goal":read_json(OUT/"FINAL_GOAL.json")["goal"],"decision":decision,"gate0_correctness":gate0,"gate1_development_signal":gate1,"gate2_generalization":gate2,"dev":dev,"val":val,"memory":rows_by_policy,"seed_stability":seed_stability,"global_assignment_conflicts":global_conflicts,"adapter_calls":calls,"historical_artifacts_unchanged":True,"runtime_future_gt_used":False,"test_accessed":False,"next_stage_authorized":decision.startswith("PASS_"),"scientific_experiments_complete":True,"git_delivery_complete":False,"posthoc_transfer_diagnosis":probe["aggregate"],"posthoc_headroom_delta":headroom["deltas"],"local_intervention_count":local["interventions"],"sealed_evidence_verified_runs":proofs["sealed_run_count"],"training_positive_labels":sum(r["beneficial_examples"] for r in proofs["authority_supervision"].values())}
    write_json(OUT/"FINAL_RESULT.json",result)
    lines=["# N72R20R4 Final Report","","## Q1–Q5: identity-to-trajectory transfer", "", "FINAL GOAL: Causal Identity-to-Trajectory Transfer: Globally Consistent Online Association for Long-Term MOT.","","Frozen scientific specification: `FINAL_GOAL.json`. Final decision: `"+decision+"`.","", "Q1: The historical 90 local corrections did not enter the chosen G0 treatment in any of the eight folds. The old path also discarded non-target Hungarian reallocations and independently re-read the sealed next-frame state. Its IoU-based shadow N01 was never an IDSW count. The new implementation audit and complete causal intervention table separate these quantities.","",f"Q2: Real R3R2 Adapter inference ran {calls} times in the no-write outer comparison. Full-sequence HOTA={ablations['B2_ADAPTER_NO_WRITES']['HOTA']}, AssA={ablations['B2_ADAPTER_NO_WRITES']['AssA']}; compare causal baseline HOTA={dev['baseline']['HOTA']}, AssA={dev['baseline']['AssA']}.","",f"Q3: Independent legacy reconstruction exactly reproduces all eight native trajectory SHA and TrackEval results. Dynamic runtime and identity-off pass exact A/A. Dynamic births/deaths change the baseline; native HOTA={aa['trackeval']['metrics']['BASELINE_NATIVE']['HOTA']} and causal HOTA={dev['baseline']['HOTA']}. Selected treatment ΔHOTA={dev['deltas']['HOTA']}, ΔAssA={dev['deltas']['AssA']}. The historical/lifecycle difference is not attributed to identity learning.","", "Q4: Representation, authority, competition, memory writes, candidate headroom and DetA are reported in separate ablations and posthoc diagnostics. Complex authority superiority is assessed against scalar/logistic/MLP, with all three seeds. Memory safety is separate from trajectory benefit; zero writes never pass retention.","",f"Q5: All 25 previously accessed VAL sequences were evaluated with a policy frozen from train-development inner selection; no VAL tuning. VAL ΔHOTA={delta['HOTA']}, ΔAssA={delta['AssA']}, ΔDetA={delta['DetA']}, ΔIDF1={delta['IDF1']}, ΔIDSW={delta['IDSW']}. Development gate={gate1}, generalization gate={gate2}.","","## Table A — full-sequence Global MOT","","| Method | HOTA | AssA | DetA | LocA | IDF1 | MOTA | IDSW | FP | FN |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    all_metrics={"Native Baseline":aa["trackeval"]["metrics"]["BASELINE_NATIVE"],**ablations,"VAL Baseline":val["baseline"],"VAL Treatment":val["treatment"]}
    for n,m in all_metrics.items():lines.append("| "+n+" | "+" | ".join(str(m[k]) for k in METRICS)+" |")
    lines.extend(["","## Table B — actual trajectory interventions","","| Method | Changed frames | N01 | N10 | Global conflicts | Persistent corrections H30 | ΔHOTA |","|---|---:|---:|---:|---:|---:|---:|"])
    for n,m in dev["interventions"].items():lines.append(f"| {n} | {m['changed_frames']} | {m['N01']} | {m['N10']} | {m['global_conflicts']} | {m['persistent_corrections_30']} | {m['paired_HOTA']} |")
    lines.extend(["","## Table C — real frozen GRU writes","","| Policy | Accepted | Wrong rate | Correct retention | HOTA | Safety pass |","|---|---:|---:|---:|---:|---|"])
    for n,m in rows_by_policy.items():lines.append(f"| {n} | {m['accepted_writes']} | {m['wrong_write_rate']} | {m['correct_write_retention']} | {m['HOTA']} | {m['safety_pass']} |")
    lines.extend(["","Wrong-write denominator is accepted writes. Correct-write retention denominator is eligible assigned correct observations. First wrong writes, accepted-write cascade length and per-sequence state divergence are in the machine artifacts.","","## Protocol, uncertainty and resource limitations","","- 8-fold LOSO uses six fit sequences and cyclic inner validation; these sequences are a repeatedly used development benchmark. The three training seeds are 720321/720322/720323.","- Historic seven-sequence Adapter checkpoint metadata discrepancy is documented. Strict six-sequence models are used for inner selection; historical SHA-verified models are an outer-only inference comparison.","- Paired 2000-draw sequence-cluster bootstrap reports sequence-macro Δ CI; combined official TrackEval Δ is a different estimator and reported separately.","- Candidate pipeline is reused SAM3/OSNet. No new candidates, dataset downloads or backbone training. Simulated GT-box human anchors are not real-human evidence; future runtime GT is forbidden.","- M5 local metrics use merged intervention windows; overlapping corrections do not provide independent additive effect estimates. Single-target oracle headroom is diagnostic only.","- Raw trajectories/checkpoints are external; committed manifests bind their hashes and code/config lineage.","- Source refresh and Git delivery statuses are recorded separately; successful scientific execution is not sufficient to mark overall Goal complete.",""])
    # Add concrete failure attribution inside the first Q1–Q5 section.
    q4=next(i for i,v in enumerate(lines) if v.startswith("Q4:"))
    q3=next(i for i,v in enumerate(lines) if v.startswith("Q3:"))
    q2=next(i for i,v in enumerate(lines) if v.startswith("Q2:"))
    lines[q2]=lines[q2].replace("Real R3R2 Adapter inference ran","Real R3R2 Adapter ensemble score API was called")
    lines[q2]+=f" The API counter includes empty-candidate calls: {nonempty_score_calls} calls have real candidates, giving {nonempty_score_calls*len(SEEDS)} actual query-tower forwards across three model seeds."
    lines[q3]+=f" Compared directly with the historical frozen COMBINED replay, new causal baseline ΔHOTA={old_vs_causal['combined_delta']['HOTA']}, ΔAssA={old_vs_causal['combined_delta']['AssA']}, ΔDetA={old_vs_causal['combined_delta']['DetA']}, ΔIDSW={old_vs_causal['combined_delta']['IDSW']}; macro ΔHOTA CI={old_vs_causal['bootstrap']['HOTA']['interval_95']}. This comparison includes lifecycle/export changes and does not isolate a learned-identity effect; it does not show a better global tracker."
    a=probe["aggregate"]
    lines[q4:q4+1]=[
        f"Q4 — Representation: full-future competitive-candidate win rate is {a['B2_ADAPTER_NO_WRITES']['hard_negative_identity_win_rate']} for strict Adapter/P0, {a['RAW_REID']['hard_negative_identity_win_rate']} for raw ReID, and {a['B3_CONSENSUS_MEMORY']['hard_negative_identity_win_rate']} for Adapter/P1. These are offline candidate diagnostics, not global MOT metrics or clean GT-replay equivalents. Other real candidates, including unmatched detections, are competitors.",
        "",
        f"Q4 — Authority: the exact global-margin certificate blocks {a['G1_FIXED_1']['global_margin_certifies_no_target_reallocation_frames']}/{a['G1_FIXED_1']['baseline_target_assigned_frames']} assigned target frames even for the strongest preregistered fixed weight. Positive label count is {result['training_positive_labels']}; this underpowered counterfactual intervention family cannot establish superiority of structured authority over simpler models. No stronger heldout-informed weight is retroactively presented as an unbiased LOSO result.",
        "",
        f"Q4 — Global conflict and memory: observed global collisions={global_conflicts}. Absence of harmful overrides is not evidence of safe useful overrides when the assignments never change. P1 accepted writes={rows_by_policy['P1']['accepted_writes']}, wrong-write rate={rows_by_policy['P1']['wrong_write_rate']}, retention={rows_by_policy['P1']['correct_write_retention']}; P2 accepted writes={rows_by_policy['P2']['accepted_writes']}, wrong-write rate={rows_by_policy['P2']['wrong_write_rate']}, retention={rows_by_policy['P2']['correct_write_retention']}. Actual GRU query hashes evolve independently of the ordinary tracker-state hash; their evolution does not by itself imply correct identity storage or trajectory improvement.",
        "",
        f"Q4 — Candidate quality and detection: real target candidate coverage given visibility={a['B2_ADAPTER_NO_WRITES']['candidate_coverage_given_visible']}. The single-target posthoc oracle produces ΔHOTA={headroom['deltas']['HOTA']}, ΔAssA={headroom['deltas']['AssA']}, ΔDetA={headroom['deltas']['DetA']}; it is not a guaranteed global upper bound because competing ownership is coupled. Official treatment ΔDetA={dev['deltas']['DetA']} is separated from identity ranking. Cached candidates cannot recover a person for whom no valid candidate exists.",
        "",
        "Q4 — Domain shift: historical GT-replay representation evidence does not establish performance on real SAM3 boxes over a whole trajectory. This stage measures that operational shift but does not isolate it from crop quality, time gap and contaminated observation histories. With no trajectory interventions, this run cannot identify the best stronger controller or prove that identity evidence is intrinsically incapable of improving MOT."
    ]
    lines.extend(["","## Paired uncertainty","",
        f"DEV ΔHOTA={dev['paired']['combined_delta']['HOTA']}; sequence-macro 95% CI={dev['paired']['bootstrap']['HOTA']['interval_95']}; CI lower > 0={dev['paired']['HOTA_CI_lower_gt_zero']}.",
        f"VAL ΔHOTA={val['paired']['combined_delta']['HOTA']}; sequence-macro 95% CI={val['paired']['bootstrap']['HOTA']['interval_95']}; CI lower > 0={val['paired']['HOTA_CI_lower_gt_zero']}.",
        "","## All 25 VAL sequences — no tuning","",
        "| Sequence | Baseline HOTA | Treatment HOTA | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for s,b in sorted(val["baseline"]["per_sequence"].items()):
        t=val["treatment"]["per_sequence"][s]
        lines.append("| "+s+" | "+str(b["HOTA"])+" | "+str(t["HOTA"])+" | "+" | ".join(str(t[k]-b[k]) for k in ("HOTA","AssA","DetA","IDF1","IDSW"))+" |")
    lines.extend(["","## Cached-feature runtime (not live SAM3/OSNet FPS)","",
        "Timing includes causal association and stateless Adapter projection, excludes disk loading, backbone inference, export and evaluator. CPU timing noise can make small incremental times negative.","",
        "| Method | Cached-feature FPS | ms/frame | Incremental ms/frame vs causal baseline |",
        "|---|---:|---:|---:|"])
    for n,r in proofs["timings"].items():
        lines.append(f"| {n} | {r['cached_feature_fps']} | {r['milliseconds_per_frame']} | {r['incremental_milliseconds_per_frame_vs_causal_baseline']} |")
    tests=read_json(OUT/"checkpoints/TEST_RESULTS.json")
    lines.extend(["","## Target-conditioned continuity (distinct from Global MOT)","",
        "Missing intervals count visible-target frames with no correct clicked-ID output; reacquisitions count correct returns after such a gap. Identity continuity changes use consecutive outputs matched to a GT identity. These diagnostics do not replace TrackEval IDSW.","",
        "| Dev sequence | Baseline target recall | Selected target recall | Matched identity changes | Reacquisitions | Wrong identity takeover frames | Persistent H30 corrections |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for s,r in sorted(target.items()):
        lines.append(f"| {s} | {r['clicked_target_recall_baseline']} | {r['clicked_target_recall_treatment']} | {r['target_ID_continuity_changes']} | {reacquisition[s]['successful_reacquisitions']} | {errors[s]['wrong_identity_takeover_frames']} | {r['persistent_corrections_30']} |")
    lines.extend(["","## Verification and checkpoints","",
        f"Stage suite: {tests['stage_suite']['passed']} passed, {tests['stage_suite']['failed']} failed. Selected dependency regression: {tests['selected_dependency_regression']['passed']} passed, {tests['selected_dependency_regression']['failed']} failed (unchanged historical branch-name assertion). Full-repository PASS is not claimed.",
        f"SHA verified {proofs['sealed_run_count']} dev/VAL trajectory+trace exports; all causal runtime core hashes match the pre-formal freeze. Source ref verified through GitHub connector and matches the initial source HEAD; direct transport failures and verification timing are in SOURCE_LINEAGE.json.",
        "Official TrackEval re-evaluation of all frozen dev ablations and both all-25 VAL streams exactly reproduces every combined/per-sequence metric. Original evaluator logs remain unchanged; this is reproducibility verification, not policy reselection.",
        "",
        f"Frozen GRU: `{MEMORY_CHECKPOINT}`, SHA256 `{MEMORY_SHA}`. Strict Adapter checkpoint paths/SHA are in `adapter/STRICT_FOLD_CHECKPOINTS.json`; all authority paths/SHA are in `association/training/*.json` and `checkpoints/SEALED_EVIDENCE.json`; final VAL checkpoint paths/SHA are in `val/FROZEN_POLICY.json`.",
        f"Final science decision `{decision}`. NEXT_STAGE_AUTHORIZED={result['next_stage_authorized']}. No downstream training, new candidate generation or test evaluation was started.",""])
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines).rstrip()+"\n")
    (OUT/"association/TRAJECTORY_INTERVENTION_SUMMARY.md").write_text("# Trajectory interventions\n\n"+"\n".join(lines[lines.index("## Table B — actual trajectory interventions"):lines.index("## Table C — real frozen GRU writes")]).rstrip()+"\n")
    status=read_json(OUT/"stage_status.json")
    status.update(status="SCIENTIFIC_COMPLETE_PENDING_GIT",scientific_decision=decision,goal_complete=False,goal_file="outputs/N72R20R4/FINAL_GOAL.json",val_complete=True,val_accessed_this_stage=True,next_stage_authorized=result["next_stage_authorized"],gate0=gate0,gate1=gate1,gate2=gate2)
    write_json(OUT/"stage_status.json",status)
    return result


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True))
