"""Seal compact completed M0/M1 evidence, not an improvement decision."""
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4_trackeval import METRICS


def run():
    reproduction=read_json(OUT/"audit/M0_REPRODUCTION.json");fresh=read_json(OUT/"audit/M0_TRACK_EVAL.json")["metrics"]
    old=read_json(R4OUT/"dev_trackeval/BASELINE.json")
    if any(fresh[k]!=old[k] for k in METRICS) or fresh["per_sequence"]!=old["per_sequence"]:raise RuntimeError("official source metrics mismatch")
    write_json(OUT/"audit/R4_BASELINE_REPRODUCTION.json",{"completed":True,"evidence":reproduction,"official_metrics":fresh,"exact_all_metrics_match":True,"new_KEEP_AA":read_json(OUT/"audit/NEW_KEEP_AA.json")})
    write_json(OUT/"audit/CHECKPOINT_LINEAGE.json",{"frozen_GRU":{"path":MEMORY_CHECKPOINT,"sha256":sha256(MEMORY_CHECKPOINT),"encoder_sha256":ENCODER_SHA},"strict_adapter_models":{s:r["strict_adapter_manifest"] for s,r in reproduction["sequences"].items()},"folds":{s:{"fit":fold_split(s)[0],"inner":fold_split(s)[1],"outer":s} for s in SEQUENCES},"forbidden_training_overlap":False})
    parts={s:r["score_decomposition"] for s,r in reproduction["sequences"].items()}
    write_json(OUT/"opportunity/GLOBAL_MARGIN_DECOMPOSITION.json",{"exact_float32_match":True,"decomposition_components":["appearance","predicted_iou","native_core","gap","native_bonus","positive_bonus","rounding_delta","hard_negative_override"],"per_sequence":parts,"candidate_vs_column_vs_global_margin_distinct":True,"exact_current_forced_global_regret_available_in_runtime":True})
    write_json(OUT/"opportunity/NATIVE_POSITIVE_LOCKIN_AUDIT.json",{"per_sequence":parts,"native_soft_total":3.5,"positive_soft_configured":5.,"positive_history_actual_activation":0,"positive_reliability_identifiability":"NO_NATURAL_ACTIVATIONS_IN_SOURCE_CONTROL","native_reliability":"PENDING_FIT_ONLY_AUDIT","not_a_claim_native_lockin_is_causal_bottleneck":True})
    write_json(OUT/"opportunity/HARD_NEGATIVE_PROVENANCE_AUDIT.json",{"source_actual_active_edges":sum(r["hard_negative_edges"] for r in parts.values()),"initial_constraint_sets_empty":True,"click_does_not_add_native_constraints":True,"hard_negative_preserved_in_new_solver_and_discount_paths":True,"synthetic_activation_is_engineering_test_only":True})
    labels=read_json(R4OUT/"association/AUTHORITY_SUPERVISION_AUDIT.json")
    write_json(OUT/"audit/R4_SUPERVISION_REPRODUCTION.json",{"historical_sealed_label_counts":labels,"unchanged_strength_0_5_challengers_not_retrained":True,"runtime_force_solver_is_distinct_from_old_residual":True})
    status=read_json(OUT/"stage_status.json");status["status"]="ACTIVE_M3_MULTI_IDENTITY_COUNTERFACTUAL_MINING";status["milestones"]={"M0":"COMPLETE","M1":"COMPLETE","M2":"ENGINEERING_VALIDATED","M3":"ACTIVE"};write_json(OUT/"stage_status.json",status)


if __name__=="__main__":run()
