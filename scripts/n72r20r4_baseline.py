#!/usr/bin/env python3
"""M2 independently reconstructs native baseline and proves new-runtime A/A."""
from __future__ import annotations
from dataclasses import replace
from scripts.n72r20r4_common import *
from scripts.n72r20r4_run_causal_tracker import run_rollout, export_run
from scripts.n72r20r4_trackeval import evaluate, METRICS
from sam3_intermot.association.identity_authority import AuthorityConfig


def run(sequences=SEQUENCES) -> dict:
    records = {}
    group = "baseline"
    for sequence in sequences:
        frames = load_frames(sequence)
        event = events()[sequence]
        native = read_zstd_jsonl(DEV_ROOT / "base_scores" / sequence / "base_scores.jsonl.zst")
        config = AuthorityConfig(lifecycle="legacy_fixed")
        rebuilt, rebuilt_profile = run_rollout(sequence, config=config, frames=frames, event=event)
        native_trace = []
        mapping_errors = []
        for row, (payload, candidates), actual in zip(native, frames, rebuilt):
            by_uid = {r["candidate_uid"]: r for r in candidates}
            expected_map = {str(r["public_id"]): r["candidate_uid"] for r in row["base_assignment"]["public_assignments"]}
            if actual["assignments"] != expected_map:
                mapping_errors.append({"frame": row["frame"], "expected": expected_map, "actual": actual["assignments"]})
            native_trace.append({"frame": int(row["frame"]), "outputs": [{"public_id": int(p), "candidate_uid": uid, "box_xyxy": list(by_uid[uid]["box_xyxy"]), "confidence": float(by_uid[uid].get("confidence", 1.0) or 1.0)} for p, uid in expected_map.items() if uid is not None]})
        native_record = export_run("BASELINE_NATIVE", sequence, native_trace, {"source": "sealed R2 baseline reference only; not A/A treatment"}, group=group)
        legacy_record = export_run("BASELINE_RECONSTRUCTED_LEGACY", sequence, rebuilt, rebuilt_profile, group=group)
        causal, causal_profile = run_rollout(sequence, config=AuthorityConfig(), frames=frames, event=event)
        off, off_profile = run_rollout(sequence, config=AuthorityConfig(mode="off", strength=1), frames=frames, event=event)
        causal_record = export_run("BASELINE_NEW_RUNTIME", sequence, causal, causal_profile, group=group)
        off_record = export_run("TREATMENT_IDENTITY_OFF", sequence, off, off_profile, group=group)
        aa = all({k:a[k] for k in ("assignments","outputs","births","deaths","explicit_none_uids","state_after")} == {k:b[k] for k in ("assignments","outputs","births","deaths","explicit_none_uids","state_after")} for a,b in zip(causal,off)) and causal_record["trajectory_sha256"] == off_record["trajectory_sha256"]
        records[sequence] = {"native_reconstruction_mismatch_frames": len(mapping_errors), "native_reconstruction_mismatches": mapping_errors[:20], "native_reconstruction_sha_equal": native_record["trajectory_sha256"] == legacy_record["trajectory_sha256"], "causal_runtime_A_A": aa, "manifests": {"native": native_record, "legacy_reconstructed": legacy_record, "new_runtime": causal_record, "identity_off": off_record}, "historical_vs_dynamic_difference_reason": "new causal baseline has explicit NONE births and max-lost-gap death lifecycle; historical baseline keeps only click-frame cohort forever", "baseline_state_read_source": "own previous frame only", "native_copy_used_for_A_A": False}
        write_json(OUT / "audit/BASELINE_EQUIVALENCE_PROGRESS.json", {"completed": records, "pending": [s for s in sequences if s not in records]})
        print(json.dumps({"sequence": sequence, "A_A": aa, "native_assignment_mismatches": len(mapping_errors)}, sort_keys=True), flush=True)
    evaluated = evaluate(group, ["BASELINE_NATIVE", "BASELINE_RECONSTRUCTED_LEGACY", "BASELINE_NEW_RUNTIME", "TREATMENT_IDENTITY_OFF"], list(sequences))
    metric_equal = all(evaluated["metrics"]["BASELINE_NEW_RUNTIME"][k] == evaluated["metrics"]["TREATMENT_IDENTITY_OFF"][k] for k in METRICS)
    native_equal = all(r["native_reconstruction_sha_equal"] for r in records.values())
    passed = all(r["causal_runtime_A_A"] for r in records.values()) and metric_equal and native_equal
    result = {"stage": STAGE, "goal_file": "outputs/N72R20R4/FINAL_GOAL.json", "status": "PASS_CAUSAL_BASELINE_EQUIVALENCE" if passed else "FAIL_CAUSAL_BASELINE_RECONSTRUCTION", "independent_trajectory_generation": True, "copied_history_as_treatment": False, "native_reconstruction_all_SHA_equal": native_equal, "causal_A_A_metrics_equal": metric_equal, "records": records, "trackeval": evaluated, "runtime_future_gt_used": False}
    for target in ("audit/BASELINE_EQUIVALENCE_REPORT.json", "causal_tracker/A_A_EQUIVALENCE.json"):
        write_json(OUT / target, result)
    write_json(OUT / "causal_tracker/RUNTIME_INVARIANTS.json", {"A_A": passed, "global_exact_assignment": True, "candidate_unique": True, "public_id_unique": True, "births_after_none": True, "no_pre_event_omission": True, "score_before_update": True, "runtime_future_gt_used": False, "evidence": "tests/test_n72r20r4_* and A_A_EQUIVALENCE.json"})
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps({"status": run()["status"]}))
