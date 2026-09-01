"""Finalize the N69 evidence ledger, gate, and human-readable report."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n69"
REPORT = ROOT / "docs/N69_FINAL_REPORT.md"
FINAL_GATE = OUT / "n69_final_gate.json"
STAGE07 = OUT / "stage_07_status.json"


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256(path)}


def audit_alt_artifacts(alt_results: dict[str, Any]) -> dict[str, Any]:
    root = OUT / "stage_06_protected_guard/event_artifacts"
    artifacts = sorted(root.glob("*.json"))
    keys: set[tuple[str, str, int]] = set()
    duplicate = 0
    frame_count = 0
    variant_count = 0
    runtime_future_gt_true = 0
    malformed = 0
    for artifact_path in artifacts:
        artifact = load(artifact_path)
        for variant in ("M0", "M1", "M2", "M3", "M4"):
            frames = artifact.get("variants", {}).get(variant, {}).get("frames", [])
            if len(frames) != 100:
                malformed += 1
            variant_count += 1
            for frame in frames:
                frame_count += 1
                key = (str(artifact.get("event_id")), variant, int(frame.get("frame", -1)))
                if key in keys:
                    duplicate += 1
                keys.add(key)
                if frame.get("runtime_future_gt_used") is not False:
                    runtime_future_gt_true += 1
    expected = 24 * 5 * 100
    return {
        "artifact_count": len(artifacts),
        "expected_artifact_count": 24,
        "variant_container_count": variant_count,
        "expected_variant_container_count": 24 * 5,
        "frame_count": frame_count,
        "expected_frame_count": expected,
        "unique_event_variant_frame_keys": len(keys),
        "duplicate_key_count": duplicate,
        "malformed_variant_count": malformed,
        "runtime_future_gt_true_count": runtime_future_gt_true,
        "complete": len(artifacts) == 24 and variant_count == 120 and frame_count == expected and len(keys) == expected and duplicate == 0 and malformed == 0 and runtime_future_gt_true == 0,
        "results_path": str(OUT / "stage_06_protected_guard/paired_replay_results.json"),
        "results_sha256": sha256(OUT / "stage_06_protected_guard/paired_replay_results.json"),
        "reported_event_count": alt_results.get("event_count"),
        "reported_frame_count": alt_results.get("frame_count"),
    }


def pct(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def main() -> None:
    stage_paths = {f"stage_{index:02d}": OUT / f"stage_{index:02d}_status.json" for index in range(0, 7)}
    stage_statuses = {key: load(path).get("status") if path.is_file() else "MISSING" for key, path in stage_paths.items()}
    stage05 = load(OUT / "stage_05_status.json")
    stage06 = load(STAGE07.parent / "stage_06_status.json")
    n69_results_path = OUT / "replay/paired_replay_results.json"
    alt_results_path = OUT / "stage_06_protected_guard/paired_replay_results.json"
    n69_results = load(n69_results_path)
    alt_results = load(alt_results_path)
    training = load(OUT / "training/n69_target_conditioned_training_manifest.json")
    dataset = load(OUT / "training/n69_target_conditioned_dataset_manifest.json")
    mapping = load(OUT / "diagnosis/mapping_summary.json")
    isolation = load(OUT / "n69_isolation_regression.json")
    runtime = load(OUT / "replay/runtime_status.json")
    alt_runtime = load(OUT / "stage_06_protected_guard/runtime_status.json")
    alt_audit = audit_alt_artifacts(alt_results)
    new = n69_results["methods"]["N69_TARGET_CONDITIONED"]
    guarded = alt_results["methods"]["N69_PROTECTED_UNTOUCHED_GUARD"]
    lower = {h: new["horizons"][h]["sequence_cluster_bootstrap"]["ci95"][0] for h in ("20", "50", "100")}
    guarded_lower = {h: guarded["horizons"][h]["sequence_cluster_bootstrap"]["ci95"][0] for h in ("20", "50", "100")}
    strict_synthetic = {
        "status": "PASS" if all(float(value) > 0.0 for value in lower.values()) and new["correct_changes"] > new["incorrect_changes"] and new["untouched_regression_frame_rate"] == 0.0 and mapping.get("candidate_frame_integrity_100") is True and mapping.get("full_native_local_global_public_provenance") is True and runtime.get("runtime_future_gt_used") is False else "FAIL_FUTURE_EFFECT",
        "positive_lower_ci_all_horizons": all(float(value) > 0.0 for value in lower.values()),
        "strict_lower_ci_by_horizon": lower,
        "correct_changes": new["correct_changes"],
        "incorrect_changes": new["incorrect_changes"],
        "untouched_assignment_changed_total": new["untouched_assignment_changed_total"],
        "untouched_regression_safe": new["untouched_regression_frame_rate"] == 0.0,
        "candidate_frame_integrity_100": mapping.get("candidate_frame_integrity_100") is True,
        "target_scope_mapping_100_on_available_candidates": mapping.get("target_scope_mapping_100_on_available_candidates") is True,
        "formal_native_local_global_public_provenance_100": mapping.get("full_native_local_global_public_provenance") is True,
        "runtime_future_gt_false": runtime.get("runtime_future_gt_used") is False,
    }
    production_gate = {
        "status": "PASS" if dataset.get("real_human_tape") is True and dataset.get("real_sam3_full_loop") is True else "BLOCKED_NO_REAL_HUMAN_TAPE_OR_REAL_SAM3_FULL_LOOP",
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
    }
    artifacts = []
    for path in sorted((OUT / "attempts").rglob("*.json")):
        artifacts.append(record(path))
    key_files = [
        OUT / "protocol.json",
        OUT / "stage_03_protocol.json",
        OUT / "diagnosis/mapping_summary.json",
        OUT / "cache/candidate_cache_manifest.json",
        OUT / "training/n69_target_conditioned_dataset_manifest.json",
        OUT / "training/n69_target_conditioned_dataset.npz",
        OUT / "training/n69_target_conditioned_training_manifest.json",
        OUT / "training/n69_target_conditioned_scorer.pt",
        OUT / "replay/runtime_status.json",
        OUT / "replay/paired_replay_results.json",
        OUT / "stage_06_protected_guard/runtime_status.json",
        OUT / "stage_06_protected_guard/paired_replay_results.json",
        OUT / "n69_isolation_regression.json",
    ]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in key_files}
    final_gate = {
        "schema": "N69_FINAL_GATE_V1",
        "experiment": "N69_WEIGHTED_TARGET_CONDITIONED_ASSOCIATION_AND_PROTECTED_GUARD",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_N69_STRICT_GATE_AND_PRODUCTION_EVIDENCE",
        "stage_statuses": stage_statuses,
        "protocol": {"path": str(OUT / "stage_03_protocol.json"), "sha256": sha256(OUT / "stage_03_protocol.json")},
        "synthetic_science_gate": strict_synthetic,
        "stage06_alternative_gate": {
            "status": alt_results["synthetic_science_gate"]["status"],
            "method": "N69_PROTECTED_UNTOUCHED_GUARD",
            "strict_lower_ci_by_horizon": guarded_lower,
            "correct_changes": guarded["correct_changes"],
            "incorrect_changes": guarded["incorrect_changes"],
            "untouched_assignment_changed_total": guarded["untouched_assignment_changed_total"],
            "guard_rejections": alt_results["alternative"]["guard_rejections"],
            "runtime_complete": alt_audit["complete"],
        },
        "production_evidence_gate": production_gate,
        "production_authorized": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "mapping": {
            "audit_rows": mapping.get("audit_rows"),
            "target_scope_total": mapping.get("target_scope_total"),
            "target_scope_resolved": mapping.get("target_scope_resolved"),
            "target_candidate_absent": mapping.get("target_candidate_absent"),
            "old_target_public_conflict_frames": mapping.get("old_target_public_conflict_frames"),
            "full_native_local_global_public_provenance": mapping.get("full_native_local_global_public_provenance"),
        },
        "training": {
            "status": training.get("status"),
            "actual_gpu_training": True,
            "device": training.get("device"),
            "cuda_visible_devices": training.get("cuda_visible_devices"),
            "best_epoch": training.get("best_epoch"),
            "parameter_count": training.get("parameter_count"),
            "holdout": training.get("holdout"),
        },
        "replay": {"runtime": runtime, "paired_results": record(n69_results_path)},
        "alternative_replay": {"runtime": alt_runtime, "paired_results": record(alt_results_path), "artifact_audit": alt_audit},
        "isolation": isolation["interpretation"],
        "input_hashes": input_hashes,
        "preserved_failure_artifacts": {"count": len(artifacts), "files": artifacts},
        "resource_plan": {
            "available_gpu_count_stage00": 10,
            "max_concurrent_gpus": 4,
            "per_gpu_concurrency": 1,
            "actual_training_gpu": "GPU0",
            "actual_replay_gpu": "GPU0 for no-trim; CPU for protected guard",
            "gpu4_external_occupancy_observed": True,
            "oom_policy": "160->100->50 frame chunks, unchanged protocol; no N69 OOM occurred",
        },
        "decision": {
            "calibration_head": "NOT_AUTHORIZED",
            "selector": "NOT_AUTHORIZED",
            "decoder_lora": "NOT_AUTHORIZED",
            "TACT": "NOT_AUTHORIZED",
            "production_interface": "NOT_AUTHORIZED",
            "minimum_next_action": "Collect provenance-complete real human event tape and candidate-complete real SAM3 full-loop, then rerun mapping-first audit before any production change.",
        },
        "iclr_2027": {"current_date": "2026-09-01", "abstract_deadline_aoe": "2026-09-18", "full_paper_deadline_aoe": "2026-09-25", "calendar_note": "At most 17/24 calendar days from the stated current date; do not trade away integrity gates."},
    }
    atomic_json(FINAL_GATE, final_gate)
    report_rel = lambda path: f"[{path.relative_to(ROOT)}]({path})"
    action_lines = []
    for action, values in sorted(n69_results.get("by_action_type", {}).items()):
        item = values["N69_TARGET_CONDITIONED"]
        action_lines.append(f"| {action} | {item['frame_count']} | {item['correct_changes']} | {item['incorrect_changes']} | {item['untouched_assignment_changed_total']} | {item['horizons']['20']['mean_utility_delta_raw_event_variant']:.8f} |")
    report = f"""# N69 Final Report — Target-Conditioned Association and Protected Assignment Guard

**Date:** 2026-09-01 (Asia/Shanghai)  
**Final status:** `BLOCKED_N69_STRICT_GATE_AND_PRODUCTION_EVIDENCE`  
**Production authorization:** `false`

## Executive conclusion

N69 completed the required mapping audit, frozen-cache reuse, actual GPU training, 24-event paired replay, one root-cause-driven alternative, isolation regression, and machine-readable final gate. It does **not** establish a deployable improvement. The corrected target-conditioned scorer changed scores on {pct(new['score_change_frame_rate'])} of frames and caused {new['correct_changes']} correct versus {new['incorrect_changes']} incorrect target assignment changes, but it also changed {new['untouched_assignment_changed_total']} untouched candidate assignments. Sequence-cluster 95% CI lower bounds were `H20={lower['20']:.8f}`, `H50={lower['50']:.8f}`, `H100={lower['100']:.8f}`; none is strictly positive. The protected alternative removed collateral changes but produced zero assignment changes.

All events are `simulated_from_gt`, not historical human clicks. There is no provenance-complete real human tape or real SAM3 full-loop in N69, so production remains blocked independently of the synthetic mechanism result.

## 1. N68 failure and N69 question

N68 showed that a local target-conditioned score could change while Hungarian assignment rarely changed, and its positive changes were sparse, ADD-only, not statistically strict, and not untouched-safe. N68 Stage01 also found `29/30` old native/public mapping mismatches and one target-scope issue. N69 therefore tested whether a real target-scoped, raw-512-D model could cross the existing assignment boundary without changing unrelated identities; it kept checkpoint, candidate cache, public-ID axis, Hungarian solver, NONE handling, future windows, and evaluation definitions fixed.

Frozen parent evidence: {report_rel(ROOT / 'docs/N68_FINAL_REPORT.md')}, {report_rel(ROOT / 'outputs/n68/n68_final_gate.json')}.

## 2. Mapping-first diagnosis

Stage01 audited `12,000` event×variant×frame rows from the frozen N54 runtime cache. Candidate structure and frame integrity were 100%; target scope was resolved for `11,910` available frames, while `90` frames explicitly lacked the target candidate. The old N68 mapping conflicted with the target public ID on `3,665` frames. N69's versioned reconciliation used the explicit event target public ID at the intervention boundary and preserved absence instead of inventing a mapping. It passed target-scope reconciliation on available candidates, but the frozen cache lacks complete native→local→global provenance, so the formal full mapping gate remains false. The 90 absent frames are data/candidate recall evidence, not fabricated negatives.

Stage01: {report_rel(OUT / 'stage_01_status.json')}; mapping summary: {report_rel(OUT / 'diagnosis/mapping_summary.json')}.

## 3. Frozen data and model

The dataset contains `12,000` groups and `92,070` candidate examples: `11,910` positive and `80,160` negative. The raw contract includes candidate, human anchor, target memory prototype, hard-negative 512-D vectors, projected products/absolute differences, and 34 audited geometry/temporal/confidence/context features. No numeric public ID or target native ID is a runtime feature. The interaction source remains `simulated_from_gt`.

The trained model is a shared low-rank `512→64` projection with ten candidate-conditioned terms plus context, then `128→64→2` target/NONE logits (`128,902` parameters). It was trained on GPU0 with sequence-disjoint train/validation/holdout splits, fixed seed `6901`, AdamW, fixed loss terms, and validation early stopping; holdout was evaluated only after selection. The first training pass exposed a temporal-pair loop bug that produced NaN and is preserved. A later completed pass exposed a label contract error: dataset `1=target` was sent directly to CE while replay interprets logit 0 as target. That checkpoint/replay is preserved and excluded from the final result. After the minimal label-boundary repair, the same GPU smoke and training protocol produced holdout AUC `{training['holdout']['auc']:.6f}` and finite loss/temporal curves.

Dataset: {report_rel(OUT / 'training/n69_target_conditioned_dataset_manifest.json')}; training manifest: {report_rel(OUT / 'training/n69_target_conditioned_training_manifest.json')}; checkpoint: {report_rel(OUT / 'training/n69_target_conditioned_scorer.pt')}.

## 4. Paired replay result

The corrected no-trimming model replayed all `24 × 5 × 100 = 12,000` future frames. Runtime candidate streams and public-ID axes were identical to baseline; only the target public column was changed; the event frame did not read the new memory; memory was first visible at event+1; runtime future GT was false.

| action | frames | correct changes | incorrect changes | untouched changes | H20 utility |
|---|---:|---:|---:|---:|---:|
{chr(10).join(action_lines)}

Aggregate score/assignment evidence:

- score changed on {pct(new['score_change_frame_rate'])} of frames;
- target assignment changed on {pct(new['target_assignment_change_rate'])} of frames, while full assignment changed on {pct(new['assignment_change_rate'])};
- target changes: `{new['correct_changes']} correct`, `{new['incorrect_changes']} incorrect`, `{new['neutral_changes']} neutral`;
- target candidate recall: {pct(new['target_candidate_recall'])}; baseline target correctness `{pct(new['baseline_target_correct_rate'])}`, treated `{pct(new['target_correct_rate'])}`;
- untouched regression: `{new['untouched_assignment_changed_total']}` candidate assignment changes, frame rate `{pct(new['untouched_regression_frame_rate'])}`;
- sequence-cluster bootstrap used 21 sequence clusters, seed 7008/6928/6958 by horizon, 2,000 repetitions.

Results: {report_rel(n69_results_path)}; diagnostics: {report_rel(OUT / 'replay/assignment_diagnostics.jsonl')}.

## 5. Stage06 alternative

The boundary diagnosis was not “no score signal”: the model had a large score change but its row-wise target-column update could displace untouched candidates. The sole isolated alternative was `N69_PROTECTED_UNTOUCHED_GUARD`: recompute the unchanged Hungarian solver, but accept the proposal only if every currently assigned native candidate retains its assignment; otherwise use the frozen baseline. It used no GT or target native ID at runtime and changed no checkpoint or candidate stream.

It rejected `{alt_results['alternative']['guard_rejections']}` proposals. Its runtime audit is complete (`{alt_audit['frame_count']}/{alt_audit['expected_frame_count']} frames`, no duplicate keys, no runtime future GT), but posthoc it yielded `0` correct, `0` incorrect, `0` assignment changes and H20/H50/H100 lower bounds `0/0/0`. This demonstrates a safety/effect trade-off, not a valid efficacy result. TACT, calibration, selector, and decoder LoRA were therefore not run.

Alternative results: {report_rel(alt_results_path)}; Stage06 status: {report_rel(OUT / 'stage_06_status.json')}.

## 6. Gate and isolation

The synthetic science gate fails because the strict sequence-cluster lower CI is not greater than zero, untouched regression is nonzero for the unguarded scorer, and complete native/local/global/public provenance is unavailable in the frozen cache. The production evidence gate is separately blocked because `real_human_tape=false` and `real_sam3_full_loop=false`. `simulated_from_gt` can test a controlled mechanism hypothesis; it cannot prove historical human interaction, production online behavior, or real SAM3 correction-to-memory causality.

Isolation passed: production/third-party/configuration trees and protected checkpoints are unchanged; frozen N36–N68 evidence is unchanged; current SAM3_InterMOT tests are `113/113`; adjacent MOT/OVMOT targeted tests are `27/27`; new scripts compile. The initial isolation failures were environment/harness failures (`sys.path`, missing pytest, then incompatible interpreter), preserved under attempts and corrected by using the existing conda environment; they are not silently counted as passes.

Machine gate: {report_rel(FINAL_GATE)}; isolation: {report_rel(OUT / 'n69_isolation_regression.json')}.

## 7. Literature/alternative scope

The prior project literature audit was reused read-only rather than claiming a new method without verification. Relevant public sources and their recorded versions are:

| method | public sources | recorded reusable mechanism | N69 decision |
|---|---|---|---|
| TACT (NeurIPS 2025) | [OpenReview](https://openreview.net/forum?id=zFGdHL9pcD), [GitHub](https://github.com/NancyQuris/TACT) | identity-preserving semantic augmentation, PCA nuisance analysis, paired sample/prototype trimming | not run: no-trimming strict gate failed |
| HATReID-MOT (ECCV 2026) | [arXiv](https://arxiv.org/abs/2503.12562), [GitHub](https://github.com/MCG-NJU/HATReID-MOT) | history-guided ReID discriminative subspace | not imported; would change the frozen evidence/production stack |
| MOTIP (CVPR 2025) | [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html), [GitHub](https://github.com/MCG-NJU/MOTIP) | trajectory-ID-conditioned prediction view | not imported; arbitrary public-ID prediction is outside N69 contract |
| REMIND (2026) | [arXiv](https://arxiv.org/abs/2607.09267), [GitHub](https://github.com/cvar-vision-dl/remind-reid-tracker) | stable/work prototypes and robust memory update gate | not imported; N69 tested a narrower target-scoped guard |

The N69 alternative is explicitly a project-level guard motivated by the observed D-category collateral assignment changes; it is not presented as a published method.

## 8. Failures preserved

N69 preserves the default-environment torch smoke failure, the first CUDA `torch` scope failure, the temporal-pair NaN/manifest failure, the invalid-label-contract audit and its replay, the first Stage06 native/public audit failure, the first Stage06 guard replay failure, the first isolation harness failure, and the public-axis Stage06 attempt. They remain under {report_rel(OUT / 'attempts')} and are not overwritten by the final corrected artifacts.

## 9. ICLR 2027 calendar and minimum next step

Using the fixed project dates, the abstract deadline is **2026-09-18 AoE** and the full-paper deadline is **2026-09-25 AoE**. From 2026-09-01 this leaves at most 17 and 24 calendar days respectively. The schedule must not be used to bypass mapping integrity, runtime future-GT checks, untouched regression, or real-input evidence.

The minimum scientifically valid next action is to obtain an externally supplied, provenance-complete real human event tape: direct public ID, raw BOX/CLICK/CONFIRMED_MASK, annotator/session/timestamp, frame hash, candidate-tape reference, explicit correction transaction, and a candidate-complete real SAM3 full-loop. After that input exists, rerun the mapping-first audit. Do not promote the N69 checkpoint, add calibration/selector/LoRA, run TACT, or modify production MOT/OVMOT code before those gates pass.

## 10. N69 artifact index

- Stage00 audit/status: {report_rel(OUT / 'stage_00_readonly_audit.json')}, {report_rel(OUT / 'stage_00_status.json')}
- Stage01 mapping: {report_rel(OUT / 'stage_01_status.json')}, {report_rel(OUT / 'diagnosis/mapping_audit.jsonl')}
- Stage02 cache: {report_rel(OUT / 'stage_02_status.json')}, {report_rel(OUT / 'cache/candidate_cache_audit.json')}
- Stage03 training: {report_rel(OUT / 'stage_03_status.json')}, {report_rel(OUT / 'training/n69_model_smoke.json')}
- Stage04/05: {report_rel(OUT / 'stage_04_status.json')}, {report_rel(OUT / 'stage_05_status.json')}
- Stage06/07: {report_rel(OUT / 'stage_06_status.json')}, {report_rel(STAGE07)}

No N36–N68 evidence was rewritten.
"""
    atomic_text(REPORT, report)
    atomic_json(STAGE07, {
        "schema": "N69_STAGE_07_STATUS_V1",
        "status": "BLOCKED_FINAL_GATE_WRITTEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_gate": str(FINAL_GATE),
        "final_gate_sha256_before_stage07_write": sha256(FINAL_GATE),
        "report": str(REPORT),
        "report_sha256": sha256(REPORT),
        "synthetic_science_gate": strict_synthetic,
        "stage06_alternative_gate": final_gate["stage06_alternative_gate"],
        "production_evidence_gate": production_gate,
        "isolation": isolation["interpretation"],
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
        "next_action": "Obtain provenance-complete real human tape and real SAM3 full-loop before any production promotion.",
    })
    # Refresh the gate with the final stage/report records without changing
    # the scientific decision.
    final_gate["stage_statuses"]["stage_07"] = "BLOCKED_FINAL_GATE_WRITTEN"
    final_gate["stage_07"] = {"path": str(STAGE07), "sha256": sha256(STAGE07)}
    final_gate["report"] = {"path": str(REPORT), "sha256": sha256(REPORT)}
    atomic_json(FINAL_GATE, final_gate)
    print(json.dumps({"status": final_gate["status"], "report": str(REPORT), "final_gate": str(FINAL_GATE), "synthetic": strict_synthetic["status"], "production": production_gate["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
