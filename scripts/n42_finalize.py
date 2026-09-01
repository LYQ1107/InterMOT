#!/usr/bin/env python3
"""Freeze N42 gate artifacts and produce the final research report."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n42"
POSTHOC = OUT / "replay/posthoc_results.json"
STAGE1 = OUT / "stage_01_status.json"
STAGE2 = OUT / "stage_02_status.json"
STAGE3 = OUT / "stage_03_status.json"
ISOLATION = OUT / "isolation_regression.json"
REPORT = ROOT / "docs/N42_FINAL_REPORT.md"
GATE = OUT / "n42_final_gate.json"
STAGE4 = OUT / "stage_04_status.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    from scripts.n36_real_eval_common import atomic_json as write_json

    write_json(path, payload)


def atomic_text(path: Path, text: str) -> None:
    from scripts.n36_real_eval_common import atomic_text as write_text

    write_text(path, text)


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value = float(value)
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def link(relative: str, label: str | None = None) -> str:
    path = (ROOT / relative).resolve()
    return f"[{label or relative}](<{path}>)"


def rows_for(posthoc: dict[str, Any], mode: str, variant: str) -> dict[str, Any]:
    return posthoc["aggregates"][mode][variant]


def main() -> None:
    posthoc = load(POSTHOC)
    stage1 = load(STAGE1)
    stage2 = load(STAGE2)
    stage3 = load(STAGE3)
    isolation = load(ISOLATION)
    training = load(OUT / "training/full_training_manifest.json")
    smoke = load(OUT / "training/smoke_status.json")
    dataset = load(OUT / "training/dataset_manifest.json")
    diagnostic = load(OUT / "diagnostic/diagnostic_interpretation.json")
    source_manifest = load(OUT / "diagnostic/source_embedding_manifest.json")
    methods = load(OUT / "method_retrieval.json")
    training_protocol = load(OUT / "training/training_protocol.json")

    t1_gate_status = {
        variant: rows_for(posthoc, "t1", variant)["holdout_gate"]["status"]
        for variant in ("M2", "M3", "M4")
    }
    t1_gate_pass = all(value == "PASS" for value in t1_gate_status.values())
    t1_m2 = rows_for(posthoc, "t1", "M2")
    t1_m3 = rows_for(posthoc, "t1", "M3")
    t1_m4 = rows_for(posthoc, "t1", "M4")
    t1_scores_changed_h20 = t1_m2["transition"]["20"]["score_change_rate"]
    t1_assignment_h20 = t1_m2["transition"]["20"]["assignment_change_rate"]
    t1_correct_h20 = t1_m2["transition"]["20"]["correct_assignment_change_count"]
    t1_incorrect_h20 = t1_m2["transition"]["20"]["incorrect_assignment_change_count"]

    runtime_applied = 0
    runtime_max_delta = 0.0
    for path in sorted((OUT / "replay/runtime/t1").glob("*.json")):
        payload = load(path)
        for variant in ("M1", "M2", "M3", "M4"):
            for entry in payload["variants"][variant]["branches"]["memory_write=True"]["future_trace"]:
                audit = entry["candidate_audit"]
                runtime_applied += int(audit.get("t1_calibration", {}).get("applied", False))
                before = np.asarray(audit.get("fused_scores_before_t1", []), dtype=float)
                after = np.asarray(audit.get("fused_scores", []), dtype=float)
                if before.shape == after.shape and before.size:
                    runtime_max_delta = max(runtime_max_delta, float(np.max(np.abs(after - before))))

    allowed_new = isolation.get("project_code", {}).get("allowed_new", [])
    n36_integrity = load(ROOT / "outputs/n36/all24_integrity_audit.json")
    n38_gate = load(ROOT / "outputs/n38/n38_final_gate.json")
    failure_rows = [
        ("outputs/n42/attempts/snapshot_inspection_attempt1_failure.json", "read-only jq inspection command malformed; exit 3; no data changed"),
        ("outputs/n42/attempts/stage_01_attempt1_system_python_failure.json", "system Python lacked torch; ModuleNotFoundError; rerun in the audited intermot environment"),
        ("outputs/n42/attempts/stage_01_attempt1_failure.json", "diagnostic initially applied event-frame hidden/read predicate to future rows; fixed to enforce hidden only at event frame"),
        ("outputs/n42/attempts/source_ab_identity_attempt1_failure.json", "initial N42 A/B source construction retained N41 equal feature digests (24/24); corrected source construction rerun"),
        ("outputs/n42/attempts/dataset_build_failure.json", "training protocol metadata boolean was iterated as a sequence split; fixed to consume train/validation/holdout only"),
        ("outputs/n42/attempts/posthoc_attempt1_failure.json", "posthoc validator read event-frame wrapper metadata from the nested candidate object; fixed without touching runtime artifacts"),
        ("outputs/n42/attempts/posthoc_attempt2_failure.json", "posthoc validator required an optional T1 subfield in T0 baseline audits; fixed to require it only when present"),
        ("outputs/n42/attempts/finalize_attempt2_gate_classification_failure.json", "finalizer initially treated the explicit smoke value save_reload='PASS' as boolean-only; corrected without changing training evidence"),
    ]

    checks = {
        "stage_01_diagnostic_pass": stage1.get("status") == "PASS_DIAGNOSTIC_ONLY",
        "stage_02_training_pass": stage2.get("status") == "TRAINING_PASS",
        "training_smoke_pass": smoke.get("status") == "PASS" and smoke.get("loss_finite") is True and smoke.get("gradient_finite") is True and smoke.get("save_reload") in (True, "PASS"),
        "training_is_actual": training.get("status") == "PASS" and int(training.get("completed_epochs", 0)) >= 1 and Path(ROOT / str(training["checkpoint"])).is_file(),
        "stage_03_posthoc_pass": stage3.get("status") == "POSTHOC_PASS",
        "runtime_t0_t1_24_each": all(load(OUT / f"replay/runtime_{mode}_manifest.json").get("status") == "PASS" for mode in ("t0", "t1")),
        "posthoc_240_variant_results": posthoc.get("status") == "COMPLETED_POSTHOC" and posthoc.get("posthoc_variant_result_count") == 240,
        "runtime_future_gt_false": posthoc.get("runtime_future_gt_used") is False and posthoc.get("gt_loaded_only_after_all_runtime_validation") is True,
        "source_distinctness_fixed": source_manifest.get("distinctness", {}).get("all_pairwise_distinct") is True,
        "isolation_pass": isolation.get("status") == "PASS",
        "t1_holdout_strict_gate": t1_gate_pass,
        "real_human_tape_available": False,
    }
    # Missing real human tape is an explicit provenance limitation, not an
    # N42 effect gate: this round is allowed to use GT-controlled diagnosis
    # and is forbidden to fabricate human events.
    gating_checks = {
        key: value for key, value in checks.items()
        if key != "real_human_tape_available"
    }
    gate_status = "N42_COMPLETED_GATE_PASS" if all(gating_checks.values()) else "N42_COMPLETED_GATE_FAILED"
    final_gate = {
        "protocol": "N42_FINAL_RESEARCH_GATE_V1",
        "status": gate_status,
        "created_at": now(),
        "research_gate": "PASS" if t1_gate_pass else "FAIL_FUTURE_EFFECT",
        "execution_status": "PASS" if all(checks[key] for key in checks if key not in ("t1_holdout_strict_gate", "real_human_tape_available")) else "FAIL",
        "checks": checks,
        "event_count": int(posthoc.get("event_count", 0)),
        "independent_sequence_count": int(posthoc.get("independent_sequence_count", 0)),
        "posthoc_variant_result_count": int(posthoc.get("posthoc_variant_result_count", 0)),
        "training": {
            "candidate": "T1_PAIRWISE_CALIBRATION_HEAD",
            "status": training.get("status"),
            "smoke_status": smoke.get("status"),
            "checkpoint": training.get("checkpoint"),
            "checkpoint_sha256": training.get("checkpoint_sha256"),
            "production_authorized": False,
        },
        "t1_holdout_gate_status": t1_gate_status,
        "failure_reasons": {
            "t1_m2_h20_h50_h100_holdout_lower_ci": {
                str(h): t1_m2["holdout_gate"]["per_horizon"][str(h)]["lower_ci"] for h in (20, 50, 100)
            },
            "t1_m2_all_data_identity_utility": {str(h): t1_m2["metrics"][str(h)]["identity_utility_delta"] for h in (20, 50, 100)},
            "t1_m3_all_data_identity_utility": {str(h): t1_m3["metrics"][str(h)]["identity_utility_delta"] for h in (20, 50, 100)},
            "t1_m4_all_data_identity_utility": {str(h): t1_m4["metrics"][str(h)]["identity_utility_delta"] for h in (20, 50, 100)},
            "strict_rule": "T1 M2/M3/M4 holdout sequence-cluster 95% CI lower bound must be strictly > 0 at H20/H50/H100; untouched-ID regression must be absent",
        },
        "authorization": {
            "calibration_head": "TRAINED_ISOLATED_NOT_AUTHORIZED",
            "selector": "NOT_AUTHORIZED",
            "decoder_lora": "NOT_AUTHORIZED",
            "production_interface_changed": False,
        },
        "artifact": str(GATE.relative_to(ROOT)),
    }
    atomic_json(GATE, final_gate)

    stage4 = {
        "stage": "N42-04_FINAL_GATE",
        "status": gate_status,
        "protocol": "N42_FINAL_RESEARCH_GATE_V1",
        "final_gate": str(GATE.relative_to(ROOT)),
        "research_gate": final_gate["research_gate"],
        "execution_status": final_gate["execution_status"],
        "training_completed": True,
        "t1_holdout_gate_status": t1_gate_status,
        "production_authorized": False,
        "calibration_head": "TRAINED_ISOLATED_NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "next_action": "Do not expand weights or modify checkpoint. If pursuing the hypothesis, collect a real human event tape through N40's external ingestion path; keep N42 T1 as a negative/diagnostic result.",
    }
    atomic_json(STAGE4, stage4)

    method_lines = []
    for entry in methods.get("entries", []):
        name = entry.get("name", "unnamed")
        paper = entry.get("paper_url")
        github = entry.get("github_url")
        links = []
        if paper:
            links.append(f"[paper]({paper})")
        if github:
            links.append(f"[GitHub]({github})")
        method_lines.append(
            f"- **{name}** ({entry.get('year')}): {'; '.join(links) or 'no paper URL retained'}; "
            f"revision/date: `{entry.get('github_revision')}` / `{entry.get('github_revision_date')}`. "
            f"Mechanism: {entry.get('public_mechanism')}. Reusable in N42: {entry.get('reusable_mechanism')}. "
            f"Limit: {entry.get('fit_and_limit')}"
        )

    metric_lines = []
    for mode in ("t0", "t1"):
        for variant in VARIANTS:
            aggregate = rows_for(posthoc, mode, variant)
            metric_lines.append(
                f"| {mode.upper()} | {variant} | "
                f"{fmt(aggregate['metrics']['20']['identity_utility_delta'])} | "
                f"{fmt(aggregate['metrics']['50']['identity_utility_delta'])} | "
                f"{fmt(aggregate['metrics']['100']['identity_utility_delta'])} | "
                f"{fmt(aggregate['sequence_cluster_bootstrap']['20'].get('lower'))} | "
                f"{fmt(aggregate['sequence_cluster_bootstrap']['50'].get('lower'))} | "
                f"{fmt(aggregate['sequence_cluster_bootstrap']['100'].get('lower'))} | "
                f"{fmt(aggregate['transition']['20']['score_change_rate'])} | "
                f"{fmt(aggregate['transition']['20']['assignment_change_rate'])} | "
                f"{aggregate['transition']['20']['correct_assignment_change_count']} / {aggregate['transition']['20']['incorrect_assignment_change_count']} |"
            )

    action_lines = []
    for action, aggregate in t1_m2["actions"].items():
        m = aggregate["metrics"]["20"]
        tr = aggregate["transition"]["20"]
        action_lines.append(
            f"| {action} | {aggregate['event_count']} | {fmt(m['identity_utility_delta'])} | "
            f"{tr['assignment_changed_count']} | {tr['correct_assignment_change_count']} | {tr['incorrect_assignment_change_count']} |"
        )

    failure_lines = "\n".join(
        f"- {link(path)} — {description}." for path, description in failure_rows
    )
    report_text = f"""# N42 — Isolated T1 Association Calibration Probe

**Date:** 2026-08-30 (Asia/Shanghai)  
**Final status:** `{gate_status}`  
**Research gate:** `FAIL_FUTURE_EFFECT`  
**Production authorization:** `FALSE`

## Executive result

N42 completed the required diagnosis, an actual isolated T1 training run, paired T0/T1 replay, posthoc evaluation, and isolation regression. T1 changed the recorded future score interface, but it did not produce a positive, reliable identity effect: on the primary all-event T1/M2 result, H20 identity utility was **{fmt(t1_m2['metrics']['20']['identity_utility_delta'])}**, H50 **{fmt(t1_m2['metrics']['50']['identity_utility_delta'])}**, and H100 **{fmt(t1_m2['metrics']['100']['identity_utility_delta'])}**. The holdout sequence-cluster lower bounds for T1 M2/M3/M4 were not strictly positive, so the calibration head remains an isolated research artifact and is not promoted. Selector and decoder LoRA remain `NOT_AUTHORIZED`.

This is a completed negative/diagnostic result, not an execution block. All 24 frozen events across 21 sequences completed runtime and posthoc processing. There is still **zero real human tape**; every event is explicitly `simulated_from_gt` and is not historical human evidence.

## Frozen scope and provenance

- Input scope: N37 frozen 24 events / 21 independent train/train_fold sequences; no val/test reread, no new event selection, no real-human tape generation.
- Checkpoint, candidate stream, embedding definition, Hungarian solver, prefix, future windows, M0–M4 definitions, metrics, and sequence-cluster bootstrap were kept frozen.
- Runtime replay used `runtime_future_gt_used=false`; GT was loaded only after all T0/T1 runtime artifacts had passed structural validation, for posthoc labels and direction checks only.
- The replay is `frozen_candidate_state_interface_probe`: T1 recomputes the isolated score/assignment interface on frozen N41 candidate/state audits. It is not a production online `StateManager` deployment and does not claim a production association change.
- Controlled source labels remain mechanism probes only: ideal/current/corrupted sources are not real human annotations.

Key machine-readable inputs and outputs: {link('outputs/n42/stage_01_status.json')}, {link('outputs/n42/diagnostic/diagnostic_interpretation.json')}, {link('outputs/n42/training/training_protocol.json')}, {link('outputs/n42/stage_02_status.json')}, {link('outputs/n42/stage_03_status.json')}, {link('outputs/n42/replay/posthoc_results.json')}, {link('outputs/n42/isolation_regression.json')}, {link('outputs/n42/n42_final_gate.json')}.

## N42-01 diagnosis

The frozen N41 parameter-transfer audit passed: `lambda_assoc` values 0/1/8 scaled appearance deltas, `human_weight` values 1/4/8 scaled the human-positive term, event-frame memory read was false, t+1 was the first read, and mapping/hard-negative checks passed.

N41's source-construction failure was preserved rather than rewritten: the old A/B source manifest had exact digest equality for 24/24 events. N42's corrected sidecar has 72 finite 512-D unit-norm features, zero exact digest collisions, and pairwise cosine ranges A/B `{fmt(source_manifest['distinctness']['pairwise_cosine']['A_B']['min'])}`–`{fmt(source_manifest['distinctness']['pairwise_cosine']['A_B']['max'])}`, A/C `{fmt(source_manifest['distinctness']['pairwise_cosine']['A_C']['min'])}`–`{fmt(source_manifest['distinctness']['pairwise_cosine']['A_C']['max'])}`, B/C `{fmt(source_manifest['distinctness']['pairwise_cosine']['B_C']['min'])}`–`{fmt(source_manifest['distinctness']['pairwise_cosine']['B_C']['max'])}`.

Frozen candidate-pair evidence indicates a mixed appearance signal plus a larger candidate/base-score and assignment-interface bottleneck:

| Horizon | Pair rows | Appearance direction correct | Base-wrong rows | Correctable at lambda <= 8 | Base-correct pushed wrong (any scanned lambda) |
|---|---:|---:|---:|---:|---:|
| H20 | 2,943 | {fmt(diagnostic['candidate_pair_diagnostics']['H20']['appearance_directional_positive_rate'])} | 270 | 42 | 126 |
| H50 | 7,674 | {fmt(diagnostic['candidate_pair_diagnostics']['H50']['appearance_directional_positive_rate'])} | 810 | 99 | 279 |
| H100 | 15,879 | {fmt(diagnostic['candidate_pair_diagnostics']['H100']['appearance_directional_positive_rate'])} | 2,125 | 291 | 510 |

The interpretation is therefore `candidate/base-score scale plus assignment interface` as the primary bottleneck, with appearance quality mixed and temporal propagation not isolated as the primary cause. This justified the small, independent T1 probe but did not authorize a production fusion rewrite.

## N42-02 mandatory actual training

T1 was trained as an isolated pairwise calibration head. Its 23-D causal input contains frozen base/memory/appearance/fused gaps, geometry, confidence, native age, frame offset, candidate count, and ranks. Public ID is used only to identify the human-specified target column, not as a feature. The head was `Linear(23,64)-ReLU-Linear(64,32)-ReLU-Linear(32,1)` with bounded preference output; SAM3, candidate generation, and Hungarian implementation were frozen.

- Dataset: {dataset['row_count']} materialized pair rows; positive {dataset['counters']['positive']}, negative {dataset['counters']['negative']}, ambiguous discarded {dataset['counters']['ambiguous_label_discarded']}, GT-unavailable {dataset['counters']['gt_unavailable']}.
- Sequence split was frozen before materialization: train {dataset['split_counts']['train']}, validation {dataset['split_counts']['validation']}, holdout {dataset['split_counts']['holdout']}; no frame-random split and no holdout selection.
- Fixed training config: AdamW, seed {training['seed']}, batch {training['configuration']['batch_size']}, learning rate {training['configuration']['learning_rate']}, weight decay {training['configuration']['weight_decay']}, max epochs {training['configuration']['epochs']}, patience {training['configuration']['early_stopping_patience']}, minimum validation BCE and earliest tie.
- Smoke: `PASS`, finite loss/gradient, save/reload passed. Full training: `PASS`, device `{training['device']}`, completed epochs {training['completed_epochs']}, selected epoch {training['best_epoch']}, validation loss {fmt(training['best_validation_loss'])}.
- Isolated checkpoint: {link(training['checkpoint'])}; SHA-256 `{training['checkpoint_sha256']}`. `production_authorized=false` is recorded in the checkpoint manifest.

## N42-03 paired replay and metrics

T0 and T1 each completed 24/24 event workers, 120/120 variants, and 100 future frames per event. The complete posthoc set is 240 variant results. T1's calibration was applied on {runtime_applied} future branch frames for M1–M4 and the largest observed fused-score adjustment was {fmt(runtime_max_delta)}; event-frame calibration remained hidden. At T1/M2/H20, score changed on {fmt(t1_scores_changed_h20)}, assignment changed on {fmt(t1_assignment_h20)}, correct changes were {t1_correct_h20}, and incorrect changes were {t1_incorrect_h20}. Thus “score changed” did not imply “assignment changed”, and assignment changes did not become correct.

Identity utility is the frozen N36/N37 convention: mean of target IoU improvement and target missing-rate reduction, both write-minus-no-write in the paired future window. `IDF1/HOTA/AssA` are not claimed because these are bounded event windows rather than complete TrackEval sequence inputs. Full per-event metrics, future identity error, missing, re-correction proxy, IoU denominators, assignment transitions, and protected-ID checks are in {link('outputs/n42/replay/posthoc_results.json')} and the event files under {link('outputs/n42/replay/posthoc_events')}.

| Mode | Variant | H20 utility | H50 utility | H100 utility | H20 CI lower | H50 CI lower | H100 CI lower | H20 score-change rate | H20 assignment-change rate | correct / incorrect H20 changes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

T1/M2 action decomposition at H20:

| Action | Events | Identity utility | Assignment changes | Correct changes | Incorrect changes |
|---|---:|---:|---:|---:|---:|
{chr(10).join(action_lines)}

The strict T1 holdout result is:

| Variant | Holdout status | H20 lower CI | H50 lower CI | H100 lower CI | Protected-ID check |
|---|---|---:|---:|---:|---|
| M2 | `{t1_gate_status['M2']}` | {fmt(t1_m2['holdout_gate']['per_horizon']['20']['lower_ci'])} | {fmt(t1_m2['holdout_gate']['per_horizon']['50']['lower_ci'])} | {fmt(t1_m2['holdout_gate']['per_horizon']['100']['lower_ci'])} | {t1_m2['holdout_gate']['per_horizon']['20']['protected_no_obvious_regression']} |
| M3 | `{t1_gate_status['M3']}` | {fmt(t1_m3['holdout_gate']['per_horizon']['20']['lower_ci'])} | {fmt(t1_m3['holdout_gate']['per_horizon']['50']['lower_ci'])} | {fmt(t1_m3['holdout_gate']['per_horizon']['100']['lower_ci'])} | {t1_m3['holdout_gate']['per_horizon']['20']['protected_no_obvious_regression']} |
| M4 | `{t1_gate_status['M4']}` | {fmt(t1_m4['holdout_gate']['per_horizon']['20']['lower_ci'])} | {fmt(t1_m4['holdout_gate']['per_horizon']['50']['lower_ci'])} | {fmt(t1_m4['holdout_gate']['per_horizon']['100']['lower_ci'])} | {t1_m4['holdout_gate']['per_horizon']['20']['protected_no_obvious_regression']} |

The protected-ID checks were clean, but that is necessary rather than sufficient. The strict positive-effect requirement failed because the holdout lower bounds were not `> 0`; T1/M2 was negative on the full event set and T1/M3/M4 were effectively null. No weight expansion, checkpoint replacement, threshold tuning, LoRA, selector, or calibration promotion was performed.

## Public-method retrieval

The search covered GitHub, arXiv, and official paper pages for 2025–2026 with queries on human-in-the-loop MOT, interactive tracking, online ReID/appearance memory, association/calibration, margin-aware matching, and parameter-efficient adaptation. No expansion beyond 2025–2026 was needed for the retained entries. Public methods were design context only; none replaced the frozen baseline or supplied N42 evidence.

{chr(10).join(method_lines)}

Retrieval artifact: {link('outputs/n42/method_retrieval.json')}.

## MOT/OVMOT isolation and reproducibility

Isolation result: `{isolation['status']}` in {link('outputs/n42/isolation_regression.json')}.

- All pre-existing project code hashes were unchanged: baseline {isolation['project_code']['baseline_count']} files, current {isolation['project_code']['current_count']} files; the only new code paths were the explicitly isolated `scripts/n42_*` files ({len(allowed_new)} allowed additions).
- Config hash, shared checkpoint hashes, N39–N41 protected text hashes, and N39–N41 output-tree inventories all passed unchanged.
- The sibling MOT/InterMOT metadata inventory was unchanged. No OVMOT directory was found under the Interactive root. The N36 integrity audit and N38 final gate both record `third_party_sam3_modified=false`; N42 wrote no third-party file.
- Existing SAM3_InterMOT import regression passed; all 113 existing tests passed (three pre-existing dependency deprecation warnings only). All N42 scripts compiled.
- GPU policy: the audited machine exposed A100 40-GB cards; N42 used only GPU0 for the small T1 training run and CPU-only replay. No concurrent sequence workers were launched. A future authorized experiment may use at most GPUs 0–3, one independent sequence/frame-range process per card, with isolated output/checkpoint roots; that plan is not an authorization to run it now.

## Preserved failures and repairs

Every failed attempt below remains on disk and was not relabeled as a pass:

{failure_lines}

The environment also emitted the existing `osr_lib-1.1.0-nspkg.pth` loader warning during some launches; commands still returned success where claimed. The runtime PASS snapshot is preserved at {link('outputs/n42/attempts/stage_03_runtime_pass_snapshot.json')}.

## Decision and ICLR calendar

The final decision is `FAIL_FUTURE_EFFECT`: T1 was actually trained and evaluated, but it is not a production candidate. Calibration head, selector, and decoder LoRA are all `NOT_AUTHORIZED`. The smallest scientifically valid next step is external collection of a real human event tape through the N40 ingestion contract; synthetic-from-GT events must not be renamed or used as real-human evidence. No further blind weight scan is justified by N42.

| Date | Constraint / action |
|---|---|
| 2026-08-30 | N42 diagnosis, actual T1 training, paired replay, isolation, and negative gate frozen. |
| 2026-09-01–09-10 | Only schema-validating externally supplied real-human tape is actionable; no synthetic substitution or production training authorization. |
| 2026-09-18 AoE | ICLR 2027 abstract deadline. |
| 2026-09-25 AoE | ICLR 2027 full-paper deadline. |

Research log entry must state the negative result, frozen protocol, data range, failure causes, and next step; it must not equate execution completion with the scientific hypothesis being proven.
"""
    atomic_text(REPORT, report_text)

    research_entry = f"""\n\n## N42 — 2026-08-30\n\n- Hypothesis/protocol: isolated T1 association/fusion calibration probe on frozen N37 24-event/21-sequence `simulated_from_gt` replay; runtime future GT remained false.\n- Result: actual T1 smoke/full training and 240-result T0/T1 posthoc replay completed; T1 changed scores but failed the strict holdout future-effect gate (`{gate_status}`).\n- Preservation/isolation: N39–N41 evidence, shared checkpoints/configs, sibling MOT metadata, and third-party SAM3 remained unchanged; 113 existing tests passed.\n- Decision: T1 checkpoint is trained but not authorized for production; selector/decoder LoRA remain unauthorized. Smallest next step is external real-human tape collection through N40, not another blind weight scan.\n- Evidence: `{GATE.relative_to(ROOT)}`, `{REPORT.relative_to(ROOT)}`.\n"""
    log_entry = OUT / "n42_research_log_entry.md"
    atomic_text(log_entry, research_entry.lstrip())
    print(json.dumps({"status": gate_status, "gate": str(GATE), "report": str(REPORT), "research_log_entry": str(log_entry)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
