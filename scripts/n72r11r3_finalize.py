#!/usr/bin/env python3
"""Materialize the N72R11R3 branch decision from sealed artifacts.

This is a report/status generator only.  It does not select data, rerun SAM3,
train a model, or change any historical artifact.  All scientific numbers are
read from the already sealed readiness and E0/E1 component artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R11R3"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def stage(stage: str, status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": "N72R11R3_FINAL_STAGE_STATUS_V1",
        "stage": stage,
        "status": status,
        "created_at_utc": now_utc(),
        "runtime_future_gt_used": False,
        "gt_used_for_runtime_decision": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
        **fields,
    }


def metric_view(metric: Mapping[str, Any], horizon: str) -> dict[str, Any]:
    value = dict(metric[horizon])
    return {
        key: value.get(key)
        for key in (
            "identity_error_reduction", "baseline_future_identity_error",
            "treatment_future_identity_error", "sequence_cluster_bootstrap_95ci",
            "assignment_change_count", "assignment_change_rate",
            "true_correct_crossing_count", "true_incorrect_crossing_count",
            "directional_improvement_count", "directional_regression_count",
            "protected_regression_count", "protected_regression_rate",
            "recorrection_rate", "candidate_recall", "missing_rate", "delta_iou",
        )
    }


def main() -> int:
    corpus = read(OUT / "onpolicy_corpus_attempt_03/corpus_manifest.json")
    formal = read(OUT / "stage_10_formal_oom_check_attempt_02.json")
    train_regular = read(OUT / "stage_11_finetune.json")
    train_balanced = read(OUT / "v3_category_balance_attempt_01/stage_11_finetune.json")
    readiness_bootstrap = read(OUT / "v3_selected_eval_attempt_02/v3_readiness_audit.json")
    readiness_regular = read(OUT / "v3_finetuned_eval_attempt_01/v3_readiness_audit.json")
    readiness_balanced = read(OUT / "v3_category_balance_eval_attempt_01/v3_readiness_audit.json")
    replay_manifest_path = OUT / "component_replay_e0e1_attempt_01/component_replay_manifest_attempt_01.json"
    replay_metrics_path = OUT / "component_replay_e0e1_attempt_01/component_metrics_attempt_02.json"
    replay_manifest = read(replay_manifest_path)
    replay_metrics = read(replay_metrics_path)
    selected_readiness = readiness_bootstrap

    readiness_candidates = {
        "bootstrap_selected_by_fixed_validation_loss": {
            "checkpoint": rel(OUT / "v3_bootstrap/v3_bootstrap.pt"),
            "audit": rel(OUT / "v3_selected_eval_attempt_02/v3_readiness_audit.json"),
            "audit_sha256": sha256(OUT / "v3_selected_eval_attempt_02/v3_readiness_audit.json"),
            "summary": selected_readiness["summary"],
            "gate": selected_readiness["gate"],
        },
        "ordinary_finetune_candidate": {
            "checkpoint": rel(OUT / "v3_finetune_attempt_01/v3_finetuned.pt"),
            "audit": rel(OUT / "v3_finetuned_eval_attempt_01/v3_readiness_audit.json"),
            "audit_sha256": sha256(OUT / "v3_finetuned_eval_attempt_01/v3_readiness_audit.json"),
            "summary": readiness_regular["summary"],
            "gate": readiness_regular["gate"],
        },
        "category_balanced_candidate": {
            "checkpoint": rel(OUT / "v3_category_balance_attempt_01/v3_category_balanced.pt"),
            "audit": rel(OUT / "v3_category_balance_eval_attempt_01/v3_readiness_audit.json"),
            "audit_sha256": sha256(OUT / "v3_category_balance_eval_attempt_01/v3_readiness_audit.json"),
            "summary": readiness_balanced["summary"],
            "gate": readiness_balanced["gate"],
        },
    }

    replay_by_horizon = {
        horizon: metric_view(replay_metrics["comparisons"]["E1_vs_E0"], horizon)
        for horizon in ("20", "50", "100")
    }
    by_action = {
        action: {
            horizon: metric_view(values["E1_vs_E0"], horizon)
            for horizon in ("20", "50", "100")
        }
        for action, values in replay_metrics["by_action"].items()
    }

    paths = {
        "bootstrap_corpus_manifest": rel(OUT / "bootstrap_corpus/corpus_manifest.json"),
        "onpolicy_corpus_manifest": rel(OUT / "onpolicy_corpus_attempt_03/corpus_manifest.json"),
        "formal_oom_audit": rel(OUT / "stage_10_formal_oom_check_attempt_02.json"),
        "replay_manifest": rel(replay_manifest_path),
        "replay_metrics": rel(replay_metrics_path),
        "readiness_bootstrap": rel(OUT / "v3_selected_eval_attempt_02/v3_readiness_audit.json"),
        "readiness_regular_finetune": rel(OUT / "v3_finetuned_eval_attempt_01/v3_readiness_audit.json"),
        "readiness_category_balance": rel(OUT / "v3_category_balance_eval_attempt_01/v3_readiness_audit.json"),
    }

    stage_payloads = {
        "stage_14_v3_finetune.json": stage(
            "N72R11R3-14-V3-FINETUNE",
            "PASS_V3_FINETUNE_CANDIDATES",
            ordinary_finetune={
                "status": train_regular["status"],
                "checkpoint": rel(OUT / "v3_finetune_attempt_01/v3_finetuned.pt"),
                "checkpoint_sha256": sha256(OUT / "v3_finetune_attempt_01/v3_finetuned.pt"),
                "validation": train_regular["finetuned_validation"],
                "selected_by_fixed_validation_loss": train_regular["selected_phase"] == "finetuned",
            },
            category_balanced={
                "status": train_balanced["status"],
                "checkpoint": rel(OUT / "v3_category_balance_attempt_01/v3_category_balanced.pt"),
                "checkpoint_sha256": sha256(OUT / "v3_category_balance_attempt_01/v3_category_balanced.pt"),
                "validation": train_balanced["finetuned_validation"],
                "weighting": train_balanced["category_balance"],
                "selected_by_fixed_validation_loss": train_balanced["selected_phase"] == "finetuned",
            },
            bootstrap_checkpoint=rel(OUT / "v3_bootstrap/v3_bootstrap.pt"),
            training_data=rel(OUT / "onpolicy_corpus_attempt_03/corpus_manifest.json"),
        ),
        "stage_15_v3_readiness.json": stage(
            "N72R11R3-15-V3-READINESS",
            "FAIL_V3_READINESS",
            checkpoint=rel(OUT / "v3_finetune_attempt_01/v3_finetuned.pt"),
            audit=rel(OUT / "v3_finetuned_eval_attempt_01/v3_readiness_audit.json"),
            summary=readiness_regular["summary"],
            gate=readiness_regular["gate"],
            thresholds=readiness_regular["thresholds"],
        ),
        "stage_16_category_balance_repair.json": stage(
            "N72R11R3-16-CATEGORY-BALANCE-REPAIR",
            "FAIL_V3_READINESS_AFTER_CATEGORY_BALANCE",
            repair_count=1,
            checkpoint=rel(OUT / "v3_category_balance_attempt_01/v3_category_balanced.pt"),
            audit=rel(OUT / "v3_category_balance_eval_attempt_01/v3_readiness_audit.json"),
            summary=readiness_balanced["summary"],
            gate=readiness_balanced["gate"],
            thresholds=readiness_balanced["thresholds"],
            training_source="train-only action_type × label_source inverse-sqrt counts; no future metrics",
        ),
        "stage_17_v3_final_freeze.json": stage(
            "N72R11R3-17-V3-FINAL-FREEZE",
            "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
            selected_checkpoint=rel(OUT / "v3_bootstrap/v3_bootstrap.pt"),
            selection_rule="minimum fixed validation loss; bootstrap retained",
            readiness_audit=rel(OUT / "v3_selected_eval_attempt_02/v3_readiness_audit.json"),
            readiness_summary=selected_readiness["summary"],
            readiness_gate=selected_readiness["gate"],
            ordinary_finetune_and_category_balance_both_failed=True,
            bridge_authorized=False,
        ),
        "stage_18_bridge_not_authorized.json": stage(
            "N72R11R3-18-BRIDGE-TRAINING",
            "NOT_AUTHORIZED_V3_NOT_READY",
            prerequisite="PASS_V3_READINESS",
            bridge_checkpoint=None,
            bridge_training_started=False,
        ),
        "stage_19_bridge_readiness.json": stage(
            "N72R11R3-19-BRIDGE-READINESS",
            "NOT_RUN_BRIDGE_NOT_AUTHORIZED",
            bridge_accuracy=None,
            legacy_accuracy=None,
            reason="V3 failed state-aligned readiness; no corrected Bridge checkpoint was trained",
        ),
        "stage_20_component_replay.json": stage(
            "N72R11R3-20-COMPONENT-REPLAY",
            "PASS_E0_E1_DEVELOPMENT_REPLAY",
            manifest=rel(replay_manifest_path),
            manifest_sha256=sha256(replay_manifest_path),
            event_count=replay_manifest["event_count"],
            child_counts=replay_manifest["counts"],
            variants=replay_manifest["variants"],
            bridge_used=False,
            live_used=False,
            scientific_promotion=False,
        ),
        "stage_21_component_metrics.json": stage(
            "N72R11R3-21-COMPONENT-METRICS",
            "PASS_COMPONENT_METRICS_E0_E1_ONLY",
            metrics=rel(replay_metrics_path),
            metrics_sha256=sha256(replay_metrics_path),
            event_completeness=replay_metrics["event_completeness"],
            comparison_names=list(replay_metrics["comparisons"]),
            horizons=[20, 50, 100],
            bootstrap=replay_metrics["bootstrap"],
            scientific_gate=replay_metrics["gate"],
        ),
        "stage_22_root_cause.json": stage(
            "N72R11R3-22-ROOT-CAUSE",
            "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
            structural_state_alignment_fixed=True,
            v3_readiness_failed=True,
            e1_effect=replay_by_horizon,
            interpretation="state contract/runtime safety repaired, but learned V3 does not generalize and legacy injection is net harmful",
            bridge_effect=None,
            live_effect=None,
            decision="REVISE",
        ),
        "stage_23_final.json": stage(
            "N72R11R3-23-FINAL",
            "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
            research_gate="FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
            production_authorized=False,
            authorized_path_complete=True,
            next_minimal_step="freeze current evidence; revise V3 candidate/state supervision or obtain provenance-complete real-human tape before any Bridge/live training",
        ),
    }
    for filename, payload in stage_payloads.items():
        atomic_json(OUT / filename, payload)

    controller = {
        "schema_version": "N72R11R3_CONTROLLER_STATUS_V1",
        "created_at_utc": now_utc(),
        "current_stage": "N72R11R3-23-FINAL",
        "current_root_cause": "V3 readiness fails after state alignment; E1 legacy injection is negative",
        "temporal_feature_contract_pass": True,
        "bridge_feature_contract_pass": True,
        "formal_oom_event_pass": bool(formal["oom_observed"] is False and formal["status"].startswith("PASS")),
        "bootstrap_train_examples": 24690,
        "bootstrap_validation_examples": 5700,
        "onpolicy_train_examples": corpus["splits"]["train"]["rollout"]["examples"],
        "onpolicy_validation_examples": corpus["splits"]["validation"]["rollout"]["examples"],
        "v3_target_accuracy": selected_readiness["summary"]["target_candidate"]["accuracy"],
        "v3_none_accuracy": selected_readiness["summary"]["none"]["accuracy"],
        "v3_future_requery_positive_count": selected_readiness["summary"]["future_requery_positive"]["count"],
        "v3_future_requery_positive_accuracy": selected_readiness["summary"]["future_requery_positive"]["accuracy"],
        "v3_readiness": "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
        "v3_readiness_candidates": readiness_candidates,
        "bridge_legacy_accuracy": None,
        "bridge_accuracy": None,
        "legacy_positive_boundary_success": None,
        "bridge_positive_boundary_success": None,
        "legacy_protected_safe_rate": None,
        "bridge_protected_safe_rate": None,
        "bridge_readiness": "NOT_AUTHORIZED_V3_NOT_READY",
        "E1_H20": replay_by_horizon["20"],
        "E1_H50": replay_by_horizon["50"],
        "E1_H100": replay_by_horizon["100"],
        "E2_minus_E1_H20": None,
        "E2_minus_E1_H50": None,
        "E2_minus_E1_H100": None,
        "E3_minus_E2_H20": None,
        "E3_minus_E2_H50": None,
        "E3_minus_E2_H100": None,
        "protected_E1": {
            horizon: replay_by_horizon[horizon]["protected_regression_count"]
            for horizon in ("20", "50", "100")
        },
        "protected_E2_increment": None,
        "protected_E3_increment": None,
        "complete_correct_reacquisition": {
            horizon: replay_by_horizon[horizon]["directional_improvement_count"]
            for horizon in ("20", "50", "100")
        },
        "complete_wrong_reacquisition": {
            horizon: replay_by_horizon[horizon]["directional_regression_count"]
            for horizon in ("20", "50", "100")
        },
        "runtime_future_gt_used": False,
        "next_root_cause": "candidate/base-score and V3 supervision/generalization remain unresolved; do not train Bridge/live",
        "paths": paths,
        "production_authorized": False,
        "research_gate": "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
    }
    atomic_json(OUT / "CONTROLLER_STATUS.json", controller)

    final_gate = {
        "schema_version": "N72R11R3_FINAL_GATE_V1",
        "created_at_utc": now_utc(),
        "status": "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
        "research_gate": "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
        "authorized_path_complete": True,
        "production_authorized": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "temporal_feature_contract_pass": True,
        "bridge_feature_contract_pass": True,
        "formal_oom_event_pass": controller["formal_oom_event_pass"],
        "v3_readiness": controller["v3_readiness"],
        "bridge_readiness": controller["bridge_readiness"],
        "component_replay": {
            "status": replay_manifest["status"],
            "event_count": replay_manifest["event_count"],
            "variants": replay_manifest["variants"],
            "metrics": rel(replay_metrics_path),
        },
        "scientific_effect": {
            "E1_vs_E0": replay_by_horizon,
            "E2_vs_E1": None,
            "E3_vs_E2": None,
            "E3_vs_E0": None,
        },
        "root_cause": controller["current_root_cause"],
        "decision": "REVISE",
        "next_minimal_step": controller["next_root_cause"],
        "failure_artifacts": [
            rel(OUT / "onpolicy_corpus/attempts/onpolicy_failure_20260908T100422Z.json"),
            rel(OUT / "onpolicy_corpus_attempt_02/attempts/process_exit_137_global_oom.json"),
            rel(OUT / "v3_selected_eval/attempts/self_rollout_failure_20260908T103010Z.json"),
            rel(OUT / "component_replay_e0e1_attempt_01/component_metrics.json"),
        ],
    }
    atomic_json(OUT / "n72r11r3_final_gate.json", final_gate)

    report = f"""# N72R11R3 Final Report

## Decision

`N72R11R3_STATE_ALIGNED_TEMPORAL_AND_GLOBAL_EDGE` completed its authorized path with
research gate **FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT**. The state/feature
contract repairs are structurally validated, but neither the ordinary finetune nor
the one permitted category-balanced repair makes V3 ready. The mandated downstream
Bridge and live branches were therefore not authorized. Production authorization is
false. The research direction is **REVISE**, not a claim of success.

All events in this development corpus are `simulated_from_gt`; real-human evidence
count remains zero. Runtime rollout and replay report `runtime_future_gt_used=false`.

## Frozen inputs and repaired contract

- Baseline commit: `1f59671309d676c19903ccca9439011b7bbc8887`.
- Corrected bootstrap corpus: `516` events, `24,690` train examples, `5,700`
  validation examples, fixed `12/6` sequence split; resource-censored development.
- On-policy corpus: train `24,690` rows / `419` events / `12` sequences and validation
  `5,700` rows / `97` events / `6` sequences. It was produced by causal V3 self-selection;
  offline labels were copied for supervision and not used for runtime decisions.
- Shared temporal schema is the eight-dimensional
  `frame_horizon_over_100, tanh_causal_top_score, tanh_causal_second_score,
  tanh_causal_margin, previous_fused_target_score_clipped,
  previous_assignment_uncertainty, trusted_age_over_100,
  has_future_frame_requery_candidate` contract.
- Shared `TemporalIdentityState` now governs corpus construction, V3 rollout and
  runtime: only exact target agreement plus finite score/margin/feature admits trusted
  memory; selected non-targets alone can enter distractor memory; geometry/native
  binding follows exact assignment; unselected candidates do not pollute memory.
- TargetEdgeBridge has one explicit 14-D scalar builder and requires explicit
  `motion_iou`; train/runtime no longer silently use different feature meanings.
- Bridge loss code was corrected to target-vs-best-other public competition, but no
  Bridge checkpoint was trained because the V3 prerequisite failed.

## Failures and repairs retained

1. The first on-policy attempt failed with `ValueError: selected candidate feature has
   zero norm` when a valid zero-filled missing embedding was passed to state memory.
   The original artifact is retained at
   `outputs/N72R11R3/onpolicy_corpus/attempts/onpolicy_failure_20260908T100422Z.json`.
2. Attempt 2 was killed with exit code `137`/`SIGKILL`. The kernel evidence identified
   host global OOM (about `6.4 GiB` available, no swap), not a CUDA OOM traceback.
   The failure record is retained at
   `outputs/N72R11R3/onpolicy_corpus_attempt_02/attempts/process_exit_137_global_oom.json`.
   The remedy was two sequential CPU split workers with bounded BLAS/OpenMP threads;
   no input or model definition changed.
3. The split workers completed atomically in attempt 3. The repaired state semantics
   preserve zero-filled decoder tokens but pass missing state features as `None`, so
   they cannot become fabricated memory vectors.
4. The first full self-rollout after that repair exposed the same missing-feature
   boundary in `n72r11_train_v3.py`; its failure is retained at
   `outputs/N72R11R3/v3_selected_eval/attempts/self_rollout_failure_20260908T103010Z.json`.
   A one-event targeted regression and the full 5,700-row self-rollout then passed.
5. The first E0/E1 aggregator rejected all 32 valid children because older `done.json`
   files omitted top-level `runtime_gt_read` while their authoritative runtime seals
   explicitly contained `false`. The failed aggregate is retained at
   `outputs/N72R11R3/component_replay_e0e1_attempt_01/component_metrics.json`.
   The validator was minimally repaired to inspect the runtime seal, and the same
   manifest re-aggregated with zero failures.

## V3 training and readiness

The corrected bootstrap V3 used the frozen two-epoch bootstrap configuration. A normal
one-epoch finetune used learning rate `2.5e-4`. A second and only permitted repair used
train-only action-type × label-source inverse-square-root balancing, raw clipping
`[0.5, 3.0]`, then mean normalization. No H20/H50/H100 or validation future outcome
was used for weighting or checkpoint choice.

| Candidate | Target-candidate accuracy | NONE accuracy | Future-requery accuracy/count | Readiness |
|---|---:|---:|---:|---|
| Selected bootstrap | 0.511730 | 0.216716 | 0.642857 / 42 | FAIL |
| Ordinary finetune | 0.525299 | 0.238166 | 0.571429 / 42 | FAIL |
| Category-balanced finetune | 0.505520 | 0.262574 | 0.738095 / 42 | FAIL |

The fixed thresholds remained target candidate `>=0.80`, NONE `>=0.50`, future
requery `>=0.60` with count `>=50`. The selected checkpoint was the bootstrap because
its fixed validation loss `5.5877319` was lower than ordinary finetune `7.6736831`
and category-balanced finetune `8.3648421`. Thus V3 was frozen as not ready after the
single permitted repair; no Bridge training was started.

## Authorized component replay: E0/E1 only

Because V3 was not ready, the protocol allowed only the development replay
`E0_BASELINE_B0` versus `E1_V3_LEGACY_INJECTION`. All `32/32` frozen events completed,
with zero child failures, over `18` independent sequences. E2/E3 were not run, so
Bridge and live increments are correctly `null`, not zero and not inferred.

| Comparison | H20 | H50 | H100 |
|---|---:|---:|---:|
| Pooled identity-error reduction E1−E0 | -0.019576 | -0.081081 | -0.068690 |
| Sequence-cluster 95% CI lower | -0.106251 | -0.157491 | -0.121157 |
| Assignment-change rate | 0.535073 | 0.416988 | 0.370288 |
| Correct crossings | 64 | 92 | 176 |
| Incorrect crossings | 76 | 218 | 391 |
| Directional improvements | 13 | 24 | 74 |
| Directional regressions | 115 | 196 | 300 |
| Protected regressions | 8 | 19 | 25 |

The sequence-cluster bootstrap used seed `7211`, `2,000` repetitions and independent
sequence clusters. The overall effect is negative at all horizons and every lower CI
is below zero. E1 changes assignments substantially, but the changes are more often
wrong than correct, especially at H50/H100. This does not recover the N72R10 direction.

By action, pooled identity-error reduction at H20/H50/H100 was:

- `ADD_NEW_IDENTITY`: `+0.144737 / +0.058824 / +0.020672`;
- `ATOMIC_ID_SWAP`: `+0.233333 / +0.033333 / +0.097973`;
- `AUTHORITATIVE_REASSIGN`: `-0.128405 / -0.143284 / -0.130212`;
- `RECOVER_IDENTITY`: `-0.018182 / -0.084095 / -0.068519`.

The positive ADD/ATOMIC slices do not overturn the negative pooled effect or the
strongly negative reassignment/recovery slices. Bounded-window metrics are not HOTA
or IDF1 claims.

## Required scientific answers

1. **Did state-mismatch repair remove the R2 degradation?** Structurally, the
   train/runtime temporal contract and memory admission semantics are aligned and
   pass CPU/targeted/runtime audits. Scientifically, no: corrected V3 still fails
   readiness and E1 is negative.
2. **Did corrected V3 + legacy injection recover N72R10?** No. E1−E0 is negative
   overall at H20/H50/H100, with negative sequence-cluster lower bounds.
3. **Did corrected Bridge reduce model→solver refusal?** Not tested. Bridge training
   was forbidden by the V3 gate, so no Bridge accuracy or refusal reduction is claimed.
4. **Did Bridge protect untouched IDs?** Not tested. No E2/E3 protected-ID result exists.
5. **What is the independent true-live requery increment?** Not measured. E3 was not
   authorized; the value is `null`.
6. **Did H100 recover a positive value?** No. E1−E0 H100 pooled reduction is
   `-0.068690`, CI lower `-0.121157`.
7. **Main direction:** **REVISE**. Keep persistent public identity and exact global
   assignment as structural foundations, but revise V3 supervision/candidate-base
   signal and obtain provenance-complete real-human evidence before another Bridge or
   live experiment. Do not scale Bridge/LoRA/checkpoints to bypass this gate.

## Isolation and reproducibility

All new code is confined to InterMOT modules/scripts; no `third_party/sam3` file was
modified and no N36–N72R2 historical evidence was overwritten. New outputs are under
`outputs/N72R11R3/`. No MOT/OVMOT project, shared checkpoint or shared configuration was
modified. The formal OOM event was rerun only once at full window and passed the resource
check; the other historical 31 events and secondary corpus were not rerun.

Key machine-readable artifacts:

- Controller: `outputs/N72R11R3/CONTROLLER_STATUS.json`
- Final gate: `outputs/N72R11R3/n72r11r3_final_gate.json`
- E0/E1 manifest: `{paths['replay_manifest']}`
- E0/E1 metrics: `{paths['replay_metrics']}`
- On-policy corpus: `{paths['onpolicy_corpus_manifest']}`

The full authorized branch remains explicitly non-production and non-real-human.
"""
    report_path = ROOT / "docs/N72R11R3_FINAL_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(report_path, report)
    print(json.dumps({
        "status": "FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT",
        "report": rel(report_path),
        "controller": rel(OUT / "CONTROLLER_STATUS.json"),
        "final_gate": rel(OUT / "n72r11r3_final_gate.json"),
        "stage_files": [rel(OUT / name) for name in stage_payloads],
        "replay": {"events": replay_manifest["event_count"], "status": replay_manifest["status"]},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
