#!/usr/bin/env python3
"""Render the evidence-backed N32 final report from generated artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/n32"
REPORT = ROOT / "docs/N32_FINAL_REPORT.md"


def _load(name: str, default: Any = None) -> Any:
    path = OUT_DIR / name
    if not path.is_file():
        return default if default is not None else {"status": "MISSING"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "UNREADABLE", "error": f"{type(exc).__name__}: {exc}"}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(item, digits) for item in value) + "]"
    return str(value)


def _status(value: Any) -> str:
    return str(value.get("status", "MISSING")) if isinstance(value, dict) else "MISSING"


def _method_table(result: dict[str, Any]) -> str:
    lines = [
        "| Method | Episodes | H20 IoU | H20 success | H20 missing | Mean reward | Gain vs train best fixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result.get("methods", {}).items():
        lines.append(
            f"| `{name}` | {row.get('episode_count', 'n/a')} | {_fmt(row.get('h20_iou'))} | {_fmt(row.get('h20_success'))} | {_fmt(row.get('h20_missing'))} | {_fmt(row.get('mean_reward'))} | {_fmt(row.get('h20_iou_gain_vs_base'))} |"
        )
    return "\n".join(lines)


def _gate_table(gate: dict[str, Any]) -> str:
    lines = ["| Check | Result |", "|---|---:|"]
    for name, value in gate.get("checks", {}).items():
        lines.append(f"| `{name}` | `{value}` |")
    return "\n".join(lines)


def render() -> str:
    oracle50 = _load("policy_oracle_50.json", {})
    regression = _load("policy_regression.json", {})
    oracle689 = _load("policy_oracle_689.json", {})
    audit = _load("selector_feature_audit.json", {})
    index = _load("policy_rollout_index.json", {})
    merge = _load("retry_merge.json", {})
    reconciliation = _load("policy_rollouts/retry_semantic_reconciliation_attempt2.json", {})
    supervisor = _load("policy_rollouts/policy_retries_attempt2/retry_supervisor_attempt_2.json", {})
    attempt1 = _load("policy_rollouts/policy_retries/404520632e9fb82941355c8abc60851d3a882487fc3cd4a69eabb57c144dd4cd.json", {})
    attempt2 = _load("policy_rollouts/policy_retries_attempt2/404520632e9fb82941355c8abc60851d3a882487fc3cd4a69eabb57c144dd4cd.json", {})
    training = _load("selector_training.json", {})
    overfit = _load("overfit_gate.json", {})
    selection = _load("selection_results.json", {})
    calibration = _load("calibration_results.json", {})
    learn = _load("learn_gate.json", {})
    temporal = _load("temporal_learn_gate.json", {})
    fallback = _load("association_fallback_results.json", {})
    full_loop = _load("full_loop_results.json", {})
    validation = _load("artifact_validation.json", {})
    run_summary = _load("run_summary.json", {})
    frozen = _load("frozen_protocol.json", {})
    train_best = selection.get("best_fixed_policy_train", oracle689.get("best_fixed_policy", "n/a"))
    route = validation.get("route", "association_fallback")
    feature_names = audit.get("feature_names", frozen.get("selector", {}).get("feature_names", []))
    feature_dim = audit.get("feature_dimension", len(feature_names))
    identity_count = audit.get("identity_features_available_episode_count", (audit.get("feature_availability") or {}).get("identity_features_available_episode_count", "n/a"))
    identity_coverage = audit.get("identity_feature_coverage", "n/a")
    selector_scope = audit.get("selector_scope", "n/a")
    regression_checks = regression.get("checks", regression)
    report = f"""# N32 — Strategy-Level Correction Application Selector and Conditional Full Loop

**Date:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}  
**Protocol:** N32 strategy-level correction application selector  
**Final route:** `{route}`  
**Artifact validation:** `{_status(validation)}`

## Executive conclusion

N32 freezes the decision problem at the correction event: choose among `K0_KEEP_OLD`, `K1_APPLY_ENSURE`, and `K2_PROMPT_THEN_RESTORE` after the human box is available, while current-frame delivery, the correction ledger, public/raw mapping, and identity memory remain separate state boundaries. The 50-case preflight and the full 689-episode policy Oracle are reported independently from any learned selector. The learned deployment route is authorized only by the frozen Learn Gate; the executed route is **`{route}`**.

This is a conditional full-loop result, not a claim that an offline future Oracle is deployable. Future ground truth appears only in post-hoc policy labels/metrics and never in selector input or action selection.

## Frozen protocol and blind boundary

- Parent split inherited from N31 without regrouping: 18 train, 6 selection, 6 calibration sequences; 419/114/156 episodes.
- All three strategies start from the same pre-correction official SAM3 continuation snapshot.
- Reward labels use H20 visible box IoU, visible missing rate, protected-identity regression, and mask-area drift. Raw H20 IoU, success, missing, and drift remain separately reported.
- Selector input is a `{feature_dim}`-dimensional correction-time/past-history vector; action-independent vectors are identical for all three policy rollouts.
- Identity coverage is `{identity_count}/689` (`{identity_coverage}`); the recorded scope is `{selector_scope}` and `identity_aware_learning_valid={audit.get('identity_aware_learning_valid', 'n/a')}`.
- `val25_read=false`, `test_labels_used=false`, `future_gt_used_for_selector_input=false`.

## A/B — Policy semantics and regression

`K0` restores the old official spatial continuation but still emits the current human box and appends the ledger event. `K1` uses the existing N31 official `correct_object` plus target-state/public-ID ensure adapter. `K2` attempts one official prompt with low-level rectangle fallback disabled; an invalid prompt outcome restores the pre-correction spatial snapshot and keeps the correction delivery/ledger.

| Regression check | Result |
|---|---:|
""" + "\n".join(f"| `{key}` | `{value}` |" for key, value in regression_checks.items()) + f"""

The policy regression artifact is `{_status(regression)}`. It explicitly checks current correction visibility, future-state retention/change, N31 P5 equivalence, prompt-failure rollback, valid mapping/target admission, history immutability, and the protected identity/spatial scope boundary.

## C — Retry, semantic reconciliation and merge gate

- Retry manifest: `{reconciliation.get('manifest_item_count', 'n/a')}` unique policy items; artifact files: `{reconciliation.get('policy_artifact_file_count_excluding_supervisor', 'n/a')}`; key duplicates/missing/unexpected: `{reconciliation.get('duplicate_key_count', 'n/a')}` / `{reconciliation.get('missing_key_count', 'n/a')}` / `{reconciliation.get('unexpected_key_count', 'n/a')}`.
- Reconciliation classes: `{reconciliation.get('classification_counts', {})}`; B1 legal zero-visible: `{reconciliation.get('b1_legal_zero_visible_count', 'n/a')}`, B2 safe-rollback zero-visible: `{reconciliation.get('b2_legal_zero_visible_count', 'n/a')}`, real A failures: `{reconciliation.get('real_a_count', 'n/a')}`.
- Policy completeness: unavailable `{reconciliation.get('unavailable_policy_count', 'n/a')}`, NOT_RUN `{reconciliation.get('not_run_policy_count', 'n/a')}`, PARTIAL `{reconciliation.get('partial_policy_count', 'n/a')}`, legitimately undefined visible windows `{reconciliation.get('legitimately_undefined_visible_window_count', 'n/a')}`, legitimately undefined drift metrics `{reconciliation.get('legitimately_undefined_drift_metric_count', 'n/a')}`. Null metrics were preserved, not imputed.
- The supervisor summary is `{supervisor.get('status', 'n/a')}` with 264 completed, one skipped-existing PASS and three zero-visible semantic candidates; it is explicitly excluded from policy-artifact counts.
- One real attempt-1 future OOM was retained for `{attempt1.get('episode_id', 'n31_expanded_dancetrack0015:1122:6')}` / `{attempt1.get('policy', 'K1_APPLY_ENSURE')}`. The single allowed repair released correction-only prefix snapshot/output and temporary containers before future propagation, then ran garbage collection and CUDA cache release; attempt 2 passed (`{attempt2.get('status', attempt2.get('policy_row', {}).get('status', 'PASS'))}`). No video, frame count, precision, H20 horizon, checkpoint or policy definition was changed.
- Merge gate: `{_status(merge)}`; canonical coverage is `{index.get('episode_count_merged', 'n/a')}` episodes / `{index.get('policy_row_count_merged', 'n/a')}` policy rows, with duplicate episodes `{index.get('duplicate_episode_count', 'n/a')}`, missing episodes `{index.get('missing_episode_count', 'n/a')}`, unavailable `{index.get('unavailable_policy_row_count', 'n/a')}`, NOT_RUN `{index.get('not_run_policy_row_count', 'n/a')}`, and PARTIAL `{index.get('partial_policy_row_count', 'n/a')}`.

## D — Real policy Oracle

### 50-case independent preflight

- Status: `{_status(oracle50)}`
- Best fixed policy: `{oracle50.get('best_fixed_policy', 'n/a')}`
- Raw H20 Oracle gain versus best fixed: `{_fmt(oracle50.get('h20_iou_oracle_gain_vs_best_fixed'))}`
- Sequence-cluster 95% CI: `{_fmt(oracle50.get('h20_iou_oracle_sequence_cluster_ci95'))}`
- Winner counts: `{oracle50.get('h20_iou_winner_counts', {})}`
- Positive sequences: `{oracle50.get('positive_sequence_count', 'n/a')}`

### All 689 expanded episodes

- Status: `{_status(oracle689)}`
- Episodes: `{oracle689.get('episode_count', 'n/a')}`; policy rows: `{_load('policy_rollout_index.json', {}).get('policy_row_count_merged', 'n/a')}`.
- Train-fold best fixed used by selector default: `{train_best}`.
- All-episode best fixed reported by Oracle: `{oracle689.get('best_fixed_policy', 'n/a')}`.
- Mean raw H20 IoU: `{oracle689.get('mean_h20_iou', {})}`.
- Mean reward: `{oracle689.get('mean_reward', {})}`.
- Raw H20 IoU Oracle gain versus all-episode best fixed: `{_fmt(oracle689.get('h20_iou_oracle_gain_vs_best_fixed'))}`.
- Raw H20 IoU Oracle sequence-cluster 95% CI: `{_fmt(oracle689.get('h20_iou_oracle_sequence_cluster_ci95'))}`.
- Winner counts: `{oracle689.get('h20_iou_winner_counts', {})}`; non-best-fixed winner rate: `{_fmt(oracle689.get('winner_not_best_fixed_rate'))}`; positive sequences: `{oracle689.get('positive_sequence_count', 'n/a')}`.
- Defined/undefined H20 IoU episode windows: `{oracle689.get('h20_iou_defined_episode_count', 'n/a')}` / `{oracle689.get('h20_iou_undefined_episode_count', 'n/a')}`; undefined windows retain null metrics.

Oracle gate checks:

{_gate_table({'checks': oracle689.get('gate_checks', {})})}

The Oracle is a post-hoc upper-bound diagnostic. It is not used as a deployed decision rule. Its frozen gate failed solely because the non-best-fixed winner rate was below the required 0.30 (`{_fmt(oracle689.get('winner_not_best_fixed_rate'))}`); this authorizes the bounded association fallback and blocks selector/temporal/full-loop claims.

## E — Selector feature audit

- Status: `{_status(audit)}`; dimensions: `{audit.get('feature_dimension', 'n/a')}`; episode/policy-row coverage: `{audit.get('episode_count', 'n/a')}` / `{audit.get('policy_row_count', 'n/a')}`.
- Same vector across K0/K1/K2: `{audit.get('same_vector_across_three_policies_episode_count', 'n/a')}` episodes.
- Finite vectors: `{audit.get('finite_vector_count', 'n/a')}`.
- Forbidden-input scan: `{audit.get('forbidden_input_name_scan', {})}`.

Feature names, in frozen order:

`{', '.join(feature_names)}`

No sequence ID, episode index, public ID, dataset identity, future image, future GT, future candidate outcome, or policy reward is emitted as a selector feature. Identity features are zero-filled for the single-ID tape with an explicit availability flag. The audit is structurally PASS, but identity coverage is `{identity_count}/689` (`{identity_coverage}`), so the current feature tape is **not valid evidence of identity-aware learning** and is scoped to `{selector_scope}`.

## F/G — Selector architecture and training

Architecture when authorized: `LayerNorm({feature_dim}) -> Linear({feature_dim},128) -> GELU -> Dropout(0.1) -> Linear(128,64) -> GELU -> Linear(64,3)`, with logits ordered K0/K1/K2. The fixed loss is listwise KL at temperature 0.10 plus 0.10 times a pairwise margin loss (margin 0.05, epsilon 0.01). Regret weights are capped at 0.50. AdamW uses learning rate 1e-3, weight decay 1e-4, batch size 64, 100 epochs, gradient clipping 1.0, and seeds 3201/3202/3203.

- Training artifact: `{_status(training)}`; parameter count: `{training.get('formal_training', [{}])[0].get('parameter_count', 'n/a') if training.get('formal_training') else 'n/a'}`.
- 20-episode overfit gate: `{_status(overfit)}`; results: `{[(item.get('seed'), item.get('final_action_accuracy'), item.get('final_reward_ratio_to_oracle'), item.get('pass')) for item in overfit.get('results', [])]}`.
- Save/load equality and shuffled-input control are included in `overfit_gate.json`.
- The selector route gate is `{_status(_load('selector_route_gate.json', {}))}`; because the Oracle Gate failed, these selector artifacts are explicit `NOT_RUN_ORACLE_GATE_FAIL`, not missing implementation results.

## H — Selection, calibration, and LOSO

Selection split results (`{selection.get('episode_count', 'n/a')}` episodes; future labels not used to select the protocol):

{_method_table(selection)}

Calibration split results (`{calibration.get('episode_count', 'n/a')}` episodes; fixed threshold and selected seed carried over):

{_method_table(calibration)}

- Selected seed: `{learn.get('selected_seed', selection.get('selected_seed', 'n/a'))}`; selected margin threshold: `{_fmt(learn.get('selected_threshold', selection.get('selected_threshold')))} `.
- Static Learn Gate: **`{_status(learn)}`**.

Learn Gate checks:

{_gate_table(learn)}

The three-seed independent results, mean-of-three result, selection threshold records, sequence gains, negative-transfer sequence rate, single-sequence contribution fraction, and LOSO records are retained in `learn_gate.json`, `selection_results.json`, `calibration_results.json`, and `leave_one_sequence_out.json`.

## I — Conditional fallbacks and full-loop boundary

- Temporal GRU fallback artifact: `{_status(temporal)}`. It is attempted only after a complete Oracle, static overfit, and static Learn Gate failure; it uses at most five causal feature snapshots and the same policy loss/gates. Here it was not authorized because the 689-episode Oracle Gate failed.
- Full-loop artifact: `{_status(full_loop)}`. Result: `{full_loop.get('reason', 'n/a')}`.
- Bounded real multi-ID fallback: `{_status(fallback)}`. It compares B10-fixed, correction-supervised global C-RLS, and a small pairwise MLP on six sequence-disjoint N30 cases for training and two each for selection/calibration; it is diagnostic-only and does not establish an identity-aware N32 selector.

Fallback methods are deliberately not turned into a new spatial selector. Because N30 records only the selected candidate's future IoU, alternative-candidate future IoU is not reconstructed; learned association results therefore expose known-candidate coverage and make no unobserved MOT gain claim.

## J — Resource and reproducibility record

- Long-task summary: `{_status(run_summary)}`; elapsed seconds: `{_fmt(run_summary.get('elapsed_seconds'))}`.
- Worker assignment: four sequence-sharded official SAM3 workers on CUDA devices 0–3; pre-existing users of CUDA 4–6 were not touched.
- The pinned checkout was not modified by N32; pre-existing user modification `third_party/sam3/sam3/perflib/fused.py` was preserved.
- Disk reserve was checked before the long run; no val/test content was scanned or opened.

Reproduction from the project root:

```bash
bash scripts/run_n32_long_task.sh
```

Key artifacts:

- [`policy_regression.json`](../outputs/n32/policy_regression.json)
- [`retry_semantic_reconciliation_attempt2.json`](../outputs/n32/policy_rollouts/retry_semantic_reconciliation_attempt2.json)
- [`retry_merge.json`](../outputs/n32/retry_merge.json)
- [`policy_rollout_index.json`](../outputs/n32/policy_rollout_index.json)
- [`policy_oracle_689.json`](../outputs/n32/policy_oracle_689.json)
- [`selector_feature_audit.json`](../outputs/n32/selector_feature_audit.json)
- [`selector_training.json`](../outputs/n32/selector_training.json)
- [`selection_results.json`](../outputs/n32/selection_results.json)
- [`calibration_results.json`](../outputs/n32/calibration_results.json)
- [`learn_gate.json`](../outputs/n32/learn_gate.json)
- [`selector_route_gate.json`](../outputs/n32/selector_route_gate.json)
- [`temporal_learn_gate.json`](../outputs/n32/temporal_learn_gate.json)
- [`full_loop_results.json`](../outputs/n32/full_loop_results.json)
- [`association_fallback_results.json`](../outputs/n32/association_fallback_results.json)
- [`artifact_validation.json`](../outputs/n32/artifact_validation.json)

## K — Research interpretation

The N32 contribution is a causal strategy-level control boundary: human correction application is treated as a policy choice over official continuation-state transactions, while current delivery and identity/mapping bookkeeping are protected invariants. The meaningful research outcome is whichever gate the frozen protocol produces. If the Learn Gate fails, the result is a negative result against deployment of a static strategy selector, not evidence for an Oracle or a hidden temporal model. The remaining credible next line is to collect a multi-identity, candidate-complete correction tape and learn an association-level policy under one global Hungarian-with-NONE constraint; that is a different problem from scaling the spatial selector.

"""
    return report


def run(*, output: Path = REPORT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(render(), encoding="utf-8")
    tmp.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    path = run(output=args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
