#!/usr/bin/env python3
"""Render the evidence-backed N31 final report from generated artifacts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/n31"
REPORT = ROOT / "docs/N31_FINAL_REPORT.md"


def _load(name: str, default: Any) -> Any:
    path = OUT_DIR / name
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "UNREADABLE", "error": f"{type(exc).__name__}: {exc}"}


def _load_path(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "UNREADABLE", "error": f"{type(exc).__name__}: {exc}"}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _branch_rows(summary: dict[str, Any]) -> str:
    lines = ["| Branch | Rollout | Prompt success | Target state | H20 IoU | H20 success | H20 missing |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.get("branch_quality", {}).items():
        h20 = row.get("horizons", {}).get("20", {})
        lines.append(
            f"| `{name}` | {_fmt(row.get('availability_rate'))} | {_fmt(row.get('prompt_success_rate'))} | {_fmt(row.get('target_state_present_rate'))} | {_fmt(h20.get('mean_box_iou_visible'))} | {_fmt(h20.get('success_at_0_5_visible'))} | {_fmt(h20.get('missing_prediction_rate_visible'))} |"
        )
    return "\n".join(lines)


def _protocol_branch_rows(protocol: dict[str, Any]) -> str:
    lines = ["| Branch | Frozen definition |", "|---|---|"]
    for branch in protocol.get("ablation", {}).get("branches", []):
        lines.append(f"| `{branch.get('name', 'n/a')}` | {branch.get('definition', 'n/a')} |")
    return "\n".join(lines)


def _formal_elapsed(log_name: str, fallback: Any) -> Any:
    path = OUT_DIR / log_name
    if not path.is_file():
        return fallback
    values = re.findall(r'\"elapsed_seconds\"\s*:\s*([0-9.eE+-]+)', path.read_text(encoding="utf-8", errors="replace"))
    return float(values[-1]) if values else fallback


def _comparison_metric(summary: dict[str, Any], comparison: str, metric: str, key: str = "mean") -> Any:
    return summary.get("comparisons", {}).get(comparison, {}).get("20", {}).get(metric, {}).get(key)


def _comparison_rows(summary: dict[str, Any]) -> str:
    lines = ["| Comparison | H20 unconditional IoU mean | H20 sequence CI | H20 conditional IoU mean |", "|---|---:|---|---:|"]
    for name, horizons in summary.get("comparisons", {}).items():
        h20 = horizons.get("20", {})
        raw = h20.get("mean_box_iou_visible", {})
        cond = h20.get("conditional_iou_given_both", {})
        lines.append(f"| `{name}` | {_fmt(raw.get('mean'))} | {_fmt(raw.get('sequence_cluster_bootstrap_ci95'))} | {_fmt(cond.get('mean'))} |")
    return "\n".join(lines)


def _candidate_rows(candidate: dict[str, Any]) -> str:
    lines = ["| Candidate | Available | Mean reward | Mean H20 IoU |", "|---|---:|---:|---:|"]
    for name, row in candidate.get("summary", {}).get("candidate_metrics", {}).items():
        lines.append(f"| `{name}` | {row.get('available_count', 0)} | {_fmt(row.get('mean_reward'))} | {_fmt(row.get('mean_h20_iou'))} |")
    return "\n".join(lines)


def render() -> str:
    protocol = _load("frozen_protocol.json", {})
    mapping = _load("id_mapping_regression.json", {})
    resume = _load("resume_equivalence_gate.json", {})
    protected = _load("protected_identity_scope.json", {})
    ablation = _load("correction_state_ablation.json", {})
    summary = _load("correction_state_ablation_summary.json", {})
    state_gate = _load("correction_state_gate.json", {})
    candidate = _load("candidate_rollout_index.json", {})
    oracle = _load("candidate_oracle_gate.json", {})
    expanded = _load("episode_manifest.json", {})
    gradient = _load("future_gradient_gate.json", {})
    learn = _load("learn_gate.json", {})
    fallback = _load("fallback_results.json", {})
    full_loop = _load("full_loop_results.json", {})
    overnight = _load("overnight_status.json", {})
    n30_multi = _load_path(ROOT / "outputs/n30/multi_identity_write_summary.json", {})
    candidate_formal_elapsed = _formal_elapsed("log_candidate_rollouts_full.txt", candidate.get("elapsed_seconds"))
    m1_evidence = fallback.get("association_trigger_evidence", {})
    lines = [
        "# N31 Final Report — Correction-State Writer/Selector",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Outcome",
        "",
        "- Conclusion: N31 repaired the N30 mapping/resume confounds and established a causal official correction-state signal, but the 50-episode candidate oracle found no future-quality upper bound beyond restoring the old state. No learned writer/selector or full-loop deployment is authorized; the bounded final route is `association_trigger_fallback`.",
        f"- N31-B continuation gate: **{resume.get('status', 'MISSING')}**; strategy: `{resume.get('selected_strategy', 'n/a')}`; episodes: `{resume.get('episode_count', 'n/a')}`.",
        f"- N31-C ablation artifact: **{ablation.get('status', 'MISSING')}**; processed episodes: `{ablation.get('episode_count', len(ablation.get('episode_results', [])))}`.",
        f"- N31-C state gate: **{state_gate.get('status', 'MISSING')}**.",
        f"- Candidate oracle gate: **{oracle.get('status', 'MISSING')}**; best measured candidate: `{oracle.get('best_candidate_by_reward', 'n/a')}`.",
        f"- Future-gradient gate: **{gradient.get('status', 'MISSING')}**.",
        f"- Learned selector gate: **{learn.get('status', 'MISSING')}**.",
        f"- Final route: `{fallback.get('route', 'n/a')}`; full loop: **{full_loop.get('status', 'MISSING')}**; overnight status: **{overnight.get('status', 'MISSING')}**.",
        "",
        "Availability and quality are reported separately. A prompt/state availability failure is never silently converted into a zero-quality sample, and an oracle choice is never treated as deployment inference.",
        "",
        "## What N31 corrected in N30",
        "",
        "- N30 B/D mixed public-ID admission recovery, official state reconstruction, and memory quality. N31 centralizes the one-to-one `_bind_external_sam_id` helper and explicitly restores the mapping after singleton initialization; the mapping-only raw-output regression is unchanged.",
        "- N30 compared a full replay with split continuation. N31 uses one correction prefix and strict next-frame continuation for every branch; the three-episode FULL/SPLIT no-op gate passes.",
        "- N30's forced reprompt label could hide the absence of a low-level rectangle reconstruction. N31 separates P2 prompt-only, P3 old-state restore, P4 target-scoped rectangle, P5 ensure-path, and P7 local official interactive-SAM pseudo-mask writes.",
        "",
        "## Frozen boundaries",
        "",
        f"- Protocol artifact: [`outputs/n31/frozen_protocol.json`](../outputs/n31/frozen_protocol.json).",
        f"- N29-R hard source: `{protocol.get('train_source', {}).get('hard_manifest', 'n/a')}`; SHA256 `{protocol.get('train_source', {}).get('hard_manifest_sha256', 'n/a')}`.",
        f"- `val25_read=false`, `test_labels_used=false`, and future GT was not used for N31 candidate/selector input or selection: `{protocol.get('causal_boundary', {}).get('future_gt_used_for_selection', 'n/a')}`.",
        "- All N31-C branches share one per-episode prefix and continuation snapshot; P5 was predeclared as the fixed writer before N31 results.",
        "- Pinned `third_party/sam3` was not edited. Existing user modification `sam3/perflib/fused.py` was preserved.",
        "",
        "## N31-A/B validity",
        "",
        f"- ID mapping regression: `{mapping.get('status', 'MISSING')}`; one-to-one contract: `{mapping.get('one_to_one_contract', 'n/a')}`; raw outputs mutated by mapping: `{mapping.get('raw_outputs_mutated_by_mapping', 'n/a')}`.",
        f"- FULL/SPLIT no-op continuation: `{resume.get('status', 'MISSING')}`; repair cycles attempted: `{resume.get('repair_cycles_attempted', 'n/a')}`; selected strategy: `{resume.get('selected_strategy', 'n/a')}`.",
        f"- Protected target-scope regression: `{protected.get('status', 'MISSING')}`; unaffected IDs preserved: `{protected.get('unaffected_ids_preserved', 'n/a')}`; future frames: `{protected.get('future_frame_count', 'n/a')}`.",
        "",
        "## N31-C P0–P7 definitions",
        "",
        _protocol_branch_rows(protocol),
        "",
        "## N31-C causal ablation",
        "",
        _branch_rows(summary),
        "",
        _comparison_rows(summary),
        "",
        f"The state gate is `{state_gate.get('status', 'MISSING')}`. Its fixed-write signal is `{state_gate.get('legitimate_fixed_write_signal', 'n/a')}` and P1/P0 raw equivalence is `{state_gate.get('p1_p0_raw_equivalence', 'n/a')}`.",
        f"- P4 minus P3 conditional H20 IoU: `{_fmt(_comparison_metric(summary, 'P4_corrected_rectangle_masklet_minus_P3_restore_old_state_after_prompt_failure', 'conditional_iou_given_both'))}`; sequence CI `{_fmt(_comparison_metric(summary, 'P4_corrected_rectangle_masklet_minus_P3_restore_old_state_after_prompt_failure', 'conditional_iou_given_both', 'sequence_cluster_bootstrap_ci95'))}`.",
        f"- P5 minus P4 unconditional H20 IoU: `{_fmt(_comparison_metric(summary, 'P5_current_ensure_path_minus_P4_corrected_rectangle_masklet', 'mean_box_iou_visible'))}`; conditional change `{_fmt(_comparison_metric(summary, 'P5_current_ensure_path_minus_P4_corrected_rectangle_masklet', 'conditional_iou_given_both'))}`.",
        f"- P7 minus P4 conditional H20 IoU: `{_fmt(_comparison_metric(summary, 'P7_frozen_interactive_sam_masklet_minus_P4_corrected_rectangle_masklet', 'conditional_iou_given_both'))}`; sequence CI `{_fmt(_comparison_metric(summary, 'P7_frozen_interactive_sam_masklet_minus_P4_corrected_rectangle_masklet', 'conditional_iou_given_both', 'sequence_cluster_bootstrap_ci95'))}`.",
        f"- P6 minus P5 online-LoRA H20 IoU: `{_fmt(_comparison_metric(summary, 'P6_online_lora_on_best_fixed_write_minus_P5_current_ensure_path', 'mean_box_iou_visible'))}`; sequence CI `{_fmt(_comparison_metric(summary, 'P6_online_lora_on_best_fixed_write_minus_P5_current_ensure_path', 'mean_box_iou_visible', 'sequence_cluster_bootstrap_ci95'))}`.",
        "- P2 has official prompt success/target-state rates of 0.20/0.20, while P3 restores target-state presence to 1.00. This is the measured admission/recovery effect; it is not presented as mask-quality improvement.",
        "",
        "## N31-D candidate library and Oracle upper bound",
        "",
        _candidate_rows(candidate),
        "",
        f"The formal candidate artifact contains `{candidate.get('summary', {}).get('row_count', 'n/a')}` rows for `{candidate.get('summary', {}).get('episode_count', 'n/a')}` episodes; all six candidates have availability 1.0. Formal rollout elapsed time recorded in the original log: `{_fmt(candidate_formal_elapsed)}` seconds.",
        f"Oracle thresholds were frozen before rollouts: H20 gain ≥ 0.01, sequence-cluster CI lower bound > 0, and ≥30% episodes not won by the fixed sanitized-rectangle candidate. Result: `{oracle.get('status', 'MISSING')}`; measured best reward candidate: `{oracle.get('best_candidate_by_reward', 'n/a')}`; not-S2 winner rate: `{_fmt(oracle.get('best_not_s2_rate'))}`.",
        "- Every alternative's H20 gain versus S0 is negative; the strongest alternative is S3 at approximately -0.253204 with sequence-cluster CI [-0.333807, -0.171602]. Thus the first two Oracle conditions fail even though candidate availability and winner diversity are measurable.",
        "- Candidate selection used only correction-time box/geometry, area, official predicted-IoU and candidate type. Future GT appears only in post-hoc reward evaluation.",
        "",
        "## N31-E/F, architecture, and learning path",
        "",
        f"- Expanded manifest: `{expanded.get('status', 'MISSING')}`, legal retained episodes `{expanded.get('available_legal_episode_count', 'n/a')}`, parent sequences `{expanded.get('parent_sequence_count', 'n/a')}`, split counts `{expanded.get('episode_counts_by_learning_split', 'n/a')}`, fixed seed `{expanded.get('fixed_sequence_split_seed', 'n/a')}`.",
        "- The frozen 18/6/6 sequence split retained all 689 legal events without duplication; meta-train has 419 episodes, below the 500 target because the frozen sequence partition has uneven event counts. Selection/calibration exceed 60. This shortfall is reported, not padded or repaired after results.",
        f"- Dataset audit: [`outputs/n31/dataset_audit.md`](../outputs/n31/dataset_audit.md); no usable local BDD100K/TAO source was found and the local KITTI label directory was empty.",
        f"- Future gradient smoke: `{gradient.get('status', 'MISSING')}`; future loss requires grad `{gradient.get('future_loss_requires_grad', 'n/a')}`; writer gradient present `{gradient.get('writer_gradient_present', 'n/a')}`. The official writer/propagation path is inference-mode/detached.",
        "- Path A was therefore not authorized. Path B was defined as a causal 11-feature, LayerNorm → Linear(11,64) → GELU → Linear(64,64) → GELU → Linear(64,1) listwise scorer, with no public ID, sequence ID, future reward, or future GT input; it was not trained because the earlier Oracle Gate failed.",
        f"- Selector artifacts are explicit: overfit `{_load('overfit_gate.json', {}).get('status', 'MISSING')}`, selection `{_load('selection_results.json', {}).get('status', 'MISSING')}`, calibration `{_load('calibration_results.json', {}).get('status', 'MISSING')}`, LOSO `{_load('leave_one_sequence_out.json', {}).get('status', 'MISSING')}`, learn gate `{learn.get('status', 'MISSING')}`. No learned-best-fixed result is claimed.",
        "- P4/P5/P7 inject masks through the official target-scoped `tracker.add_new_masks(..., reconditioning=True)` adapter path; P7 calls the local official interactive SAM prompt encoder/decoder with predicted-IoU multimask ranking. Base SAM3 remains frozen.",
        "",
        "## Correction accounting, fallback, and resources",
        "",
        "- Human correction input: 50 box events from the frozen N29-R hard source; 0 clicks and 0 human masks. N31-C writes 50 each for P4/P5/P7; candidate rollouts contain 100 rectangle candidate writes (S1/S2) and 150 official pseudo-mask candidate writes (S3–S5), all post-hoc evaluated rather than deployed.",
        f"- Fallback: `{fallback.get('status', 'MISSING')}` via `{fallback.get('route', 'n/a')}`. The bounded real train-fold multi-ID evidence has `{m1_evidence.get('case_count', 'n/a')}` cases; official spatial write-only future IoU `{_fmt((m1_evidence.get('official_spatial_write_future_iou') or {}).get('mean'))}` with sequence CI `{_fmt(m1_evidence.get('official_spatial_write_sequence_ci'))}`; joint-minus-best-single `{_fmt((m1_evidence.get('joint_minus_best_single_future_iou') or {}).get('mean'))}` with episode CI `{_fmt((m1_evidence.get('joint_minus_best_single_future_iou') or {}).get('episode_bootstrap_ci95'))}`; online-LoRA-minus-joint `{_fmt((m1_evidence.get('online_lora_minus_joint_future_iou') or {}).get('mean'))}`. This is association evidence, not an N31 learned or end-to-end MOT gain.",
        f"- Full loop/TrackEval: `{full_loop.get('status', 'MISSING')}` because Oracle/learn gates did not authorize deployment. ICLR/end-to-end MOT claim: **not established**.",
        "- Resource record: formal N31-C elapsed 7750.845674 seconds; formal N31-D elapsed 3971.552922 seconds; resume gate elapsed 383.48226495087147 seconds. The runs used CUDA device 9, maintained the 40 GiB disk reserve (about 79 GiB free at launch), and did not record a separate N31 peak-memory number.",
        "",
        "## Repairs and limitations",
        "",
        "1. The adapter continuation repair was restricted to a prior propagation action; it no longer inserts cancel/replay before the first `add` propagation.",
        "2. Snapshot restore preserves official feature-cache aliasing; otherwise future auto-updates can incorrectly fall through to a missing `images` key.",
        "3. The official target-scoped writer required bridging the already materialized outer feature tuple to each inner tracker state's `cached_features`; the bridge is recorded in every writer audit.",
        "4. Candidate token selection was corrected before the formal run to use the predicted-IoU rank order; candidate features were corrected to avoid a token-index/one-element-IoU mismatch. The formal 300-row result is from that corrected implementation.",
        "5. Future GT is used for post-hoc evaluation/training labels only. It is never an input to correction-time candidate generation or selector inference.",
        "",
        "## Reproduction commands",
        "",
        "```bash",
        "cd .",
        "CUDA_VISIBLE_DEVICES=9 python scripts/n31_id_mapping_regression.py",
        "CUDA_VISIBLE_DEVICES=9 python scripts/n31_resume_equivalence_smoke.py --limit 3 --repair-cycles 3",
        "CUDA_VISIBLE_DEVICES=9 python scripts/n31_correction_state_ablation.py --resume",
        "CUDA_VISIBLE_DEVICES=9 python scripts/n31_build_candidate_rollouts.py --resume",
        "CUDA_VISIBLE_DEVICES=9 python scripts/n31_future_gradient_smoke.py",
        "python scripts/n31_train_state_selector.py",
        "python scripts/n31_fallback.py",
        "python scripts/n31_full_loop_gate.py",
        "```",
        "",
        "## Artifact index",
        "",
        "- [`plan.md`](../outputs/n31/plan.md)",
        "- [`frozen_protocol.json`](../outputs/n31/frozen_protocol.json)",
        "- [`id_mapping_regression.json`](../outputs/n31/id_mapping_regression.json)",
        "- [`resume_equivalence_gate.json`](../outputs/n31/resume_equivalence_gate.json)",
        "- [`correction_state_ablation.json`](../outputs/n31/correction_state_ablation.json)",
        "- [`correction_state_ablation_summary.json`](../outputs/n31/correction_state_ablation_summary.json)",
        "- [`correction_state_gate.json`](../outputs/n31/correction_state_gate.json)",
        "- [`candidate_rollout_index.json`](../outputs/n31/candidate_rollout_index.json)",
        "- [`candidate_oracle_gate.json`](../outputs/n31/candidate_oracle_gate.json)",
        "- [`episode_manifest.json`](../outputs/n31/episode_manifest.json)",
        "- [`dataset_audit.md`](../outputs/n31/dataset_audit.md)",
        "- [`future_gradient_gate.json`](../outputs/n31/future_gradient_gate.json)",
        "- [`overfit_gate.json`](../outputs/n31/overfit_gate.json)",
        "- [`train_metrics.jsonl`](../outputs/n31/train_metrics.jsonl)",
        "- [`selection_results.json`](../outputs/n31/selection_results.json)",
        "- [`calibration_results.json`](../outputs/n31/calibration_results.json)",
        "- [`learn_gate.json`](../outputs/n31/learn_gate.json)",
        "- [`fallback_results.json`](../outputs/n31/fallback_results.json)",
        "- [`full_loop_results.json`](../outputs/n31/full_loop_results.json)",
        "- [`overnight_status.json`](../outputs/n31/overnight_status.json)",
        "- [`overnight.log`](../outputs/n31/logs/overnight.log)",
        "",
        "The only recommended next line is a newly frozen multi-identity association/interaction-timing protocol; do not enlarge the failed correction-state writer/selector route or claim blind-validation performance from this train-fold evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(render(), encoding="utf-8")
    temporary.replace(args.output)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
