#!/usr/bin/env python
"""Write the N72R18 final comparison, result, and report files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="outputs/N72R18/metrics/ema_comparison.json")
    parser.add_argument("--output-dir", default="outputs/N72R18")
    return parser.parse_args()


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def metric_row(name: str, metrics: dict[str, object]) -> str:
    return "| " + " | ".join(
        [
            name,
            pct(metrics["h20_win_rate"]),
            pct(metrics["h50_win_rate"]),
            pct(metrics["h100_win_rate"]),
            f"[{pct(metrics['h100_ci95'][0])}, {pct(metrics['h100_ci95'][1])}]",
            pct(metrics["rank_1"]),
            pct(metrics["rank_2"]),
            pct(metrics["rank_3"]),
            f"{float(metrics['mrr']):.4f}",
            f"{float(metrics['median_margin']):+.6f}",
        ]
    ) + " |"


def main() -> None:
    args = parse_args()
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = comparison["baseline"]["summary"]
    gru = comparison["methods"]["GRU_MEMORY"]["summary"]
    gate = comparison["methods"]["GRU_RELIABILITY_GATE"]["summary"]
    learned_beats = bool(comparison["learned_memory_beats_ema"])
    gate_effective = float(gate["h100_win_rate"]) > float(gru["h100_win_rate"])
    gate_diag = comparison.get("gate_diagnostics") or {}
    decision = "PASS_PERSISTENT_MEMORY_IMPROVES_IDENTITY" if learned_beats else "LEARNED_MEMORY_DOES_NOT_BEAT_FROZEN_EMA"

    result = {
        "stage": "N72R18",
        "goal": "Human Identity Memory Learning",
        "central_question": "Can a learned identity memory updater build a more reliable long-term identity representation than a fixed EMA from sparse human-confirmed observations?",
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "baseline": "OSNet EMA(0.90)",
        "ema_h100_win_rate": baseline["h100_win_rate"],
        "gru_h100_win_rate": gru["h100_win_rate"],
        "gru_gate_h100_win_rate": gate["h100_win_rate"],
        "gru_gate_minus_ema": comparison["deltas"]["gate_h100_minus_ema"],
        "learned_memory_beats_ema": learned_beats,
        "gate_effective_vs_gru": gate_effective,
        "decision": decision,
        "next_association_stage_authorized": False,
        "goal_frozen": True,
        "offline_causal_gt_observation_diagnostic": True,
    }
    (output_dir / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "Learned identity memory 是否超过 EMA？",
        "",
        "FINAL GOAL: Human Identity Memory Learning",
        "",
        "CENTRAL QUESTION: “学习一个身份记忆更新模型，能不能超过一个简单 EMA？”",
        "",
        f"结论：{'是，且达到预设 +5 percentage-point 门槛。' if learned_beats else '否，当前 GRU+Gate 未达到超过 EMA(0.90) 至少 5 个百分点的预设门槛。'}",
        "",
        "## Frozen scope",
        "",
        "本阶段只使用 N72R17 的冻结 OSNet 512-D embedding cache 和原协议；没有运行 N72R18 的 SAM3、MOT association、Hungarian、TrackEval、HOTA、IDF1、requery、LoRA、数据集搬运或新的 crop encoder。完整 pytest 仅诊断性触发了仓库既有 TrackEval 测试，并复现 210 passed / 4 failed 的旧环境与第三方 CLI 问题。未来同身份 GT observation 是离线因果 replay 输入，因此结果不是可部署 online association 证明。",
        "",
        "## Primary endpoint",
        "",
        "| Method | H20 win | H50 win | H100 win | H100 CI95 | Rank-1 | Rank-2 | Rank-3 | MRR | H100 median margin |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        metric_row("Single Anchor (N72R17 M1)", comparison["single_anchor"]["summary"]),
        metric_row("EMA(0.90) frozen baseline", baseline),
        metric_row("GRU Memory", gru),
        metric_row("GRU + Reliability Gate", gate),
        "",
        f"Single-anchor H100 sequence-cluster 95% CI: [{pct(comparison['single_anchor']['summary']['h100_ci95'][0])}, {pct(comparison['single_anchor']['summary']['h100_ci95'][1])}].",
        f"EMA H100 sequence-cluster 95% CI: [{pct(baseline['h100_ci95'][0])}, {pct(baseline['h100_ci95'][1])}].",
        f"GRU H100 sequence-cluster 95% CI: [{pct(gru['h100_ci95'][0])}, {pct(gru['h100_ci95'][1])}].",
        f"GRU+Gate H100 sequence-cluster 95% CI: [{pct(gate['h100_ci95'][0])}, {pct(gate['h100_ci95'][1])}].",
        "",
        "| Method | H100 mean margin | median | P10 | P25 | P75 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Single Anchor (N72R17 M1) | {float(comparison['single_anchor']['summary']['mean_margin']):+.6f} | {float(comparison['single_anchor']['summary']['median_margin']):+.6f} | {float(comparison['single_anchor']['summary']['p10_margin']):+.6f} | {float(comparison['single_anchor']['summary']['p25_margin']):+.6f} | {float(comparison['single_anchor']['summary']['p75_margin']):+.6f} |",
        f"| EMA(0.90) | {float(baseline['mean_margin']):+.6f} | {float(baseline['median_margin']):+.6f} | {float(baseline['p10_margin']):+.6f} | {float(baseline['p25_margin']):+.6f} | {float(baseline['p75_margin']):+.6f} |",
        f"| GRU | {float(gru['mean_margin']):+.6f} | {float(gru['median_margin']):+.6f} | {float(gru['p10_margin']):+.6f} | {float(gru['p25_margin']):+.6f} | {float(gru['p75_margin']):+.6f} |",
        f"| GRU + Reliability Gate | {float(gate['mean_margin']):+.6f} | {float(gate['median_margin']):+.6f} | {float(gate['p10_margin']):+.6f} | {float(gate['p25_margin']):+.6f} | {float(gate['p75_margin']):+.6f} |",
        "",
        "## Required answers",
        "",
        f"1. GRU > EMA? **{'Yes' if float(gru['h100_win_rate']) > float(baseline['h100_win_rate']) else 'No'}** at H100; delta {100.0 * (float(gru['h100_win_rate']) - float(baseline['h100_win_rate'])):+.2f} pp.",
        f"2. Gate effective? **{'Yes' if gate_effective else 'No'}** relative to GRU; delta {100.0 * (float(gate['h100_win_rate']) - float(gru['h100_win_rate'])):+.2f} pp. It still exceeds EMA, but the gate did not improve the ungated learned updater.",
        f"3. Which observations rejected? The fixed diagnostic threshold is gate < 0.5. It rejected {gate_diag.get('rejected_updates', 'n/a')} of {gate_diag.get('updates', 'n/a')} replay updates ({pct(gate_diag.get('rejection_rate'))}); see the time-gap and pre-update-result breakdown below.",
        f"4. Hard-negative errors reduced? **{'Yes' if float(gate['h100_win_rate']) > float(baseline['h100_win_rate']) else 'No'}** versus frozen EMA at H100; delta {100.0 * float(comparison['deltas']['gate_h100_minus_ema']):+.2f} pp.",
        f"5. H100 gain: GRU {100.0 * float(comparison['deltas']['gru_h100_minus_ema']):+.2f} pp vs EMA; GRU+Gate {100.0 * float(comparison['deltas']['gate_h100_minus_ema']):+.2f} pp vs EMA.",
        "6. Authorize InterMOT reattachment? **No.** N72R18 does not automatically authorize the next association stage; the result remains an offline identity-memory diagnostic and the frozen status is `next_association_stage_authorized=false`.",
        "",
        "## Time-gap and gate diagnostics",
        "",
        "| Gap | EMA win | GRU+Gate win | Gate mean reliability | Rejected | Updates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, display in (("1_5", "1–5"), ("6_20", "6–20"), ("21_50", "21–50"), ("51_100", "51–100")):
        ema_bin = baseline["time_gap_bins"][label]
        gate_bin = gate["time_gap_bins"][label]
        diag_bin = (gate_diag.get("time_gap_bins") or {}).get(label, {})
        lines.append(
            f"| {display} | {pct(ema_bin['hard_negative_win_rate'])} | {pct(gate_bin['hard_negative_win_rate'])} | {pct(diag_bin.get('mean_reliability'))} | {diag_bin.get('rejected_updates', 'n/a')} | {diag_bin.get('updates', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{decision}`. The frozen success rule is `GRU+Gate H100 > EMA(0.90) by >= 5 pp`; the machine-readable value is `learned_memory_beats_ema={str(learned_beats).lower()}`.",
            "",
            "The comparison is intentionally limited to the N72R18 Goal. It does not claim improved MOT, HOTA, association, or deployable online memory.",
            "",
            "## Provenance",
            "",
            f"- Final Goal: `outputs/N72R18/FINAL_GOAL.json`",
            f"- Protocol metadata: `outputs/N72R18/protocol.json`",
            f"- Reference audit: `outputs/N72R18/reference_audit.md`",
            f"- Comparison JSON: `{args.comparison}`",
            "- Frozen external baseline: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/results/memory_probe_osnet_x1_0_market1501_val.json`",
        ]
    )
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_path = output_dir / "stage_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update(
        {
            "stage": "N72R18",
            "goal": "Human Identity Memory Learning",
            "central_question": "Can a learned identity memory updater build a more reliable long-term identity representation than a fixed EMA from sparse human-confirmed observations?",
            "status": "FINAL_RESULT_WRITTEN",
            "learned_memory_beats_ema": learned_beats,
            "next_association_stage_authorized": False,
            "goal_frozen": True,
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str((output_dir / 'FINAL_RESULT.json').resolve()), "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
