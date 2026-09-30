#!/usr/bin/env python
"""Compare learned N72R18 memory summaries with frozen N72R17 EMA(0.90)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru-summary", default="outputs/N72R18/metrics/gru_val_summary.json")
    parser.add_argument("--gate-summary", default="outputs/N72R18/metrics/gru_gate_val_summary.json")
    parser.add_argument(
        "--ema-summary",
        default="/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/results/memory_probe_osnet_x1_0_market1501_val.json",
    )
    parser.add_argument("--output", default="outputs/N72R18/metrics/ema_comparison.json")
    return parser.parse_args()


def learned_metrics(summary: dict[str, object]) -> dict[str, object]:
    h = summary["horizons"]
    h100 = h["H100"]
    return {
        "h20_win_rate": h["H20"]["hard_negative_win_rate"],
        "h50_win_rate": h["H50"]["hard_negative_win_rate"],
        "h100_win_rate": h100["hard_negative_win_rate"],
        "h100_ci95": [
            h100["hard_negative_win_rate_sequence_cluster_ci95"]["lower"],
            h100["hard_negative_win_rate_sequence_cluster_ci95"]["upper"],
        ],
        "rank_1": h100["rank_1_accuracy"],
        "rank_2": h100["rank_2_accuracy"],
        "rank_3": h100["rank_3_accuracy"],
        "mrr": h100["mrr"],
        "mean_margin": h100["mean_margin"],
        "median_margin": h100["median_margin"],
        "p10_margin": h100["p10_margin"],
        "p25_margin": h100["p25_margin"],
        "p75_margin": h100["p75_margin"],
        "time_gap_bins": summary["time_gap_bins"],
    }


def main() -> None:
    args = parse_args()
    gru = json.loads(Path(args.gru_summary).read_text(encoding="utf-8"))
    gate = json.loads(Path(args.gate_summary).read_text(encoding="utf-8"))
    ema_document = json.loads(Path(args.ema_summary).read_text(encoding="utf-8"))
    ema = learned_metrics(ema_document["variants"]["EMA_0.90"])
    single_anchor = learned_metrics(ema_document["variants"]["M1"])
    gru_metrics = learned_metrics(gru)
    gate_metrics = learned_metrics(gate)
    ema_h100 = float(ema["h100_win_rate"])
    gate_h100 = float(gate_metrics["h100_win_rate"])
    gru_h100 = float(gru_metrics["h100_win_rate"])
    comparison = {
        "stage": "N72R18",
        "goal": "Human Identity Memory Learning",
        "central_question": "Can a learned identity memory updater build a more reliable long-term identity representation than a fixed EMA from sparse human-confirmed observations?",
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "single_anchor": {"name": "M1", "summary": single_anchor, "source": str(Path(args.ema_summary).resolve())},
        "baseline": {"name": "EMA_0.90", "summary": ema, "source": str(Path(args.ema_summary).resolve())},
        "methods": {
            "GRU_MEMORY": {"summary": gru_metrics, "source": str(Path(args.gru_summary).resolve())},
            "GRU_RELIABILITY_GATE": {"summary": gate_metrics, "source": str(Path(args.gate_summary).resolve())},
        },
        "gate_diagnostics": gate.get("gate_diagnostics"),
        "deltas": {
            "gru_h100_minus_ema": gru_h100 - ema_h100,
            "gate_h100_minus_ema": gate_h100 - ema_h100,
            "gate_h100_minus_gru": gate_h100 - gru_h100,
        },
        "success_threshold": {
            "baseline": ema_h100,
            "required_gate_h100": ema_h100 + 0.05,
            "definition": "GRU+ReliabilityGate H100 must exceed EMA(0.90) by at least 5 percentage points",
        },
        "learned_memory_beats_ema": bool(gate_h100 > ema_h100 and gate_h100 - ema_h100 >= 0.05),
        "goal_frozen": True,
        "next_association_stage_authorized": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "learned_memory_beats_ema": comparison["learned_memory_beats_ema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
