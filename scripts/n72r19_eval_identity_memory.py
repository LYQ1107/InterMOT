#!/usr/bin/env python
"""Evaluate N72R19 EMA, clean GRU, and robust GRU+Reliability on noisy episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sam3_intermot.identity_memory.dataset import FrozenIdentityEpisodeDataset
from sam3_intermot.identity_memory.noise import NoiseSpec
from sam3_intermot.identity_memory.robust_eval import (
    METHOD_EMA,
    METHOD_GRU,
    METHOD_ROBUST,
    evaluate_condition,
)
from sam3_intermot.identity_memory.updater import build_updater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_val.json",
    )
    parser.add_argument(
        "--store",
        default="/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_val",
    )
    parser.add_argument("--n72r18-gru-checkpoint", default="outputs/N72R18/checkpoints/identity_memory_gru.pt")
    parser.add_argument("--robust-checkpoint", default="outputs/N72R19/checkpoints/identity_memory_robust_gate.pt")
    parser.add_argument("--output-dir", default="outputs/N72R19")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7219)
    return parser.parse_args()


def _load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    variant = str(checkpoint["variant"])
    model = build_updater(
        variant,
        feature_dim=int(config.get("feature_dimension", 512)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rate_slug(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def _cell_summary(summary: dict[str, object]) -> dict[str, object]:
    h100 = summary["horizons"]["H100"]
    ci = h100["hard_negative_win_rate_sequence_cluster_ci95"]
    return {
        "samples": summary["sample_count"],
        "h20_win_rate": summary["horizons"]["H20"]["hard_negative_win_rate"],
        "h50_win_rate": summary["horizons"]["H50"]["hard_negative_win_rate"],
        "h100_win_rate": h100["hard_negative_win_rate"],
        "h100_ci95": [ci["lower"], ci["upper"]],
        "rank_1": h100["rank_1_accuracy"],
        "rank_2": h100["rank_2_accuracy"],
        "rank_3": h100["rank_3_accuracy"],
        "mrr": h100["mrr"],
        "mean_margin": h100["mean_margin"],
        "median_margin": h100["median_margin"],
        "p10_margin": h100["p10_margin"],
        "p25_margin": h100["p25_margin"],
        "p75_margin": h100["p75_margin"],
        "update_count": summary["update_count"],
        "noise_applied_count": summary["noise_applied_count"],
        "missing_count": summary["missing_count"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = FrozenIdentityEpisodeDataset(args.protocol, args.store, split="val")
    robust_model, robust_checkpoint = _load_model(args.robust_checkpoint, device)
    gru_model, gru_checkpoint = _load_model(args.n72r18_gru_checkpoint, device)

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    noise_dir = output_dir / "noise_results"
    drift_dir = output_dir / "memory_drift"
    for path in (metrics_dir, noise_dir, drift_dir):
        path.mkdir(parents=True, exist_ok=True)

    models = {
        METHOD_EMA: None,
        METHOD_GRU: gru_model,
        METHOD_ROBUST: robust_model,
    }
    conditions: list[tuple[str, float]] = [("clean", 0.0)]
    for kind in ("wrong_identity_injection", "missing_observation", "hard_negative_replacement"):
        conditions.extend((kind, rate) for rate in (0.1, 0.2, 0.3, 0.5))

    all_summaries: dict[str, dict[str, dict[str, object]]] = {}
    all_drift: dict[str, dict[str, dict[str, object]]] = {}
    for kind, rate in conditions:
        noise = NoiseSpec(kind, rate, seed=args.seed)
        condition_slug = f"{kind}_{_rate_slug(rate)}"
        print(json.dumps({"condition": condition_slug, "status": "started"}), flush=True)
        for method, model in models.items():
            records, updates, summary, drift = evaluate_condition(
                dataset,
                method,
                noise,
                device=device,
                model=model,
                batch_size=args.batch_size,
                bootstrap_reps=args.bootstrap_reps,
                bootstrap_seed=args.seed,
            )
            method_slug = method.lower()
            summary_path = metrics_dir / f"{method_slug}__{condition_slug}.json"
            record_path = noise_dir / f"{method_slug}__{condition_slug}.jsonl"
            update_path = noise_dir / f"{method_slug}__{condition_slug}__updates.jsonl"
            drift_path = drift_dir / f"{method_slug}__{condition_slug}.json"
            summary["protocol"] = str(Path(args.protocol).resolve())
            summary["embedding_store"] = str(Path(args.store).resolve())
            summary["checkpoint"] = (
                None
                if method == METHOD_EMA
                else str(
                    Path(args.n72r18_gru_checkpoint if method == METHOD_GRU else args.robust_checkpoint).resolve()
                )
            )
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write_jsonl(record_path, records)
            _write_jsonl(update_path, updates)
            drift_path.write_text(json.dumps(drift, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            all_summaries.setdefault(method, {})[condition_slug] = {
                "summary_path": str(summary_path.resolve()),
                "summary": _cell_summary(summary),
            }
            all_drift.setdefault(method, {})[condition_slug] = {
                "drift_path": str(drift_path.resolve()),
                "drift": drift,
            }
            print(
                json.dumps(
                    {
                        "condition": condition_slug,
                        "method": method,
                        "h100_win_rate": summary["horizons"]["H100"]["hard_negative_win_rate"],
                        "records": len(records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    matrix = {
        "stage": "N72R19",
        "goal": "Robust Human Identity Memory Learning",
        "goal_family": "Human Identity Memory Learning",
        "goal_reference": "outputs/N72R19/FINAL_GOAL.json",
        "central_question": "Can learned identity memory outperform EMA under noisy observation conditions?",
        "primary_endpoint": "H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "noise_seed": args.seed,
        "methods": all_summaries,
        "memory_drift": all_drift,
        "training_checkpoints": {
            "n72r18_gru": str(Path(args.n72r18_gru_checkpoint).resolve()),
            "n72r19_robust": str(Path(args.robust_checkpoint).resolve()),
            "n72r18_gru_stage": gru_checkpoint.get("stage"),
            "n72r19_robust_stage": robust_checkpoint.get("stage"),
        },
        "sam3_required": False,
        "mot_required": False,
        "goal_frozen": True,
        "next_association_stage_authorized": False,
    }
    (metrics_dir / "noise_matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"matrix": str((metrics_dir / 'noise_matrix.json').resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
