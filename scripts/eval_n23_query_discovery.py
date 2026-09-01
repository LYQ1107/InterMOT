#!/usr/bin/env python3
"""Evaluate N23 query-conditioned whole-frame discovery on calibration10.

Proposal generation and ranking never read GT.  GT is loaded after scoring to
measure target-inclusive bank recall, top-k ranking, explicit-NONE behavior,
and identity-safe recovery.  The generic-pool columns are copied from the
N17 episode manifest and are reported separately because they describe a
different candidate namespace.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3_intermot.recovery.n23_query_discovery import (  # noqa: E402
    NoneGate,
    PairRanker,
    gate_features,
    pair_geometry,
)


DT = Path("/path/to/dancetrack")
if not DT.exists():
    DT = Path("/path/to/dancetrack")


def iou_xyxy(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    return float((ranks[y > 0.5].mean() - ranks[y <= 0.5].mean()) / len(score) + 0.5)


def load_models(ranker_path: Path, gate_path: Path, device: torch.device):
    rpayload = torch.load(ranker_path, map_location="cpu", weights_only=False)
    ranker = PairRanker(**rpayload.get("arch", {})).to(device)
    ranker.load_state_dict(rpayload["state"])
    ranker.eval()
    gpayload = torch.load(gate_path, map_location="cpu", weights_only=False)
    gate = NoneGate(**gpayload.get("arch", {})).to(device)
    gate.load_state_dict(gpayload["state"])
    gate.eval()
    return ranker, gate, float(gpayload.get("threshold", 0.5))


def load_gt(seq: str):
    from sam3_intermot.identity_anchor.identity_benchmark import load_gt as load_benchmark_gt

    return load_benchmark_gt(DT / "train" / seq)


def score_row(
    row,
    cache_dir: Path,
    ranker: PairRanker,
    gate: NoneGate,
    device,
    rank_batch: int,
    score_mode: str,
):
    with np.load(cache_dir / row["cache_file"]) as item:
        q = torch.from_numpy(item["query"].astype(np.float32)).to(device)
        boxes = torch.from_numpy(item["boxes"].astype(np.float32)).to(device)
        emb = torch.from_numpy(item["embeddings"].astype(np.float32)).to(device)
    qbox = torch.tensor(json.loads(row["human_box"]), dtype=torch.float32, device=device)
    delta = float(row["delta"])
    raw = emb @ q
    geom = pair_geometry(qbox, boxes, delta)
    with torch.inference_mode():
        if score_mode == "raw":
            score_logits = raw
        else:
            logits_parts = []
            for start in range(0, len(boxes), rank_batch):
                n = min(rank_batch, len(boxes) - start)
                logits_parts.append(
                    ranker(
                        q.expand(n, -1),
                        emb[start : start + n],
                        geom[start : start + n],
                    )
                )
            score_logits = torch.cat(logits_parts)
        feats = gate_features(score_logits, raw, boxes, qbox, delta)
        gate_prob = float(torch.sigmoid(gate(feats)).item())
    return (
        boxes.cpu().numpy(),
        raw.cpu().numpy(),
        score_logits.cpu().numpy(),
        gate_prob,
    )


def top_hit(scores: np.ndarray, boxes: np.ndarray, target, k: int):
    order = np.argsort(scores)[::-1][:k]
    if target is None:
        return False, 0.0
    vals = [iou_xyxy(boxes[i], target) for i in order]
    return bool(vals and max(vals) >= 0.5), float(max(vals) if vals else 0.0)


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def aggregate(detail):
    def subset(rows, pred=lambda r: True):
        return [r for r in rows if pred(r)]

    def rate(rows, field):
        return mean_or_nan([float(r[field]) for r in rows])

    out = {"rows": len(detail)}
    for name, pred in {
        "all": lambda r: True,
        "target_present": lambda r: r["target_present"] == 1,
        "target_absent": lambda r: r["target_present"] == 0,
        "generic_miss": lambda r: r["target_present"] == 1 and r["generic_miss"] == 1,
    }.items():
        rows = subset(detail, pred)
        out[name] = {
            "n": len(rows),
            "bank_available_iou05": rate(rows, "bank_available"),
            "raw_top1": rate(rows, "raw_top1"),
            "raw_top3": rate(rows, "raw_top3"),
            "raw_top5": rate(rows, "raw_top5"),
            "raw_top10": rate(rows, "raw_top10"),
            "adapter_top1": rate(rows, "adapter_top1"),
            "adapter_top3": rate(rows, "adapter_top3"),
            "adapter_top5": rate(rows, "adapter_top5"),
            "adapter_top10": rate(rows, "adapter_top10"),
            "adapter_selected_iou": rate(rows, "adapter_selected_iou"),
            "gate_accept": rate(rows, "gate_accept"),
            "correct_recovery": rate(rows, "correct_recovery"),
            "wrong_recovery": rate(rows, "wrong_recovery"),
            "other_person_overlap": rate(rows, "other_person_overlap"),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/n23/window_cache/calibration"))
    parser.add_argument("--ranker", default=str(ROOT / "outputs/n23/models/query_ranker.pt"))
    parser.add_argument("--gate", default=str(ROOT / "outputs/n23/models/none_gate.pt"))
    parser.add_argument("--out-prefix", default=str(ROOT / "outputs/n23/n23_query_discovery_calibration"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank-batch", type=int, default=256)
    parser.add_argument("--score-mode", choices=["adapter", "raw"], default="adapter")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ranker, gate, threshold = load_models(Path(args.ranker), Path(args.gate), device)
    cache_dir = Path(args.cache_dir)
    index_paths = sorted(cache_dir.glob("index_shard*.csv"))
    if not index_paths:
        raise FileNotFoundError(f"no completed cache index under {cache_dir}")

    rows = []
    for path in index_paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda r: int(r["row_id"]))
    if args.max_rows:
        rows = rows[: args.max_rows]
    gt_cache = {}
    detail = []
    for n, row in enumerate(rows, 1):
        boxes, raw, adapter, gate_prob = score_row(
            row, cache_dir, ranker, gate, device, args.rank_batch, args.score_mode
        )
        present = int(row["target_present"]) == 1
        target = np.asarray(json.loads(row["target_box"]), dtype=float) if present else None
        all_gt = gt_cache.setdefault(row["sequence"], load_gt(row["sequence"]))
        gt_at_f = all_gt.get(int(row["f"]), [])
        other_boxes = [box for gid, box in gt_at_f if int(gid) != int(row["gid"])]
        best_bank_iou = max((iou_xyxy(b, target) for b in boxes), default=0.0) if present else 0.0
        raw_metrics = {}
        adapter_metrics = {}
        for k in (1, 3, 5, 10):
            raw_metrics[k], _ = top_hit(raw, boxes, target, k)
            adapter_metrics[k], _ = top_hit(adapter, boxes, target, k)
        selected_scores = raw if args.score_mode == "raw" else adapter
        adapter_order = np.argsort(selected_scores)[::-1]
        selected = boxes[adapter_order[0]]
        selected_iou = iou_xyxy(selected, target) if present else 0.0
        other_overlap = max((iou_xyxy(selected, b) for b in other_boxes), default=0.0)
        accepted = gate_prob >= threshold
        correct = int(accepted and present and selected_iou >= 0.5)
        wrong = int(accepted and (not present or selected_iou < 0.5))
        detail.append(
            {
                "row_id": int(row["row_id"]),
                "sequence": row["sequence"],
                "t": int(row["t"]),
                "f": int(row["f"]),
                "gid": int(row["gid"]),
                "delta": int(float(row["delta"])),
                "target_present": int(present),
                "generic_miss": int(row.get("generic_miss") or 0),
                "generic_candidate_present": int(row.get("generic_candidate_present") or 0),
                "bank_available": int(present and best_bank_iou >= 0.5),
                "bank_best_iou": float(best_bank_iou),
                "gate_prob": gate_prob,
                "gate_accept": int(accepted),
                "selected_iou": float(selected_iou),
                "other_person_overlap": int(other_overlap >= 0.5),
                "correct_recovery": correct,
                "wrong_recovery": wrong,
                "adapter_selected_iou": float(selected_iou),
                **{f"raw_top{k}": int(raw_metrics[k]) for k in (1, 3, 5, 10)},
                **{f"adapter_top{k}": int(adapter_metrics[k]) for k in (1, 3, 5, 10)},
            }
        )
        if n % 100 == 0 or n == len(rows):
            print(f"scored={n}/{len(rows)}", flush=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if detail:
        with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
            writer.writeheader()
            writer.writerows(detail)

    y = np.asarray([r["target_present"] for r in detail], dtype=np.float32)
    p = np.asarray([r["gate_prob"] for r in detail], dtype=np.float32)
    summary = {
        "method": f"N23 correction-driven query discovery ({args.score_mode})",
        "split": "calibration10",
        "cache_dir": str(cache_dir),
        "rows": len(detail),
        "threshold": threshold,
        "score_mode": args.score_mode,
        "gate_auc": auc_score(y, p),
        "aggregate": aggregate(detail),
        "by_delta": {
            str(d): aggregate([r for r in detail if r["delta"] == d])
            for d in sorted({r["delta"] for r in detail})
        },
        "controls": {
            "gfn_pool_columns": "generic_candidate_present/generic_miss are N17 GFN namespace labels; not treated as N23 proposals",
            "raw_clip": "frozen CLIP-ReID cosine over the same whole-frame window bank",
            "adapter": "trained PairRanker over frozen CLIP-ReID plus geometry",
            "proposed": f"{args.score_mode} top-1 with train30-only NONE gate at checkpoint threshold",
            "upper_bound": "bank_available: any generated window IoU >= 0.5 with target",
        },
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
