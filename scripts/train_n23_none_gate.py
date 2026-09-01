#!/usr/bin/env python3
"""Train the N23 explicit-NONE episode gate from cached train30 proposals.

The cache contains only frozen visual features and geometry.  Candidate boxes
were generated without GT and without the GFN/SAM3 proposal list.  GT is used
here only as the training label for whether the target is visible at the
future frame; calibration sequences remain untouched until evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3_intermot.recovery.n23_query_discovery import (  # noqa: E402
    NoneGate,
    PairRanker,
    gate_features,
    pair_geometry,
)


def load_ranker(checkpoint: Path, device: torch.device) -> PairRanker:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PairRanker(**payload.get("arch", {})).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model


def shard_indices(cache_dir: Path):
    return sorted(cache_dir.glob("index_shard*.csv"))


def build_gate_table(
    cache_dir: Path,
    ranker: PairRanker,
    device: torch.device,
    rank_batch: int,
    score_mode: str,
):
    index_paths = shard_indices(cache_dir)
    if not index_paths:
        raise FileNotFoundError(f"no completed cache index under {cache_dir}")
    records = []
    with torch.inference_mode():
        for index_path in index_paths:
            with index_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    cache_path = cache_dir / row["cache_file"]
                    with np.load(cache_path) as item:
                        q = torch.from_numpy(item["query"].astype(np.float32)).to(device)
                        boxes = torch.from_numpy(item["boxes"].astype(np.float32)).to(device)
                        emb = torch.from_numpy(item["embeddings"].astype(np.float32)).to(device)
                        qbox = torch.tensor(
                            json.loads(row["human_box"]), dtype=torch.float32, device=device
                        )
                        delta = float(row["delta"])
                        raw = emb @ q
                        geom = pair_geometry(qbox, boxes, delta)
                        if score_mode == "raw":
                            score_logits = raw
                        else:
                            logits_parts = []
                            for start in range(0, len(boxes), rank_batch):
                                qpart = q.expand(min(rank_batch, len(boxes) - start), -1)
                                logits_parts.append(
                                    ranker(
                                        qpart,
                                        emb[start : start + rank_batch],
                                        geom[start : start + rank_batch],
                                    )
                                )
                            score_logits = torch.cat(logits_parts)
                        feats = gate_features(score_logits, raw, boxes, qbox, delta)
                    records.append(
                        {
                            "x": feats.cpu().numpy().astype(np.float32),
                            "y": int(row["target_present"]),
                            "sequence": row["sequence"],
                            "row_id": int(row["row_id"]),
                        }
                    )
    records.sort(key=lambda r: r["row_id"])
    if not records:
        raise RuntimeError("cache indexes contained no rows")
    x = np.stack([r["x"] for r in records]).astype(np.float32)
    y = np.asarray([r["y"] for r in records], dtype=np.float32)
    seq = np.asarray([r["sequence"] for r in records])
    row_ids = np.asarray([r["row_id"] for r in records], dtype=np.int64)
    return x, y, seq, row_ids


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    return float((ranks[y > 0.5].mean() - ranks[y <= 0.5].mean()) / len(score) + 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/n23/window_cache/train"))
    parser.add_argument("--ranker", default=str(ROOT / "outputs/n23/models/query_ranker.pt"))
    parser.add_argument("--out", default=str(ROOT / "outputs/n23/models/none_gate.pt"))
    parser.add_argument("--feature-out", default=str(ROOT / "outputs/n23/none_gate_features.npz"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--rank-batch", type=int, default=256)
    parser.add_argument("--score-mode", choices=["adapter", "raw"], default="adapter")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ranker = load_ranker(Path(args.ranker), device)
    x, y, seq, row_ids = build_gate_table(
        Path(args.cache_dir), ranker, device, args.rank_batch, args.score_mode
    )

    # Sequence-disjoint validation prevents nearly identical adjacent frames
    # from making the gate look better than it is.
    sequences = sorted(set(seq.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(sequences)
    cut = max(1, int(round(len(sequences) * 0.8)))
    train_sequences = set(sequences[:cut])
    train_mask = np.asarray([s in train_sequences for s in seq], dtype=bool)
    val_mask = ~train_mask
    if not val_mask.any():
        val_mask = train_mask.copy()
    xt = torch.from_numpy(x[train_mask]).to(device)
    yt = torch.from_numpy(y[train_mask]).to(device)
    xv = torch.from_numpy(x[val_mask]).to(device)
    yv = torch.from_numpy(y[val_mask]).to(device)
    gate = NoneGate().to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    pos = float(yt.sum().detach().cpu())
    neg = float(len(yt) - pos)
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
    history = []
    for epoch in range(args.epochs):
        gate.train()
        opt.zero_grad(set_to_none=True)
        logits = gate(xt)
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
        opt.step()
        gate.eval()
        with torch.inference_mode():
            val_prob = torch.sigmoid(gate(xv)).cpu().numpy()
        vy = y[val_mask]
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "val_auc": auc_score(vy, val_prob),
                "val_accuracy": float(((val_prob >= args.threshold) == (vy > 0.5)).mean()),
                "val_accept_rate": float((val_prob >= args.threshold).mean()),
            }
        )
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 10 == 0:
            print(history[-1], flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": gate.state_dict(),
            "arch": {"in_dim": 7, "hidden": 32},
            "threshold": args.threshold,
            "data": {
                "cache_dir": str(args.cache_dir),
                "rows": int(len(y)),
                "positives": int(y.sum()),
                "negatives": int(len(y) - y.sum()),
                "train_sequences": sorted(train_sequences),
                "val_sequences": sorted(set(seq[val_mask].tolist())),
                "score_mode": args.score_mode,
                "threshold": args.threshold,
                "feature_definition": "selected score top/second/margin + raw cosine + geometry/delta/window count",
            },
            "history": history,
            "args": vars(args),
        },
        out,
    )
    np.savez_compressed(args.feature_out, x=x, y=y, sequence=seq, row_id=row_ids)
    (ROOT / "outputs/n23/none_gate_train_metrics.json").write_text(
        json.dumps(
            {
                "rows": int(len(y)),
                "positives": int(y.sum()),
                "negatives": int(len(y) - y.sum()),
                "train_sequences": sorted(train_sequences),
                "val_sequences": sorted(set(seq[val_mask].tolist())),
                "score_mode": args.score_mode,
                "threshold": args.threshold,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
