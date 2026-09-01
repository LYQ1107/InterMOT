#!/usr/bin/env python3
"""Train the N23 correction-query compatibility adapter.

Training uses only train30 human-correction episodes and frozen CLIP-ReID
crop features already audited in N15.  The adapter is deliberately small;
its job is to learn query/candidate compatibility and geometry, not to
replace the visual backbone.
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
    IMAGE_H,
    IMAGE_W,
    PairRanker,
    pair_geometry,
)


def load_crop_tables():
    payload = json.loads(
        (ROOT / "outputs/n15/identity_benchmark/benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    features = np.load(ROOT / "outputs/n15/features/clipreid.npy", mmap_mode="r")
    by_key = {}
    by_frame = {}
    for crop in payload["crops"]:
        key = (crop["seq"], int(crop["frame"]), int(crop["gid"]))
        by_key.setdefault(key, crop)
        by_frame.setdefault((crop["seq"], int(crop["frame"])), []).append(crop)
    return features, by_key, by_frame


def make_pair_dataset(manifest: Path, max_pairs: int, seed: int):
    features, by_key, by_frame = load_crop_tables()
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rng = random.Random(seed)
    rng.shuffle(rows)
    pairs = []
    episode_used = 0
    positive = 0
    negative = 0
    for row in rows:
        seq = row["sequence"]
        t = int(row["t"])
        f = int(row["f"])
        gid = int(row["gid"])
        qcrop = by_key.get((seq, t, gid))
        if qcrop is None:
            continue
        candidates = list(by_frame.get((seq, f), []))
        if not candidates:
            continue
        q = np.asarray(features[int(qcrop["crop_id"])], dtype=np.float32)
        # Positive candidate, when the target is visible at f.
        present = row["target_present"] == "1"
        target_crop = by_key.get((seq, f, gid)) if present else None
        if target_crop is not None:
            ordered = [target_crop]
            negatives = [c for c in candidates if int(c["gid"]) != gid]
            rng.shuffle(negatives)
            ordered.extend(negatives[:5])
        else:
            rng.shuffle(candidates)
            ordered = candidates[:6]
        hb = np.asarray(json.loads(row["human_box"]), dtype=np.float32)
        for crop in ordered:
            cb = np.asarray(crop["box"], dtype=np.float32)
            y = int(target_crop is not None and int(crop["gid"]) == gid)
            pairs.append(
                {
                    "q": q,
                    "c": np.asarray(features[int(crop["crop_id"])], dtype=np.float32),
                    "qbox": hb,
                    "cbox": cb,
                    "delta": float(row["delta"]),
                    "y": y,
                    "sequence": seq,
                }
            )
            positive += y
            negative += 1 - y
        episode_used += 1
        if len(pairs) >= max_pairs:
            break
    # Balance the pair loss without discarding episode diversity.
    pos = [p for p in pairs if p["y"] == 1]
    neg = [p for p in pairs if p["y"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n = min(len(pos), len(neg), max_pairs // 2)
    selected = pos[:n] + neg[:n]
    rng.shuffle(selected)
    return selected, {
        "manifest": str(manifest),
        "episodes_used": episode_used,
        "pairs_before_balance": len(pairs),
        "pairs_after_balance": len(selected),
        "positive_after_balance": n,
        "negative_after_balance": n,
        "feature_source": "outputs/n15/features/clipreid.npy",
        "query_source": "train30 human-correction episodes",
        "image_width": IMAGE_W,
        "image_height": IMAGE_H,
    }


def tensors(pairs):
    q = torch.from_numpy(np.stack([p["q"] for p in pairs])).float()
    c = torch.from_numpy(np.stack([p["c"] for p in pairs])).float()
    qb = torch.from_numpy(np.stack([p["qbox"] for p in pairs])).float()
    cb = torch.from_numpy(np.stack([p["cbox"] for p in pairs])).float()
    delta = torch.tensor([p["delta"] for p in pairs], dtype=torch.float32)
    y = torch.tensor([p["y"] for p in pairs], dtype=torch.float32)
    g = torch.stack(
        [pair_geometry(qb[i], cb[i : i + 1], delta[i])[0] for i in range(len(pairs))]
    )
    return q, c, g, y


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    # Mann-Whitney formulation, with ties counted as half.
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    return float((ranks[y > 0.5].mean() - ranks[y <= 0.5].mean()) / len(score) + 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "outputs/n17/train_episodes.csv"))
    parser.add_argument("--max-pairs", type=int, default=180000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(ROOT / "outputs/n23/models/query_ranker.pt"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pairs, data_meta = make_pair_dataset(Path(args.manifest), args.max_pairs, args.seed)
    if not pairs:
        raise RuntimeError("no train pairs were constructed")
    q, c, g, y = tensors(pairs)
    n = len(y)
    split = max(1, int(round(n * 0.9)))
    train_idx = torch.arange(split)
    val_idx = torch.arange(split, n)
    model = PairRanker().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.from_numpy(rng.permutation(len(train_idx)))]
        losses = []
        for start in range(0, len(order), args.batch_size):
            ix = order[start : start + args.batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(q[ix].to(device), c[ix].to(device), g[ix].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, y[ix].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_logits = model(q[val_idx].to(device), c[val_idx].to(device), g[val_idx].to(device)).cpu()
        vy = y[val_idx].numpy()
        vs = torch.sigmoid(val_logits).numpy()
        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_auc": auc_score(vy, vs),
            "val_accuracy": float(((vs >= 0.5) == (vy > 0.5)).mean()) if len(vy) else None,
        }
        history.append(metrics)
        print(metrics, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "arch": {
                "anchor_dim": 1280,
                "projection_dim": 128,
                "hidden": 256,
                "geom_dim": 10,
            },
            "data": data_meta,
            "args": vars(args),
            "history": history,
        },
        out,
    )
    metrics_path = ROOT / "outputs/n23/query_adapter_train_metrics.json"
    metrics_path.write_text(
        json.dumps({"data": data_meta, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
