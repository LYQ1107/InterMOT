#!/usr/bin/env python
"""N15 Identity Backbone Bake-Off on the Human Seed Identity Benchmark."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
BENCH = ROOT / "outputs/n15/identity_benchmark/benchmark.json"
FEAT_DIR = ROOT / "outputs/n15/features"
OUT_DIR = ROOT / "outputs/n15"

BACKBONES = ["osnet", "clipreid", "dinov2"]


def nfc(feats: np.ndarray, k1: int = 2, k2: int = 2) -> np.ndarray:
    """Pose2ID Neighbor Feature Centralization (CVPR 2025), numpy port."""
    f = feats.astype(np.float64)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    sim = f @ f.T
    np.fill_diagonal(sim, -1e9)
    idx = np.argsort(-sim, axis=1)[:, :k1]
    mutual = []
    for i in range(feats.shape[0]):
        m = [j for j in idx[i] if i in idx[j][:k2]]
        mutual.append(np.asarray(m, dtype=int))
    out = f.copy()
    for i, m in enumerate(mutual):
        if m.size:
            out[i] = out[i] + f[m].sum(axis=0)
    out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out.astype(np.float32)


def auc_from_scores(pos_sim: float, neg_sims: np.ndarray) -> float:
    if neg_sims.size == 0:
        return 1.0
    return float(np.mean(neg_sims < pos_sim) + 0.5 * np.mean(neg_sims == pos_sim))


def main() -> None:
    payload = json.loads(BENCH.read_text(encoding="utf-8"))
    queries = payload["queries"]
    crops = payload["crops"]
    box_by_crop = {c["crop_id"]: c["box"] for c in crops}
    feats = {b: np.load(FEAT_DIR / f"{b}.npy") for b in BACKBONES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_seq_rows = []
    hard_rows = []
    detail_rows = []
    fpr95_cache = {}
    for backbone in BACKBONES:
        base = feats[backbone]
        for variant in ("raw", "nfc"):
            detail = []
            all_scores = []
            all_labels = []
            agg = defaultdict(lambda: [0, 0])
            for q in queries:
                qf = base[q["query_crop_id"]]
                gids = q["gallery_crop_ids"]
                gf = base[gids]
                labels = np.zeros(len(gids), dtype=int)
                labels[: len(q["positive_crop_ids"])] = 1
                if variant == "nfc":
                    cfeat = nfc(np.concatenate([qf[None], gf], axis=0))
                    qc, gc = cfeat[0], cfeat[1:]
                else:
                    qc, gc = qf, gf
                sims = gc @ qc
                all_scores.extend(sims.tolist())
                all_labels.extend(labels.tolist())
                order = np.argsort(-sims)
                hit1 = 1 if labels[order[0]] == 1 else 0
                topk = min(5, len(sims))
                hit5 = 1 if labels[order[:topk]].sum() > 0 else 0
                pos_sim = float(sims[labels == 1][0])
                neg_sims = sims[labels == 0]
                neg_max = float(neg_sims.max()) if neg_sims.size else pos_sim
                margin = pos_sim - neg_max
                auc = auc_from_scores(pos_sim, neg_sims)
                qbox = np.asarray(box_by_crop[q["query_crop_id"]])
                pbox = np.asarray(box_by_crop[q["positive_crop_ids"][0]])
                qarea = max(1e-6, (qbox[2] - qbox[0]) * (qbox[3] - qbox[1]))
                parea = max(1e-6, (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]))
                scale_change = abs(np.log(parea / qarea))
                qc_center = (qbox[:2] + qbox[2:]) / 2
                cross = any(
                    np.linalg.norm(qc_center - (np.asarray(box_by_crop[i][:2]) + np.asarray(box_by_crop[i][2:])) / 2) < 80
                    for i in q["negative_crop_ids"]
                )
                hard_app = 1 if (neg_sims.size and neg_max > pos_sim - 0.05) else 0
                detail.append(
                    {
                        "split": q["split"], "seq": q["seq"], "query_id": q["query_id"],
                        "delta": q["delta"], "crowd": q["crowd"], "hit1": hit1,
                        "hit5": hit5, "pos_sim": pos_sim, "neg_max_sim": neg_max,
                        "margin": margin, "auc": auc, "n_neg": int(len(neg_sims)),
                        "scale_change": scale_change, "cross_neg": int(cross),
                        "hard_appearance": hard_app,
                    }
                )
            detail_rows.extend(
                {
                    "backbone": backbone, "variant": variant, **d,
                }
                for d in detail
            )
            scores_arr = np.asarray(all_scores, dtype=np.float64)
            labels_arr = np.asarray(all_labels, dtype=int)
            thr = np.percentile(scores_arr[labels_arr == 1], 5)
            fpr95 = float(np.mean(scores_arr[labels_arr == 0] >= thr))
            fpr95_cache[(backbone, variant)] = fpr95
            for split in ("train", "calibration", "all"):
                rows = detail if split == "all" else [d for d in detail if d["split"] == split]
                if not rows:
                    continue
                r1 = float(np.mean([d["hit1"] for d in rows]))
                r5 = float(np.mean([d["hit5"] for d in rows]))
                auc = float(np.mean([d["auc"] for d in rows]))
                pos_sim = float(np.mean([d["pos_sim"] for d in rows]))
                neg_sim = float(np.mean([d["neg_max_sim"] for d in rows]))
                margin = float(np.mean([d["margin"] for d in rows]))
                fpr1 = float(np.mean([1 - d["hit1"] for d in rows]))
                delta_r1 = {}
                for dt in (1, 3, 5, 10, 30):
                    sub = [d for d in rows if d["delta"] == dt]
                    delta_r1[dt] = float(np.mean([d["hit1"] for d in sub])) if sub else float("nan")
                summary_rows.append(
                    {
                        "backbone": backbone, "variant": variant, "split": split,
                        "n_queries": len(rows),
                        "R@1": r1, "R@5": r5, "AUC": auc, "FPR@1": fpr1,
                        "FPR@95TPR": fpr95,
                        "pos_sim": pos_sim, "neg_sim": neg_sim, "margin": margin,
                        **{f"R1_d{dt}": delta_r1[dt] for dt in (1, 3, 5, 10, 30)},
                    }
                )
            for seq in sorted({d["seq"] for d in detail}):
                sub = [d for d in detail if d["seq"] == seq]
                per_seq_rows.append(
                    {
                        "backbone": backbone, "variant": variant, "seq": seq,
                        "split": sub[0]["split"], "R@1": float(np.mean([d["hit1"] for d in sub])),
                        "n": len(sub),
                    }
                )
            strata = {}
            for d in detail:
                crowd_bin = "0-4" if d["crowd"] <= 4 else ("5-9" if d["crowd"] <= 9 else "10+")
                scale_bin = "<0.3" if d["scale_change"] < 0.3 else ("0.3-0.7" if d["scale_change"] < 0.7 else ">0.7")
                for key, val in [
                    (f"crowd:{crowd_bin}", d["hit1"]),
                    (f"scale:{scale_bin}", d["hit1"]),
                    (f"cross:{'yes' if d['cross_neg'] else 'no'}", d["hit1"]),
                    (f"app:{'hard' if d['hard_appearance'] else 'easy'}", d["hit1"]),
                    (f"split:{d['split']}", d["hit1"]),
                ]:
                    s = strata.setdefault(key, [0, 0])
                    s[0] += val
                    s[1] += 1
            for key, (h, t) in sorted(strata.items()):
                hard_rows.append(
                    {"backbone": backbone, "variant": variant, "stratum": key, "R@1": h / max(1, t), "n": t}
                )
            print(f"{backbone}_{variant}: R@1(train)="
                  f"{float(np.mean([d['hit1'] for d in detail if d['split']=='train'])):.4f} "
                  f"R@1(cal)="
                  f"{float(np.mean([d['hit1'] for d in detail if d['split']=='calibration'])):.4f}")

    def write_csv(name, fieldnames, rows):
        with open(OUT_DIR / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_csv(
        "identity_benchmark.csv",
        ["backbone", "variant", "split", "seq", "query_id", "delta", "crowd", "hit1",
         "hit5", "pos_sim", "neg_max_sim", "margin", "auc", "n_neg", "scale_change",
         "cross_neg", "hard_appearance"],
        detail_rows,
    )
    write_csv(
        "backbone_bakeoff.csv",
        ["backbone", "variant", "split", "n_queries", "R@1", "R@5", "AUC", "FPR@1", "FPR@95TPR",
         "pos_sim", "neg_sim", "margin", "R1_d1", "R1_d3", "R1_d5", "R1_d10", "R1_d30"],
        summary_rows,
    )
    write_csv(
        "backbone_per_sequence.csv",
        ["backbone", "variant", "seq", "split", "R@1", "n"],
        per_seq_rows,
    )
    write_csv(
        "hard_negative_analysis.csv",
        ["backbone", "variant", "stratum", "R@1", "n"],
        hard_rows,
    )


if __name__ == "__main__":
    main()
