#!/usr/bin/env python
"""N22 Phase-I: audit identity sources and evaluate raw representations.

This audit intentionally reconstructs the *live* evidence convention used by
N20/N21: a shadow starts at f0 and evidence step 1 is f0+1.  It does not use
the historical N21 ``cal10.npz`` because that builder included f0 as the
first visual token, which is one frame earlier than the live runner.

The experiment uses only cached GFN/R0 embeddings and real SAM3 shadow boxes.
GT labels are read from the already-built offline shadow feature CSV strictly
for evaluation.  No label participates in a score or ranking decision.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(".")
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"
R0_CACHE = ROOT / "outputs/n20/gfn_cache_r0"
SHADOW = ROOT / "outputs/n20/full_shadow_cache_cal10"
FEATURES = ROOT / "outputs/n20/features/shadow_kplus1_cal10.csv"
OUT = ROOT / "outputs/n22"
HORIZONS = (1, 3, 5, 8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(1e-8, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1e-8, (bx2 - bx1) * (by2 - by1))
    return inter / (aa + bb - inter)


def safe_float(value, default=-1.0):
    if value is None or value == "":
        return default
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


class SequenceCache:
    def __init__(self, seq: str):
        z = np.load(CACHE / f"{seq}.npz")
        qz = np.load(CACHE / f"{seq}_queries.npz")
        rz = np.load(R0_CACHE / f"{seq}.npz")
        self.frames = z["frames"].astype(np.int64)
        self.offsets = z["offsets"].astype(np.int64)
        self.boxes = z["boxes"].astype(np.float32)
        emb = z["emb"].astype(np.float32)
        self.gfn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        r0 = rz["r0g"].astype(np.float32)
        self.r0 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
        qg = qz["qemb"].astype(np.float32)
        self.qg = qg / (np.linalg.norm(qg, axis=1, keepdims=True) + 1e-8)
        qr = rz["r0q"].astype(np.float32)
        self.qr = qr / (np.linalg.norm(qr, axis=1, keepdims=True) + 1e-8)
        self.qgids = [int(x) for x in qz["gids"]]
        self.qindex = {gid: i for i, gid in enumerate(self.qgids)}
        z.close()
        qz.close()
        rz.close()

    def query(self, gid: int):
        idx = self.qindex.get(gid)
        if idx is None:
            return None
        return self.qg[idx], self.qr[idx]

    def detection(self, frame: int, box):
        if box is None:
            return None
        pos = int(np.searchsorted(self.frames, frame))
        if pos >= len(self.frames) or int(self.frames[pos]) != int(frame):
            return None
        lo = int(self.offsets[pos - 1]) if pos > 0 else 0
        hi = int(self.offsets[pos])
        if hi <= lo:
            return None
        box = np.asarray(box, dtype=np.float32)
        scores = np.asarray([iou(b, box) for b in self.boxes[lo:hi]])
        best = int(np.argmax(scores))
        if float(scores[best]) < 0.5:
            return None
        idx = lo + best
        return self.gfn[idx], self.r0[idx]


def load_feature_labels():
    """Index evaluation labels and memory scores by attempt/horizon/rank."""
    out = defaultdict(lambda: defaultdict(dict))
    with FEATURES.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            att = row["attempt"]
            h = int(row["evidence_step"])
            rank = int(row["candidate_rank"])
            out[att][h][rank] = {
                "label": int(row["label_correct"]),
                "gfn_memory": safe_float(row["gfn_sim_mem_max"]),
                "r0_memory": safe_float(row["r0_sim_mem_max"]),
                "csv_gfn": safe_float(row["gfn_sim_human_root"]),
                "csv_r0": safe_float(row["r0_sim_human_root"]),
            }
    return out


def collect_records(labels):
    """Build live-aligned per-candidate prefix scores for every shadow."""
    seq_cache = {}
    records = defaultdict(lambda: defaultdict(dict))
    n_shadow = 0
    n_missing_query = 0
    n_missing_frame = 0
    for path in sorted(SHADOW.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                seq = row["sequence"]
                f0 = int(row["frame"])
                gid = int(row["gid"])
                rank = int(row["candidate_rank"])
                att = f"{seq}:{f0}:{gid}"
                if att not in labels:
                    continue
                if seq not in seq_cache:
                    seq_cache[seq] = SequenceCache(seq)
                cache = seq_cache[seq]
                root = cache.query(gid)
                if root is None:
                    n_missing_query += 1
                    continue
                root_gfn, root_r0 = root
                frame_map = {
                    int(x["frame"]): x.get("box")
                    for x in row.get("frames", [])
                }
                per_step = {}
                step_gvecs = []
                step_rvecs = []
                for step in range(1, max(HORIZONS) + 1):
                    frame = f0 + step
                    box = frame_map.get(frame)
                    de = cache.detection(frame, box)
                    if de is not None:
                        step_gvecs.append(de[0])
                        step_rvecs.append(de[1])
                    else:
                        step_gvecs.append(None)
                        step_rvecs.append(None)
                        n_missing_frame += int(box is not None)
                for h in HORIZONS:
                    gvecs = [x for x in step_gvecs[:h] if x is not None]
                    rvecs = [x for x in step_rvecs[:h] if x is not None]
                    if gvecs:
                        g = np.mean(np.asarray(gvecs), axis=0)
                        g = g / (np.linalg.norm(g) + 1e-8)
                        g_score = float(g @ root_gfn)
                    else:
                        g_score = -1.0
                    if rvecs:
                        r = np.mean(np.asarray(rvecs), axis=0)
                        r = r / (np.linalg.norm(r) + 1e-8)
                        r_score = float(r @ root_r0)
                    else:
                        r_score = -1.0
                    info = labels[att].get(h, {}).get(rank)
                    if info is None:
                        continue
                    mem_g = info["gfn_memory"]
                    mem_r = info["r0_memory"]
                    # The live runner always has the Human Root as the
                    # initial memory slot. Older offline rows leave the
                    # memory columns blank when no dynamic writer slot was
                    # added, so use root similarity as the causal K=1
                    # fallback instead of dropping the whole attempt.
                    if mem_g <= -0.99:
                        mem_g = info["csv_gfn"] if info["csv_gfn"] > -0.99 else g_score
                    if mem_r <= -0.99:
                        mem_r = info["csv_r0"] if info["csv_r0"] > -0.99 else r_score
                    mem_values = [x for x in (mem_g, mem_r) if x > -0.99]
                    per_step[h] = {
                        "label": info["label"],
                        "gfn": g_score,
                        "r0": r_score,
                        "fused": 0.5 * (g_score + r_score)
                        if g_score > -0.99 and r_score > -0.99 else -1.0,
                        "memory_gfn": mem_g,
                        "memory_r0": mem_r,
                        "memory_fused": float(np.mean(mem_values))
                        if mem_values else -1.0,
                    }
                records[att][rank] = per_step
                n_shadow += 1
    return records, {
        "shadow_rows_used": n_shadow,
        "missing_queries": n_missing_query,
        "candidate_box_embedding_misses": n_missing_frame,
        "sequences_loaded": sorted(seq_cache),
    }


def auc(scores, labels):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    pos_rank_sum = float(ranks[pos].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_representation(records, rep, horizon):
    scores = []
    labels = []
    top1_hits = []
    top5_hits = []
    margins = []
    hard_errors = []
    usable_groups = 0
    for att, by_rank in records.items():
        candidates = []
        for rank in range(1, 6):
            item = by_rank.get(rank, {}).get(horizon)
            if item is None or item[rep] <= -0.99:
                break
            candidates.append((rank, float(item[rep]), int(item["label"])))
        if len(candidates) != 5:
            continue
        usable_groups += 1
        candidates_sorted = sorted(candidates, key=lambda x: (-x[1], x[0]))
        top1_hits.append(int(candidates_sorted[0][2] == 1))
        top5_hits.append(int(any(x[2] == 1 for x in candidates)))
        positives = [x[1] for x in candidates if x[2] == 1]
        negatives = [x[1] for x in candidates if x[2] == 0]
        if positives and negatives:
            margin = max(positives) - max(negatives)
            margins.append(margin)
            hard_errors.append(int(max(negatives) >= max(positives)))
        for _, score, label in candidates:
            scores.append(score)
            labels.append(label)
    return {
        "representation": rep,
        "horizon": horizon,
        "attempts": usable_groups,
        "candidate_rows": len(scores),
        "positive_candidates": int(sum(labels)),
        "auc": round(auc(scores, labels), 6),
        "top1": round(float(np.mean(top1_hits)) if top1_hits else float("nan"), 6),
        "top5_hit": round(float(np.mean(top5_hits)) if top5_hits else float("nan"), 6),
        "mean_hard_negative_margin": round(float(np.mean(margins)) if margins else float("nan"), 6),
        "hard_negative_error_rate": round(float(np.mean(hard_errors)) if hard_errors else float("nan"), 6),
    }


def write_alignment_audit():
    builder = (ROOT / "scripts/build_n21_tracklet_identity_dataset.py").read_text()
    live = (ROOT / "scripts/run_n21_live_final_gate.py").read_text()
    historical_includes_f0 = 'r["frames"][:h]' in builder
    live_starts_f1 = "range(1, args.h + 1)" in live
    result = {
        "historical_n21_builder_first_shadow_frame": "f0" if historical_includes_f0 else "not_detected",
        "n21_live_runner_first_evidence_frame": "f0+1" if live_starts_f1 else "not_detected",
        "status": "MISMATCH" if historical_includes_f0 and live_starts_f1 else "NOT_CONFIRMED",
        "impact": "historical CATIL offline tokens are shifted one frame earlier than live evidence",
        "n22_policy": "reconstruct f0+1..f0+8 from shadow JSONL and use this audit for the first representation gate",
    }
    (OUT / "n21_frame_alignment_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_feature_labels()
    records, collection = collect_records(labels)
    alignment = write_alignment_audit()
    reps = ("gfn", "r0", "fused", "memory_gfn", "memory_r0", "memory_fused")
    summary = []
    for h in HORIZONS:
        for rep in reps:
            summary.append(evaluate_representation(records, rep, h))

    with (OUT / "identity_representation_eval.csv").open(
            "w", newline="", encoding="utf-8") as f:
        fields = list(summary[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    payload = {
        "protocol": {
            "split": "cal10",
            "shadow_source": str(SHADOW),
            "feature_label_source": str(FEATURES),
            "candidate_pool": "real top-5 shadow candidates",
            "horizons": list(HORIZONS),
            "score": "masked mean of GFN/R0 embeddings at f0+1..f0+h, cosine to human root",
            "fused": "mean of GFN and R0 cosine scores",
            "memory": "per-step max memory similarity from the same causal feature stream",
            "gt_used": "offline labels only; never used for scores",
        },
        "collection": collection,
        "alignment": alignment,
        "results": summary,
    }
    (OUT / "identity_representation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    write_identity_audit_doc(payload)
    print(json.dumps(payload, indent=2), flush=True)
    print("N22_IDENTITY_AUDIT_DONE", flush=True)


def write_identity_audit_doc(payload):
    rows = payload["results"]
    lines = [
        "# N22 Identity Pipeline Audit",
        "",
        "Project: `.`",
        "",
        "Date: 2026-08-21. This is the first N22 evidence gate, reconstructed from the live-aligned shadow convention.",
        "",
        "## 1. Scope and protocol",
        "",
        "The audit follows the actual N20/N21 causal convention: a shadow starts at `f0`, and evidence step `h` uses only `f0+1 ... f0+h`. The candidate pool is the real top-5 shadow cache from cal10. GT-derived `label_correct` values are used only offline to score the representations.",
        "",
        "The raw identity score is the cosine between the masked mean candidate embedding and the Human Root embedding. `GFN+R0` is the mean of the two cosine scores. Memory rows use the causal max similarity to the active memory slots already recorded by the N20 feature builder; blank historical memory fields fall back to the live runner's mandatory Human Root slot.",
        "",
        "## 2. Identity source audit",
        "",
        "| component | source in code/cache | role | trainable in N21 live gate |",
        "| --- | --- | --- | --- |",
        "| Human Root | `outputs/n18/route_c/gfn_cache/*_queries.npz`: `qemb`; `outputs/n20/gfn_cache_r0`: `r0q` | initial identity authority/query | no |",
        "| GFN candidate | `gfn_cache/*.npz`: `emb`, matched to shadow boxes at IoU >= 0.5 | global recovery appearance | no |",
        "| R0 candidate | `gfn_cache_r0/*.npz`: `r0g`, produced by the N18 R0 embedding head | adapted identity projection used by recovery | no in live gate |",
        "| SAM3 feature | SAM3 propagation session outputs/boxes; no stable per-object identity vector is persisted in this pipeline | generates/propagates shadow hypotheses | no |",
        "| Memory similarity | `mem_state` slots in `run_n20_onpolicy_full_loop.py` and `run_n21_live_final_gate.py`; slots contain GFN/R0 vectors from prior causal writes | compares current candidate to past identity evidence | slot content is not learned online |",
        "| CATIL | N21 `TrackletIdentityModel`: 4096-d concatenated GFN+R0 per-frame vectors plus a temporal encoder | verifier residual over frozen identity inputs | C0/C1/C2 updated, but upstream cache vectors remain fixed |",
        "",
        "The source audit confirms that SAM3 is a recovery/tracking engine, not the identity representation being adapted. The deployed identity information enters through GFN and R0; memory only stores vectors in those same spaces; CATIL receives a late, already-computed representation.",
        "",
        "## 3. Historical N21 alignment confound",
        "",
        f"- Historical N21 dataset builder status: `{payload['alignment']['historical_n21_builder_first_shadow_frame']}`.",
        f"- N21 live runner status: `{payload['alignment']['n21_live_runner_first_evidence_frame']}`.",
        f"- Alignment verdict: **`{payload['alignment']['status']}`**.",
        "",
        "`build_n21_tracklet_identity_dataset.py` iterated `r[\"frames\"][:h]`, whose first entry is the shadow start frame `f0`; the live runner iterated `range(1, args.h + 1)`, whose first evidence frame is `f0+1`. N22 therefore does not reuse the historical CATIL NPZ for the first representation gate. This is a protocol confound to repair and report, not a reason to discard the N21 live result.",
        "",
        "## 4. Representation evaluation",
        "",
        "| representation | h | attempts | AUC | top-1 | top-5 hit | hard-neg margin | hard-neg error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['representation']} | {r['horizon']} | {r['attempts']} | {r['auc']} | {r['top1']} | {r['top5_hit']} | {r['mean_hard_negative_margin']} | {r['hard_negative_error_rate']} |"
        )
    lines += [
        "",
        "`top-5 hit` means that at least one of the five real recovery candidates is labeled correct; it measures candidate-pool availability, not a claim that a new method can create a missing candidate.",
        "",
        "## 5. Initial diagnosis",
        "",
        "The comparison separates three possible locations of identity information: a frozen per-frame appearance space (GFN/R0), a fused appearance space, and causal memory similarity. If memory beats the raw root scores while raw GFN/R0 remain weak, the next method should learn a memory-conditioned identity representation from positive identity evidence and explicit hard negatives, rather than only updating CATIL's final decision boundary. If all raw spaces have poor hard-negative separation, a direct R0/tracklet encoder experiment is necessary before any live-loop claim.",
        "",
        "The exact quantitative conclusion and all provenance are in `outputs/n22/identity_representation_eval.csv` and `outputs/n22/identity_representation_summary.json`.",
        "",
    ]
    (ROOT / "docs/N22_identity_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
