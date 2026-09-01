#!/usr/bin/env python
"""N20.17-18: on-policy K+1 decision evaluation on the cal10 attempt
distribution (offline simulation with real shadow hypotheses).

For each cal10 attempt with a complete top-K shadow set, run the trained
K+1 verifier at H=5 and apply the deployment decision rule
(commit if P(best)>=threshold and margin>=margin; else reject-all).
Labels come from offline GT only.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_n20_kplus1 import SharedGRUSet, load_groups, groups_to_tensors  # noqa

N20 = ROOT / "outputs/n20"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/kplus1_gru.pt")
    ap.add_argument("--dataset", default="features/shadow_kplus1_cal10.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="kplus1_onpolicy_metrics.csv")
    args = ap.parse_args()

    bundle = torch.load(N20 / args.model, map_location="cpu")
    feats = bundle["feature_cols"]
    mu = bundle["mu"]
    sd = bundle["sd"]
    model = SharedGRUSet(len(feats))
    model.load_state_dict(bundle["model"])
    model.eval()
    groups = load_groups(N20 / args.dataset, args.h, feats)
    # keep only attempts with complete ranks 1..K
    full = []
    for seqs in groups:
        ranks = {k for k, *_ in seqs}
        if all(r in ranks for r in range(1, args.k + 1)):
            full.append(seqs)
    print(f"groups={len(groups)} full_top{args.k}={len(full)}", flush=True)
    X, M, Y = groups_to_tensors(full, args.h, len(feats), args.k)
    X = (X - mu) / sd
    with torch.inference_mode():
        P = torch.softmax(model(X, M), 1).numpy()
    ytrue = Y.argmax(1).numpy()
    rows = []
    stats = {"n": len(full), "commits": 0, "correct_commits": 0,
             "false_commits": 0, "rejects": 0, "correct_rejects": 0,
             "missed_commits": 0}
    for i, seqs in enumerate(full):
        seq = seqs[0][4].split(":")[0]
        att = seqs[0][4]
        p = P[i]
        best = int(np.argmax(p))
        decision = "REJECT"
        commit_ok = None
        if best >= 1 and p[best] >= args.threshold:
            others = np.delete(p, best)
            if p[best] - others.max() >= args.margin:
                decision = f"COMMIT_{best}"
        if decision.startswith("COMMIT"):
            stats["commits"] += 1
            if best == ytrue[i]:
                stats["correct_commits"] += 1
                commit_ok = 1
            else:
                stats["false_commits"] += 1
                commit_ok = 0
        else:
            stats["rejects"] += 1
            if ytrue[i] == 0:
                stats["correct_rejects"] += 1
            else:
                stats["missed_commits"] += 1
        rows.append({"attempt": att, "sequence": seq, "true_class": ytrue[i],
                     "decision": decision,
                     "p_none": round(float(p[0]), 4),
                     "p_best": round(float(p[1:].max()), 4),
                     "commit_ok": commit_ok,
                     "probs": json.dumps([round(float(v), 4) for v in p])})
    stats["commit_precision"] = round(
        stats["correct_commits"] / max(1, stats["commits"]), 4)
    stats["reject_all_accuracy"] = round(
        stats["correct_rejects"] / max(1, stats["rejects"]), 4)
    stats["coverage"] = round(len(full) / max(1, len(groups)), 4)
    with (N20 / args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (N20 / "kplus1_onpolicy_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print("ONPOLICY_DONE", flush=True)


if __name__ == "__main__":
    main()
