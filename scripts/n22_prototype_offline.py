#!/usr/bin/env python
"""N22 correction-driven compatible identity memory offline gate.

This is the post-CDCIA fallback hypothesis.  The frozen R0 encoder remains
the canonical coordinate.  Each identity keeps a causal positive prototype
and up to two explicit wrong-identity prototypes.  A human correction updates
only that identity's base-space memory, so old cached vectors never become
stale under a global encoder update.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from n22_cdcia_offline import load_groups


ROOT = Path(".")
OUT = ROOT / "outputs/n22"


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def candidate_vectors(g):
    x = (g["r0"] * g["mask"][..., None]).sum(1)
    x = x / np.maximum(g["mask"].sum(1, keepdims=True), 1.0)
    return normalize(x)


def decide(scores, threshold, margin):
    order = np.argsort(-scores, kind="stable")
    if scores[order[0]] >= threshold and \
            scores[order[0]] - scores[order[1]] >= margin:
        return int(order[0]) + 1
    return 0


def stream(groups, alpha, beta, threshold=0.5, margin=0.05):
    state = {}
    rows = []
    corrected_keys = set()
    repeat_wrong = 0
    for g in groups:
        key = (g["seq"], int(g["y"]), g["att"].split(":")[-1])
        # The public gid is encoded in the attempt key; using the final
        # component above would be ambiguous only if the dataset key were
        # malformed, which is checked in the loader.  Keep a separate parse
        # for clarity and stable sequence order.
        parts = g["att"].split(":")
        state_key = (g["seq"], int(parts[2]))
        if state_key not in state:
            state[state_key] = {"pos": normalize(g["root"]), "negs": []}
        st = state[state_key]
        cand = candidate_vectors(g)
        scores = cand @ st["pos"]
        if st["negs"] and beta > 0:
            neg_sim = np.max(np.stack([cand @ n for n in st["negs"]]), axis=0)
            scores = scores - beta * np.maximum(neg_sim, 0.0)
        decision = decide(scores, threshold, margin)
        target = int(g["y"])
        wrong = bool((decision >= 1 and decision != target) or
                     (decision == 0 and target >= 1))
        repeated = int(wrong and state_key in corrected_keys)
        repeat_wrong += repeated
        rows.append({
            "att": g["att"], "seq": g["seq"],
            "frame": int(g["frame"]), "gid": int(parts[2]),
            "decision": decision, "target": target,
            "wrong": int(wrong), "repeated_wrong": repeated,
        })
        if not wrong:
            continue
        corrected_keys.add(state_key)
        # The completed human correction supplies the positive target box;
        # an ID_WRONG commit also supplies the committed candidate as an
        # explicit negative.  This update happens after this attempt only.
        if target >= 1:
            p = cand[target - 1]
            st["pos"] = normalize((1.0 - alpha) * st["pos"] + alpha * p)
        if decision >= 1 and decision != target:
            n = cand[decision - 1]
            st["negs"].append(n)
            if len(st["negs"]) > 2:
                st["negs"] = st["negs"][-2:]
    false = sum(int(r["decision"] >= 1 and r["decision"] != r["target"])
                for r in rows)
    missed = sum(int(r["decision"] == 0 and r["target"] >= 1) for r in rows)
    correct = sum(int(r["decision"] >= 1 and r["decision"] == r["target"])
                  for r in rows)
    return {
        "attempts": len(rows), "correct_commits": correct,
        "false_commits": false, "missed_commits": missed,
        "corrections": false + missed,
        "commit_precision": correct / max(1, correct + false),
        "commit_recall": correct / max(1, correct + missed),
        "same_identity_repeated_wrong": repeat_wrong,
        "alpha": alpha, "beta": beta,
        "threshold": threshold, "margin": margin,
    }, rows


def main():
    horizon = 5
    train = load_groups(OUT / "datasets/train30_aligned.npz", horizon)
    cal = load_groups(OUT / "datasets/cal10_aligned.npz", horizon)
    # Small hyperparameter grid is selected on train30 only.  Cal10 remains
    # sequence-disjoint and is never used to choose alpha/beta.
    grid = []
    for alpha, beta in itertools.product((0.05, 0.10, 0.20, 0.40),
                                          (0.0, 0.05, 0.10, 0.20)):
        tr, _ = stream(train, alpha, beta)
        grid.append({"split": "train30", **tr})
    grid.sort(key=lambda x: (x["corrections"], -x["commit_precision"],
                             -x["commit_recall"]))
    best = grid[0]
    selected = (float(best["alpha"]), float(best["beta"]))
    cal_result, cal_rows = stream(cal, *selected)
    frozen_result, _ = stream(train, 0.0, 0.0)
    frozen_cal, _ = stream(cal, 0.0, 0.0)
    rows = [{"method": "prototype_frozen", "split": "train30",
             **frozen_result},
            {"method": "prototype_frozen", "split": "cal10",
             **frozen_cal},
            {"method": "prototype_correction_memory", "split": "train30",
             **stream(train, *selected)[0]},
            {"method": "prototype_correction_memory", "split": "cal10",
             **cal_result}]
    path = OUT / "prototype_offline_grid.csv"
    fields = sorted({k for row in grid + rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(grid)
        writer.writerows(rows)
    summary = {
        "horizon": horizon,
        "train_groups": len(train), "cal10_groups": len(cal),
        "selected_alpha": selected[0], "selected_beta": selected[1],
        "train_selected": rows[2], "cal10_selected": rows[3],
        "frozen_cal10": frozen_cal,
        "output_csv": str(path),
        "hypothesis": "causal per-identity base-R0 positive prototype plus explicit wrong prototypes",
        "human_signal": "target positive and committed wrong candidate are applied after the current decision",
    }
    (OUT / "prototype_offline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "prototype_cal10_trace.jsonl").open("w", encoding="utf-8") as h:
        for row in cal_rows:
            h.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("N22_PROTOTYPE_OFFLINE_DONE", flush=True)


if __name__ == "__main__":
    main()
