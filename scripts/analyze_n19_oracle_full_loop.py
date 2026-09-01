#!/usr/bin/env python
"""N19.4 Oracle FULL_LOOP gate analysis.

Aggregates the N19.3 oracle-anchor + oracle-verifier FULL_LOOP shards,
compares with N18 V0 and human-control baselines, and emits the gate
summary under outputs/n19/.

This is an offline diagnostic over causal oracle writes/verification:
future GT is used only to score what happened, never inside the loop.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import iou  # noqa: E402

N18 = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
DT = Path("/path/to/dancetrack")


def load_gt(seq):
    return DanceTrackDataset(str(DT), sequences=[], split="train").load_gt(seq)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_transactions(pattern):
    rows = []
    for p in sorted(N18.glob(pattern)):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def summarize(tag, tx):
    n_tx = len(tx)
    n_accept = sum(1 for t in tx if t.get("accepted"))
    n_correct = sum(1 for t in tx
                    if t.get("accepted") and t.get("correct_accept"))
    n_false = n_accept - n_correct
    n_react = sum(1 for t in tx if t.get("reactivated"))
    return {
        "attempts": n_tx,
        "accepted": n_accept,
        "accept_rate": n_accept / n_tx if n_tx else 0.0,
        "correct_accepts": n_correct,
        "false_accepts": n_false,
        "accept_correct_rate": n_correct / n_accept if n_accept else 0.0,
        "reactivated": n_react,
        "tag": tag,
    }


def main():
    N19.mkdir(exist_ok=True)

    # ---- N19.3 oracle run (score accepts against GT at the same frame)
    oracle_tx = load_transactions(
        "reactivation_transactions_oracle_n19_s*.jsonl")
    gt_cache = {}

    def gt_for(seq):
        if seq not in gt_cache:
            gt_cache[seq] = load_gt(seq)
        return gt_cache[seq]

    for t in oracle_tx:
        if t.get("recovery_box") is None:
            continue
        gf = gt_for(t["sequence"]).get(t["frame"])
        gid = t["gid"]
        if gf is None or gid not in gf.gt_ids:
            t["correct_accept"] = False
            continue
        box = [float(v) for v in t["recovery_box"]]
        tgt = [float(v) for v in gf.boxes[gf.gt_ids.index(gid)]]
        t["correct_accept"] = iou(box, tgt) >= 0.5

    oracle_sum = summarize("oracle_n19", oracle_tx)

    # baseline V0 (deployed verifier) and human control
    base_sum = {}
    for name, pattern in [("n18_v0", "reactivation_transactions_full_s*.jsonl"),
                          ("human", "reactivation_transactions_human_s*.jsonl")]:
        tx = load_transactions(pattern)
        for t in tx:
            if t.get("recovery_box") is None:
                continue
            gf = gt_for(t["sequence"]).get(t["frame"])
            gid = t["gid"]
            if gf is None or gid not in gf.gt_ids:
                t["correct_accept"] = False
                continue
            box = [float(v) for v in t["recovery_box"]]
            tgt = [float(v) for v in gf.boxes[gf.gt_ids.index(gid)]]
            t["correct_accept"] = iou(box, tgt) >= 0.5
        base_sum[name] = summarize(name, tx)

    # ---- metrics CSV aggregation
    def metrics_rows(prefix):
        rows = []
        for p in sorted(N18.glob(f"{prefix}_s*.csv")):
            rows.extend(read_csv(p))
        return rows

    def agg_metrics(prefix):
        rows = metrics_rows(prefix)
        if not rows:
            return {}
        out = {"sequences": len(rows)}
        for key in ["recovery_attempts", "accepted_recoveries",
                    "verifier_accepts", "lost_episodes", "runtime_s"]:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            out[key] = sum(vals) if vals else 0.0
        rec = [float(r["mean_recorrection_prob"])
               for r in rows if r.get("mean_recorrection_prob")]
        out["mean_recorrection_prob"] = sum(rec) / len(rec) if rec else None
        wden = sum(float(r["frames"]) for r in rows
                   if r.get("mean_recorrection_prob"))
        out["mean_recorrection_weighted"] = (
            sum(float(r["mean_recorrection_prob"]) * float(r["frames"])
                for r in rows if r.get("mean_recorrection_prob"))
            / wden) if wden else None
        for k in ["retention_1", "retention_3", "retention_5",
                  "retention_10", "retention_30", "retention_60",
                  "retention_120"]:
            vals = [float(r[k]) for r in rows if r.get(k) not in (None, "")]
            out[k] = sum(vals) / len(vals) if vals else None
        return out

    oracle_metrics = agg_metrics("full_loop_v0_metrics_oracle_n19")
    base_v0_metrics = agg_metrics("full_loop_v0_metrics_full")
    human_metrics = agg_metrics("full_loop_v0_metrics_human")

    # ---- write outputs
    table_rows = []
    for name, sm, mm in [
            ("oracle_n19", oracle_sum, oracle_metrics),
            ("n18_v0", base_sum["n18_v0"], base_v0_metrics),
            ("human_control", base_sum["human"], human_metrics)]:
        row = {"method": name}
        row.update(sm)
        for k in ["mean_recorrection_prob", "retention_1", "retention_3",
                  "retention_5", "retention_10", "retention_30",
                  "retention_60", "retention_120", "runtime_s"]:
            v = mm.get(k)
            row[k] = "" if v is None else round(float(v), 4)
        table_rows.append(row)

    (N19 / "oracle_full_loop.csv").write_text(
        _to_csv(table_rows), encoding="utf-8")

    summary = {
        "gate": "N19.4_ORACLE_FULL_LOOP",
        "oracle": {**oracle_sum, **oracle_metrics},
        "n18_v0": {**base_sum["n18_v0"], **base_v0_metrics},
        "human_control": {**base_sum["human"], **human_metrics},
    }
    (N19 / "oracle_gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- gate verdict (diagnostic; final decision needs the stage report)
    o, v = oracle_sum, base_sum["n18_v0"]
    gain_recorr = None
    if oracle_metrics.get("mean_recorrection_prob") is not None and \
            base_v0_metrics.get("mean_recorrection_prob") is not None:
        gain_recorr = (base_v0_metrics["mean_recorrection_prob"] -
                       oracle_metrics["mean_recorrection_prob"])
    verdict = "INCONCLUSIVE"
    if o["attempts"] > 0:
        improved = (o["accept_correct_rate"] >= 0.9 and
                    o["false_accepts"] <= 3 and
                    (gain_recorr is None or gain_recorr > 0.05))
        verdict = "PASS_ORACLE_REFRESH" if improved \
            else "FAIL_REFRESH_HYPOTHESIS"
    summary["verdict"] = verdict
    (N19 / "oracle_gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _to_csv(rows):
    if not rows:
        return ""
    keys = list(rows[0].keys())
    out = [",".join(keys)]
    for r in rows:
        out.append(",".join(str(r.get(k, "")) for k in keys))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
