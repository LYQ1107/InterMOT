#!/usr/bin/env python
"""N20.4: aggregate Oracle Shadow Gate tables from delayed-loop runs."""

import csv
import glob as globmod
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

N20 = ROOT / "outputs/n20"
OUT = N20 / "full_loop_oracle_shadow"


def load_metrics(prefix):
    rows = []
    for p in map(Path, sorted(globmod.glob(str(OUT / f"metrics_{prefix}*.csv")))):
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def load_tx(prefix):
    rows = []
    for p in map(Path, sorted(globmod.glob(str(OUT / f"transactions_{prefix}*.jsonl")))):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    variants = []
    for p in sorted(globmod.glob(str(OUT / "metrics_*.csv"))):
        name = Path(p).stem[len("metrics_"):]
        if name.startswith(("smoke", "dump_only", "probe")):
            continue
        variants.append(name)
    variants = sorted(set(variants))

    gate_rows = []
    latency_rows = []
    full_rows = []
    for v in variants:
        ms = load_metrics(v)
        tx = load_tx(v)
        attempts = len([t for t in tx
                        if t.get("shadow_event") != "VERDICT"])
        shadow_started = [t for t in tx
                          if t.get("shadow_event") == "START"]
        accepts = [t for t in tx if t.get("reactivated")]
        shadow_accepts = [t for t in tx if t.get("shadow_commit")]
        rejects = [t for t in tx if t.get("shadow_verdict")
                   in ("REJECT", "TIMEOUT")]
        # correctness of shadow accepts from GT (offline)
        # transactions do not carry GT; use the events trace instead
        ev = []
        for p in map(Path, sorted(globmod.glob(str(OUT / f"events_{v}*.jsonl")))):
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    ev.append(json.loads(line))
        ev_by = defaultdict(dict)
        for e in ev:
            ev_by[(e["sequence"], e["gid"])][int(e["frame"])] = e
        correct = 0
        for t in shadow_accepts:
            f0 = int(t.get("commit_frame", t["frame"]))
            e = ev_by.get((t["sequence"], t["gid"]), {}).get(f0)
            if e is not None and e.get("correct") == 1:
                correct += 1
        false_accepts = len(shadow_accepts) - correct
        latency = [int(t["commit_frame"]) - int(t["frame"])
                   for t in shadow_accepts]
        lost_frames = sum(latency)
        recorr = None
        if ms:
            vals = [float(r["mean_recorrection_prob"])
                    for r in ms if r.get("mean_recorrection_prob")]
            if vals:
                recorr = sum(vals) / len(vals)
        gate_rows.append({
            "variant": v,
            "attempts": attempts,
            "shadow_started": len(shadow_started),
            "accepts": len(accepts),
            "shadow_accepts": len(shadow_accepts),
            "correct_accepts": correct,
            "false_accepts": false_accepts,
            "rejects": len(rejects),
            "confirmation_latency_mean": round(
                sum(latency) / len(latency), 3) if latency else None,
            "confirmation_latency_median": round(
                sorted(latency)[len(latency) // 2], 3) if latency else None,
            "confirmation_latency_p90": round(
                sorted(latency)[int(len(latency) * 0.9)], 3)
            if latency else None,
            "lost_frames_by_delay": lost_frames,
            "mean_recorrection_prob": recorr,
            "retention_5": None,
            "retention_30": None,
        })
        for t in shadow_accepts:
            latency_rows.append({
                "variant": v, "sequence": t["sequence"], "frame": t["frame"],
                "gid": t["gid"], "commit_frame": t.get("commit_frame"),
                "latency": int(t["commit_frame"]) - int(t["frame"]),
            })
        for r in ms:
            full_rows.append({"variant": v, **r})

    with (N20 / "oracle_shadow_gate.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        w.writeheader()
        w.writerows(gate_rows)
    with (N20 / "oracle_shadow_latency.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
        w.writeheader()
        w.writerows(latency_rows)
    with (N20 / "oracle_shadow_full_loop.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        w.writeheader()
        w.writerows(full_rows)
    print(json.dumps(gate_rows, indent=2), flush=True)
    print("GATE_AGG_DONE", flush=True)


if __name__ == "__main__":
    main()
