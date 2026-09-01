#!/usr/bin/env python
"""N20.4: Oracle Gate verdict against N18 V0 / Human / N19 baselines."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(".")
N20 = ROOT / "outputs/n20"

BASELINES = {
    "n18_v0": 0.7025318591132063,
    "human_control": 0.7154244078619095,
    "oracle_n19": 0.6291904104850248,
}


def main():
    rows = []
    with (N20 / "oracle_shadow_gate.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    summary = {}
    for r in rows:
        v = r["variant"]
        rec = {
            "attempts": int(r["attempts"]),
            "shadow_started": int(r["shadow_started"]),
            "accepts": int(r["accepts"]),
            "shadow_accepts": int(r["shadow_accepts"]),
            "correct_accepts": int(r["correct_accepts"]),
            "false_accepts": int(r["false_accepts"]),
            "mean_recorrection": (None if r["mean_recorrection_prob"] == ""
                                  else float(r["mean_recorrection_prob"])),
            "latency_mean": r["confirmation_latency_mean"],
            "lost_frames": int(r["lost_frames_by_delay"]),
        }
        summary[v] = rec
    best_delayed = None
    for k in ("3", "5"):
        for h in ("1", "3", "5", "8"):
            key = f"k{k}_h{h}"
            if key in summary and summary[key]["mean_recorrection"] is not None:
                if best_delayed is None or \
                        summary[key]["mean_recorrection"] < \
                        best_delayed["mean_recorrection"]:
                    best_delayed = {"variant": key, **summary[key]}
    h0 = summary.get("k5_h0")
    verdicts = []
    if h0 and h0["mean_recorrection"] is not None:
        d = BASELINES["n18_v0"] - h0["mean_recorrection"]
        verdicts.append(f"immediate oracle (k5_h0) recorr={h0['mean_recorrection']:.4f} "
                        f"delta_v0={d:+.4f}")
    if best_delayed:
        d = BASELINES["n18_v0"] - best_delayed["mean_recorrection"]
        verdicts.append(f"best delayed {best_delayed['variant']} "
                        f"recorr={best_delayed['mean_recorrection']:.4f} "
                        f"delta_v0={d:+.4f}")
    any_false = any(v["false_accepts"] > 0
                    for name, v in summary.items()
                    if name.startswith("k"))
    gate_pass = (
        h0 is not None and h0["mean_recorrection"] is not None and
        BASELINES["n18_v0"] - h0["mean_recorrection"] > 0.005 and
        best_delayed is not None and
        BASELINES["n18_v0"] - best_delayed["mean_recorrection"] > 0.005 and
        not any_false
    )
    out = {
        "gate": "PASS_ORACLE_SHADOW" if gate_pass
        else "FAIL_ORACLE_SHADOW",
        "baselines": BASELINES,
        "variants": summary,
        "best_delayed": best_delayed,
        "verdicts": verdicts,
    }
    (N20 / "oracle_gate_verdict.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    print("GATE_VERDICT_DONE", flush=True)


if __name__ == "__main__":
    main()
