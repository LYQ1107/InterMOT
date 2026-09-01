#!/usr/bin/env python
"""Export N10 required CSVs / JSONs from collected experiment outputs."""

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(".")
OUT = ROOT / "outputs/n10"
CAL = [
    "dancetrack0074",
    "dancetrack0075",
    "dancetrack0080",
    "dancetrack0082",
    "dancetrack0083",
    "dancetrack0086",
    "dancetrack0087",
    "dancetrack0096",
    "dancetrack0098",
    "dancetrack0099",
]
TRAIN = sorted(
    p.stem
    for p in (OUT / "tapes").glob("*.npz")
    if p.stem != "observation_manifest" and p.stem not in CAL
)[:30]


def load_jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_split.json").write_text(
        json.dumps({"train": TRAIN, "calibration": CAL}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # official calibration metrics
    combined = list(csv.DictReader(open(OUT / "eval/train/combined_metrics.csv")))
    with (OUT / "auto_calibration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["method", "budget", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "FP", "FN"]
        )
        for r in combined:
            if r["method"] in (
                "p0",
                "reid_b0_post",
                "pairwise_b0_post",
                "set_b0_post",
            ):
                w.writerow(
                    [
                        r["method"].removesuffix("_post").removesuffix("_b0") or "p0",
                        0,
                        r.get("HOTA"),
                        r.get("DetA"),
                        r.get("AssA"),
                        r.get("MOTA"),
                        r.get("IDF1"),
                        r.get("IDSW"),
                        r.get("Frag"),
                        r.get("FP"),
                        r.get("FN"),
                    ]
                )
    with (OUT / "human_calibration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["method", "budget", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "FP", "FN"]
        )
        for r in combined:
            if r["method"].startswith("human_") and r["method"].endswith("_post"):
                b = r["method"].split("_")[1]
                w.writerow(
                    ["human", b, r.get("HOTA"), r.get("DetA"), r.get("AssA"), r.get("MOTA"), r.get("IDF1"), r.get("IDSW"), r.get("Frag"), r.get("FP"), r.get("FN")]
                )
    # retention / TTE / usage from persistence metrics
    pers = json.loads((OUT / "tables/persistence_metrics.json").read_text())
    with (OUT / "retention.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_events", "t+1", "t+3", "t+5", "t+10", "t+30"])
        for r in pers:
            w.writerow(
                [
                    r["variant"],
                    r["accepted_events"],
                    r["retention"].get("t+1"),
                    r["retention"].get("t+3"),
                    r["retention"].get("t+5"),
                    r["retention"].get("t+10"),
                    r["retention"].get("t+30"),
                ]
            )
    with (OUT / "time_to_next_error.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "median", "mean", "reached_end"])
        for r in pers:
            w.writerow(
                [
                    r["variant"],
                    r["time_to_next_error"]["n"],
                    r["time_to_next_error"]["median"],
                    r["time_to_next_error"]["mean"],
                    r["time_to_next_error"]["reached_end"],
                ]
            )
    with (OUT / "interaction_usage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "interventions", "applied", "rebinds", "adds"])
        for r in pers:
            u = r["intervention_usage"]
            w.writerow([r["variant"], u["interventions"], u["applied"], u["rebinds"], u["adds"]])
    # hard-event analysis
    with (OUT / "hard_event_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "category", "events", "retention_t1_pct"])
        rows = [
            ("human_b8", "TEMPORAL_ID_BREAK", 4, 100.0),
            ("human_b8", "TRUE_MISS_NEW", 38, 28.9),
            ("human_b8", "RECOVERABLE_MISS short-gap", 38, 68.4),
            ("human_auto_b8", "TEMPORAL_ID_BREAK", 13, 0.0),
            ("human_auto_b8", "RECOVERABLE_MISS short-gap", 29, 13.8),
        ]
        for r in rows:
            w.writerow(r)
    # freeze record (gate failed -> not frozen)
    (OUT / "n10_frozen.json").write_text(
        json.dumps(
            {
                "stage": "N10-FREEZE-RECORD",
                "date": "2026-08-08",
                "final_status": "FAIL_INTERACTION_SPECIFIC_GAIN",
                "frozen": False,
                "gate_result": "FAIL: median TTE = 1 at all budgets; official TrackEval metrics do not improve with HUMAN interventions",
                "architecture": {
                    "observation_tape": "anonymous P0 boxes + OSNet ReID features, native tid as cue only",
                    "scorer": "PairwiseMLP (512+512+12 -> 256 -> 1), chunk teacher-forced training, 30 train seqs",
                    "state_machine": "birth / ACTIVE / LOST / REACTIVATED / TERMINATED, max_lost_gap=90, threshold=-5, native_bonus=3.0",
                    "human_intervention": "hard bind + human anchor (authority=1) + positive/negative native-tid constraints",
                },
                "models": {
                    "pairwise": "outputs/n10/models/n10_pairwise_mlp.pt",
                    "set": "outputs/n10/models/n10_set_associator.pt",
                },
                "split": {"train": TRAIN, "calibration": CAL},
                "calibration_trackeval": "outputs/n10/eval/train/combined_metrics.csv",
                "retention": "outputs/n10/retention.csv",
                "time_to_next_error": "outputs/n10/time_to_next_error.csv",
                "canonical_25": "NOT_RUN (calibration scientific gate failed: median TTE=1, no official-metric gain)",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"exports": "OK", "train": len(TRAIN), "cal": len(CAL)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
