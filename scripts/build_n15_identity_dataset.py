#!/usr/bin/env python
"""Build the N15 Human Seed Identity Benchmark from DanceTrack train/calibration."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam3_intermot.identity_anchor.identity_benchmark import build_benchmark

ROOT = Path(".")
DT = Path("/path/to/dancetrack")

TRAIN30 = [
    "dancetrack0001", "dancetrack0002", "dancetrack0006", "dancetrack0008",
    "dancetrack0012", "dancetrack0015", "dancetrack0016", "dancetrack0020",
    "dancetrack0023", "dancetrack0024", "dancetrack0027", "dancetrack0029",
    "dancetrack0032", "dancetrack0033", "dancetrack0037", "dancetrack0039",
    "dancetrack0044", "dancetrack0045", "dancetrack0049", "dancetrack0051",
    "dancetrack0052", "dancetrack0053", "dancetrack0055", "dancetrack0057",
    "dancetrack0061", "dancetrack0062", "dancetrack0066", "dancetrack0068",
    "dancetrack0069", "dancetrack0072",
]
CAL10 = [
    "dancetrack0074", "dancetrack0075", "dancetrack0080", "dancetrack0082",
    "dancetrack0083", "dancetrack0086", "dancetrack0087", "dancetrack0096",
    "dancetrack0098", "dancetrack0099",
]


def main() -> None:
    out = ROOT / "outputs/n15/identity_benchmark/benchmark.json"
    tmp = ROOT / "outputs/n15/identity_benchmark/_all.json"
    crops, queries = build_benchmark(
        DT, TRAIN30 + CAL10, "train", tmp,
        max_queries_per_seq=50,
        max_ids_per_frame=2,
        deltas=(1, 3, 5, 10, 30),
        max_negs=6,
    )
    cal_set = set(CAL10)
    qs = []
    for q in queries:
        d = q.__dict__
        if q.seq in cal_set:
            d["split"] = "calibration"
        qs.append(d)
    merged = {
        "crops": [
            {"crop_id": c.crop_id, "seq": c.seq, "frame": c.frame, "gid": c.gid, "box": list(c.box)}
            for c in crops
        ],
        "queries": qs,
    }
    out.write_text(json.dumps(merged), encoding="utf-8")
    tmp.unlink(missing_ok=True)
    n_train = sum(1 for q in qs if q["split"] == "train")
    n_cal = len(qs) - n_train
    print(f"crops={len(crops)} queries={len(qs)} train={n_train} cal={n_cal}")
    frozen = {
        "stage": "N15-FREEZE-RECORD",
        "date": "2026-08-10",
        "split": {"train30": TRAIN30, "calibration10": CAL10, "val25": "locked"},
        "benchmark": {
            "path": str(out),
            "crops": len(crops),
            "queries": len(qs),
            "train_queries": n_train,
            "calibration_queries": n_cal,
            "deltas": [1, 3, 5, 10, 30],
        },
    }
    (ROOT / "outputs/n15/n15_frozen.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
