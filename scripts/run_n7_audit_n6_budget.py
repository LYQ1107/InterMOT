#!/usr/bin/env python
"""N7-A: read-only re-audit of N6 budget results from real CSVs."""

import csv
import json
from pathlib import Path

from sam3_intermot.datasets.dancetrack import DanceTrackDataset


ROOT = Path(".")
OUT = ROOT / "outputs/n7/audit"
STATS = ROOT / "outputs/n6/full25_stats"
DS = DanceTrackDataset("/path/to/dancetrack", split="val")
SEQS = sorted(
    p.name
    for p in Path("/path/to/dancetrack/val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)


def load_per_seq():
    rows = {}
    with (STATS / "per_sequence_metrics_post.csv").open() as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["protocol"], {})[r["sequence"]] = r
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_per_seq()
    out_rows = []
    summary = {"n_sequences": len(SEQS), "b_ge_p0": {}, "b_lt_p0": {}}
    for seq in SEQS:
        gt = DS.load_gt(seq)
        gt_ids = sorted({gid for f in gt.values() for gid in f.gt_ids})
        row = {
            "sequence": seq,
            "frames": DS.num_frames(seq),
            "gt_identities": len(gt_ids),
        }
        for proto, label in [
            ("p0", "P0"),
            ("p4_b1", "B1"),
            ("p4_b2", "B2"),
            ("p4_b4", "B4"),
            ("p4_b8", "B8"),
        ]:
            r = data.get(proto, {}).get(seq)
            if r is None:
                continue
            for m in ["HOTA", "AssA", "IDF1", "IDSW"]:
                row[f"{label}_{m}"] = r.get(m, "")
        out_rows.append(row)
    with (OUT / "n6_budget_vs_p0.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    # aggregate comparisons
    for b, label in [(1, "B1"), (2, "B2"), (4, "B4"), (8, "B8")]:
        ge = sum(
            1
            for r in out_rows
            if float(r[f"{label}_HOTA"]) + 1e-9 >= float(r["P0_HOTA"])
        )
        lt = len(out_rows) - ge
        summary[f"b_ge_p0"][label] = ge
        summary[f"b_lt_p0"][label] = lt
        summary[f"mean_delta_{label}_HOTA"] = float(
            sum(float(r[f"{label}_HOTA"]) - float(r["P0_HOTA"]) for r in out_rows)
            / len(out_rows)
        )
    (OUT / "n6_budget_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("worst sequences by B1-P0 HOTA:")
    diffs = sorted(
        (
            (float(r["B1_HOTA"]) - float(r["P0_HOTA"]), r["sequence"])
            for r in out_rows
        )
    )
    for d, s in diffs[:5]:
        print(s, round(d, 2))


if __name__ == "__main__":
    main()
