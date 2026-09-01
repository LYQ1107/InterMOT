"""Aggregate N9 evaluation, persistence, cost and statistics outputs."""

import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(".")
EVAL = Path(os.environ.get("N9_EVAL_OUT", ROOT / "outputs/n9/eval"))
REAL = ROOT / "outputs/n9/real"
TABLES = ROOT / "outputs/n9/tables"
DATA = ROOT / "outputs/n9"


def load_jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    combined = {}
    with (EVAL / "combined_metrics.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            combined[r["method"]] = r
    variants = ["n8", "reid", "pairwise", "auto", "proposed"]
    budgets = [0, 1, 2, 4, 8]
    seqs = sorted(
        p.name
        for p in (Path("/path/to/dancetrack") / "val").iterdir()
        if p.is_dir()
    )
    qc_rows = []
    for v in variants:
        for b in budgets:
            key = f"{v}_b{b}_post" if v != "n8" else f"n8_b{b}_post"
            c = combined.get(key, {})
            accepted = 0
            for s in seqs:
                root = (
                    ROOT / "outputs/n8/real" / f"route_a_b{b}"
                    if v == "n8"
                    else REAL / f"{v}_b{b}"
                )
                p = root / s / "summary.json"
                if p.exists():
                    accepted += json.loads(p.read_text()).get("accepted_count", 0)
            qc_rows.append(
                {
                    "variant": v,
                    "budget": b,
                    "accepted_interactions": accepted,
                    "interactions_per_sequence": round(accepted / max(1, len(seqs)), 3),
                    **{k: c.get(k) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "FP", "FN")},
                }
            )
    write_csv(TABLES / "n9_quality_cost_curve.csv", qc_rows)
    # event type statistics from proposed B8 (or n8 B0 verified stream)
    ev_counter = Counter()
    acc_counter = Counter()
    for s in seqs:
        d = REAL / f"proposed_b8" / s
        for e in load_jsonl(d / "verified_errors.jsonl"):
            ev_counter[e["event_type"]] += 1
        for e in load_jsonl(d / "interaction_events.jsonl"):
            acc_counter[e["event_type"]] += 1
    ev_rows = [
        {"event_type": t, "count": ev_counter.get(t, 0), "accepted": acc_counter.get(t, 0)}
        for t in sorted(set(ev_counter) | set(acc_counter))
    ]
    write_csv(TABLES / "n9_event_type_statistics.csv", ev_rows)
    # persistence table copy
    src = TABLES / "persistence_metrics.csv"
    if src.exists():
        shutil.copyfile(src, DATA / "n9_persistence_metrics.csv")
    stats_src = EVAL / "stats.json"
    if stats_src.exists():
        shutil.copyfile(stats_src, DATA / "n9_stats.json")
    print(json.dumps({"quality_cost_rows": len(qc_rows), "events": dict(ev_counter)}))


if __name__ == "__main__":
    main()
