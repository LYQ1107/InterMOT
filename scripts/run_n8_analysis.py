#!/usr/bin/env python
"""N8 event-level, cost and statistical analysis over the canonical 25 runs."""

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows
from sam3_intermot.interaction.n8_temporal_observer import EventType


ROOT = Path(".")
REAL = ROOT / "outputs/n8/real"
EVAL = ROOT / "outputs/n8/eval_full25"
TABLES = ROOT / "outputs/n8/tables"
DATA = ROOT / "outputs/n8"
DATASET = DanceTrackDataset("/path/to/dancetrack", split="val")
SEQS = sorted(p.name for p in REAL.glob("route_a_unlimited/*") if (p / "summary.json").exists())
BUDGETS = ["b0", "b1", "b2", "b4", "b8", "unlimited"]
BUDGET_VAL = {"b0": 0, "b1": 1, "b2": 2, "b4": 4, "b8": 8, "unlimited": -1}
COSTED_TYPES = {
    EventType.TRUE_MISS_NEW,
    EventType.RECOVERABLE_MISS,
    EventType.TEMPORAL_ID_BREAK,
    EventType.TEMPORAL_ID_SWAP,
}
NEXT_ERROR_TYPES = {
    EventType.TEMPORAL_ID_BREAK,
    EventType.RECOVERABLE_MISS,
    EventType.TEMPORAL_ID_SWAP,
}


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def best_pid_for_gt(gt_box: np.ndarray, cand: list) -> int:
    """Greedy best-IoU public id for one GT box (diagnostic, not official)."""
    best_iou = 0.5
    best = None
    gx1, gy1, gx2, gy2 = gt_box
    garea = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    for pid, b in cand:
        b = np.asarray(b, dtype=float)
        x1, y1, x2, y2 = b
        ix1, iy1 = max(gx1, x1), max(gy1, y1)
        ix2, iy2 = min(gx2, x2), min(gy2, y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = garea + area - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best = int(pid)
    return best


def load_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> None:
    summaries = {}
    for seq in SEQS:
        for bn in BUDGETS:
            s = REAL / f"route_a_{bn}" / seq / "summary.json"
            if s.exists():
                summaries[(seq, bn)] = json.loads(s.read_text())
    # --- event type statistics (verified stream is budget independent) ---
    event_counter = Counter()
    accepted_counter = Counter()
    for seq in SEQS:
        for e in load_jsonl(REAL / f"route_a_unlimited" / seq / "verified_errors.jsonl"):
            event_counter[e["event_type"]] += 1
        for e in load_jsonl(REAL / f"route_a_unlimited" / seq / "interaction_events.jsonl"):
            accepted_counter[e["event_type"]] += 1
    event_rows = [
        {
            "event_type": t,
            "count": event_counter.get(t, 0),
            "accepted_count": accepted_counter.get(t, 0),
            "budget_consuming": t in COSTED_TYPES,
        }
        for t in [
            EventType.FIRST_APPEARANCE_MATCHED,
            EventType.TRUE_MISS_NEW,
            EventType.RECOVERABLE_MISS,
            EventType.TEMPORAL_ID_BREAK,
            EventType.TEMPORAL_ID_SWAP,
            EventType.LOCALIZATION_ONLY_ERROR,
            EventType.FALSE_POSITIVE,
        ]
    ]
    write_csv(TABLES / "event_type_statistics.csv", event_rows)
    # --- event frame distribution ---
    bin_rows = []
    bins = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100)]
    for seq in SEQS:
        nf = DATASET.num_frames(seq)
        dist = Counter()
        for e in load_jsonl(REAL / f"route_a_unlimited" / seq / "verified_errors.jsonl"):
            frac = 100.0 * (e["frame"] - 1) / max(1, nf)
            for lo, hi in bins:
                if lo <= frac < hi or (hi == 100 and frac >= lo):
                    dist[f"{lo}-{hi}"] += 1
                    break
        bin_rows.append({"sequence": seq, "num_frames": nf, **{f"{lo}-{hi}": dist.get(f"{lo}-{hi}", 0) for lo, hi in bins}})
    write_csv(TABLES / "event_frame_distribution.csv", bin_rows)
    # --- interaction costs per sequence/budget ---
    cost_rows = []
    for seq in SEQS:
        for bn in BUDGETS:
            key = (seq, bn)
            if key not in summaries:
                continue
            s = summaries[key]
            nf = DATASET.num_frames(seq)
            by_type = s["accepted_by_type"]
            row = {
                "sequence": seq,
                "budget": BUDGET_VAL[bn],
                "budget_name": bn,
                "accepted": s["accepted_count"],
                "num_frames": nf,
                "interactions_per_1000_frames": round(1000.0 * s["accepted_count"] / nf, 3),
                **{f"accepted_{t}": by_type.get(t, 0) for t in COSTED_TYPES},
            }
            cost_rows.append(row)
    write_csv(TABLES / "interaction_costs.csv", cost_rows)
    # --- repeated correction (unlimited run, per gid) ---
    repeat_rows = []
    for seq in SEQS:
        cnt = Counter()
        for e in load_jsonl(REAL / f"route_a_unlimited" / seq / "interaction_events.jsonl"):
            cnt[e["dataset_gt_id"]] += 1
        dist = Counter(cnt.values())
        for k in sorted(dist):
            repeat_rows.append({"sequence": seq, "corrections_per_identity": k, "identity_count": dist[k]})
    write_csv(TABLES / "repeated_correction.csv", repeat_rows)
    # --- time to next error (unlimited accepted events) ---
    tte_rows = []
    for seq in SEQS:
        errs = load_jsonl(REAL / f"route_a_unlimited" / seq / "verified_errors.jsonl")
        by_gid = defaultdict(list)
        for e in errs:
            if e["event_type"] in NEXT_ERROR_TYPES and e.get("dataset_gt_id") is not None:
                by_gid[e["dataset_gt_id"]].append(e["frame"])
        for e in load_jsonl(REAL / f"route_a_unlimited" / seq / "interaction_events.jsonl"):
            gid = e.get("dataset_gt_id")
            nxt = [f for f in by_gid.get(gid, []) if f > e["frame"]]
            tte_rows.append(
                {
                    "sequence": seq,
                    "event_frame": e["frame"],
                    "event_type": e["event_type"],
                    "dataset_gt_id": gid,
                    "time_to_next_error": min(nxt) - e["frame"] if nxt else None,
                    "reached_sequence_end": not nxt,
                }
            )
    write_csv(TABLES / "time_to_next_error.csv", tte_rows)
    # --- retention at t/t+1/t+3/t+5/t+10/t+30 (unlimited post stream) ---
    ret_rows = []
    for seq in SEQS:
        gt = DATASET.load_gt(seq)
        post = read_mot_rows(REAL / f"route_a_unlimited" / seq / "post_mot" / f"{seq}.txt")
        frame_cache = {}
        for e in load_jsonl(REAL / f"route_a_unlimited" / seq / "interaction_events.jsonl"):
            gid = e.get("dataset_gt_id")
            canon = e.get("canonical_public_id") or e.get("public_mot_id")
            if gid is None or canon is None:
                continue
            for off in (0, 1, 3, 5, 10, 30):
                f0 = e["frame"] - 1 + off
                cached = frame_cache.get(f0)
                if cached is None:
                    cached = {}
                    gtf = gt.get(f0)
                    cand = post.get(f0, [])
                    if gtf is not None:
                        for gi, gg in enumerate(gtf.gt_ids):
                            cached[gg] = best_pid_for_gt(np.asarray(gtf.boxes[gi], dtype=float), cand)
                    frame_cache[f0] = cached
                gtf = gt.get(f0)
                pid = cached.get(gid)
                matched = pid is not None
                correct = bool(pid is not None and int(pid) == int(canon))
                ret_rows.append(
                    {
                        "sequence": seq,
                        "event_frame": e["frame"],
                        "event_type": e["event_type"],
                        "dataset_gt_id": gid,
                        "offset": off,
                        "target_frame": f0 + 1,
                        "gt_present": bool(gtf is not None and gid in gtf.gt_ids),
                        "matched": matched,
                        "identity_correct": correct,
                    }
                )
    write_csv(TABLES / "retention_metrics.csv", ret_rows)
    # --- accepted counts for curve/marginal ---
    accepted_tot = {bn: sum(summaries[(s, bn)]["accepted_count"] for s in SEQS) for bn in BUDGETS}
    combined = {}
    with (EVAL / "combined_metrics_post.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            combined[r["method"]] = r
    qc_rows = []
    for bn in BUDGETS:
        m = f"route_a_{bn}_post"
        c = combined.get(m, {})
        qc_rows.append(
            {
                "budget": BUDGET_VAL[bn],
                "method": m,
                "accepted_interactions": accepted_tot[bn],
                "interactions_per_sequence": round(accepted_tot[bn] / len(SEQS), 3),
                **{k: c.get(k) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "FP", "FN")},
            }
        )
    write_csv(TABLES / "quality_cost_curve.csv", qc_rows)
    metrics = ["HOTA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]
    order = ["b0", "b1", "b2", "b4", "b8", "unlimited"]
    mg_rows = []
    for lo, hi in zip(order, order[1:]):
        delta = {m: float(combined[f"route_a_{hi}_post"].get(m, 0) or 0) - float(combined[f"route_a_{lo}_post"].get(m, 0) or 0) for m in metrics}
        add = accepted_tot[hi] - accepted_tot[lo]
        mg_rows.append(
            {
                "step": f"{lo}_to_{hi}",
                "additional_interactions": add,
                **{f"delta_{m}": round(delta[m], 4) for m in metrics},
                **{f"gain_per_interaction_{m}": (round(delta[m] / add, 6) if add else None) for m in metrics},
            }
        )
    write_csv(TABLES / "marginal_gain.csv", mg_rows)
    # --- gt audit + invariants ---
    gt_audit = {
        "sequences": len(SEQS),
        "gt_read_current_after_prediction": 0,
        "gt_read_before_prediction": 0,
        "gt_read_future": 0,
        "system_mutation_without_accepted_action": 0,
    }
    inv_rows = []
    for (seq, bn), s in summaries.items():
        ga = s["gt_audit"]
        for k in gt_audit:
            if k in ga:
                gt_audit[k] += int(ga[k])
        for v in s["invariant_violations"]:
            inv_rows.append({"sequence": seq, "budget": bn, "violation": v})
        for v in s["namespace_violations"]:
            inv_rows.append({"sequence": seq, "budget": bn, "violation": f"namespace: {v}"})
    write_csv(TABLES / "invariant_violations.csv", inv_rows)
    (TABLES / "gt_access_audit.json").write_text(
        json.dumps(gt_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # --- statistical tests (from eval) ---
    stats = json.loads((EVAL / "stats.json").read_text())
    (TABLES / "statistical_tests.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # --- required root-level copies ---
    for name in [
        "combined_metrics_pre.csv",
        "combined_metrics_post.csv",
        "per_sequence_metrics_pre.csv",
        "per_sequence_metrics_post.csv",
    ]:
        src = EVAL / name
        if src.exists():
            shutil.copyfile(src, DATA / name)
    shutil.copyfile(TABLES / "event_type_statistics.csv", DATA / "event_type_statistics.csv")
    shutil.copyfile(TABLES / "event_frame_distribution.csv", DATA / "event_frame_distribution.csv")
    shutil.copyfile(TABLES / "interaction_costs.csv", DATA / "interaction_costs.csv")
    shutil.copyfile(TABLES / "retention_metrics.csv", DATA / "retention_metrics.csv")
    shutil.copyfile(TABLES / "time_to_next_error.csv", DATA / "time_to_next_error.csv")
    shutil.copyfile(TABLES / "repeated_correction.csv", DATA / "repeated_correction.csv")
    shutil.copyfile(TABLES / "quality_cost_curve.csv", DATA / "quality_cost_curve.csv")
    shutil.copyfile(TABLES / "marginal_gain.csv", DATA / "marginal_gain.csv")
    shutil.copyfile(TABLES / "statistical_tests.json", DATA / "statistical_tests.json")
    shutil.copyfile(TABLES / "gt_access_audit.json", DATA / "gt_access_audit.json")
    shutil.copyfile(TABLES / "invariant_violations.csv", DATA / "invariant_violations.csv")
    summary = {
        "sequences": len(SEQS),
        "event_counts": dict(event_counter),
        "accepted_counts": dict(accepted_counter),
        "accepted_total_unlimited": sum(accepted_counter.values()),
        "accepted_by_budget": accepted_tot,
        "first_error_frame_stats": {
            "min": None,
            "median": None,
        },
    }
    first_errs = []
    for seq in SEQS:
        errs = load_jsonl(REAL / f"route_a_unlimited" / seq / "verified_errors.jsonl")
        costed = [e for e in errs if e["event_type"] in COSTED_TYPES]
        if costed:
            first_errs.append(min(e["frame"] for e in costed))
    if first_errs:
        summary["first_error_frame_stats"] = {
            "min": min(first_errs),
            "median": float(np.median(first_errs)),
            "mean": float(np.mean(first_errs)),
        }
    (TABLES / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
