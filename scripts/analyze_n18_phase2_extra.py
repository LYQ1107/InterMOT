#!/usr/bin/env python
"""N18 Phase-II supplemental tables from the frozen V0 loop trace.

Produces:
  collateral_analysis.csv
  public_id_continuity.csv
  verifier_online.csv
  recovery_precision_recall.csv
  lost_trigger_precision_recall.csv
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n18"


def read_jsonl(pattern, modes):
    rows = []
    for mode in modes:
        for p in sorted(OUT.glob(pattern.format(mode=mode))):
            for line in p.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def main():
    modes = ("full", "human", "gfn")
    trace = read_jsonl("full_loop_v0_events_{mode}_s*.jsonl", modes)
    tx = read_jsonl("reactivation_transactions_{mode}_s*.jsonl", modes)

    # --- public id continuity
    cont_rows = []
    by_seq_gid = defaultdict(lambda: defaultdict(list))
    for r in trace:
        by_seq_gid[r["sequence"]][r["gid"]].append(r)
    for seq, g in by_seq_gid.items():
        for gid, evs in g.items():
            evs.sort(key=lambda e: e["frame"])
            pubs = sorted({e["public_id"] for e in evs})
            states = sorted({e["state"] for e in evs})
            dup_frames = sum(e.get("duplicate_flag", 0) for e in evs)
            cont_rows.append({
                "sequence": seq, "gid": gid,
                "public_ids": ";".join(str(x) for x in pubs),
                "n_public_ids": len(pubs),
                "first_frame": evs[0]["frame"], "last_frame": evs[-1]["frame"],
                "frames": len(evs),
                "states_seen": ";".join(sorted(states)),
                "duplicate_flag_frames": dup_frames,
            })

    # --- verifier online: all full-mode attempts with recovered IoU
    ver_rows = []
    for t in tx:
        if t.get("accepted") is None:
            continue
        ver_rows.append({
            "sequence": t["sequence"], "frame": t["frame"], "gid": t["gid"],
            "public_id": t.get("public_id"),
            "anchor_frame": t.get("anchor_frame"),
            "verifier_score": t.get("verifier_score"),
            "accepted": int(bool(t.get("accepted"))),
            "reactivated": int(bool(t.get("reactivated"))),
            "state_after": t.get("state_after"),
            "recovery_box": json.dumps(t.get("recovery_box")),
        })

    # --- recovery precision/recall by threshold (full mode)
    pr_rows = []
    full_ver = [t for t in ver_rows if True]
    # verifier_online rows do not carry mode; recompute from full-mode files
    full_ver = []
    for t in read_jsonl("reactivation_transactions_{mode}_s*.jsonl",
                        ("full",)):
        if t.get("accepted") is None:
            continue
        full_ver.append({
            "sequence": t["sequence"], "frame": t["frame"], "gid": t["gid"],
            "verifier_score": t.get("verifier_score"),
            "recovery_box": t.get("recovery_box"),
        })
    for th in (0.4, 0.5, 0.6, 0.7, 0.8):
        sel = [t for t in full_ver if t["verifier_score"] is not None
               and t["verifier_score"] >= th]
        n_accept = len(sel)
        # recompute candidate correctness with GT
        n_correct = 0
        from sam3_intermot.datasets.dancetrack import DanceTrackDataset
        dt = DanceTrackDataset(
            str(Path("/path/to/dancetrack")),
            sequences=[], split="train")
        for t in sel:
            gt = dt.load_gt(t["sequence"]).get(t["frame"])
            if gt is not None and t["gid"] in gt.gt_ids:
                gbox = np.asarray(
                    gt.boxes[gt.gt_ids.index(t["gid"])], dtype=float)
                rbox = t["recovery_box"] if t["recovery_box"] else None
                if rbox is not None and iou(rbox, gbox) >= 0.5:
                    n_correct += 1
        pr_rows.append({
            "threshold": th, "accepts": n_accept,
            "correct_accepts": n_correct,
            "precision": n_correct / n_accept if n_accept else None,
        })

    # --- collateral analysis around full-mode accepted reactivations
    full_trace = [r for r in trace if r["mode"] == "full"] if False else trace
    # trace rows do not carry mode; rebuild by grouping the per-mode files
    full_trace = read_jsonl("full_loop_v0_events_{mode}_s*.jsonl", ("full",))
    by_frame = defaultdict(lambda: defaultdict(dict))
    for r in full_trace:
        by_frame[r["sequence"]][r["frame"]][r["gid"]] = r
    coll_rows = []
    for t in read_jsonl("reactivation_transactions_{mode}_s*.jsonl",
                        ("full",)):
        if not t.get("reactivated"):
            continue
        seq, f, gid = t["sequence"], t["frame"], t["gid"]
        others = [g for g in by_frame[seq][f] if g != gid]
        before = after = 0
        b_present = a_present = 0
        for o in others:
            for df in range(-5, 0):
                e = by_frame[seq].get(f + df, {}).get(o)
                if e is not None and e["gt_present"]:
                    b_present += 1
                    before += (not e["correct"])
            for df in range(1, 6):
                e = by_frame[seq].get(f + df, {}).get(o)
                if e is not None and e["gt_present"]:
                    a_present += 1
                    after += (not e["correct"])
        coll_rows.append({
            "sequence": seq, "frame": f, "gid": gid,
            "other_ids": len(others), "before_present": b_present,
            "before_errors": before, "after_present": a_present,
            "after_errors": after, "error_delta": after - before,
        })

    # --- lost trigger precision/recall (full mode)
    lost_rows = []
    by_seq = defaultdict(lambda: defaultdict(list))
    idx = {}
    for r in full_trace:
        by_seq[r["sequence"]][r["gid"]].append(r)
        idx[(r["sequence"], r["frame"], r["gid"])] = r
    for seq, g in by_seq.items():
        for gid, evs in g.items():
            evs.sort(key=lambda e: e["frame"])
            prev = None
            start = None
            for e in evs:
                if e["state"] == "LOST" and prev != "LOST":
                    start = e["frame"]
                if start is not None and e["state"] != "LOST":
                    lost_rows.append((seq, gid, start, e["frame"]))
                    start = None
                prev = e["state"]
            if start is not None:
                lost_rows.append((seq, gid, start, evs[-1]["frame"]))
    lpr_rows = []
    for seq, gid, s, e in lost_rows:
        fp = tp = 0
        for f in range(s, e + 1):
            row = idx.get((seq, f, gid))
            if row is not None and row["gt_present"] and row["correct"]:
                fp += 1
            elif row is not None and row["gt_present"]:
                tp += 1
        lpr_rows.append({
            "sequence": seq, "gid": gid, "start": s, "end": e,
            "fp_frames": fp, "tp_frames": tp,
        })

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    def write(name, rows, fields):
        p = tables / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p} ({len(rows)} rows)")

    write("public_id_continuity.csv", cont_rows, [
        "sequence", "gid", "public_ids", "n_public_ids", "first_frame",
        "last_frame", "frames", "states_seen", "duplicate_flag_frames"])
    write("verifier_online.csv", ver_rows, [
        "sequence", "frame", "gid", "public_id", "anchor_frame",
        "verifier_score", "accepted", "reactivated", "state_after",
        "recovery_box"])
    write("recovery_precision_recall.csv", pr_rows, [
        "threshold", "accepts", "correct_accepts", "precision"])
    write("collateral_analysis.csv", coll_rows, [
        "sequence", "frame", "gid", "other_ids", "before_present",
        "before_errors", "after_present", "after_errors", "error_delta"])
    write("lost_trigger_precision_recall.csv", lpr_rows, [
        "sequence", "gid", "start", "end", "fp_frames", "tp_frames"])


if __name__ == "__main__":
    main()
