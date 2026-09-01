#!/usr/bin/env python
"""N18 Phase-II FULL_LOOP_V0 analysis and failure accounting.

Reads the per-shard trace/transaction/metrics files produced by
run_n18_full_loop_v0.py and produces the tables required by the N18 prompt
(sections 22-27, 112), including the F1-F12 failure taxonomy.

GT is only used for offline evaluation of the already-recorded causal
decisions; it is never fed back into the loop.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n18"
DT = Path("/path/to/dancetrack")


def iou(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_gt(seq):
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    return DanceTrackDataset(str(DT), sequences=[], split="train").load_gt(seq)


def read_jsonl(pattern, modes=("full", "human", "gfn")):
    rows = []
    for mode in modes:
        for p in sorted(OUT.glob(pattern.format(mode=mode))):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def group_trace(rows):
    by_seq = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_seq[r["sequence"]][r["gid"]].append(r)
    for seq, g in by_seq.items():
        for gid in g:
            g[gid].sort(key=lambda r: r["frame"])
    return by_seq


def gt_box(gt, f, gid):
    gf = gt.get(f)
    if gf is None or gid not in gf.gt_ids:
        return None
    return np.asarray(gf.boxes[gf.gt_ids.index(gid)], dtype=float)


def load_p0(seq):
    from dataclasses import dataclass
    p = ROOT / "outputs/n9/p0_train" / f"{seq}.txt"
    rows = defaultdict(list)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 7:
            continue
        f0 = int(float(parts[0])) - 1
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]),
                      float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        box = np.asarray([x, y, x + w, y + h], dtype=float)
        rows[f0].append(box)
    return rows


def auto_coverage(seq):
    """Fraction of GT-present identity-frames covered by a frozen P0 row."""
    gt = load_gt(seq)
    p0 = load_p0(seq)
    n_frames = max(gt) + 1 if gt else 0
    den = hit = 0
    for f in range(n_frames):
        gf = gt.get(f)
        if gf is None:
            continue
        rows = p0.get(f, [])
        for gbox in gf.boxes:
            den += 1
            if any(iou(r, gbox) >= 0.5 for r in rows):
                hit += 1
    return hit / den if den else None


def analyze_sequence(seq, gt, trace, tx):
    """Return a dict of per-sequence diagnostics and per-episode records."""
    by_frame = defaultdict(dict)
    for r in trace:
        by_frame[r["frame"]][r["gid"]] = r
    n_frames = max(by_frame) + 1 if by_frame else 0
    gids = sorted({r["gid"] for r in trace})

    # ---- lost trigger episodes
    lost_rows = []
    for gid in gids:
        evs = sorted([by_frame[f][gid] for f in range(n_frames)
                      if gid in by_frame[f]], key=lambda e: e["frame"])
        prev_state = None
        ep_start = None
        for e in evs:
            f = e["frame"]
            if e["state"] == "LOST" and prev_state != "LOST":
                ep_start = f
            if ep_start is not None and e["state"] != "LOST":
                f_ep = ep_start
                ep_start = None
                r0 = by_frame[f_ep][gid]
                false_trigger = bool(r0["gt_present"] and r0["correct"])
                delay = None
                if not false_trigger:
                    for f2 in range(f_ep, -1, -1):
                        r2 = by_frame.get(f2, {}).get(gid)
                        if r2 is None:
                            break
                        if r2["gt_present"] and r2["correct"]:
                            break
                        delay = f_ep - f2
                lost_rows.append({
                    "sequence": seq, "gid": gid, "frame": f_ep,
                    "false_trigger": int(false_trigger), "delay": delay,
                })
            prev_state = e["state"]
        if ep_start is not None:
            r0 = by_frame[ep_start][gid]
            lost_rows.append({
                "sequence": seq, "gid": gid, "frame": ep_start,
                "false_trigger": int(bool(r0["gt_present"] and r0["correct"])),
                "delay": None,
            })
    # missed-lost events: a >=3-frame true-loss streak with no LOST state
    missed_lost = 0
    true_lost_streaks = 0
    for gid in gids:
        streak = 0
        triggered = False
        for f in range(n_frames):
            e = by_frame[f].get(gid)
            truly_lost = e is not None and not e["gt_present"] \
                and not e["correct"]
            if truly_lost:
                streak += 1
                if e["state"] == "LOST":
                    triggered = True
            else:
                if streak >= 3:
                    true_lost_streaks += 1
                    if not triggered:
                        missed_lost += 1
                streak = 0
                triggered = False
        if streak >= 3:
            true_lost_streaks += 1
            if not triggered:
                missed_lost += 1

    # ---- recovery / verifier / reactivation
    rec_rows = []
    for t in tx:
        f = t["frame"]
        gid = t["gid"]
        target = gt_box(gt, f, gid)
        box = t["recovery_box"]
        rec_iou = iou(box, target) if (box is not None and target is not None) \
            else None
        gt_present = target is not None
        rec_correct = rec_iou is not None and rec_iou >= 0.5
        rec_rows.append({
            **{k: t[k] for k in ("frame", "gid", "public_id", "verifier_score",
                                 "accepted", "reactivated", "state_after")},
            "sequence": seq, "gt_present": int(gt_present),
            "recovered_iou": None if rec_iou is None else round(rec_iou, 4),
            "rec_correct": int(rec_correct),
        })

    # ---- retention / re-correction horizons
    by_gid_frame = {}
    for gid in gids:
        by_gid_frame[gid] = {e["frame"]: e for e in trace if e["gid"] == gid}
    horizons = (1, 3, 5, 10, 30, 60, 120)
    retention = {}
    react_ts = [t for t in tx if t.get("reactivated")]
    for h in horizons:
        n = hit = 0
        for t in react_ts:
            e = by_gid_frame[t["gid"]].get(t["frame"] + h)
            if e is not None and e["gt_present"]:
                n += 1
                hit += e["correct"]
        retention[h] = hit / n if n else None
    recorr = {}
    for gid in gids:
        evs = sorted(by_gid_frame[gid].values(), key=lambda e: e["frame"])
        if not evs:
            continue
        base = evs[0]["frame"]
        for h in horizons:
            n = hit = 0
            for e in evs:
                if e["frame"] >= base + h and e["gt_present"]:
                    n += 1
                    hit += (not e["correct"])
            recorr.setdefault(gid, {})[h] = hit / n if n else None
    return {
        "lost_rows": lost_rows, "rec_rows": rec_rows,
        "retention": retention, "recorr": recorr,
        "missed_lost": missed_lost, "true_lost_streaks": true_lost_streaks,
        "n_frames": n_frames, "gids": gids,
    }


def failure_taxonomy(seq, gt, trace, tx, per_seq):
    """Classify every recovery episode into F1-F12 (multi-label per episode)."""
    by_frame = defaultdict(dict)
    for r in trace:
        by_frame[r["frame"]][r["gid"]] = r
    rows = []
    react_ts = [t for t in tx if t.get("reactivated")]
    for t in tx:
        f = t["frame"]
        gid = t["gid"]
        target = gt_box(gt, f, gid)
        box = t["recovery_box"]
        rec_iou = iou(box, target) if (box is not None and target is not None) \
            else None
        rec_correct = int(rec_iou is not None and rec_iou >= 0.5)
        labels = []
        if target is not None and rec_correct:
            # verifier
            if not t["accepted"]:
                labels.append("F5_verifier_false_reject")
        else:
            if t["accepted"]:
                labels.append("F6_verifier_false_accept")
            if target is not None:
                labels.append("F3_F4_candidate_wrong")
        if t["accepted"] and not t["reactivated"]:
            labels.append("F7_reactivation_init_fail")
        if t["reactivated"]:
            e1 = by_frame.get(f + 1, {}).get(gid)
            if e1 is not None and e1["gt_present"] and not e1["correct"]:
                labels.append("F8_immediate_propagation_fail")
            e30 = by_frame.get(f + 30, {}).get(gid)
            if e1 is not None and e1["gt_present"] and e1["correct"] \
                    and e30 is not None and e30["gt_present"] \
                    and not e30["correct"]:
                labels.append("F9_longterm_drift")
        rows.append({"sequence": seq, "frame": f, "gid": gid,
                     "labels": ";".join(labels) if labels else "none"})
    # F1/F2 from lost trigger rows
    for r in per_seq["lost_rows"]:
        if r["false_trigger"]:
            rows.append({"sequence": seq, "frame": r["frame"], "gid": r["gid"],
                         "labels": "F2_lost_trigger_false_positive"})
        elif r["delay"] is not None and r["delay"] > 10:
            rows.append({"sequence": seq, "frame": r["frame"], "gid": r["gid"],
                         "labels": "F1_lost_trigger_late"})
    # F10 duplicate public-id boxes
    dup_frames = [r["frame"] for r in trace if r.get("duplicate_flag")]
    for f in sorted(set(dup_frames)):
        rows.append({"sequence": seq, "frame": f, "gid": -1,
                     "labels": "F10_public_id_conflict"})
    # F11 collateral: non-target correctness around accepted reactivations
    for t in react_ts:
        f = t["frame"]
        gid = t["gid"]
        others = [g for g in per_seq["gids"] if g != gid]
        before = after = 0
        for o in others:
            for df in range(-5, 0):
                e = by_frame.get(f + df, {}).get(o)
                if e is not None and e["gt_present"]:
                    before += (not e["correct"])
            for df in range(1, 6):
                e = by_frame.get(f + df, {}).get(o)
                if e is not None and e["gt_present"]:
                    after += (not e["correct"])
        if after > before:
            rows.append({"sequence": seq, "frame": f, "gid": gid,
                         "labels": "F11_non_target_collateral"})
    # F12 oscillation: rapid repeated recoveries per identity
    per_gid = defaultdict(list)
    for t in tx:
        per_gid[t["gid"]].append(t["frame"])
    for gid, fs in per_gid.items():
        if len(fs) >= 3 and max(fs) - min(fs) <= 60:
            rows.append({"sequence": seq, "frame": max(fs), "gid": gid,
                         "labels": "F12_repeat_recovery_oscillation"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="")
    ap.add_argument("--modes", default="full,human,gfn")
    args = ap.parse_args()
    modes = tuple(m for m in args.modes.split(",") if m)
    seqs = args.seqs.split(",") if args.seqs else json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text()
    )["split"]["calibration10"]

    out_tables = OUT / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    all_tax = []
    all_lost = []
    all_rec = []
    all_summary = []
    all_ret = []
    all_recorr = []
    auto_rows = []
    for seq in seqs:
        cov = auto_coverage(seq)
        auto_rows.append({"mode": "auto", "sequence": seq,
                          "gt_coverage": cov})
    for mode in modes:
        mode_trace = read_jsonl(f"full_loop_v0_events_{{mode}}_s*.jsonl",
                                modes=(mode,))
        mode_tx = read_jsonl(
            f"reactivation_transactions_{{mode}}_s*.jsonl", modes=(mode,))
        if not mode_trace:
            print(f"[skip] mode={mode}: no trace files")
            continue
        by_seq = group_trace(mode_trace)
        tx_by_seq = defaultdict(list)
        for t in mode_tx:
            tx_by_seq[t["sequence"]].append(t)
        for seq in seqs:
            if seq not in by_seq:
                print(f"[warn] {mode} {seq}: no trace")
                continue
            gt = load_gt(seq)
            seq_trace = [r for rows in by_seq[seq].values() for r in rows]
            per_seq = analyze_sequence(seq, gt, seq_trace, tx_by_seq.get(seq, []))
            all_lost += [{"mode": mode, **r} for r in per_seq["lost_rows"]]
            all_rec += [{"mode": mode, **r} for r in per_seq["rec_rows"]]
            for h, v in per_seq["retention"].items():
                all_ret.append({"mode": mode, "sequence": seq,
                                "horizon": h, "retention": v})
            for gid, hs in per_seq["recorr"].items():
                for h, v in hs.items():
                    all_recorr.append({"mode": mode, "sequence": seq,
                                       "gid": gid, "horizon": h,
                                       "recorrection_prob": v})
            tax = failure_taxonomy(seq, gt, seq_trace,
                                   tx_by_seq.get(seq, []), per_seq)
            all_tax += [{"mode": mode, **r} for r in tax]
            # per-sequence summary from metrics CSV
            mrows = []
            for p in sorted(OUT.glob(f"full_loop_v0_metrics_{mode}_s*.csv")):
                with p.open(encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row["sequence"] == seq:
                            mrows.append(row)
            if mrows:
                all_summary.append({"mode": mode, **mrows[-1]})
    all_summary += auto_rows

    def write_csv(name, rows, fields):
        p = out_tables / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p} ({len(rows)} rows)")

    write_csv("full_loop_v0_summary.csv", all_summary, [
        "mode", "sequence", "n_identities", "frames", "recovery_attempts",
        "accepted_recoveries", "verifier_accepts", "accept_rate",
        "lost_episodes", "mean_recorrection_prob", "retention_1",
        "retention_3", "retention_5", "retention_10", "retention_30",
        "retention_60", "retention_120", "runtime_s", "gt_coverage"])
    write_csv("lost_trigger_v0.csv", all_lost, [
        "mode", "sequence", "gid", "frame", "false_trigger", "delay"])
    write_csv("recovery_attempts.csv", all_rec, [
        "mode", "sequence", "frame", "gid", "public_id", "verifier_score",
        "accepted", "reactivated", "state_after", "gt_present",
        "recovered_iou", "rec_correct"])
    write_csv("post_reactivation_retention.csv", all_ret, [
        "mode", "sequence", "horizon", "retention"])
    write_csv("recorrection.csv", all_recorr, [
        "mode", "sequence", "gid", "horizon", "recorrection_prob"])
    write_csv("full_loop_failure_accounting.csv", all_tax, [
        "mode", "sequence", "frame", "gid", "labels"])

    # ---- mode-level aggregate table
    agg_rows = []
    for mode in modes:
        if not any(r["mode"] == mode for r in all_summary):
            continue
        mr = [r for r in all_summary if r["mode"] == mode]
        def num(f):
            vals = [float(r[f]) for r in mr if r.get(f) not in (None, "", "null")]
            return np.mean(vals) if vals else None
        rec = [r for r in all_rec if r["mode"] == mode]
        lost = [r for r in all_lost if r["mode"] == mode]
        attempts = len(rec)
        accepts = sum(r["accepted"] for r in rec)
        react = sum(r["reactivated"] for r in rec)
        fp = sum(r["false_trigger"] for r in lost)
        correct_rec = sum(r["rec_correct"] for r in rec)
        agg_rows.append({
            "mode": mode, "sequences": len(mr),
            "recovery_attempts": attempts,
            "verifier_accepts": accepts,
            "reactivations": react,
            "recovery_precision": accepts / attempts if attempts else None,
            "false_triggers": fp,
            "correct_recovery_boxes": correct_rec,
            "mean_recorrection_prob": num("mean_recorrection_prob"),
            "mean_gt_coverage": num("gt_coverage"),
            "retention_1": num("retention_1"),
            "retention_3": num("retention_3"),
            "retention_5": num("retention_5"),
            "retention_10": num("retention_10"),
            "retention_30": num("retention_30"),
            "retention_60": num("retention_60"),
            "retention_120": num("retention_120"),
            "mean_runtime_s": num("runtime_s"),
        })
    write_csv("full_loop_mode_comparison.csv", agg_rows, [
        "mode", "sequences", "recovery_attempts", "verifier_accepts",
        "reactivations", "recovery_precision", "false_triggers",
        "correct_recovery_boxes", "mean_recorrection_prob",
        "mean_gt_coverage", "retention_1",
        "retention_3", "retention_5", "retention_10", "retention_30",
        "retention_60", "retention_120", "mean_runtime_s"])

    # ---- taxonomy counts
    c = Counter()
    full_mode = next((m for m in modes if m.startswith("full")), "full")
    for r in all_tax:
        if r["mode"] != full_mode:
            continue
        for lab in r["labels"].split(";"):
            if lab != "none":
                c[lab] += 1
    tax_rows = [{"failure": k, "count": v} for k, v in sorted(c.items())]
    write_csv("failure_taxonomy_counts.csv", tax_rows, ["failure", "count"])


if __name__ == "__main__":
    main()
