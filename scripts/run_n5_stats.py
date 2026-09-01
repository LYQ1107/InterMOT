#!/usr/bin/env python
"""N5-6 aggregate metrics, costs, retention and statistics."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes
from sam3_intermot.tracking.association import box_iou


ROOT = Path(".")
OUT = ROOT / "outputs/n5"
EVAL_ROOT = OUT / "tmp_trackeval_full25"
if not EVAL_ROOT.exists():
    EVAL_ROOT = OUT / "tmp_trackeval"
GT_ROOT = Path("/path/to/dancetrack/val")
DS = DanceTrackDataset(str(Path("/path/to/dancetrack")), split="val")

PROTOCOLS = {
    "p0": {"dir": OUT / "p0_auto", "budget": 0},
    "p1": {"dir": OUT / "p1_oracle_frame_all", "budget": 0},
    "p2": {"dir": OUT / "p2_oracle_state_all", "budget": 0},
    "p3": {"dir": OUT / "p3_continuous_id_miss", "budget": 0},
    "p4_b1": {"dir": OUT / "p4_budget_b1", "budget": 1},
    "p4_b2": {"dir": OUT / "p4_budget_b2", "budget": 2},
    "p4_b4": {"dir": OUT / "p4_budget_b4", "budget": 4},
    "p4_b8": {"dir": OUT / "p4_budget_b8", "budget": 8},
}


def parse_log(log: Path, tracker: str):
    text = log.read_text(encoding="utf-8").splitlines()
    section = None
    combined = {}
    per_seq = {}
    for line in text:
        if line.startswith(f"HOTA: {tracker}-pedestrian"):
            section = "HOTA"
            continue
        if line.startswith(f"CLEAR: {tracker}-pedestrian"):
            section = "CLEAR"
            continue
        if line.startswith(f"Identity: {tracker}-pedestrian"):
            section = "ID"
            continue
        if section is None or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[0] == "COMBINED":
            if section == "HOTA":
                combined = {
                    "HOTA": float(parts[1]),
                    "DetA": float(parts[2]),
                    "AssA": float(parts[3]),
                    "LocA": float(parts[8]),
                }
            elif section == "CLEAR":
                combined.update(
                    {
                        "MOTA": float(parts[1]),
                        "MOTP": float(parts[2]),
                        "IDSW": float(parts[13]),
                        "MT": float(parts[14]),
                        "PT": float(parts[15]),
                        "ML": float(parts[16]),
                        "Frag": float(parts[17]),
                        "FP": float(parts[11]),
                        "FN": float(parts[10]),
                    }
                )
            elif section == "ID":
                combined.update({"IDF1": float(parts[1]), "IDR": float(parts[2]), "IDP": float(parts[3])})
            section = None
            continue
        seq = parts[0]
        if section == "HOTA":
            per_seq[seq] = {"HOTA": float(parts[1]), "DetA": float(parts[2]), "AssA": float(parts[3])}
        elif section == "CLEAR":
            d = per_seq.setdefault(seq, {})
            d.update(
                {
                    "MOTA": float(parts[1]),
                    "IDSW": float(parts[13]),
                    "Frag": float(parts[16]),
                    "FP": float(parts[11]),
                    "FN": float(parts[10]),
                }
            )
        elif section == "ID":
            d = per_seq.setdefault(seq, {})
            d["IDF1"] = float(parts[1])
    return combined, per_seq


def read_mot(path: Path):
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split(",")
        f = int(float(p[0])) - 1
        tid = int(float(p[1]))
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        rows.setdefault(f, []).append((tid, np.asarray([x, y, x + w, y + h], float)))
    return rows


def events_for(proto: str, seq: str):
    p = PROTOCOLS[proto]["dir"] / seq / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    seqs = sorted(
        p.name for p in GT_ROOT.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").is_file()
    )
    combined_rows = []
    per_seq_rows = []
    interaction_events = []
    costs = []
    primitive = {}
    budget_metrics = {}

    for proto, cfg in PROTOCOLS.items():
        log = EVAL_ROOT / f"{proto}.log"
        if not log.exists():
            continue
        for stream in ("pre_mot", "post_mot"):
            combined, per_seq = parse_log(log, stream)
            row = {"protocol": proto, "budget": cfg["budget"], "stream": stream, **combined}
            combined_rows.append(row)
            for seq, m in per_seq.items():
                per_seq_rows.append(
                    {"protocol": proto, "budget": cfg["budget"], "stream": stream, "sequence": seq, **m}
                )
        for seq in seqs:
            events = events_for(proto, seq)
            for e in events:
                interaction_events.append(e)
            n = len(events)
            accepted = sum(1 for e in events)
            by_type = {}
            for e in events:
                by_type[e["action_type"]] = by_type.get(e["action_type"], 0) + 1
            num_frames = DS.num_frames(seq)
            fps = 20
            video_minutes = num_frames / fps / 60.0
            costs.append(
                {
                    "protocol": proto,
                    "budget": cfg["budget"],
                    "sequence": seq,
                    "NoI_total": n,
                    "NoI_accepted": accepted,
                    "NoAddNew": by_type.get("ADD_NEW_IDENTITY", 0),
                    "NoRecover": by_type.get("RECOVER_IDENTITY", 0),
                    "NoReassign": by_type.get("AUTHORITATIVE_REASSIGN", 0),
                    "NoSwap": by_type.get("ATOMIC_ID_SWAP", 0),
                    "NoCorrect": by_type.get("AUTHORITATIVE_CORRECT", 0),
                    "NoDelete": by_type.get("AUTHORITATIVE_DELETE", 0),
                    "interactions_per_sequence": n,
                    "interactions_per_1000_frames": 1000.0 * n / num_frames,
                    "interactions_per_video_minute": n / video_minutes if video_minutes else 0.0,
                    "human_seconds": "NOT_MEASURED",
                    "mean_interaction_interval_frames": 0.0,
                    "median_interaction_interval_frames": 0.0,
                    "mean_interaction_interval_video_seconds": 0.0,
                }
            )
            if proto == "p3":
                frames = sorted(e["frame"] for e in events)
                intervals = np.diff(frames) if len(frames) > 1 else np.array([0])
                costs[-1]["mean_interaction_interval_frames"] = float(np.mean(intervals))
                costs[-1]["median_interaction_interval_frames"] = float(np.median(intervals))
                costs[-1]["mean_interaction_interval_video_seconds"] = float(np.mean(intervals)) / fps

    # primitive costs
    prim = {"number_of_box_draws": 0, "number_of_track_selections": 0,
            "number_of_identity_selections": 0, "number_of_pair_selections": 0,
            "number_of_delete_clicks": 0, "number_of_confirmations": 0}
    for e in interaction_events:
        t = e["action_type"]
        if t == "ADD_NEW_IDENTITY":
            prim["number_of_box_draws"] += 1
            prim["number_of_confirmations"] += 1
        elif t == "RECOVER_IDENTITY":
            prim["number_of_box_draws"] += 1
            prim["number_of_identity_selections"] += 1
        elif t == "AUTHORITATIVE_REASSIGN":
            prim["number_of_track_selections"] += 1
            prim["number_of_identity_selections"] += 1
        elif t == "ATOMIC_ID_SWAP":
            prim["number_of_track_selections"] += 2
            prim["number_of_confirmations"] += 1
        elif t == "AUTHORITATIVE_CORRECT":
            prim["number_of_box_draws"] += 1
        elif t == "AUTHORITATIVE_DELETE":
            prim["number_of_track_selections"] += 1
            prim["number_of_delete_clicks"] += 1
    primitive_rows = [{"primitive": k, "count": v} for k, v in prim.items()]

    # quality-cost curve
    qc = []
    for proto in ("p0", "p4_b1", "p4_b2", "p4_b4", "p4_b8", "p3"):
        row = next((r for r in combined_rows if r["protocol"] == proto and r["stream"] == "post_mot"), None)
        if row is None and proto == "p0":
            row = next((r for r in combined_rows if r["protocol"] == "p0" and r["stream"] == "pre_mot"), None)
        if row is None:
            continue
        avg_noi = np.mean([c["NoI_total"] for c in costs if c["protocol"] == proto]) if costs else 0
        qc.append(
            {
                "protocol": proto,
                "accepted_interactions_per_sequence": float(avg_noi),
                "HOTA": row.get("HOTA"),
                "DetA": row.get("DetA"),
                "AssA": row.get("AssA"),
                "MOTA": row.get("MOTA"),
                "IDF1": row.get("IDF1"),
                "IDSW": row.get("IDSW"),
                "Frag": row.get("Frag"),
            }
        )
        if proto.startswith("p4") or proto == "p3":
            budget_metrics[proto] = row

    # AUC (normalized trapezoid over accepted interactions per sequence)
    auc = {}
    for metric in ("HOTA", "AssA", "IDF1"):
        xs = [c["accepted_interactions_per_sequence"] for c in qc]
        ys = [c.get(metric) or 0.0 for c in qc]
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        if len(xs) < 2 or xs[-1] <= xs[0]:
            auc[metric] = 0.0
            continue
        auc[metric] = float(np.trapz(ys, xs) / (xs[-1] - xs[0]))

    # statistical tests (paired, sequence-level)
    pairs = [
        ("P2_pre_vs_P1_pre", "p2", "pre_mot", "p1", "pre_mot"),
        ("P3_pre_vs_P0", "p3", "pre_mot", "p0", "pre_mot"),
        ("P3_post_vs_P0", "p3", "post_mot", "p0", "post_mot"),
    ]
    for b in (1, 2, 4, 8):
        pairs.append((f"B{b}_pre_vs_P0", f"p4_b{b}", "pre_mot", "p0", "pre_mot"))
        pairs.append((f"B{b}_post_vs_P0", f"p4_b{b}", "post_mot", "p0", "post_mot"))
    try:
        from scipy import stats
    except ImportError:
        stats = None
    stat_results = {}
    for name, pa, sa, pb, sb in pairs:
        amap = next((r for r in per_seq_rows if r["protocol"] == pa and r["stream"] == sa), None)
        bmap = next((r for r in per_seq_rows if r["protocol"] == pb and r["stream"] == sb), None)
        if amap is None or bmap is None:
            continue
        a_rows = [r for r in per_seq_rows if r["protocol"] == pa and r["stream"] == sa]
        b_rows = [r for r in per_seq_rows if r["protocol"] == pb and r["stream"] == sb]
        res = {}
        for metric in ("HOTA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"):
            da = {r["sequence"]: r.get(metric) for r in a_rows}
            db = {r["sequence"]: r.get(metric) for r in b_rows}
            common = sorted(set(da) & set(db))
            x = np.array([da[s] for s in common], dtype=float)
            y = np.array([db[s] for s in common], dtype=float)
            delta = y - x
            rng = np.random.default_rng(42)
            boot = []
            for _ in range(2000):
                idx = rng.integers(0, len(delta), len(delta))
                boot.append(delta[idx].mean())
            boot = np.asarray(boot)
            p = float(stats.wilcoxon(delta).pvalue) if stats is not None and np.any(delta) else 1.0
            sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
            effect = float(np.mean(delta) / sd) if sd > 0 else 0.0
            res[metric] = {
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "wilcoxon_p": p,
                "effect_size": effect,
                "improved": int(np.sum(delta > 1e-9)),
                "degraded": int(np.sum(delta < -1e-9)),
                "unchanged": int(np.sum(np.abs(delta) <= 1e-9)),
                "n_sequences": len(common),
            }
        stat_results[name] = res

    OUT.mkdir(parents=True, exist_ok=True)

    def write_csv(name, rows):
        if not rows:
            return
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv("combined_metrics_pre.csv",
              [r for r in combined_rows if r["stream"] == "pre_mot"])
    write_csv("combined_metrics_post.csv",
              [r for r in combined_rows if r["stream"] == "post_mot"])
    write_csv("per_sequence_metrics_pre.csv",
              [r for r in per_seq_rows if r["stream"] == "pre_mot"])
    write_csv("per_sequence_metrics_post.csv",
              [r for r in per_seq_rows if r["stream"] == "post_mot"])
    write_csv("interaction_costs.csv", costs)
    write_csv("primitive_costs.csv", primitive_rows)
    write_csv("quality_cost_curve.csv", qc)
    with (OUT / "interaction_events.jsonl").open("w", encoding="utf-8") as f:
        for e in interaction_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (OUT / "quality_cost_auc.json").write_text(
        json.dumps(auc, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "statistical_tests.json").write_text(
        json.dumps(stat_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"combined": combined_rows, "auc": auc}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
