#!/usr/bin/env python
"""N6 full-25 statistics: metrics, costs, quality-cost curve, tests."""

import csv
import json
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows


ROOT = Path(".")
OUT = ROOT / "outputs/n6/full25_stats"
EVAL = ROOT / "outputs/n6/full25_eval"
REAL = ROOT / "outputs/n6/full25/real"
DS = DanceTrackDataset("/path/to/dancetrack", split="val")
SEQS = sorted(
    p.name
    for p in Path("/path/to/dancetrack/val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)

PROTO = {
    "p0": {"dir": ROOT / "outputs/n5/integrity/canonical_mot_results/b0", "budget": 0},
    "p1": {"dir": ROOT / "outputs/n6/full25/p1_oracle_frame_all", "budget": 0},
    "p2": {"dir": REAL / "p2_oracle_state_all", "budget": 0},
    "p3": {"dir": REAL / "p3_continuous_id_miss", "budget": 0},
    "p4_b1": {"dir": REAL / "p4_budget_b1", "budget": 1},
    "p4_b2": {"dir": REAL / "p4_budget_b2", "budget": 2},
    "p4_b4": {"dir": REAL / "p4_budget_b4", "budget": 4},
    "p4_b8": {"dir": REAL / "p4_budget_b8", "budget": 8},
}


def parse_log(log: Path, tracker: str):
    text = log.read_text(encoding="utf-8").splitlines()
    section = None
    combined, per_seq = {}, {}
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
                        "FP": float(parts[12]),
                        "FN": float(parts[11]),
                    }
                )
            elif section == "ID":
                combined.update(
                    {"IDF1": float(parts[1]), "IDR": float(parts[2]), "IDP": float(parts[3])}
                )
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
                    "Frag": float(parts[17]),
                    "FP": float(parts[12]),
                    "FN": float(parts[11]),
                }
            )
        elif section == "ID":
            per_seq.setdefault(seq, {})["IDF1"] = float(parts[1])
    return combined, per_seq


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    combined_rows, per_seq_rows, costs, events_all = [], [], [], []
    identity_issues = {}
    budget_ok = True
    for proto, cfg in PROTO.items():
        log = EVAL / f"{proto}.log"
        for stream in ("pre_mot", "post_mot"):
            combined, per_seq = parse_log(log, stream)
            combined_rows.append(
                {"protocol": proto, "budget": cfg["budget"], "stream": stream, **combined}
            )
            for seq, m in per_seq.items():
                per_seq_rows.append(
                    {
                        "protocol": proto,
                        "budget": cfg["budget"],
                        "stream": stream,
                        "sequence": seq,
                        **m,
                    }
                )
        for seq in SEQS:
            ev_file = cfg["dir"] / seq / "events.jsonl"
            if not ev_file.exists():
                continue
            events = [json.loads(l) for l in ev_file.read_text().splitlines() if l.strip()]
            events_all.extend(events)
            by_type = {}
            for e in events:
                by_type[e["action_type"]] = by_type.get(e["action_type"], 0) + 1
            num_frames = DS.num_frames(seq)
            costs.append(
                {
                    "protocol": proto,
                    "budget": cfg["budget"],
                    "sequence": seq,
                    "NoI_total": len(events),
                    "NoAddNew": by_type.get("ADD_NEW_IDENTITY", 0),
                    "NoRecover": by_type.get("RECOVER_IDENTITY", 0),
                    "NoReassign": by_type.get("AUTHORITATIVE_REASSIGN", 0),
                    "NoSwap": by_type.get("ATOMIC_ID_SWAP", 0),
                    "NoCorrect": by_type.get("AUTHORITATIVE_CORRECT", 0),
                    "NoDelete": by_type.get("AUTHORITATIVE_DELETE", 0),
                    "interactions_per_1000_frames": 1000.0 * len(events) / num_frames,
                    "human_seconds": "NOT_MEASURED",
                }
            )
            if cfg["budget"] > 0:
                summary = json.loads((cfg["dir"] / seq / "summary.json").read_text())
                if summary["accepted_commands"] > cfg["budget"]:
                    budget_ok = False
            # identity namespace check on post stream
            post = read_mot_rows(cfg["dir"] / seq / "post_mot" / f"{seq}.txt")
            gt = DS.load_gt(seq)
            g2p, p2g, dup = {}, {}, 0
            for f in range(DS.num_frames(seq)):
                g = gt.get(f)
                gb = [np.asarray(b, float) for b in g.boxes] if g else []
                gids = g.gt_ids if g else []
                po = post.get(f, [])
                ids = [t for t, _ in po if t >= 1000]
                if len(ids) != len(set(ids)):
                    dup += 1
                pm = match_boxes(gb, [np.asarray(b, float) for _, b in po], 0.5)
                pmap = {pi: t for pi, (t, _) in enumerate(po)}
                for gi, pi, _ in pm:
                    pid = pmap[pi]
                    if pid < 1000:
                        continue
                    g2p.setdefault(gids[gi], set()).add(pid)
                    p2g.setdefault(pid, set()).add(gids[gi])
            bad = [k for k, v in g2p.items() if len(v) != 1] + [
                k for k, v in p2g.items() if len(v) != 1
            ]
            identity_issues.setdefault(proto, {})[seq] = {
                "gt_to_public_multi": [str(k) for k, v in g2p.items() if len(v) > 1],
                "public_to_gt_multi": [str(k) for k, v in p2g.items() if len(v) > 1],
                "duplicate_public_id_frames": dup,
                "ok": not bad and dup == 0,
            }

    def write_csv(name, rows):
        if not rows:
            return
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv("combined_metrics_pre.csv", [r for r in combined_rows if r["stream"] == "pre_mot"])
    write_csv("combined_metrics_post.csv", [r for r in combined_rows if r["stream"] == "post_mot"])
    write_csv("per_sequence_metrics_pre.csv", [r for r in per_seq_rows if r["stream"] == "pre_mot"])
    write_csv("per_sequence_metrics_post.csv", [r for r in per_seq_rows if r["stream"] == "post_mot"])
    write_csv("interaction_costs.csv", costs)
    with (OUT / "interaction_events.jsonl").open("w", encoding="utf-8") as f:
        for e in events_all:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (OUT / "identity_namespace_audit.json").write_text(
        json.dumps(identity_issues, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # quality-cost curve (post)
    qc = []
    for proto in ["p0", "p4_b1", "p4_b2", "p4_b4", "p4_b8", "p3"]:
        row = next(
            (r for r in combined_rows if r["protocol"] == proto and r["stream"] == "post_mot"),
            None,
        )
        if row is None:
            continue
        avg_noi = float(
            np.mean([c["NoI_total"] for c in costs if c["protocol"] == proto])
            if any(c["protocol"] == proto for c in costs)
            else 0
        )
        qc.append(
            {
                "protocol": proto,
                "accepted_interactions_per_sequence": avg_noi,
                "HOTA": row.get("HOTA"),
                "AssA": row.get("AssA"),
                "MOTA": row.get("MOTA"),
                "IDF1": row.get("IDF1"),
                "IDSW": row.get("IDSW"),
            }
        )
    write_csv("quality_cost_curve.csv", qc)
    auc = {}
    for metric in ["HOTA", "AssA", "IDF1"]:
        xs = np.asarray([c["accepted_interactions_per_sequence"] for c in qc], float)
        ys = np.asarray([c.get(metric) or 0.0 for c in qc], float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        auc[metric] = float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else 0.0
    (OUT / "quality_cost_auc.json").write_text(
        json.dumps(auc, indent=2) + "\n", encoding="utf-8"
    )

    # paired statistics (25 sequences)
    from scipy import stats as sps

    def per_seq_map(proto, stream):
        return {
            r["sequence"]: r
            for r in per_seq_rows
            if r["protocol"] == proto and r["stream"] == stream
        }

    pairs = [
        ("P2_pre_vs_P0", "p2", "pre_mot", "p0", "pre_mot"),
        ("P3_pre_vs_P0", "p3", "pre_mot", "p0", "pre_mot"),
        ("P3_post_vs_P0", "p3", "post_mot", "p0", "post_mot"),
    ]
    for b in (1, 2, 4, 8):
        pairs.append((f"B{b}_pre_vs_P0", f"p4_b{b}", "pre_mot", "p0", "pre_mot"))
        pairs.append((f"B{b}_post_vs_P0", f"p4_b{b}", "post_mot", "p0", "post_mot"))
    stat = {}
    rng = np.random.default_rng(42)
    for name, pa, sa, pb, sb in pairs:
        a, b = per_seq_map(pa, sa), per_seq_map(pb, sb)
        common = sorted(set(a) & set(b))
        res = {}
        for metric in ["HOTA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]:
            x = np.asarray([a[s].get(metric, np.nan) for s in common], float)
            y = np.asarray([b[s].get(metric, np.nan) for s in common], float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            delta = x - y
            boot = np.asarray(
                [
                    delta[rng.integers(0, len(delta), len(delta))].mean()
                    for _ in range(2000)
                ]
            )
            p = float(sps.wilcoxon(delta).pvalue) if np.any(delta) else 1.0
            sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
            res[metric] = {
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "wilcoxon_p": p,
                "effect_size": float(np.mean(delta) / sd) if sd > 0 else 0.0,
                "improved": int(np.sum(delta > 1e-9)),
                "degraded": int(np.sum(delta < -1e-9)),
                "unchanged": int(np.sum(np.abs(delta) <= 1e-9)),
                "n": len(delta),
            }
        stat[name] = res
    (OUT / "statistical_tests.json").write_text(
        json.dumps(stat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "combined_post": [r for r in combined_rows if r["stream"] == "post_mot"],
                "combined_pre": [r for r in combined_rows if r["stream"] == "pre_mot"],
                "auc": auc,
                "budget_contract_ok": budget_ok,
                "identity_namespace_ok": all(
                    v["ok"] for proto_map in identity_issues.values() for v in proto_map.values()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "post": [r for r in combined_rows if r["stream"] == "post_mot"],
            "auc": auc,
            "budget_contract_ok": budget_ok,
            "identity_ok": all(
                v["ok"] for proto_map in identity_issues.values() for v in proto_map.values()
            ),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
