#!/usr/bin/env python
"""N8 official TrackEval evaluation (pre and post streams)."""

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(".")
PY = "python"
TRACKEVAL = (
    Path(".")
    / "third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py"
)
GT = Path("/path/to/dancetrack/val")
OUT = Path(os.environ.get("N8_EVAL_OUT", ROOT / "outputs/n8/eval"))
REAL = Path(os.environ.get("N8_REAL_ROOT", ROOT / "outputs/n8/real"))
BUDGETS = [int(x) for x in os.environ.get("N8_BUDGETS", "0 1 2 4 8 -1").split()]
P0_DIR = Path(
    os.environ.get("N8_P0_DIR", ROOT / "outputs/n5/integrity/canonical_mot_results/b0")
)
_ALL_SEQS = sorted(
    p.name for p in GT.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
SEQS = sorted(os.environ.get("N8_SEQS", "").split()) or _ALL_SEQS
FRAME_LIMIT = int(os.environ.get("N8_EVAL_FRAMES", "0")) or None


def budget_name(b: int) -> str:
    return "unlimited" if b < 0 else f"b{b}"


TRACKERS = ["p0"] + [
    f"route_a_{budget_name(b)}_{stream}"
    for b in BUDGETS
    for stream in ("pre", "post")
]


def _link_results() -> None:
    mot_root = OUT / "mot_results"
    mot_root.mkdir(parents=True, exist_ok=True)
    for name in TRACKERS:
        dst = mot_root / name
        dst.mkdir(exist_ok=True)
        for seq in SEQS:
            if name == "p0":
                src = P0_DIR / f"{seq}.txt"
            else:
                b_str, stream = name.rsplit("_", 1)
                src = REAL / b_str / seq / f"{stream}_mot" / f"{seq}.txt"
            out_file = dst / f"{seq}.txt"
            if not src.exists():
                continue
            if FRAME_LIMIT is not None:
                out_file.write_text(
                    "".join(
                        line
                        for line in src.read_text(encoding="utf-8").splitlines(keepends=True)
                        if line.strip() and int(line.split(",")[0]) <= FRAME_LIMIT
                    ),
                    encoding="utf-8",
                )
            elif not out_file.exists():
                out_file.symlink_to(src)


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
            per_seq[seq] = {
                "HOTA": float(parts[1]),
                "DetA": float(parts[2]),
                "AssA": float(parts[3]),
            }
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


def run_trackeval() -> None:
    seqmap = OUT / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(SEQS) + "\n", encoding="utf-8")
    cmd = [
        PY,
        str(TRACKEVAL),
        "--GT_FOLDER", str(GT),
        "--TRACKERS_FOLDER", str(OUT / "mot_results"),
        "--TRACKERS_TO_EVAL", *TRACKERS,
        "--TRACKER_SUB_FOLDER", "",
        "--OUTPUT_SUB_FOLDER", "",
        "--SEQMAP_FILE", str(seqmap),
        "--BENCHMARK", "DanceTrack",
        "--SPLIT_TO_EVAL", "val",
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "False",
        "--CLASSES_TO_EVAL", "pedestrian",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--PLOT_CURVES", "False",
        "--PRINT_RESULTS", "True",
        "--PRINT_ONLY_COMBINED", "False",
        "--OUTPUT_SUMMARY", "True",
        "--OUTPUT_DETAILED", "True",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = OUT / "trackeval.log"
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    print("TRACKEVAL_RC", proc.returncode)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-3000:])


def stats(per_seq: dict):
    out = {}
    for metric in ["HOTA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]:
        vals = np.asarray([per_seq[s].get(metric, np.nan) for s in per_seq], float)
        vals = vals[np.isfinite(vals)]
        out[metric] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }
    return out


def paired_stats(a: dict, b: dict):
    common = sorted(set(a) & set(b))
    rng = np.random.default_rng(42)
    out = {}
    for metric in ["HOTA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]:
        x = np.asarray([a[s].get(metric, np.nan) for s in common], float)
        y = np.asarray([b[s].get(metric, np.nan) for s in common], float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        delta = x - y
        boot = np.asarray(
            [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)]
        )
        from scipy import stats as sps

        p = float(sps.wilcoxon(delta).pvalue) if np.any(delta) else 1.0
        sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        out[metric] = {
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
            "wilcoxon_p": p,
            "effect_size": float(np.mean(delta) / sd) if sd > 0 else 0.0,
            "improved": int(np.sum(delta > 1e-9)),
            "degraded": int(np.sum(delta < -1e-9)),
            "unchanged": int(np.sum(np.abs(delta) <= 1e-9)),
            "n": int(len(delta)),
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _link_results()
    run_trackeval()
    log = OUT / "trackeval.log"
    combined_rows = []
    per_seq_rows = []
    per_seq_map = {}
    for name in TRACKERS:
        combined, per_seq = parse_log(log, name)
        combined_rows.append({"method": name, **combined})
        per_seq_map[name] = per_seq
        for seq, m in per_seq.items():
            per_seq_rows.append({"method": name, "sequence": seq, **m})
    with (OUT / "combined_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        w.writeheader()
        w.writerows(combined_rows)
    with (OUT / "per_sequence_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seq_rows[0].keys()))
        w.writeheader()
        w.writerows(per_seq_rows)
    for stream in ("pre", "post"):
        rows = [r for r in combined_rows if r["method"].endswith(f"_{stream}")]
        with (OUT / f"combined_metrics_{stream}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        rows_seq = [r for r in per_seq_rows if r["method"].endswith(f"_{stream}")]
        with (OUT / f"per_sequence_metrics_{stream}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows_seq[0].keys()))
            w.writeheader()
            w.writerows(rows_seq)
    qc = []
    for i, name in enumerate(TRACKERS):
        if name == "p0":
            continue
        b_str, stream = name.rsplit("_", 1)
        bn = b_str.rsplit("_", 1)[1]
        b = -1 if bn == "unlimited" else int(bn[1:])
        qc.append(
            {
                "budget": b,
                "stream": stream,
                "method": name,
                **{k: combined_rows[i].get(k) for k in ("HOTA", "AssA", "MOTA", "IDF1", "IDSW", "Frag")},
            }
        )
    with (OUT / "quality_cost.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(qc[0].keys()))
        w.writeheader()
        w.writerows(qc)
    stats_out = {name: stats(per_seq_map[name]) for name in TRACKERS}
    pairs = {"route_a_b0_vs_p0": ("p0", "p0")}
    for b in BUDGETS:
        if b == 0:
            continue
        pairs[f"route_a_{budget_name(b)}_post_vs_p0"] = (
            f"route_a_{budget_name(b)}_post",
            "p0",
        )
    pair_out = {
        name: paired_stats(per_seq_map[a], per_seq_map[b])
        for name, (a, b) in pairs.items()
    }
    (OUT / "stats.json").write_text(
        json.dumps(
            {"per_method": stats_out, "paired": pair_out},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"combined": combined_rows, "paired": pair_out},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
