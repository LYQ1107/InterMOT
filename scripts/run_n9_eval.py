#!/usr/bin/env python
"""N9 official TrackEval evaluation (N8 baseline + variants, pre/post)."""

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
OUT = Path(os.environ.get("N9_EVAL_OUT", ROOT / "outputs/n9/eval"))
REAL = Path(os.environ.get("N9_REAL_ROOT", ROOT / "outputs/n9/real"))
N8_REAL = ROOT / "outputs/n8/real"
BUDGETS = [int(x) for x in os.environ.get("N9_BUDGETS", "0 1 2 4 8").split()]
VARIANTS = os.environ.get("N9_VARIANTS", "reid pairwise auto proposed").split()
P0_DIR = ROOT / "outputs/n5/integrity/canonical_mot_results/b0"
_ALL_SEQS = sorted(
    p.name for p in GT.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
SEQS = sorted(os.environ.get("N9_SEQS", "").split()) or _ALL_SEQS

TRACKERS = ["p0"] + [f"n8_b{b}_post" for b in BUDGETS] + [
    f"{v}_b{b}_{stream}"
    for v in VARIANTS
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
            elif name.startswith("n8_b"):
                bstr = name.removeprefix("n8_").removesuffix("_post")
                src = N8_REAL / f"route_a_{bstr}" / seq / "post_mot" / f"{seq}.txt"
            else:
                v, bstr, stream = name.rsplit("_", 2)
                src = REAL / f"{v}_{bstr}" / seq / f"{stream}_mot" / f"{seq}.txt"
            out_file = dst / f"{seq}.txt"
            if not src.exists():
                continue
            if not out_file.exists():
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
                combined = {"HOTA": float(parts[1]), "DetA": float(parts[2]), "AssA": float(parts[3]), "LocA": float(parts[8])}
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
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-3000:])


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
    combined_rows, per_seq_rows, per_seq_map = [], [], {}
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
    pairs = {}
    for b in BUDGETS:
        pairs[f"proposed_b{b}_vs_n8_b{b}"] = (f"proposed_b{b}_post", f"n8_b{b}_post")
        pairs[f"proposed_b{b}_vs_reid_b{b}"] = (f"proposed_b{b}_post", f"reid_b{b}_post")
        pairs[f"proposed_b{b}_vs_auto_b{b}"] = (f"proposed_b{b}_post", f"auto_b{b}_post")
        pairs[f"reid_b{b}_vs_n8_b{b}"] = (f"reid_b{b}_post", f"n8_b{b}_post")
    pair_out = {
        name: paired_stats(per_seq_map[a], per_seq_map[b])
        for name, (a, b) in pairs.items()
        if a in per_seq_map and b in per_seq_map
    }
    (OUT / "stats.json").write_text(
        json.dumps({"paired": pair_out}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"combined": combined_rows, "paired": pair_out}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
