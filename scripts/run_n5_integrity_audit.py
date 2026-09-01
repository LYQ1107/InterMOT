#!/usr/bin/env python
"""N5-0 canonical integrity audit.

Builds the canonical 25-sequence validation set from existing N4/N1.5 outputs,
re-runs official TrackEval with a corrected seqmap, and writes all integrity
artifacts under outputs/n5/integrity/.
"""

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(".")
GT_ROOT = Path("/path/to/dancetrack/val")
N4_WINNER = ROOT / "outputs/n4/full25/winner"
N4_R2_G2 = ROOT / "outputs/n4/round2/R2_G2"
A0_V2 = ROOT / "outputs/n1_5/a0_v2_mot"
OUT = ROOT / "outputs/n5/integrity"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. GT sequence set: every directory under val, excluding files.
    gt_seqs = sorted(
        p.name
        for p in GT_ROOT.iterdir()
        if p.is_dir() and (p / "gt" / "gt.txt").is_file()
    )
    (OUT / "gt_sequences.txt").write_text("\n".join(gt_seqs) + "\n", encoding="utf-8")

    # 2. A0 result sequences (auto baseline).
    a0_files = sorted(p.name for p in A0_V2.glob("*.txt"))
    (OUT / "a0_sequences.txt").write_text("\n".join(a0_files) + "\n", encoding="utf-8")

    # 3. N4 full25 result matrix (24 seqs in seqmap + files present).
    n4_seqs = [
        line.strip()
        for line in (N4_WINNER / "seqmap.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip() != "name"
    ]
    matrix_rows = []
    for seq in sorted(set(gt_seqs) | set(n4_seqs)):
        matrix_rows.append(
            {
                "sequence": seq,
                "in_gt": seq in gt_seqs,
                "in_n4_seqmap": seq in n4_seqs,
                "n4_b0_file": (N4_WINNER / "mot_results/b0" / f"{seq}.txt").is_file(),
                "n4_b1_file": (N4_WINNER / "mot_results/b1" / f"{seq}.txt").is_file(),
                "n4_b2_file": (N4_WINNER / "mot_results/b2" / f"{seq}.txt").is_file(),
                "n4_b5_file": (N4_WINNER / "mot_results/b5" / f"{seq}.txt").is_file(),
                "r2_g2_b1_file": (N4_R2_G2 / "mot_results/b1" / f"{seq}.txt").is_file(),
                "a0_v2_file": (A0_V2 / f"{seq}.txt").is_file(),
            }
        )
    with (OUT / "result_file_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)

    missing = {
        "missing_in_n4_seqmap": sorted(set(gt_seqs) - set(n4_seqs)),
        "missing_in_n4_b0": sorted(set(gt_seqs) - {
            p.name for p in (N4_WINNER / "mot_results/b0").glob("*.txt") if p.stem != "pedestrian"
        }),
        "missing_in_a0_v2": sorted(set(gt_seqs) - set(a0_files)),
    }
    (OUT / "missing_sequences.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 4. Canonical seqmap (all 25).
    canonical_seqmap = OUT / "canonical_seqmap.txt"
    canonical_seqmap.write_text("name\n" + "\n".join(gt_seqs) + "\n", encoding="utf-8")

    # 5. Build canonical MOT results: N4 full25 + reuse legal 0004 outputs.
    canonical_mot = OUT / "canonical_mot_results"
    reuse_records = []
    for budget, src_0004 in [
        ("b0", A0_V2 / "dancetrack0004.txt"),
        ("b1", N4_R2_G2 / "mot_results/b1/dancetrack0004.txt"),
        ("b2", N4_R2_G2 / "mot_results/b2/dancetrack0004.txt"),
        ("b5", N4_R2_G2 / "mot_results/b5/dancetrack0004.txt"),
    ]:
        dst_dir = canonical_mot / budget
        dst_dir.mkdir(parents=True, exist_ok=True)
        n4_budget = N4_WINNER / "mot_results" / budget
        for seq in gt_seqs:
            if (n4_budget / f"{seq}.txt").is_file():
                src = n4_budget / f"{seq}.txt"
                shutil.copy2(src, dst_dir / f"{seq}.txt")
            elif seq == "dancetrack0004" and src_0004.is_file():
                shutil.copy2(src_0004, dst_dir / f"{seq}.txt")
                reuse_records.append(
                    {
                        "sequence": seq,
                        "budget": budget,
                        "source_file": str(src_0004),
                        "source_sha256": sha256(src_0004),
                        "destination_file": str(dst_dir / f"{seq}.txt"),
                        "destination_sha256": sha256(dst_dir / f"{seq}.txt"),
                    }
                )
            else:
                raise RuntimeError(f"missing canonical result: {budget}/{seq}")
    (OUT / "reused_0004_records.json").write_text(
        json.dumps(reuse_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 6. Validate canonical files.
    validation = {}
    for budget in ("b0", "b1", "b2", "b5"):
        files = sorted((canonical_mot / budget).glob("*.txt"))
        seqs_in_dir = sorted(p.stem for p in files)
        validation[budget] = {
            "count": len(seqs_in_dir),
            "matches_canonical": seqs_in_dir == gt_seqs,
        }
    (OUT / "canonical_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert all(v["matches_canonical"] for v in validation.values()), validation

    # 7. Official TrackEval on canonical 25-seq set.
    trackeval_log = OUT / "trackeval_canonical25.log"
    cmd = [
        "python",
        "./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py",
        "--GT_FOLDER", str(GT_ROOT),
        "--TRACKERS_FOLDER", str(canonical_mot),
        "--TRACKERS_TO_EVAL", "b0", "b1", "b2", "b5",
        "--TRACKER_SUB_FOLDER", "",
        "--OUTPUT_SUB_FOLDER", "",
        "--SEQMAP_FILE", str(canonical_seqmap),
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
    trackeval_log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    summary = {
        "returncode": proc.returncode,
        "log_file": str(trackeval_log),
        "evaluating_line": [
            line for line in proc.stdout.splitlines() if "Evaluating" in line
        ],
        "on_25": "on 25 sequence(s)" in proc.stdout,
    }
    (OUT / "trackeval_canonical25.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["on_25"] or proc.returncode != 0:
        raise SystemExit("CANONICAL_TRACKEVAL_FAILED")


if __name__ == "__main__":
    main()
