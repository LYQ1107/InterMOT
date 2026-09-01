#!/usr/bin/env python
"""N1.5 window-boundary diagnostics for the frozen N1 MOT results."""

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from sam3_intermot.utils.io import atomic_write_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/path/to/dancetrack/val")
TRACKEVAL = (
    Path(".")
    / "third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py"
)
PY = "python"
SEQS = ["dancetrack0004", "dancetrack0005", "dancetrack0007"]
WINDOW = 200


def load_mot(path):
    rows = []
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 7:
            continue
        rows.append(
            {
                "frame": int(float(p[0])),
                "id": int(float(p[1])),
                "box": np.asarray([float(p[2]), float(p[3]), float(p[2]) + float(p[4]), float(p[3]) + float(p[5])]),
                "score": float(p[6]),
            }
        )
    return rows


def load_gt(path):
    rows = []
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 7 or float(p[6]) == 0:
            continue
        rows.append(
            {
                "frame": int(float(p[0])),
                "id": int(float(p[1])),
                "box": np.asarray([float(p[2]), float(p[3]), float(p[2]) + float(p[4]), float(p[3]) + float(p[5])]),
            }
        )
    return rows


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def run_trackeval_window(seq, start, end, out_dir):
    gt_rows = [r for r in gt_cache[seq] if start <= r["frame"] - 1 <= end]
    tr_rows = [r for r in mot_cache[seq] if start <= r["frame"] - 1 <= end]
    length = end - start + 1
    seq_dir = out_dir / "gt" / seq
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "trackers" / "sam3_auto").mkdir(parents=True, exist_ok=True)
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname={}\nimWidth=1920\nimHeight=1080\nimExt=jpg\nseqLength={}\n".format(seq, length),
        encoding="utf-8",
    )
    with (seq_dir / "gt" / "gt.txt").open("w", encoding="utf-8") as f:
        for r in gt_rows:
            x, y, x2, y2 = r["box"]
            f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},1,1,1\n".format(
                r["frame"] - start, r["id"], x, y, x2 - x, y2 - y
            ))
    with (out_dir / "trackers" / "sam3_auto" / (seq + ".txt")).open("w", encoding="utf-8") as f:
        for r in tr_rows:
            x, y, x2, y2 = r["box"]
            f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.3f},-1,-1,-1\n".format(
                r["frame"] - start, r["id"], x, y, x2 - x, y2 - y, r["score"]
            ))
    seqmap = out_dir / "seqmap.txt"
    seqmap.write_text("name\n{}\n".format(seq), encoding="utf-8")
    cmd = [
        PY, str(TRACKEVAL),
        "--GT_FOLDER", str(out_dir / "gt"),
        "--TRACKERS_FOLDER", str(out_dir / "trackers"),
        "--TRACKERS_TO_EVAL", "sam3_auto",
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
    text = proc.stdout + proc.stderr
    m = {}
    summary = out_dir / "trackers" / "sam3_auto" / "pedestrian_summary.txt"
    if summary.exists():
        lines = summary.read_text().splitlines()
        if len(lines) >= 2:
            header = lines[0].split()
            vals = lines[1].split()
            for key, val in zip(header, vals):
                try:
                    m[key] = float(val)
                except ValueError:
                    m[key] = val
    if not m:
        m = _parse_trackeval(text)
    return m, proc.returncode, text


def _parse_trackeval(text):
    metrics = {}
    section = None
    for line in text.splitlines():
        if line.startswith("HOTA:"):
            section = "HOTA"
            continue
        if line.startswith("CLEAR:"):
            section = "CLEAR"
            continue
        if line.startswith("Identity:"):
            section = "Identity"
            continue
        if not line.startswith("COMBINED") or section is None:
            continue
        vals = line.split()[1:]
        try:
            nums = [float(v) for v in vals]
        except ValueError:
            continue
        if section == "HOTA" and len(nums) >= 3:
            metrics["HOTA"], metrics["DetA"], metrics["AssA"] = nums[0], nums[1], nums[2]
        elif section == "CLEAR" and len(nums) >= 17:
            metrics["MOTA"], metrics["MOTP"] = nums[0], nums[1]
            metrics["IDSW"], metrics["MT"], metrics["PT"], metrics["ML"] = (
                int(nums[12]), int(nums[13]), int(nums[14]), int(nums[15])
            )
            metrics["Frag"] = int(nums[16])
        elif section == "Identity" and len(nums) >= 3:
            metrics["IDF1"], metrics["IDR"], metrics["IDP"] = nums[0], nums[1], nums[2]
    return metrics


def greedy_pairs(seq, start=None, end=None):
    pairs = set()
    mot_by_frame = {}
    gt_by_frame = {}
    for r in mot_cache[seq]:
        if start is not None and not (start <= r["frame"] - 1 <= end):
            continue
        mot_by_frame.setdefault(r["frame"], []).append(r)
    for r in gt_cache[seq]:
        if start is not None and not (start <= r["frame"] - 1 <= end):
            continue
        gt_by_frame.setdefault(r["frame"], []).append(r)
    for frame in sorted(set(mot_by_frame) | set(gt_by_frame)):
        used = set()
        for gt in gt_by_frame.get(frame, []):
            best = None
            best_iou = 0.25
            for m in mot_by_frame.get(frame, []):
                if m["id"] in used:
                    continue
                v = iou(gt["box"], m["box"])
                if v > best_iou:
                    best_iou = v
                    best = m
            if best is not None:
                used.add(best["id"])
                pairs.add((gt["id"], best["id"]))
    return pairs


def main():
    global gt_cache, mot_cache
    out = ROOT / "outputs" / "n1_5"
    out.mkdir(parents=True, exist_ok=True)
    gt_cache = {}
    mot_cache = {}
    for seq in SEQS:
        gt_cache[seq] = load_gt(DATA / seq / "gt" / "gt.txt")
        mot_cache[seq] = load_mot(ROOT / "outputs" / "n1" / "mot_results" / "sam3_auto" / (seq + ".txt"))

    per_window = []
    boundary = []
    identity_frag = []
    entrant = []
    failures = []
    for seq in SEQS:
        num_frames = max(r["frame"] for r in gt_cache[seq])
        starts = list(range(0, num_frames, WINDOW))
        for wi, start in enumerate(starts):
            end = min(start + WINDOW - 1, num_frames - 1)
            d = out / "tmp_eval" / seq / ("w%d_%d" % (start, end))
            metrics, rc, log = run_trackeval_window(seq, start, end, d)
            pairs = greedy_pairs(seq, start, end)
            per_window.append({
                "sequence": seq, "window": wi, "start_frame": start, "end_frame": end,
                "length": end - start + 1, "trackeval_rc": rc,
                **metrics,
                "gt_ids": len({g for g, _ in pairs}),
                "mot_ids": len({m for _, m in pairs}),
            })
            if rc != 0:
                failures.append({"type": "PROPAGATION_FAILURE", "sequence": seq, "start": start, "end": end, "detail": log[-500:]})
        # boundary windows around each internal boundary
        for start in starts[1:]:
            bs = max(0, start - 5)
            be = min(num_frames - 1, start + 5)
            d = out / "tmp_eval" / seq / ("b_%d_%d" % (bs, be))
            metrics, rc, log = run_trackeval_window(seq, bs, be, d)
            boundary.append({
                "sequence": seq, "boundary_frame": start, "start_frame": bs, "end_frame": be,
                "trackeval_rc": rc, **metrics,
            })
            if rc != 0:
                failures.append({"type": "WINDOW_HANDOVER_FAILURE", "sequence": seq, "start": bs, "end": be, "detail": log[-500:]})
        # identity fragmentation across windows
        gt_first = {}
        gt_last = {}
        for r in gt_cache[seq]:
            gt_first.setdefault(r["id"], r["frame"])
            gt_last[r["id"]] = r["frame"]
        mot_ids_per_gt = {}
        gt_ids_per_mot = {}
        for g, m in greedy_pairs(seq):
            mot_ids_per_gt.setdefault(g, set()).add(m)
            gt_ids_per_mot.setdefault(m, set()).add(g)
        for g, mids in mot_ids_per_gt.items():
            identity_frag.append({
                "sequence": seq, "gt_id": g, "num_mot_ids": len(mids),
                "mot_ids": sorted(mids), "first_frame": gt_first[g], "last_frame": gt_last[g],
                "window_boundary_crossed": any(
                    gt_first[g] <= s <= gt_last[g] for s in starts[1:]
                ),
            })
        # new entrant delay and lost-recreated
        mot_by_frame = {}
        for r in mot_cache[seq]:
            mot_by_frame.setdefault(r["frame"], []).append(r)
        for gid, first in sorted(gt_first.items()):
            first_mot_frame = None
            for f in range(first, min(first + 60, num_frames + 1)):
                if any(iou(gt["box"], m["box"]) > 0.25 for gt in gt_cache[seq] if gt["frame"] == f and gt["id"] == gid for m in mot_by_frame.get(f, [])):
                    first_mot_frame = f
                    break
            if first_mot_frame is not None and first_mot_frame > first:
                entrant.append({"sequence": seq, "gt_id": gid, "first_gt_frame": first, "first_mot_frame": first_mot_frame, "delay": first_mot_frame - first})
                failures.append({"type": "NEW_ENTRANT_DELAY", "sequence": seq, "gt_id": gid, "delay": first_mot_frame - first})
        # lost target recreated as new MOT ID across windows
        pairs_full = greedy_pairs(seq)
        for gid, mids in mot_ids_per_gt.items():
            if len(mids) > 1 and any(gt_first[gid] <= s <= gt_last[gid] for s in starts[1:]):
                failures.append({"type": "WINDOW_HANDOVER_FAILURE", "sequence": seq, "gt_id": gid, "mot_ids": sorted(mids)})

    write_csv(out / "per_window_metrics.csv", per_window)
    write_csv(out / "boundary_idsw.csv", boundary)
    write_csv(out / "identity_fragmentation.csv", identity_frag)
    write_csv(out / "new_entrant_delay.csv", entrant)
    with (out / "failure_cases.jsonl").open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({
        "windows": per_window,
        "boundary": boundary,
        "identity_fragmentation": identity_frag,
        "new_entrant_delay": entrant,
        "failure_cases": failures,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
