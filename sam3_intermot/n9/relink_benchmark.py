"""Build tracklet-relinking episodes from P0 outputs and DanceTrack GT."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows
from sam3_intermot.interaction.simulator import GTFrame


def build_segments(
    rows: Dict[int, List[Tuple[int, np.ndarray]]],
) -> List[dict]:
    """Group P0 rows into tracklet segments (tid runs, max 2-frame hole)."""
    by_tid = defaultdict(list)
    for f in sorted(rows):
        for tid, box in rows[f]:
            by_tid[int(tid)].append((f, np.asarray(box, float)))
    segments = []
    for tid, items in sorted(by_tid.items()):
        items.sort()
        run: List[Tuple[int, np.ndarray]] = []
        for f, box in items:
            if run and f - run[-1][0] > 2:
                segments.append(_segment(tid, run))
                run = []
            run.append((f, box))
        if run:
            segments.append(_segment(tid, run))
    return segments


def _segment(tid: int, run: List[Tuple[int, np.ndarray]]) -> dict:
    frames = [f for f, _ in run]
    boxes = [b for _, b in run]
    centers = np.asarray(
        [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes], dtype=float
    )
    vel = (
        (centers[-1] - centers[0]) / max(1, len(frames) - 1)
        if len(frames) > 1
        else np.zeros(2)
    )
    return {
        "tid": int(tid),
        "start": int(frames[0]),
        "end": int(frames[-1]),
        "frames": frames,
        "boxes": [b.tolist() for b in boxes],
        "last_box": boxes[-1].tolist(),
        "first_box": boxes[0].tolist(),
        "velocity": vel.tolist(),
        "age": len(frames),
        "gid": None,
        "gid_coverage": 0.0,
    }


def assign_gt(segments: List[dict], gt: Dict[int, GTFrame]) -> None:
    for seg in segments:
        counts: Dict[int, int] = defaultdict(int)
        total = 0
        for f, box in zip(seg["frames"], seg["boxes"]):
            gtf = gt.get(f)
            if gtf is None or not gtf.boxes:
                continue
            matches = match_boxes(
                [np.asarray(gtf.boxes[gi], float) for gi in range(len(gtf.boxes))],
                [np.asarray(box, float)],
                0.5,
            )
            if matches:
                counts[gtf.gt_ids[matches[0][0]]] += 1
                total += 1
        if total > 0:
            gid, c = max(counts.items(), key=lambda kv: kv[1])
            seg["gid"] = int(gid)
            seg["gid_coverage"] = c / total


def build_episodes(
    sequence: str,
    segments: List[dict],
    gt: Dict[int, GTFrame],
    max_candidates: int = 8,
    min_seg_len: int = 2,
    max_gap: int = 200,
) -> List[dict]:
    segs = [s for s in segments if s["gid"] is not None and s["gid_coverage"] >= 0.6]
    by_gid: Dict[int, List[dict]] = defaultdict(list)
    for s in segs:
        by_gid[s["gid"]].append(s)
    episodes = []
    for gid, gsegs in by_gid.items():
        gsegs.sort(key=lambda s: s["start"])
        for a, b in zip(gsegs, gsegs[1:]):
            if a["tid"] == b["tid"]:
                continue
            gap = b["start"] - a["end"] - 1
            if gap < 0 or gap > max_gap:
                continue
            if len(a["frames"]) < min_seg_len or len(b["frames"]) < min_seg_len:
                continue
            candidates = _candidates(by_gid, gid, b["start"], max_candidates, max_gap)
            episodes.append(
                {
                    "sequence": sequence,
                    "gid": int(gid),
                    "hist_tid": int(a["tid"]),
                    "new_tid": int(b["tid"]),
                    "gap": int(gap),
                    "hist_start": int(a["start"]),
                    "hist_end": int(a["end"]),
                    "new_start": int(b["start"]),
                    "new_end": int(b["end"]),
                    "hist_frames": a["frames"],
                    "new_frames": b["frames"],
                    "hist_last_box": a["last_box"],
                    "new_first_box": b["first_box"],
                    "hist_velocity": a["velocity"],
                    "candidates": candidates,
                }
            )
    return episodes


def build_decision_episodes(
    sequence: str,
    rows: Dict[int, List[Tuple[int, np.ndarray]]],
    gt: Dict[int, GTFrame],
    mem_len: int = 10,
    max_mem_gap: int = 60,
) -> List[dict]:
    """Per-frame decision episodes: identity memory vs current rows.

    For every GT identity present at frame f, the positive row is the P0 row
    matched by Hungarian IoU>=0.5; negatives are all other rows at f.  The
    memory is the identity's matched frames before f (at most mem_len frames,
    within max_mem_gap frames of f).
    """
    hist: Dict[int, List[int]] = {}
    gid_to_tid: Dict[int, Dict[int, int]] = {}
    episodes = []
    max_frame = max(rows) if rows else 0
    for f in range(max_frame + 1):
        gtf = gt.get(f)
        frame_rows = rows.get(f, [])
        if gtf is None or not gtf.boxes or not frame_rows:
            _prune_hist(hist, f, max_mem_gap)
            continue
        matches = match_boxes(
            [np.asarray(b, float) for b in gtf.boxes],
            [np.asarray(b, float) for _, b in frame_rows],
            0.5,
        )
        gid_to_tid[f] = {
            gtf.gt_ids[gi]: int(frame_rows[pi][0]) for gi, pi, _ in matches
        }
        matched_gi = {gi for gi, _, _ in matches}
        matched_pi = {pi for _, pi, _ in matches}
        for gi, pi, _iou in matches:
            gid = gtf.gt_ids[gi]
            tid = int(frame_rows[pi][0])
            mem = hist.get(gid, [])
            prev_tid = None
            if mem:
                prev_frame = mem[-1]
                prev_tid = gid_to_tid.get(prev_frame, {}).get(gid)
            gap = f - mem[-1] - 1 if mem else None
            episodes.append(
                {
                    "sequence": sequence,
                    "gid": int(gid),
                    "frame": int(f),
                    "pos_tid": tid,
                    "prev_tid": prev_tid,
                    "tid_changed": prev_tid is not None and prev_tid != tid,
                    "gap": gap,
                    "mem_frames": list(mem),
                    "neg_tids": [
                        int(frame_rows[j][0])
                        for j in range(len(frame_rows))
                        if j != pi and j not in matched_pi
                    ],
                    "crowd": len(frame_rows),
                    "miss": False,
                }
            )
            hist.setdefault(gid, []).append(f)
        # misses (RECOVERABLE_MISS-like) recorded for statistics
        for gi in range(len(gtf.boxes)):
            if gi in matched_gi:
                continue
            gid = gtf.gt_ids[gi]
            mem = hist.get(gid, [])
            episodes.append(
                {
                    "sequence": sequence,
                    "gid": int(gid),
                    "frame": int(f),
                    "pos_tid": None,
                    "prev_tid": gid_to_tid.get(mem[-1], {}).get(gid) if mem else None,
                    "tid_changed": False,
                    "gap": f - mem[-1] - 1 if mem else None,
                    "mem_frames": list(mem),
                    "neg_tids": [
                        int(frame_rows[j][0])
                        for j in range(len(frame_rows))
                        if j not in matched_pi
                    ],
                    "crowd": len(frame_rows),
                    "miss": True,
                }
            )
        _prune_hist(hist, f, max_mem_gap)
    return episodes


def _tid_at(
    rows: Dict[int, List[Tuple[int, np.ndarray]]],
    frame: int,
    gid: int,
    gt: Dict[int, GTFrame],
) -> Optional[int]:
    gtf = gt.get(frame)
    if gtf is None or frame not in rows:
        return None
    matches = match_boxes(
        [np.asarray(b, float) for b in gtf.boxes],
        [np.asarray(b, float) for _, b in rows[frame]],
        0.5,
    )
    for gi, pi, _ in matches:
        if gtf.gt_ids[gi] == gid:
            return int(rows[frame][pi][0])
    return None


def _prune_hist(hist: Dict[int, List[int]], frame: int, max_gap: int) -> None:
    for gid in list(hist):
        hist[gid] = [x for x in hist[gid] if frame - x <= max_gap]
        if not hist[gid]:
            del hist[gid]


def write_decision_csv(path: Path, episodes: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sequence",
                "gid",
                "frame",
                "pos_tid",
                "prev_tid",
                "tid_changed",
                "gap",
                "mem_frames",
                "neg_tids",
                "crowd",
                "miss",
            ]
        )
        for e in episodes:
            w.writerow(
                [
                    e["sequence"],
                    e["gid"],
                    e["frame"],
                    "" if e["pos_tid"] is None else e["pos_tid"],
                    "" if e["prev_tid"] is None else e["prev_tid"],
                    int(e["tid_changed"]),
                    "" if e["gap"] is None else e["gap"],
                    ",".join(str(x) for x in e["mem_frames"]),
                    ",".join(str(x) for x in e["neg_tids"]),
                    e["crowd"],
                    int(e["miss"]),
                ]
            )


def load_decision_csv(path: Path) -> List[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                {
                    "sequence": r["sequence"],
                    "gid": int(r["gid"]),
                    "frame": int(r["frame"]),
                    "pos_tid": None if r["pos_tid"] == "" else int(r["pos_tid"]),
                    "prev_tid": None if r["prev_tid"] == "" else int(r["prev_tid"]),
                    "tid_changed": r["tid_changed"] == "1",
                    "gap": None if r["gap"] == "" else int(r["gap"]),
                    "mem_frames": [int(x) for x in r["mem_frames"].split(",") if x],
                    "neg_tids": [int(x) for x in r["neg_tids"].split(",") if x],
                    "crowd": int(r["crowd"]),
                    "miss": r["miss"] == "1",
                }
            )
    return out


def _candidates(
    by_gid: Dict[int, List[dict]],
    positive_gid: int,
    frame: int,
    max_candidates: int,
    max_gap: int,
) -> List[dict]:
    cands = []
    for gid, gsegs in by_gid.items():
        if gid == positive_gid:
            continue
        active = [s for s in gsegs if s["start"] <= frame <= s["end"]]
        recent = [s for s in gsegs if s["end"] < frame and frame - s["end"] - 1 <= max_gap]
        pick = active[0] if active else (recent[-1] if recent else None)
        if pick is not None:
            cands.append(
                {
                    "gid": int(gid),
                    "hist_tid": int(pick["tid"]),
                    "hist_start": int(pick["start"]),
                    "hist_end": int(pick["end"]),
                    "hist_frames": pick["frames"],
                    "hist_last_box": pick["last_box"],
                    "hist_velocity": pick["velocity"],
                }
            )
        if len(cands) >= max_candidates:
            break
    return cands


def write_episodes_csv(path: Path, episodes: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sequence",
                "gid",
                "hist_tid",
                "new_tid",
                "gap",
                "hist_start",
                "hist_end",
                "new_start",
                "new_end",
                "hist_frames",
                "new_frames",
                "hist_last_box",
                "new_first_box",
                "hist_velocity",
                "candidate_gids",
                "candidate_tids",
                "candidate_frames",
                "candidate_last_box",
                "candidate_velocity",
            ]
        )
        for e in episodes:
            cand_gids = ";".join(str(c["gid"]) for c in e["candidates"])
            cand_tids = ";".join(str(c["hist_tid"]) for c in e["candidates"])
            cand_frames = "|".join(
                ",".join(str(x) for x in c["hist_frames"]) for c in e["candidates"]
            )
            cand_lb = "|".join(
                ";".join(f"{v:.2f}" for v in c["hist_last_box"]) for c in e["candidates"]
            )
            cand_vel = "|".join(
                ";".join(f"{v:.4f}" for v in c["hist_velocity"]) for c in e["candidates"]
            )
            w.writerow(
                [
                    e["sequence"],
                    e["gid"],
                    e["hist_tid"],
                    e["new_tid"],
                    e["gap"],
                    e["hist_start"],
                    e["hist_end"],
                    e["new_start"],
                    e["new_end"],
                    ",".join(str(x) for x in e["hist_frames"]),
                    ",".join(str(x) for x in e["new_frames"]),
                    ";".join(f"{v:.2f}" for v in e["hist_last_box"]),
                    ";".join(f"{v:.2f}" for v in e["new_first_box"]),
                    ";".join(f"{v:.4f}" for v in e["hist_velocity"]),
                    cand_gids,
                    cand_tids,
                    cand_frames,
                    cand_lb,
                    cand_vel,
                ]
            )


def load_episodes_csv(path: Path) -> List[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                {
                    "sequence": r["sequence"],
                    "gid": int(r["gid"]),
                    "hist_tid": int(r["hist_tid"]),
                    "new_tid": int(r["new_tid"]),
                    "gap": int(r["gap"]),
                    "hist_start": int(r["hist_start"]),
                    "hist_end": int(r["hist_end"]),
                    "new_start": int(r["new_start"]),
                    "new_end": int(r["new_end"]),
                    "hist_frames": [int(x) for x in r["hist_frames"].split(",") if x],
                    "new_frames": [int(x) for x in r["new_frames"].split(",") if x],
                    "hist_last_box": [float(x) for x in r["hist_last_box"].split(";")],
                    "new_first_box": [float(x) for x in r["new_first_box"].split(";")],
                    "hist_velocity": [float(x) for x in r["hist_velocity"].split(";")],
                    "candidates": [
                        {
                            "gid": int(g),
                            "hist_tid": int(t),
                            "hist_start": None,
                            "hist_end": None,
                            "hist_frames": [int(x) for x in fs.split(",") if x],
                            "hist_last_box": [float(x) for x in lb.split(";")],
                            "hist_velocity": [float(x) for x in v.split(";")],
                        }
                        for g, t, fs, lb, v in zip(
                            r["candidate_gids"].split(";") if r["candidate_gids"] else [],
                            r["candidate_tids"].split(";") if r["candidate_tids"] else [],
                            r["candidate_frames"].split("|") if r["candidate_frames"] else [],
                            r["candidate_last_box"].split("|") if r["candidate_last_box"] else [],
                            r["candidate_velocity"].split("|") if r["candidate_velocity"] else [],
                        )
                    ],
                }
            )
    return out
