#!/usr/bin/env python
"""Real SAM 3.1 automatic MOT pipeline (N1).

Uses the real mirror checkpoint on GPU 8/9.  The pipeline is intentionally
simple and auditable: one ``person`` concept prompt on frame 0, one full
propagation pass, then TrackManager assigns stable MOT IDs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.evaluation.mot_export import export_mot_file, validate_mot_file
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.tracking.association import box_iou, center_distance
from sam3_intermot.tracking.track_manager import TrackManager
from sam3_intermot.utils.io import atomic_write_json, write_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "single", "three", "all"],
        default="smoke",
    )
    parser.add_argument("--gpu", type=int, default=8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text())
    data_root = Path(cfg["dataset"]["root"]) / cfg["dataset"]["split"]
    seqmap = data_root / "val_seqmap.txt"
    all_seqs = [
        line.split()[0]
        for line in seqmap.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if args.mode == "smoke":
        jobs = [("smoke", root / "outputs" / "n0" / "backend_examples" / "video_frames")]
    elif args.mode == "single":
        jobs = [("dancetrack0004", data_root / "dancetrack0004" / "img1")]
    elif args.mode == "three":
        jobs = [
            (s, data_root / s / "img1")
            for s in ["dancetrack0004", "dancetrack0005", "dancetrack0007"]
        ]
    else:
        jobs = [(s, data_root / s / "img1") for s in all_seqs]

    out_root = root / "outputs" / "n1"
    mot_dir = out_root / "mot_results" / "sam3_auto"
    mot_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    per_seq = []
    for seq, video_dir in jobs:
        t0 = time.time()
        entry = _run_sequence(root, cfg, seq, video_dir, mot_dir, gpu=args.gpu)
        entry["wall_clock_seconds"] = round(time.time() - t0, 2)
        logs.append(entry)
        per_seq.append(
            {
                "sequence": seq,
                "frames": entry["num_frames"],
                "tracks": entry["num_tracks"],
                "rows": entry["num_rows"],
                "violations": ";".join(entry["validation_violations"]),
                "status": entry["status"],
            }
        )
        print(json.dumps(entry, ensure_ascii=False))
        if args.mode != "all":
            if entry["status"] != "PASS":
                print(f"STOP: sequence {seq} failed")
                break

    write_csv(out_root / "per_sequence_metrics.csv", per_seq)
    atomic_write_json(
        out_root / "run_log.json",
        {
            "mode": args.mode,
            "sequences": logs,
            "gpu": args.gpu,
        },
    )

    if args.mode != "smoke" and all(e["status"] == "PASS" for e in logs):
        _run_trackeval(root, out_root, mot_dir, seqmap)
    return 0 if all(e["status"] == "PASS" for e in logs) else 1


def _run_sequence(root, cfg, seq, video_dir, mot_dir, gpu):
    backend = Sam3Backend(
        checkpoint_path=cfg["backend"]["checkpoint_path"],
        max_num_objects=cfg["backend"]["max_num_objects"],
        multiplex_count=cfg["backend"]["multiplex_count"],
        use_fa3=cfg["backend"].get("use_fa3", False),
        use_rope_real=cfg["backend"].get("use_rope_real", True),
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    try:
        backend.start_video(str(video_dir))
        num_frames = len(list(video_dir.glob("*.jpg")))
        manager = TrackManager()
        lineages = IdentityLineageRegistry()
        window = 200
        detections_frame0 = 0
        for start in range(0, num_frames, window):
            end = min(start + window - 1, num_frames - 1)
            detections = backend.detect_concept(start, "person")
            if start == 0:
                detections_frame0 = len(detections)
            # Model object ids are reused every window; clear stale sam
            # bindings before associating the new window's observations.
            for track in list(manager.active_tracks()):
                manager.unbind_sam_object(track.mot_track_id)
            for obs in detections:
                _associate_or_create(manager, lineages, start, obs)
            propagated = backend.propagate(
                start, end, start_frame_index=start, keep_masks=False
            )
            backend._output_cache.clear()
            for frame_idx in range(start, end + 1):
                for obs in propagated.get(frame_idx, []):
                    _associate_or_create(manager, lineages, frame_idx, obs)
                seen = {o.sam_object_id for o in propagated.get(frame_idx, [])}
                for track in manager.active_tracks():
                    if (
                        track.sam_object_id is not None
                        and track.sam_object_id not in seen
                        and track.last_seen_frame < frame_idx
                    ):
                        manager.mark_missed(track.mot_track_id, frame_idx)
        outputs_by_frame = {
            f: list(mapping.items())
            for f, mapping in manager._outputs.items()
            if mapping
        }
        mot_path = mot_dir / f"{seq}.txt"
        export_mot_file(mot_path, outputs_by_frame)
        violations = validate_mot_file(mot_path, num_frames=num_frames)
        inv = manager.invariant_violations()
        status = "PASS" if not violations and not inv else "FAIL"
        return {
            "sequence": seq,
            "status": status,
            "num_frames": num_frames,
            "num_detections_frame0": detections_frame0,
            "num_tracks": len(manager.tracks),
            "num_rows": sum(len(v) for v in outputs_by_frame.values()),
            "validation_violations": violations,
            "invariant_violations": inv,
            "mot_path": str(mot_path),
        }
    finally:
        backend.close()


def _track_for_sam(manager, sam_object_id):
    for track in manager.active_tracks():
        if track.sam_object_id == sam_object_id:
            return track
    return None


def _associate_or_create(manager, lineages, frame_idx, obs):
    best = None
    best_score = float("-inf")
    # First pass: prefer the track that already owns this SAM object id.
    for track in manager.active_tracks():
        if track.sam_object_id != obs.sam_object_id or track.last_box is None:
            continue
        iou = box_iou(obs.box_xyxy, track.last_box)
        dist = center_distance(obs.box_xyxy, track.last_box)
        score = iou - 1e-4 * dist
        if score > best_score:
            best_score = score
            best = track
    if best is None:
        # Second pass: box-overlap association for new / re-detected objects.
        for track in manager.active_tracks():
            if track.last_box is None:
                continue
            iou = box_iou(obs.box_xyxy, track.last_box)
            if iou < 0.2:
                continue
            dist = center_distance(obs.box_xyxy, track.last_box)
            score = iou - 1e-4 * dist
            if score > best_score:
                best_score = score
                best = track
    if best is not None:
        if best.sam_object_id != obs.sam_object_id:
            manager.rebind_sam_object(best.mot_track_id, obs.sam_object_id, frame_idx)
        manager.update_track(best.mot_track_id, frame_idx, obs)
        return best
    lineage = lineages.create(frame_idx)
    track = manager.create_track(frame_idx, obs, lineage.lineage_id)
    lineage.bind_track(track.mot_track_id)
    return track


def _run_trackeval(root, out_root, mot_dir, seqmap):
    import subprocess

    trackeval_root = (
        Path(".")
        / "third_party"
        / "MOTIP"
        / "TrackEval"
    )
    script = trackeval_root / "scripts" / "run_mot_challenge.py"
    python = "python"
    gt_dir = seqmap.parent
    clean_seqmap = out_root / "dancetrack_val_seqmap_clean.txt"
    available = sorted(
        p.stem for p in mot_dir.glob("*.txt") if (gt_dir / p.stem).is_dir()
    )
    clean_seqmap.write_text(
        "name\n" + "\n".join(available) + "\n",
        encoding="utf-8",
    )
    log_dir = out_root / "trackeval"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(script),
        "--GT_FOLDER",
        str(gt_dir),
        "--TRACKERS_FOLDER",
        str(mot_dir.parent),
        "--TRACKERS_TO_EVAL",
        "sam3_auto",
        "--TRACKER_SUB_FOLDER",
        "",
        "--OUTPUT_SUB_FOLDER",
        "",
        "--SEQMAP_FILE",
        str(clean_seqmap),
        "--BENCHMARK",
        "DanceTrack",
        "--SPLIT_TO_EVAL",
        "val",
        "--SKIP_SPLIT_FOL",
        "True",
        "--DO_PREPROC",
        "False",
        "--CLASSES_TO_EVAL",
        "pedestrian",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--USE_PARALLEL",
        "False",
        "--PLOT_CURVES",
        "False",
        "--PRINT_RESULTS",
        "True",
        "--PRINT_ONLY_COMBINED",
        "False",
        "--OUTPUT_SUMMARY",
        "True",
        "--OUTPUT_DETAILED",
        "True",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (log_dir / "trackeval.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    print("TRACKEVAL_RC", proc.returncode)
    print(proc.stdout[-4000:])


if __name__ == "__main__":
    raise SystemExit(main())
