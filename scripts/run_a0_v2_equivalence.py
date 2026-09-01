#!/usr/bin/env python
"""Budget-0 A0 equivalence run using the unified identity registry.

Writes results to outputs/n1_5/a0_v2_mot so the frozen sam31_auto_v1 outputs
under outputs/n1 are never overwritten.
"""

import subprocess
from pathlib import Path

import numpy as np
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.evaluation.mot_export import export_mot_file, validate_mot_file
from sam3_intermot.identity.registry import ObjectIdentityRegistry
from sam3_intermot.utils.io import atomic_write_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
SEQS = ["dancetrack0004", "dancetrack0005", "dancetrack0007"]
WINDOW = 200


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    data = Path(cfg["dataset"]["root"]) / cfg["dataset"]["split"]
    out = ROOT / "outputs" / "n1_5" / "a0_v2_mot"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in SEQS:
        video = data / seq / "img1"
        backend = Sam3Backend(
            checkpoint_path=cfg["backend"]["checkpoint_path"],
            max_num_objects=cfg["backend"]["max_num_objects"],
            multiplex_count=cfg["backend"]["multiplex_count"],
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
        registry = ObjectIdentityRegistry()
        try:
            backend.start_video(str(video))
            num_frames = len(list(video.glob("*.jpg")))
            det0 = 0
            for start in range(0, num_frames, WINDOW):
                end = min(start + WINDOW - 1, num_frames - 1)
                dets = backend.detect_concept(start, "person")
                if start == 0:
                    det0 = len(dets)
                registry.unbind_all_for_window()
                for obs in dets:
                    registry.register_auto_object(start, obs)
                prop = backend.propagate(
                    start, end, start_frame_index=start, keep_masks=False
                )
                backend._output_cache.clear()
                for f in range(start, end + 1):
                    for obs in prop.get(f, []):
                        registry.register_auto_object(f, obs)
                    seen = {o.sam_object_id for o in prop.get(f, [])}
                    for t in registry.manager.active_tracks():
                        if (
                            t.sam_object_id is not None
                            and t.sam_object_id not in seen
                            and t.last_seen_frame < f
                        ):
                            registry.manager.mark_missed(t.mot_track_id, f)
            outputs_by_frame = {
                f: list(m.items())
                for f, m in registry.manager._outputs.items()
                if m
            }
            mot_path = out / (seq + ".txt")
            export_mot_file(mot_path, outputs_by_frame)
            violations = validate_mot_file(mot_path, num_frames=num_frames)
            inv = registry.invariant_violations()
            rows.append({
                "sequence": seq,
                "status": "PASS" if not violations and not inv else "FAIL",
                "num_frames": num_frames,
                "det0": det0,
                "num_tracks": len(registry.manager.tracks),
                "num_rows": sum(len(v) for v in outputs_by_frame.values()),
                "validation_violations": ";".join(violations),
                "invariant_violations": ";".join(inv),
            })
            print(rows[-1])
        finally:
            backend.close()
    write_csv(ROOT / "outputs" / "n1_5" / "a0_v2_sequence_stats.csv", rows)
    _trackeval(out)


def _trackeval(mot_dir):
    gt = Path("/path/to/dancetrack/val")
    seqmap = ROOT / "outputs" / "n1_5" / "a0_v2_seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(SEQS) + "\n", encoding="utf-8")
    cmd = [
        "python",
        str(Path("./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py")),
        "--GT_FOLDER", str(gt),
        "--TRACKERS_FOLDER", str(mot_dir.parent),
        "--TRACKERS_TO_EVAL", "a0_v2_mot",
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
    (ROOT / "outputs" / "n1_5" / "a0_v2_trackeval.log").write_text(
        proc.stdout + proc.stderr, encoding="utf-8"
    )
    print("A0_V2_TRACKEVAL_RC", proc.returncode)
    print(proc.stdout[-3000:])


if __name__ == "__main__":
    main()
