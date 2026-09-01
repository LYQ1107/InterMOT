#!/usr/bin/env python
"""Run frozen P0 (A0_v2 windows + registry) on one DanceTrack train sequence."""

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.evaluation.mot_export import export_mot_file, validate_mot_file
from sam3_intermot.identity.registry import ObjectIdentityRegistry


ROOT = Path(__file__).resolve().parents[1]
SEQ = os.environ["N9_SEQ"]
OUT_DIR = Path(os.environ.get("N9_OUT_DIR", ROOT / "outputs/n9/p0_train"))
SPLIT = os.environ.get("N9_SPLIT", "train")
WINDOW = 200


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    video = (
        Path("/path/to/dancetrack")
        / SPLIT
        / SEQ
        / "img1"
    )
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
    t0 = time.time()
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
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        mot_path = OUT_DIR / f"{SEQ}.txt"
        export_mot_file(mot_path, outputs_by_frame)
        violations = validate_mot_file(mot_path, num_frames=num_frames)
        inv = registry.invariant_violations()
        summary = {
            "sequence": SEQ,
            "split": SPLIT,
            "status": "PASS" if not violations and not inv else "FAIL",
            "num_frames": num_frames,
            "det0": det0,
            "num_tracks": len(registry.manager.tracks),
            "num_rows": sum(len(v) for v in outputs_by_frame.values()),
            "validation_violations": violations,
            "invariant_violations": inv,
            "wall_seconds": round(time.time() - t0, 2),
        }
        (OUT_DIR / f"{SEQ}.summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
