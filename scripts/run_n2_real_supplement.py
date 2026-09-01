#!/usr/bin/env python
"""Targeted N2 Add/Reassign real events on the 120-frame clip."""

import json
from pathlib import Path

import numpy as np
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.interaction.actions import HumanInteraction, SystemContext
from sam3_intermot.interaction.add import perform_add
from sam3_intermot.interaction.reassign import perform_reassign
from sam3_intermot.tracking.track_manager import TrackManager
from sam3_intermot.utils.io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
NUM_FRAMES = 120
SEQ = "dancetrack0004"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    data_root = Path(cfg["dataset"]["root"])
    out = ROOT / "outputs" / "n2_real"
    clip = out / "clip120"
    (clip / SEQ).mkdir(parents=True, exist_ok=True)
    src_img = data_root / cfg["dataset"]["split"] / SEQ / "img1"
    for n in sorted(p.name for p in src_img.glob("*.jpg"))[:NUM_FRAMES]:
        dst = clip / SEQ / n
        if not dst.exists():
            dst.symlink_to(src_img / n)

    dataset = DanceTrackDataset(str(data_root), sequences=[SEQ], split=cfg["dataset"]["split"])
    full_gt = dataset.load_gt(SEQ)
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
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    ctx = SystemContext(backend=backend, manager=manager, lineages=lineages)
    ctx._next_sam_object_id = 1000
    try:
        backend.start_video(str(clip / SEQ))
        dets = backend.detect_concept(0, "person")
        for obs in dets:
            backend._objects[obs.sam_object_id] = {
                "box": backend._sanitize_box(obs.box_xyxy).copy(),
                "human_box": obs.box_xyxy.copy(),
                "frame": 0,
                "source": "concept_detection",
            }
            lineage = lineages.create(0)
            manager.create_track(0, obs, lineage.lineage_id)
            lineage.bind_track(manager.tracks[manager._sam_to_track[obs.sam_object_id]].mot_track_id)
        propagated = backend.propagate(0, NUM_FRAMES - 1, start_frame_index=0, keep_masks=False)
        backend._output_cache.clear()
        for f in range(NUM_FRAMES):
            for obs in propagated.get(f, []):
                track = _by_sam(manager, obs.sam_object_id)
                if track is not None:
                    manager.update_track(track.mot_track_id, f, obs)
                else:
                    lineage = lineages.create(f)
                    track = manager.create_track(f, obs, lineage.lineage_id)
                    lineage.bind_track(track.mot_track_id)

        # ---- Add: 8 current-frame GT boxes with track state cleared to
        # simulate a detection-unavailable / missed target event. ----
        add_results = []
        add_frames = [0, 15, 30, 45, 60, 75, 90, 105]
        for i, f in enumerate(add_frames):
            gt = full_gt.get(f)
            if gt is None or not gt.boxes:
                continue
            box = gt.boxes[i % len(gt.boxes)]
            # Clear last boxes so Add is not rejected as duplicate; this
            # models "the system had no usable observation for this target".
            for t in manager.active_tracks():
                t.last_box = None
            action = HumanInteraction(
                action_id="add_%d" % i,
                frame_idx=f,
                action_type="Add",
                box_xyxy=box,
                source="sim_gt_current_frame",
            )
            res = perform_add(ctx, action)
            add_results.append(res)

        # ---- Reassign: 8 identity transactions between existing tracks. ----
        reassign_results = []
        tracks = manager.active_tracks()
        for i in range(min(8, len(tracks) // 2)):
            src = tracks[2 * i]
            dst = tracks[2 * i + 1]
            f = 10 + i * 10
            manager.remove_output(f, dst.mot_track_id)
            action = HumanInteraction(
                action_id="reassign_%d" % i,
                frame_idx=f,
                action_type="Reassign",
                target_track_id=src.mot_track_id,
                destination_track_id=dst.mot_track_id,
                source="sim_user_decision",
            )
            res = perform_reassign(ctx, action)
            reassign_results.append(res)

        result = {
            "add_accepted": sum(1 for r in add_results if r.accepted),
            "add_total": len(add_results),
            "add_rejected": [r.reason for r in add_results if not r.accepted],
            "reassign_accepted": sum(1 for r in reassign_results if r.accepted),
            "reassign_total": len(reassign_results),
            "reassign_rejected": [r.reason for r in reassign_results if not r.accepted],
            "violations": manager.invariant_violations(),
        }
        atomic_write_json(out / "supplement_result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        backend.close()


def _by_sam(manager, sam_id):
    for t in manager.active_tracks():
        if t.sam_object_id == sam_id:
            return t
    return None


if __name__ == "__main__":
    main()
