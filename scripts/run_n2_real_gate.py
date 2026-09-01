#!/usr/bin/env python
"""N2 real interaction gate on a 120-frame DanceTrack clip."""

import json
from pathlib import Path

import numpy as np
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.interaction.actions import ActionType
from sam3_intermot.interaction.simulator import (
    GTFrame,
    SimulatedInteractionDriver,
    SimulatorConfig,
)
from sam3_intermot.tracking.track_manager import TrackManager
from sam3_intermot.utils.io import atomic_write_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
NUM_FRAMES = 120
SEQ = "dancetrack0004"


class RealAutoDriver(SimulatedInteractionDriver):
    def __init__(self, backend, manager, lineages, config, propagated, num_frames):
        super().__init__(backend, manager, lineages, config)
        self.propagated = propagated
        self.num_frames = num_frames

    def _automatic_step(self, frame_idx: int) -> None:
        observations = self.propagated.get(frame_idx, [])
        for obs in observations:
            if obs.sam_object_id not in self.backend._objects:
                self.backend._objects[obs.sam_object_id] = {
                    "box": self.backend._sanitize_box(obs.box_xyxy).copy(),
                    "human_box": obs.box_xyxy.copy(),
                    "frame": frame_idx,
                    "source": "concept_detection",
                }
        seen_sam = set()
        for obs in observations:
            if obs.sam_object_id in seen_sam:
                continue
            seen_sam.add(obs.sam_object_id)
            if (
                obs.sam_object_id in self.manager._tombstones
                and frame_idx - self.manager._tombstones[obs.sam_object_id]
                < self.manager.lifecycle.tombstone_cooldown_frames
            ):
                continue
            track = self._track_for_sam(obs.sam_object_id)
            if track is not None:
                self.manager.update_track(track.mot_track_id, frame_idx, obs)
            else:
                lineage = self.lineages.create(frame_idx)
                track = self.manager.create_track(frame_idx, obs, lineage.lineage_id)
                lineage.bind_track(track.mot_track_id)
        matched = {o.sam_object_id for o in observations}
        for track in self.manager.active_tracks():
            if track.sam_object_id is not None and track.sam_object_id not in matched:
                self.manager.mark_missed(track.mot_track_id, frame_idx)

    def _interaction_step(self, gt, frame_idx, max_actions):
        super()._interaction_step(gt, frame_idx, max_actions)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    data = Path(cfg["dataset"]["root"]) / cfg["dataset"]["split"]
    out = ROOT / "outputs" / "n2_real"
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "clip120"
    (clip / SEQ).mkdir(parents=True, exist_ok=True)
    src_img = data / SEQ / "img1"
    names = sorted(p.name for p in src_img.glob("*.jpg"))[:NUM_FRAMES]
    for n in names:
        dst = clip / SEQ / n
        if not dst.exists():
            dst.symlink_to(src_img / n)

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

        dataset = DanceTrackDataset(
            str(Path(cfg["dataset"]["root"])),
            sequences=[SEQ],
            split=cfg["dataset"]["split"],
        )
        full_gt = dataset.load_gt(SEQ)
        gt_frames = {f: full_gt[f] for f in range(NUM_FRAMES) if f in full_gt}
        config = SimulatorConfig(
            enabled_actions={ActionType.ADD, ActionType.CORRECT, ActionType.REASSIGN, ActionType.DELETE},
            budget_per_100_frames=100,
            detection_interval=1000,
        )
        driver = RealAutoDriver(backend, manager, lineages, config, propagated, NUM_FRAMES)
        driver.ctx._next_sam_object_id = 1000
        summary = driver.run(gt_frames, NUM_FRAMES)
        write_csv(
            out / "interaction_events.csv",
            [
                {
                    "frame": e.frame_idx,
                    "event_type": e.event_type.value,
                    "track_id": e.track_id,
                    "gt_id": e.gt_id,
                    "detail": e.detail,
                }
                for e in summary["events"]
            ],
        )
        with (out / "transactions.jsonl").open("w", encoding="utf-8") as f:
            for r in summary["results"]:
                f.write(json.dumps(r.__dict__, ensure_ascii=False, default=str) + "\n")
        write_csv(
            out / "per_event_metrics.csv",
            [
                {
                    "action_id": r.action_id,
                    "action_type": r.action_type,
                    "frame_idx": r.frame_idx,
                    "accepted": r.accepted,
                    "rolled_back": r.rolled_back,
                    "reason": r.reason,
                }
                for r in summary["results"]
            ],
        )
        violations = list(manager.invariant_violations())
        if not driver.leakage.clean:
            violations.append("future_gt_leakage_detected")
        write_csv(
            out / "invariant_violations.csv",
            [{"frame_idx": "", "violation": v} for v in violations],
        )
        by_type = {}
        for r in summary["results"]:
            by_type.setdefault(r.action_type, {"accepted": 0, "rejected": 0, "rolled_back": 0})
            by_type[r.action_type]["accepted" if r.accepted else "rejected"] += 1
            if r.rolled_back:
                by_type[r.action_type]["rolled_back"] += 1
        result = {
            "status": "PASS" if (
                by_type.get("Add", {}).get("accepted", 0) >= 1
                and by_type.get("Correct", {}).get("accepted", 0) >= 1
                and by_type.get("Reassign", {}).get("accepted", 0) >= 1
                and by_type.get("Delete", {}).get("accepted", 0) >= 1
                and not violations
            ) else "FAIL",
            "by_type": by_type,
            "events": [e.__dict__ for e in summary["events"]],
            "actions_used": summary["actions_used"],
            "invariant_violations": violations,
            "leakage": vars(summary["leakage"]),
        }
        atomic_write_json(out / "gate_result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
