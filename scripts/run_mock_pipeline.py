#!/usr/bin/env python
"""Mock end-to-end pipeline for tests and development only.

Outputs from this script are synthetic and MUST NOT be reported as real
SAM 3.1 results.
"""

from pathlib import Path

import numpy as np

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.evaluation.mot_export import export_mot_file
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.interaction.actions import ActionType
from sam3_intermot.interaction.simulator import (
    GTFrame,
    SimulatedInteractionDriver,
    SimulatorConfig,
)
from sam3_intermot.tracking.track_manager import TrackManager
from sam3_intermot.utils.io import atomic_write_json, write_csv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "mock_demo"
    backend = MockBackend(frame_h=1080, frame_w=1920, seed=7)
    backend.start_video("mock://demo")
    # Two deterministic concept detections every frame (test double only).
    backend.set_concept_boxes(
        "person",
        [
            np.asarray([100, 100, 160, 260]),
            np.asarray([400, 120, 470, 290]),
        ],
    )
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    gt_frames = {}
    for f in range(30):
        gt = GTFrame()
        gt.boxes = [
            np.asarray([100 + f, 100 + f, 160 + f, 260 + f]),
            np.asarray([400 - f, 120, 470 - f, 290]),
        ]
        gt.gt_ids = [1, 2]
        gt_frames[f] = gt
    cfg = SimulatorConfig(
        enabled_actions={ActionType.ADD, ActionType.CORRECT, ActionType.REASSIGN, ActionType.DELETE},
        budget_per_100_frames=5,
    )
    driver = SimulatedInteractionDriver(backend, manager, lineages, cfg)
    summary = driver.run(gt_frames, num_frames=30)
    outputs_by_frame = {
        f: [(t.mot_track_id, obs) for t, obs in _frame_entries(manager, f)]
        for f in range(30)
    }
    export_mot_file(out / "mock_results.txt", outputs_by_frame)
    atomic_write_json(
        out / "mock_summary.json",
        {
            "events": [e.__dict__ for e in summary["events"]],
            "results": [r.__dict__ for r in summary["results"]],
            "actions_used": summary["actions_used"],
            "invariant_violations": manager.invariant_violations(),
        },
    )
    write_csv(
        out / "mock_events.csv",
        [{"frame": e.frame_idx, "event": e.event_type.value, "track": e.track_id, "gt": e.gt_id} for e in summary["events"]],
    )
    print("mock demo done; results are synthetic, not SAM 3.1 real results")


def _frame_entries(manager, frame_idx):
    from sam3_intermot.tracking.track import TrackState

    for track in manager.tracks.values():
        if track.state in (TrackState.TERMINATED, TrackState.DELETED):
            continue
        for obs in manager.outputs_for_frame(frame_idx):
            if obs.sam_object_id == track.sam_object_id:
                yield track, obs
                break


if __name__ == "__main__":
    main()
