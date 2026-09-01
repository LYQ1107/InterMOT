import numpy as np
import pytest

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.interaction.simulator import GTFrame, GTView
from sam3_intermot.tracking.track_manager import TrackManager


def test_gt_view_blocks_future_access():
    view = GTView({5: GTFrame()}, current=5)
    with pytest.raises(RuntimeError):
        view.frame(6)
    assert view.frame(5) is not None


def test_driver_results_only_reference_current_frames():
    from sam3_intermot.interaction.actions import SystemContext
    from sam3_intermot.interaction.simulator import (
        SimulatedInteractionDriver,
        SimulatorConfig,
    )

    backend = MockBackend()
    backend.start_video("x")
    backend.set_concept_boxes("person", [np.asarray([10, 10, 50, 80])])
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    gt = {}
    for f in range(10):
        g = GTFrame()
        g.boxes = [np.asarray([10 + f, 10, 50 + f, 80])]
        g.gt_ids = [1]
        gt[f] = g
    driver = SimulatedInteractionDriver(
        backend, manager, lineages, SimulatorConfig(budget_per_100_frames=0)
    )
    summary = driver.run(gt, num_frames=10)
    for r in summary["results"]:
        assert 0 <= r.frame_idx < 10
    assert summary["leakage"].clean
