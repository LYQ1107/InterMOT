"""Invariant checks shared by tests and evaluators."""

from typing import List

from sam3_intermot.tracking.track_manager import TrackManager


def collect_invariant_violations(manager: TrackManager) -> List[str]:
    return manager.invariant_violations()


def assert_no_same_frame_duplicate_ids(manager: TrackManager) -> List[str]:
    violations = []
    for frame_idx, mapping in manager._outputs.items():
        track_ids = list(mapping.keys())
        if len(track_ids) != len(set(track_ids)):
            violations.append(f"frame {frame_idx}: duplicate track ids")
    return violations
