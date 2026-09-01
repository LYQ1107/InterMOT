"""Track state machine transitions."""

from dataclasses import dataclass

from sam3_intermot.tracking.track import Track, TrackState


@dataclass
class LifecycleConfig:
    confirm_after_frames: int = 5
    lost_after_frames: int = 10
    max_lost_frames: int = 45
    tombstone_cooldown_frames: int = 30


def update_state(track: Track, frame_idx: int, matched: bool, cfg: LifecycleConfig) -> None:
    if track.state in (TrackState.TERMINATED, TrackState.DELETED):
        return
    if matched:
        track.time_since_update = 0
        if track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.TENTATIVE and track.age >= cfg.confirm_after_frames:
            track.state = TrackState.CONFIRMED
        return
    track.time_since_update += 1
    if track.state in (TrackState.TENTATIVE, TrackState.CONFIRMED):
        if track.time_since_update >= cfg.lost_after_frames:
            track.state = TrackState.LOST
    elif track.state == TrackState.LOST:
        if track.time_since_update >= cfg.max_lost_frames:
            track.state = TrackState.TERMINATED
