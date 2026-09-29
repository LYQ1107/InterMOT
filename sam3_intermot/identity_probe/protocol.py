"""Deterministic, score-blind N72R16 anchor protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

from .dataset import DanceTrackSequence, GTBox


PROTOCOL_VERSION = "N72R16-probe-v1"
FINAL_GOAL = "Human Identity Representation Probe"
CENTRAL_QUESTION = "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?"


@dataclass(frozen=True)
class FutureObservation:
    frame: int
    target: GTBox
    competitors: tuple[GTBox, ...]


@dataclass(frozen=True)
class Anchor:
    sequence: str
    track_id: int
    anchor_frame: int
    anchor: GTBox
    future: tuple[FutureObservation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "track_id": self.track_id,
            "anchor_frame": self.anchor_frame,
            "anchor": self.anchor.as_dict(),
            "future": [
                {
                    "frame": item.frame,
                    "target": item.target.as_dict(),
                    "competitors": [box.as_dict() for box in item.competitors],
                }
                for item in self.future
            ],
        }


def _box_from_dict(data: dict[str, object]) -> GTBox:
    left, top, width, height = (float(value) for value in data["tlwh"])
    return GTBox(
        sequence=str(data["sequence"]),
        frame=int(data["frame"]),
        track_id=int(data["track_id"]),
        left=left,
        top=top,
        width=width,
        height=height,
        confidence=float(data.get("confidence", 1.0)),
        class_id=int(data.get("class_id", 1)),
        visibility=float(data.get("visibility", 1.0)),
    )


def anchor_from_dict(data: dict[str, object]) -> Anchor:
    future = tuple(
        FutureObservation(
            frame=int(item["frame"]),
            target=_box_from_dict(item["target"]),
            competitors=tuple(_box_from_dict(box) for box in item["competitors"]),
        )
        for item in data["future"]
    )
    return Anchor(
        sequence=str(data["sequence"]),
        track_id=int(data["track_id"]),
        anchor_frame=int(data["anchor_frame"]),
        anchor=_box_from_dict(data["anchor"]),
        future=future,
    )


def build_anchors(
    sequences: list[DanceTrackSequence],
    horizon: int = 100,
    min_future_observations: int = 20,
) -> list[Anchor]:
    """Build one earliest eligible anchor per identity without model scores."""
    if horizon < 1 or min_future_observations < 1:
        raise ValueError("horizon and min_future_observations must be positive")
    anchors: list[Anchor] = []
    for sequence in sequences:
        by_id: dict[int, list[GTBox]] = {}
        for box in sequence.boxes():
            by_id.setdefault(box.track_id, []).append(box)
        for track_id in sorted(by_id):
            observations = sorted(by_id[track_id], key=lambda box: box.frame)
            for anchor_box in observations:
                end_frame = min(sequence.seq_length, anchor_box.frame + horizon)
                future: list[FutureObservation] = []
                all_future_ids: set[int] = set()
                for frame in range(anchor_box.frame + 1, end_frame + 1):
                    frame_boxes = tuple(sequence.boxes(frame))
                    target = next((box for box in frame_boxes if box.track_id == track_id), None)
                    if target is None:
                        continue
                    competitors = tuple(box for box in frame_boxes if box.track_id != track_id and box.track_id >= 0)
                    all_future_ids.update(box.track_id for box in frame_boxes if box.track_id >= 0)
                    future.append(FutureObservation(frame, target, competitors))
                competitive_frames = sum(bool(item.competitors) for item in future)
                if (
                    len(future) >= min_future_observations
                    and len(all_future_ids) >= 2
                    and competitive_frames >= 1
                ):
                    anchors.append(
                        Anchor(
                            sequence=sequence.name,
                            track_id=track_id,
                            anchor_frame=anchor_box.frame,
                            anchor=anchor_box,
                            future=tuple(future),
                        )
                    )
                    break
    return anchors


def protocol_document(
    split: str,
    anchors: list[Anchor],
    dataset_root: str,
    horizon: int = 100,
    min_future_observations: int = 20,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "N72R16",
        "goal": FINAL_GOAL,
        "central_question": CENTRAL_QUESTION,
        "split": split,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "horizon": horizon,
        "min_future_observations": min_future_observations,
        "anchor_rule": "For each sequence and identity, choose the earliest GT anchor frame whose next H100 contains at least 20 same-ID observations and at least one competing identity; selection uses no embedding or future similarity score.",
        "id_policy": "track_id >= 0; DanceTrack release contains valid zero-based identity ID 0; negative IDs are invalid.",
        "future_frames_are_gt_references_only": True,
        "anchors": [anchor.to_dict() for anchor in anchors],
    }


def write_protocol(path: str | Path, document: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_protocol(path: str | Path) -> dict[str, object]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {document.get('protocol_version')!r}")
    if document.get("goal") != FINAL_GOAL:
        raise ValueError("protocol Goal does not match the frozen N72R16 Final Goal")
    return document


def iter_anchors(document: dict[str, object]) -> Iterator[Anchor]:
    for item in document.get("anchors", []):
        yield anchor_from_dict(item)
