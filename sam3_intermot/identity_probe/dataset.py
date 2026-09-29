"""Read-only DanceTrack MOT data access for N72R16.

The probe reads the official MOT-format files in place.  It never writes
cropped images or modifies the supplied annotations.  DanceTrack's released
annotations use zero-based identity IDs in some sequences, so ID ``0`` is a
valid identity here; negative IDs remain invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class GTBox:
    sequence: str
    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float
    class_id: int
    visibility: float

    @property
    def tlwh(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.width, self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width / 2.0, self.top + self.height / 2.0

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "frame": self.frame,
            "track_id": self.track_id,
            "tlwh": [self.left, self.top, self.width, self.height],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "visibility": self.visibility,
        }


def _read_seqinfo(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


class DanceTrackSequence:
    """One sequence with lazily read images and eagerly parsed GT boxes."""

    def __init__(self, sequence_dir: Path, split: str):
        self.sequence_dir = sequence_dir
        self.split = split
        self.name = sequence_dir.name
        self.seqinfo_path = sequence_dir / "seqinfo.ini"
        self.gt_path = sequence_dir / "gt" / "gt.txt"
        self.image_dir = sequence_dir / "img1"
        if not self.seqinfo_path.is_file():
            raise FileNotFoundError(f"missing seqinfo.ini: {self.seqinfo_path}")
        if not self.gt_path.is_file():
            raise FileNotFoundError(f"missing gt.txt: {self.gt_path}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"missing img1 directory: {self.image_dir}")

        self.seqinfo = _read_seqinfo(self.seqinfo_path)
        required = ("seqLength", "imWidth", "imHeight")
        missing = [key for key in required if key not in self.seqinfo]
        if missing:
            raise ValueError(f"{self.seqinfo_path} missing keys: {missing}")
        self.seq_length = int(self.seqinfo["seqLength"])
        self.image_width = int(self.seqinfo["imWidth"])
        self.image_height = int(self.seqinfo["imHeight"])
        self.image_paths = self._index_images()
        self.gt_by_frame = self._read_gt()

    def _index_images(self) -> dict[int, Path]:
        paths: dict[int, Path] = {}
        for path in sorted(self.image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                frame = int(path.stem)
            except ValueError:
                continue
            if frame in paths:
                raise ValueError(f"duplicate image frame {self.name}:{frame}")
            paths[frame] = path
        return paths

    def _read_gt(self) -> dict[int, list[GTBox]]:
        by_frame: dict[int, list[GTBox]] = {}
        for line_number, raw in enumerate(self.gt_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                raise ValueError(f"{self.gt_path}:{line_number}: fewer than 6 fields")
            try:
                frame = int(float(fields[0]))
                track_id = int(float(fields[1]))
                left, top, width, height = (float(fields[i]) for i in range(2, 6))
                confidence = float(fields[6]) if len(fields) >= 7 else 1.0
                class_id = int(float(fields[7])) if len(fields) >= 8 else 1
                visibility = float(fields[8]) if len(fields) >= 9 else 1.0
            except ValueError as exc:
                raise ValueError(f"{self.gt_path}:{line_number}: invalid numeric field") from exc
            box = GTBox(
                sequence=self.name,
                frame=frame,
                track_id=track_id,
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=confidence,
                class_id=class_id,
                visibility=visibility,
            )
            by_frame.setdefault(frame, []).append(box)
        for frame in by_frame:
            by_frame[frame].sort(key=lambda box: box.track_id)
        return by_frame

    def validate(self, verify_images: bool = True) -> dict[str, object]:
        invalid_frame = 0
        invalid_id = 0
        invalid_geometry = 0
        invalid_class = 0
        duplicate_frame_id = 0
        gt_rows = 0
        id_zero_rows = 0
        ids: set[int] = set()
        referenced_frames: set[int] = set()
        for frame, boxes in self.gt_by_frame.items():
            seen: set[int] = set()
            for box in boxes:
                gt_rows += 1
                ids.add(box.track_id)
                referenced_frames.add(frame)
                invalid_frame += int(frame < 1 or frame > self.seq_length)
                invalid_id += int(box.track_id < 0)
                id_zero_rows += int(box.track_id == 0)
                invalid_geometry += int(box.width <= 0 or box.height <= 0)
                invalid_class += int(box.class_id <= 0)
                if box.track_id in seen:
                    duplicate_frame_id += 1
                seen.add(box.track_id)

        missing_images = sorted(frame for frame in referenced_frames if frame not in self.image_paths)
        image_frames = sorted(self.image_paths)
        unreadable_images: list[int] = []
        if verify_images:
            for frame, path in self.image_paths.items():
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception:
                    unreadable_images.append(frame)

        result = {
            "sequence": self.name,
            "split": self.split,
            "seq_length": self.seq_length,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_count": len(self.image_paths),
            "first_image_frame": image_frames[0] if image_frames else None,
            "last_image_frame": image_frames[-1] if image_frames else None,
            "gt_rows": gt_rows,
            "identity_count": len(ids),
            "min_identity_id": min(ids) if ids else None,
            "max_identity_id": max(ids) if ids else None,
            "id_zero_rows": id_zero_rows,
            "id_zero_policy": "valid_zero_based_identity_id",
            "invalid_frame_rows": invalid_frame,
            "invalid_negative_id_rows": invalid_id,
            "invalid_geometry_rows": invalid_geometry,
            "invalid_class_rows": invalid_class,
            "duplicate_frame_id_rows": duplicate_frame_id,
            "missing_images_for_gt_frames": missing_images,
            "unreadable_images": unreadable_images,
        }
        errors = {
            key: value
            for key, value in result.items()
            if key.startswith("invalid_") and value
        }
        if duplicate_frame_id:
            errors["duplicate_frame_id_rows"] = duplicate_frame_id
        if missing_images:
            errors["missing_images_for_gt_frames"] = missing_images
        if unreadable_images:
            errors["unreadable_images"] = unreadable_images
        if errors:
            raise ValueError(f"invalid DanceTrack sequence {self.name}: {errors}")
        return result

    def boxes(self, frame: int | None = None) -> Iterable[GTBox]:
        if frame is not None:
            return tuple(self.gt_by_frame.get(frame, ()))
        return (box for current in sorted(self.gt_by_frame) for box in self.gt_by_frame[current])

    def image_path(self, frame: int) -> Path:
        try:
            return self.image_paths[frame]
        except KeyError as exc:
            raise FileNotFoundError(f"missing image {self.name}:{frame}") from exc


def discover_sequences(dataset_root: str | Path, split: str) -> list[DanceTrackSequence]:
    root = Path(dataset_root).expanduser().resolve()
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing DanceTrack split directory: {split_dir}")
    sequence_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise FileNotFoundError(f"no sequences under {split_dir}")
    return [DanceTrackSequence(path, split) for path in sequence_dirs]
