"""Small shared utilities for the N36 real sharded tape pipeline.

The helpers in this module are deliberately format-only.  They do not read
DanceTrack annotations and do not make identity decisions; runtime candidate
rows contain only official SAM3 observations plus independent machine
box-crop features.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/path/to/dancetrack")
CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
DEFAULT_SEQUENCE_LIST = ROOT / "outputs/n34/selected_sequences.json"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def image_files(sequence_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in (sequence_dir / "img1").iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ],
        key=lambda path: int(path.stem),
    )


def load_sequences(path: Path, explicit: str = "") -> list[str]:
    if explicit:
        return sorted({item.strip() for item in explicit.split(",") if item.strip()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        str(item["sequence"])
        for item in payload.get("sequences", [])
        if isinstance(item, dict) and item.get("sequence")
    )


def encode_mask(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    array = np.asarray(mask, dtype=bool)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    compressed = zlib.compress(packed.tobytes(), level=1)
    return {
        "encoding": "packbits_zlib_base64",
        "shape": [int(v) for v in array.shape],
        "bitorder": "little",
        "data": base64.b64encode(compressed).decode("ascii"),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def decode_mask(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, list):
        array = np.asarray(value, dtype=bool)
        return array if array.ndim == 2 else None
    if not isinstance(value, dict) or value.get("encoding") != "packbits_zlib_base64":
        return None
    try:
        shape = tuple(int(v) for v in value["shape"])
        if len(shape) != 2 or min(shape) <= 0:
            return None
        packed = zlib.decompress(base64.b64decode(value["data"].encode("ascii")))
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
        count = int(shape[0] * shape[1])
        if bits.size < count:
            return None
        array = bits[:count].reshape(shape).astype(bool, copy=False)
        expected = value.get("sha256")
        if expected and hashlib.sha256(array.tobytes()).hexdigest() != str(expected):
            return None
        return array.copy()
    except (TypeError, ValueError, KeyError, zlib.error):
        return None


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=float).reshape(-1)
    b = np.asarray(list(right), dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union > 0 else 0.0


def cosine(left: Any, right: Any) -> float | None:
    try:
        a = np.asarray(left, dtype=np.float32).reshape(-1)
        b = np.asarray(right, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.size == 0 or a.size != b.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-6 or nb <= 1e-6:
        return None
    return float(np.dot(a, b) / (na * nb))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)
