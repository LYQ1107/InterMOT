"""Atomic, hash-chained JSONL storage for server-generated N72R1 artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from sam3_intermot.provenance.path_safety import resolve_within_root


class AppendOnlyJSONLError(ValueError):
    pass


class AppendOnlyJSONL:
    """Append JSON objects under a confined root without overwriting rows."""

    def __init__(self, path: str | Path, *, root: str | Path, key_field: str = "event_id") -> None:
        self.path = resolve_within_root(path, root)
        self.key_field = str(key_field)
        if not self.key_field:
            raise AppendOnlyJSONLError("key_field is required")

    @staticmethod
    def _canonical(record: Mapping[str, Any]) -> bytes:
        return (json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

    @classmethod
    def _hash(cls, record: Mapping[str, Any]) -> str:
        return hashlib.sha256(cls._canonical(record)).hexdigest()

    def _read(self) -> tuple[str | None, set[str]]:
        previous = None
        keys: set[str] = set()
        if not self.path.exists():
            return previous, keys
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AppendOnlyJSONLError(f"invalid JSONL at line {line_number}") from exc
                if not isinstance(item, dict):
                    raise AppendOnlyJSONLError(f"row {line_number} is not an object")
                key = item.get(self.key_field)
                if key is not None:
                    key = str(key)
                    if key in keys:
                        raise AppendOnlyJSONLError(f"duplicate {self.key_field}: {key}")
                    keys.add(key)
                stored = item.get("event_sha256")
                if stored is not None:
                    unsigned = dict(item)
                    unsigned.pop("event_sha256", None)
                    if self._hash(unsigned) != stored:
                        raise AppendOnlyJSONLError(f"hash mismatch at line {line_number}")
                    if item.get("previous_event_sha256") != previous:
                        raise AppendOnlyJSONLError(f"hash-chain discontinuity at line {line_number}")
                    previous = str(stored)
        return previous, keys

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise TypeError("record must be an object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                previous, keys = self._read()
                enriched = dict(record)
                key = enriched.get(self.key_field)
                if key is not None and str(key) in keys:
                    raise AppendOnlyJSONLError(f"duplicate {self.key_field}: {key}")
                enriched["previous_event_sha256"] = previous
                enriched["event_sha256"] = self._hash(enriched)
                with self.path.open("ab") as handle:
                    handle.write(self._canonical(enriched))
                    handle.flush()
                    os.fsync(handle.fileno())
                directory_fd = os.open(str(self.path.parent), os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return enriched
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def rows(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


__all__ = ["AppendOnlyJSONL", "AppendOnlyJSONLError"]
