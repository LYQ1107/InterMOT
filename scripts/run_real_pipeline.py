#!/usr/bin/env python
"""Real SAM 3.1 pipeline entry point.

Exits with BLOCKED until a legal official checkpoint is provided.
"""

import sys
from pathlib import Path

import yaml

from sam3_intermot.backend.sam3_backend import CheckpointUnavailableError, Sam3Backend


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "default.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backend = Sam3Backend(checkpoint_path=cfg["backend"]["checkpoint_path"])
    try:
        backend.start_video("<video_source>")
    except CheckpointUnavailableError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
