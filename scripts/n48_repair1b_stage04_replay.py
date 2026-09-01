#!/usr/bin/env python3
"""Run unchanged N48 paired replay against the isolated repair2 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n48_stage04_replay as replay  # noqa: E402

R2 = ROOT / "outputs/n48/repair1b"
replay.OUT = R2
replay.RUNTIME = R2 / "replay/runtime"
replay.POSTHOC = R2 / "replay/posthoc"
replay.CHECKPOINT = R2 / "training/n48_r1_repair2_risk_aware_512d_bce.pt"
replay.MEMORY_MANIFEST = ROOT / "outputs/n48/training/simulated_event_memory.json"


if __name__ == "__main__":
    replay.main()
