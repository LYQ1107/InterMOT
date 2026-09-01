#!/usr/bin/env python3
"""Run the unchanged N48 full replay against the isolated R1 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n48_stage04_replay as replay  # noqa: E402

R1 = ROOT / "outputs/n48/repair1"
replay.OUT = R1
replay.RUNTIME = R1 / "replay/runtime"
replay.POSTHOC = R1 / "replay/posthoc"
replay.CHECKPOINT = R1 / "training/n48_r1_risk_aware_512d_bce.pt"
replay.MEMORY_MANIFEST = ROOT / "outputs/n48/training/simulated_event_memory.json"


if __name__ == "__main__":
    replay.main()
