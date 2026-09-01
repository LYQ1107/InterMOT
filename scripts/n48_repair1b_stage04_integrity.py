#!/usr/bin/env python3
"""Run independent N48 integrity checks in the isolated repair2 tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n48_stage04_integrity as integrity  # noqa: E402

R2 = ROOT / "outputs/n48/repair1b"
integrity.OUT = R2
integrity.RUNTIME = R2 / "replay/runtime"
integrity.MEMORY = ROOT / "outputs/n48/training/simulated_event_memory.json"
integrity.CHECKPOINT = R2 / "training/n48_r1_repair2_risk_aware_512d_bce.pt"


if __name__ == "__main__":
    integrity.main()
