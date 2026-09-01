#!/usr/bin/env python3
"""Run the repaired N48 Stage-01 refinement in the isolated R1 tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n48_stage01_refine_n47_diagnosis as refine  # noqa: E402

refine.OUT = ROOT / "outputs/n48/repair1"
refine.FRAME_PATH = refine.OUT / "diagnosis/n47_m2_frame_diagnostics.jsonl"
refine.DIAG_PATH = refine.OUT / "diagnosis/n47_m2_structural_diagnosis.json"
refine.REFINED_PATH = refine.OUT / "diagnosis/n47_m2_refined_diagnosis.json"
refine.STATUS_PATH = refine.OUT / "stage_01_refined_status.json"


if __name__ == "__main__":
    refine.main()
