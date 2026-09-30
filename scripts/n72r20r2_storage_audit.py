#!/usr/bin/env python3
"""Read-only storage authorization audit for N72R20R2 development tapes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WARNING_BYTES = 120_000_000_000
HARD_STOP_BYTES = 100_000_000_000
TARGET_NEW_BYTES = 25_000_000_000


def du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    result = subprocess.run(["du", "-sb", "--", str(path)], check=True, capture_output=True, text=True)
    return int(result.stdout.split()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R2/storage_audit_before.json")
    args = parser.parse_args()
    usage = shutil.disk_usage(ROOT)
    free = int(usage.free)
    status = "HARD_STOP" if free < HARD_STOP_BYTES else "WARNING" if free < WARNING_BYTES else "OK"
    safe_budget = max(0, free - HARD_STOP_BYTES)
    report = {
        "stage": "N72R20R2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "policy": {
            "warning_free_bytes": WARNING_BYTES,
            "hard_stop_free_bytes": HARD_STOP_BYTES,
            "target_new_storage_bytes": TARGET_NEW_BYTES,
            "deletion_performed": False,
            "download_performed": False,
            "test_dataset_allowed": False,
        },
        "filesystem": {
            "path": str(ROOT),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": free,
            "free_gb_decimal": free / 1_000_000_000,
            "free_gib": free / (1024**3),
            "storage_status": status,
            "safe_generation_budget_bytes_before_hard_stop": safe_budget,
            "target_budget_fits_hard_stop_headroom": TARGET_NEW_BYTES <= safe_budget,
            "df_h_output": subprocess.run(["df", "-h", str(ROOT)], check=True, capture_output=True, text=True).stdout,
        },
        "existing_assets": {
            "r1_fresh_asset_bytes": du_bytes(Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")),
            "dataset_asset_bytes": du_bytes(Path("/data3/liuyeqiang/InterMOT_N72R16_assets")),
            "reference_bytes": du_bytes(Path("/data3/liuyeqiang/research_references/N72R20")),
        },
        "authorization": {
            "development_generation_allowed": status != "HARD_STOP",
            "development_generation_must_remain_below_bytes": safe_budget,
            "r2_target_sequences": 8,
            "runtime_future_gt_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "free_gib": report["filesystem"]["free_gib"], "safe_budget_gb": safe_budget / 1_000_000_000}, sort_keys=True))
    return 0 if status != "HARD_STOP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
