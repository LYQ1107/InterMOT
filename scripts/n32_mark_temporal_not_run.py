#!/usr/bin/env python3
"""Record why the optional temporal fallback was not attempted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "protocol": "N32-G-TEMPORAL-FALLBACK",
        "status": "NOT_RUN_UPSTREAM_CONDITION",
        "reason": args.reason,
        "future_gt_used_for_selector_input": False,
        "val25_read": False,
        "test_labels_used": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
