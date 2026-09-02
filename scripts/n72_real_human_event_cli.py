#!/usr/bin/env python3
"""Validate an externally collected N72 real-human JSONL event tape."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.interaction.n72_real_human import N72RealHumanEventAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--failure-dir", type=Path, default=Path("outputs/N72/human_tape/attempts"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    adapter = N72RealHumanEventAdapter(candidate_root=args.candidate_root, raw_root=args.raw_root, failure_dir=args.failure_dir)
    report = adapter.validate_jsonl(args.input)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "total_records": report["total_records"], "accepted_records": report["accepted_records"], "rejected_records": report["rejected_records"], "report": str(args.report.resolve())}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
