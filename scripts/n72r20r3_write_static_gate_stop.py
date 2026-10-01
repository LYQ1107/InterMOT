#!/usr/bin/env python3
"""Write the auditable static-gate stop artifacts for N72R20R3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.n72r20r3_common import ROOT, SEQUENCES, write_zstd_jsonl


FINAL_DECISION = "FAIL_OPEN_SET_CROSS_SEQUENCE_GENERALIZATION"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/N72R20R3/causal")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "stage": "N72R20R3",
        "status": "NOT_RUN_STATIC_PRESENCE_GATE_FAILED",
        "final_decision": FINAL_DECISION,
        "reason": "No LOSO static presence policy satisfied both pooled and all-eight-sequence safety/usefulness gates.",
        "runtime_future_gt_used": False,
        "association_rescue": False,
        "association_override": False,
    }
    (args.output_dir / "causal_replay.json").write_text(
        json.dumps(
            {
                **common,
                "static_presence_gate_required": True,
                "static_presence_gate_passed": False,
                "replay_rows": 0,
                "causal_wrong_write_rate": None,
                "causal_correct_write_retention": None,
                "memory_write_audit": "not_run",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "causal_per_sequence.json").write_text(
        json.dumps(
            {
                **common,
                "sequences": [
                    {"sequence": sequence, "status": "NOT_RUN_STATIC_PRESENCE_GATE_FAILED"}
                    for sequence in SEQUENCES
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_zstd_jsonl(args.output_dir / "memory_write_audit.jsonl.zst", [])
    print(json.dumps({"status": common["status"], "final_decision": FINAL_DECISION}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
