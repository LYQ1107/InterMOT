#!/usr/bin/env python3
"""Apply the frozen assignment-shadow stop rule to train-smoke replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-manifest", type=Path, default=ROOT / "outputs/N72R20R1/causal_replay/train_smoke_replay_manifest.json")
    parser.add_argument("--shadow-metrics", type=Path, default=ROOT / "outputs/N72R20R1/shadow/train_smoke_identity_metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R1/assignment_shadow/train_smoke_summary.json")
    args = parser.parse_args()
    replay = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
    shadow = json.loads(args.shadow_metrics.read_text(encoding="utf-8"))
    per_variant: dict[str, dict[str, int]] = {}
    for sequence in replay["sequences"]:
        for variant, audit in sequence["variants"].items():
            result = per_variant.setdefault(
                variant,
                {
                    "frames": 0,
                    "future_frames": 0,
                    "score_changed_frames": 0,
                    "score_changed_cells": 0,
                    "solver_changed_frames": 0,
                    "changed_public_id_count": 0,
                    "candidate_none_change_count": 0,
                    "row_max_preservation_failures": 0,
                    "memory_write_count": 0,
                },
            )
            result["frames"] += int(audit["frames"])
            result["future_frames"] += max(0, int(audit["frames"]) - 1)
            for key in result:
                if key in {"frames", "future_frames"}:
                    continue
                source_key = key
                result[key] += int(audit.get(source_key, 0))
    e2 = per_variant["FRESH_LEARNED_IDENTITY_TRUSTED"]
    e2_activity_rate = float(e2["score_changed_frames"] / max(e2["future_frames"], 1))
    e2_solver_activity_rate = float(e2["solver_changed_frames"] / max(e2["future_frames"], 1))
    inactive = e2_activity_rate > 0.10 and e2_solver_activity_rate < 0.01
    status = "FAIL_LEARNED_MEMORY_DECISION_INACTIVE" if inactive else "PASS_LEARNED_MEMORY_DECISION_ACTIVE"
    payload: dict[str, Any] = {
        "stage": "N72R20R1",
        "status": status,
        "decision_scope": "train_smoke_assignment_shadow_gate",
        "shadow_status": shadow.get("status"),
        "variants": per_variant,
        "learned_score_activity_rate": e2_activity_rate,
        "learned_solver_activity_rate": e2_solver_activity_rate,
        "stop_rule": {
            "rule": "if learned scores change materially but exact public assignment is nearly unchanged, stop",
            "score_activity_threshold": 0.10,
            "solver_activity_threshold": 0.01,
            "triggered": inactive,
        },
        "candidate_tape_shared": True,
        "base_score_tape_shared": True,
        "exact_solver_unchanged": True,
        "public_authority_unchanged": True,
        "runtime_future_gt_used": False,
        "formal_causal_replay_authorized": False if inactive else True,
        "val_formal_authorized": False,
        "next_action": "STOP_BEFORE_FORMAL_CAUSAL_REPLAY_AND_VAL" if inactive else "CONTINUE_TO_CAUSAL_INTEGRATION",
        "interpretation": (
            "The learned identity signal is non-trivial in the fresh tape shadow, but the current exact public-ID decision boundary is inactive; this is not association gain."
            if inactive
            else "The learned identity signal crosses the current exact public-ID decision boundary and may proceed to the frozen causal integration gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "score_activity_rate": e2_activity_rate, "solver_activity_rate": e2_solver_activity_rate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
