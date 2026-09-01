#!/usr/bin/env python3
"""Targeted regression for N47 assignment-transition taxonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_stage04_global_probe_replay import classify_assignment_transition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "pure_two_row_swap": ([101, 202, None], [202, 101, None], {"pure_swap_changes": True, "id_set_changes": False}),
        "none_row_exchange_multiset_preserving": ([101, None], [None, 101], {"pure_swap_changes": True, "id_set_changes": False}),
        "id_removal_and_none": ([101, 202], [101, None], {"pure_swap_changes": False, "id_set_changes": True}),
        "unchanged": ([101, None], [101, None], {"assignment_changed": False, "pure_swap_changes": False, "id_set_changes": False}),
    }
    results = {}
    for name, (before, after, expected) in cases.items():
        actual = classify_assignment_transition(before, after)
        for key, value in expected.items():
            if actual[key] != value:
                raise AssertionError(f"{name}: {key}={actual[key]!r}, expected {value!r}")
        results[name] = actual
    payload = {
        "status": "PASS",
        "protocol": "N47_SWAP_METRIC_TARGETED_REGRESSION_V1",
        "cases": results,
        "invariants": {
            "pure_swap_requires_non_none_multiset_equality": True,
            "pure_swap_requires_at_least_two_changed_rows": True,
            "id_set_changes_exposes_multiset_changes": True,
            "none_new_removal_multiset_changes_are_not_pure_swap": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
