#!/usr/bin/env python3
"""Execute N28-B and conditionally continue into N28-C without a pause."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n28_cached_causal_replay import ROOT, atomic_json, run_n28b  # noqa: E402
from n28_meta_train_lcia import run_n28c  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcia-steps", type=int, default=5)
    parser.add_argument("--b-output", type=Path, default=ROOT / "outputs/n28/n28b_result.json")
    parser.add_argument("--c-output", type=Path, default=ROOT / "outputs/n28/n28c_result.json")
    args = parser.parse_args()
    b_result = run_n28b(output=args.b_output, lcia_steps=args.lcia_steps)

    # The conditional is deliberately in the same top-level process: a
    # passed B gate starts C immediately, while a failed B gate writes an
    # explicit authorization record and never initializes meta-training.
    c_result = run_n28c(
        b_result,
        output=args.c_output,
        device_name="cpu",
        full_sweep=True,
    )
    b_result["transition"] = {
        **b_result.get("transition", {}),
        "n28c_started": bool(c_result.get("meta_training_started", False)),
        "n28c_status": c_result.get("status"),
        "n28c_result": str(args.c_output.relative_to(ROOT)),
    }
    atomic_json(args.b_output, b_result)
    print(
        json.dumps(
            {
                "n28b_status": b_result.get("status"),
                "n28c_status": c_result.get("status"),
                "n28c_started": c_result.get("meta_training_started", False),
                "val25_read": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("N28_B_THEN_C_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
