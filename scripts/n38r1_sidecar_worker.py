#!/usr/bin/env python3
"""Run exactly one N38R1 event×variant lossless sidecar in one process."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n38r1_sidecar_common import (  # noqa: E402
    N37_MANIFEST,
    build_sidecar,
    load_manifest_item,
    write_failure,
)
from scripts.n36_real_eval_common import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=N37_MANIFEST)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(
            json.dumps(
                {
                    "event_id": args.event_id,
                    "variant": args.variant,
                    "status": "FAIL",
                    "error": f"FileExistsError: refusing to overwrite existing sidecar: {args.output}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        _manifest, item = load_manifest_item(args.manifest, args.event_id)
        payload = build_sidecar(item, args.variant)
        atomic_json(args.output, payload)
        print(json.dumps({"event_id": args.event_id, "variant": args.variant, "status": "PASS"}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        try:
            write_failure(
                args.output,
                event_id=str(args.event_id),
                variant=str(args.variant),
                exc=exc,
                traceback_text=traceback.format_exc(),
            )
        except Exception:
            traceback.print_exc()
        print(
            json.dumps(
                {
                    "event_id": args.event_id,
                    "variant": args.variant,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
