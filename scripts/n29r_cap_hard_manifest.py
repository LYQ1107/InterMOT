#!/usr/bin/env python3
"""Apply a deterministic, pre-paired size cap to a frozen hard manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/n29r/hard_episode_manifest.json"))
    parser.add_argument("--max-per-sequence", type=int, default=5)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    original = list(payload.get("episodes", []))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for episode in original:
        grouped[str(episode["sequence"])].append(episode)
    selected: list[dict] = []
    for sequence in dict.fromkeys(str(item["sequence"]) for item in original):
        selected.extend(grouped[sequence][: args.max_per_sequence])
    payload["pre_cap_episode_count"] = len(original)
    payload["episodes"] = selected
    payload["episode_count"] = len(selected)
    payload["selection_policy"]["deterministic_cap"] = (
        f"first {args.max_per_sequence} already-selected episodes per sequence; applied before paired replay"
    )
    payload["selection_frozen_before_paired"] = True
    payload["pairing_manifest_sha256_pending"] = True
    for audit in payload.get("sequence_audits", []):
        audit["selected_for_paired_count"] = sum(
            1
            for episode in selected
            if episode.get("sequence") == audit.get("sequence")
        )
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pre_cap_episode_count": len(original), "episode_count": len(selected), "sequence_count": len(grouped)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
