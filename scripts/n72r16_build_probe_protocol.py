#!/usr/bin/env python3
"""Build the frozen, score-blind N72R16 anchor protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam3_intermot.identity_probe.dataset import discover_sequences
from sam3_intermot.identity_probe.protocol import build_anchors, protocol_document, write_protocol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-sequences", type=int)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--min-future-observations", type=int, default=20)
    args = parser.parse_args()

    sequences = discover_sequences(args.dataset_root, args.split)
    if args.smoke_sequences is not None:
        if args.smoke_sequences < 1:
            raise ValueError("--smoke-sequences must be positive")
        sequences = sequences[: args.smoke_sequences]
    for sequence in sequences:
        sequence.validate(verify_images=False)
    anchors = build_anchors(
        sequences,
        horizon=args.horizon,
        min_future_observations=args.min_future_observations,
    )
    document = protocol_document(
        split=args.split,
        anchors=anchors,
        dataset_root=str(args.dataset_root),
        horizon=args.horizon,
        min_future_observations=args.min_future_observations,
    )
    document["sequence_count"] = len(sequences)
    document["sequence_names"] = [sequence.name for sequence in sequences]
    write_protocol(args.output, document)
    print(json.dumps({"output": str(args.output), "sequences": len(sequences), "anchors": len(anchors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
