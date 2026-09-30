#!/usr/bin/env python3
"""Prepare R2's simulated human-anchor protocol for the frozen train dev set.

GT is opened only here to choose one frame-0 target and to produce the
post-hoc label sidecar.  Candidate generation, scoring, commit replay, and
rescue replay consume the event file without opening future GT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from sam3_intermot.identity_probe.encoders import OSNetEncoder
from scripts.n72r20r1_prepare_interaction_events import prepare_sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")
DEFAULT_OSNET = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth")
DEV_SEQUENCES = (
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0039",
    "dancetrack0057",
    "dancetrack0062",
    "dancetrack0072",
)


def relabel(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("N72R20R2" if key == "stage" else relabel(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [relabel(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--osnet-checkpoint", type=Path, default=DEFAULT_OSNET)
    parser.add_argument("--sequences", nargs="+", default=list(DEV_SEQUENCES))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    asset_root = args.asset_root.resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    encoder = OSNetEncoder(args.osnet_checkpoint.resolve(), device=args.device)
    events: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for sequence in args.sequences:
        event, label = prepare_sequence(
            dataset_root=args.dataset_root.resolve(),
            asset_root=asset_root,
            sequence=str(sequence),
            encoder=encoder,
        )
        events.append(relabel(event))
        labels.append(relabel(label))
    event_path = asset_root / "interaction_events.json"
    label_path = asset_root / "posthoc_event_labels.json"
    event_path.write_text(json.dumps({"stage": "N72R20R2", "events": events}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label_path.write_text(json.dumps({"stage": "N72R20R2", "labels": labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_N72R20R2_INTERACTION_EVENTS", "events": len(events), "event_path": str(event_path)}, sort_keys=True))
    del encoder
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
