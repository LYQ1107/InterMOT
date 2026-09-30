#!/usr/bin/env python3
"""Prepare the small simulated human-event input for the R1 train smoke.

This is the only preparation step that reads DanceTrack GT.  It creates a
human-observation protocol (event frame, confirmed box and frozen OSNet
anchor) plus a separate post-hoc label sidecar.  The base-score generator and
all runtime replay scripts consume only ``interaction_events.json`` and never
open GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from scripts.n72r20r1_candidate_stream_smoke import crop_batch
from sam3_intermot.identity_probe.encoders import OSNetEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
DEFAULT_OSNET = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth")
SEQUENCES = ("dancetrack0001", "dancetrack0002")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_sha256(feature: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(feature, dtype=np.float32).tobytes(order="C")).hexdigest()


def read_metadata(path: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def box_xyxy(gt_row: list[str]) -> list[float]:
    x, y, width, height = (float(value) for value in gt_row[2:6])
    return [x, y, x + width, y + height]


def iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def gt_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        if len(values) < 6:
            continue
        if int(float(values[0])) == 1:
            rows.append(values)
    return sorted(rows, key=lambda row: int(float(row[1])))


def prepare_sequence(
    *,
    dataset_root: Path,
    asset_root: Path,
    sequence: str,
    encoder: OSNetEncoder,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = asset_root / "candidates" / sequence / "metadata.jsonl.zst"
    frames = read_metadata(metadata_path)
    if not frames or int(frames[0].get("frame", -1)) != 0:
        raise ValueError(f"{sequence}: candidate tape has no frame 0")
    candidates = list(frames[0].get("candidates", []))
    if not candidates:
        raise ValueError(f"{sequence}: frame 0 has no candidates")
    gt_path = dataset_root / "train" / sequence / "gt" / "gt.txt"
    image_path = dataset_root / "train" / sequence / "img1" / "00000001.jpg"
    rows = gt_rows(gt_path)
    if not rows:
        raise ValueError(f"{sequence}: frame 1 has no GT identity")

    matches: list[tuple[float, int, str, list[float], list[str]]] = []
    for row in rows:
        target_box = box_xyxy(row)
        best = max(
            candidates,
            key=lambda candidate: (iou(candidate["box_xyxy"], target_box), str(candidate["candidate_uid"])),
        )
        best_iou = iou(best["box_xyxy"], target_box)
        if best_iou >= 0.50:
            matches.append((best_iou, int(float(row[1])), str(best["candidate_uid"]), target_box, row))
    if not matches:
        raise ValueError(f"{sequence}: no frame-1 GT identity has candidate IoU >= 0.50")
    _, target_gt_id, target_uid, target_box, target_row = sorted(matches, key=lambda item: (item[1], item[2]))[0]
    target_candidate = next(item for item in candidates if str(item["candidate_uid"]) == target_uid)

    state_bindings: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        state_id = int(index)
        state_bindings.append(
            {
                "association_state_id": state_id,
                "public_id": 100000 + state_id,
                "initial_candidate_uid": str(candidate["candidate_uid"]),
                "initial_candidate_index": int(index - 1),
                "initial_native_id": int(candidate["native_tid"]),
                "initial_box_xyxy": [float(value) for value in candidate["box_xyxy"]],
            }
        )
    target_binding = next(item for item in state_bindings if item["initial_candidate_uid"] == target_uid)
    batch = crop_batch(image_path, [target_box])
    with torch.inference_mode():
        anchor = encoder.encode(batch).cpu().numpy().astype(np.float32)[0]
    if anchor.shape != (512,) or not np.isfinite(anchor).all():
        raise ValueError(f"{sequence}: invalid human anchor")
    anchor = anchor / max(float(np.linalg.norm(anchor)), 1.0e-6)
    event = {
        "stage": "N72R20R1",
        "sequence": sequence,
        "split": "train",
        "event_frame": 0,
        "target_public_id": int(target_binding["public_id"]),
        "target_association_state_id": int(target_binding["association_state_id"]),
        "target_candidate_uid": target_uid,
        "target_box_xyxy": target_box,
        "human_anchor": anchor.tolist(),
        "human_anchor_sha256": feature_sha256(anchor),
        "human_anchor_source": "simulated_human_confirmed_gt_box_frozen_osnet",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "association_state_axis": [item["association_state_id"] for item in state_bindings],
        "public_id_axis": [item["public_id"] for item in state_bindings],
        "state_bindings": state_bindings,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "event_preparation_used_gt": True,
        "source_gt_sha256": sha256(gt_path),
    }
    label = {
        "stage": "N72R20R1",
        "sequence": sequence,
        "split": "train",
        "event_frame": 0,
        "target_gt_id": int(target_gt_id),
        "target_public_id": int(target_binding["public_id"]),
        "target_association_state_id": int(target_binding["association_state_id"]),
        "target_candidate_uid_at_event": target_uid,
        "target_box_xyxy": target_box,
        "event_candidate_iou": float(iou(target_candidate["box_xyxy"], target_box)),
        "gt_path": str(gt_path),
        "gt_sha256": sha256(gt_path),
        "posthoc_only": True,
        "runtime_future_gt_used": False,
    }
    return event, label


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--osnet-checkpoint", type=Path, default=DEFAULT_OSNET)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    encoder = OSNetEncoder(args.osnet_checkpoint.resolve(), device=args.device)
    events: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for sequence in args.sequences:
        event, label = prepare_sequence(
            dataset_root=args.dataset_root.resolve(),
            asset_root=args.asset_root.resolve(),
            sequence=str(sequence),
            encoder=encoder,
        )
        events.append(event)
        labels.append(label)
    args.asset_root.mkdir(parents=True, exist_ok=True)
    event_path = args.asset_root / "interaction_events.json"
    label_path = args.asset_root / "posthoc_event_labels.json"
    event_path.write_text(json.dumps({"stage": "N72R20R1", "events": events}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label_path.write_text(json.dumps({"stage": "N72R20R1", "labels": labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_N72R20R1_INTERACTION_EVENTS", "events": len(events), "event_path": str(event_path)}, sort_keys=True))
    del encoder
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
