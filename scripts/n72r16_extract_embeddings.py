#!/usr/bin/env python3
"""Extract frozen OSNet embeddings from GT boxes without saving crops."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sam3_intermot.identity_probe.crop import CropConfig, load_person_crops
from sam3_intermot.identity_probe.dataset import DanceTrackSequence, discover_sequences
from sam3_intermot.identity_probe.encoders import OSNetEncoder
from sam3_intermot.identity_probe.storage import EmbeddingStoreWriter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-sequences", type=int)
    parser.add_argument("--context", type=float, default=1.0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    sequences = discover_sequences(args.dataset_root, args.split)
    if args.smoke_sequences is not None:
        sequences = sequences[: args.smoke_sequences]
    for sequence in sequences:
        sequence.validate(verify_images=False)
    # Count without decoding images, then decode each image exactly once while
    # extracting all boxes belonging to that frame.
    total = sum(len(sequence.gt_by_frame[frame]) for sequence in sequences for frame in sequence.gt_by_frame)
    encoder = OSNetEncoder(args.checkpoint, device=args.device)
    writer = EmbeddingStoreWriter(args.output_dir, args.split, count=total, dimension=encoder.embedding_dim)
    crop_config = CropConfig(height=encoder.input_size[0], width=encoder.input_size[1], context=args.context)
    pending_boxes = []
    pending_tensors: list[torch.Tensor] = []
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not pending_boxes:
            return
        batch = torch.stack(pending_tensors, dim=0)
        vectors = encoder.encode(batch).cpu().numpy()
        for (sequence, box), vector in zip(pending_boxes, vectors, strict=True):
            writer.add(sequence.name, box.frame, box.track_id, box.tlwh, vector)
            processed += 1
        pending_boxes.clear()
        pending_tensors.clear()

    try:
        for sequence in sequences:
            for frame in sorted(sequence.gt_by_frame):
                boxes = list(sequence.boxes(frame))
                crops = load_person_crops(sequence.image_path(frame), boxes, crop_config)
                for box, crop in zip(boxes, crops, strict=True):
                    pending_boxes.append((sequence, box))
                    pending_tensors.append(crop)
                    if len(pending_boxes) >= args.batch_size:
                        flush()
                        if processed % (args.batch_size * 8) == 0:
                            print(f"processed {processed}/{total}", flush=True)
        flush()
        writer.close(
            encoder_metadata=encoder.metadata(),
            extra={
                "dataset_root": str(args.dataset_root.expanduser().resolve()),
                "sequence_names": [sequence.name for sequence in sequences],
                "gt_box_count": total,
                "context": args.context,
                "split_is_smoke_subset": args.smoke_sequences is not None,
            },
        )
    except Exception:
        # Leave any partial files for diagnosis; do not silently mark them as a
        # complete store.
        raise
    print(f"wrote {processed} float16 embeddings to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
