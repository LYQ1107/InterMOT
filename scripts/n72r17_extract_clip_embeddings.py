#!/usr/bin/env python3
"""Extract frozen public CLIP embeddings from DanceTrack GT crops."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sam3_intermot.identity_research.crop import (
    ClipCropConfig,
    load_clip_person_crops,
    load_clip_person_crops_gpu,
)
from sam3_intermot.identity_research.dataset import discover_sequences
from sam3_intermot.identity_research.encoder import OpenAIClipEncoder
from sam3_intermot.identity_research.storage import EmbeddingStoreWriter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-sequences", type=int)
    parser.add_argument("--context", type=float, default=1.0)
    parser.add_argument("--crop-backend", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.context <= 0:
        raise ValueError("--context must be positive")

    sequences = discover_sequences(args.dataset_root, args.split)
    if args.smoke_sequences is not None:
        if args.smoke_sequences < 1:
            raise ValueError("--smoke-sequences must be positive")
        sequences = sequences[: args.smoke_sequences]
    for sequence in sequences:
        sequence.validate(verify_images=False)
    total = sum(len(sequence.gt_by_frame[frame]) for sequence in sequences for frame in sequence.gt_by_frame)
    encoder = OpenAIClipEncoder(device=args.device)
    writer = EmbeddingStoreWriter(args.output_dir, args.split, count=total, dimension=encoder.embedding_dim)
    crop_config = ClipCropConfig(height=encoder.input_size[0], width=encoder.input_size[1], context=args.context)
    device = torch.device(args.device)
    if args.crop_backend == "gpu" and device.type != "cuda":
        raise ValueError("--crop-backend gpu requires a CUDA device")
    pending_boxes: list[tuple[str, object]] = []
    pending_tensors: list[torch.Tensor] = []
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not pending_boxes:
            return
        batch = torch.stack(pending_tensors, dim=0)
        vectors = encoder.encode(batch).cpu().numpy()
        for (sequence_name, box), vector in zip(pending_boxes, vectors, strict=True):
            writer.add(sequence_name, box.frame, box.track_id, box.tlwh, vector)
            processed += 1
        pending_boxes.clear()
        pending_tensors.clear()

    for sequence in sequences:
        for frame in sorted(sequence.gt_by_frame):
            boxes = list(sequence.boxes(frame))
            if args.crop_backend == "gpu":
                crops = load_clip_person_crops_gpu(sequence.image_path(frame), boxes, device, crop_config)
            else:
                crops = load_clip_person_crops(sequence.image_path(frame), boxes, crop_config)
            for index, box in enumerate(boxes):
                pending_boxes.append((sequence.name, box))
                pending_tensors.append(crops[index])
                if len(pending_boxes) >= args.batch_size:
                    flush()
                    if processed % (args.batch_size * 16) == 0:
                        print(f"processed {processed}/{total}", flush=True)
    flush()
    writer.close(
        encoder_metadata=encoder.metadata(),
        extra={
            "dataset_root": str(args.dataset_root.expanduser().resolve()),
            "sequence_names": [sequence.name for sequence in sequences],
            "gt_box_count": total,
            "context": args.context,
            "crop_backend": args.crop_backend,
            "split_is_smoke_subset": args.smoke_sequences is not None,
        },
    )
    print(f"wrote {processed} float16 CLIP embeddings to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
