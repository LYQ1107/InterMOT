#!/usr/bin/env python3
"""Prepare N72R17 metadata while copying the N72R16 protocol byte-for-byte."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from sam3_intermot.identity_research.protocol import FINAL_GOAL, CENTRAL_QUESTION, protocol_sha256, read_frozen_protocol


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n72r16-output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--osnet-embedding-root", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_entries: dict[str, object] = {}
    for split in ("train", "val"):
        source = args.n72r16_output_root.expanduser().resolve() / f"protocol_{split}.json"
        document = read_frozen_protocol(source)
        destination = output_dir / f"protocol_{split}.json"
        shutil.copyfile(source, destination)
        if protocol_sha256(source) != protocol_sha256(destination):
            raise RuntimeError(f"protocol copy changed bytes for {split}")
        protocol_entries[split] = {
            "path": str(destination),
            "source_path": str(source),
            "sha256": protocol_sha256(destination),
            "anchor_count": len(document.get("anchors", [])),
            "sequence_count": document.get("sequence_count"),
            "protocol_version": document.get("protocol_version"),
        }

    protocol_document = {
        "stage": "N72R17",
        "goal": FINAL_GOAL,
        "central_question": CENTRAL_QUESTION,
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "protocol_inheritance": "byte_for_byte_copy_of_N72R16_protocol_v1",
        "protocol_semantics_changed": False,
        "anchor_rule_changed": False,
        "future_frame_rule_changed": False,
        "hard_negative_definition_changed": False,
        "splits": protocol_entries,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    n72r16_manifest = args.asset_root.expanduser().resolve() / "outputs" / "N72R16" / "asset_manifest.json"
    clip_path = args.clip_model_path.expanduser().resolve() if args.clip_model_path else None
    manifest = {
        "stage": "N72R17",
        "asset_lineage": "N72R17_REUSED_N72R16_ASSETS_PLUS_PUBLIC_CLIP",
        "goal": FINAL_GOAL,
        "central_question": CENTRAL_QUESTION,
        "dataset": describe(args.dataset_root.expanduser().resolve()),
        "n72r16_asset_manifest": describe(n72r16_manifest),
        "frozen_protocol": describe(output_dir / "protocol.json"),
        "osnet_baseline": {
            "embedding_root": describe(args.osnet_embedding_root.expanduser().resolve()),
            "source": "reused N72R16 float16 embeddings; no OSNet re-extraction",
            "training": False,
        },
        "clip_encoder": {
            "model_id": "timm/vit_base_patch32_clip_224.openai",
            "model_path": describe(clip_path) if clip_path else None,
            "source_url": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai",
            "original_source_url": "https://github.com/openai/CLIP",
            "reid_reference_url": "https://github.com/OPA067/ReID",
            "reid_reference_note": "The public OPA067/ReID pipeline was reviewed, but no compatible trained ReID checkpoint was available in this environment; this result is explicitly generic frozen OpenAI CLIP, not CLIP-ReID fine-tuning.",
            "training": False,
            "identity_reid_finetuning": False,
            "input": "GT box crop, context=1.0, integer crop rectangle, torchvision roi_align GPU resize to 224x224, CLIP mean/std, L2-normalized 512-D output",
            "crop_backend": "torchvision_roi_align_gpu",
        },
        "video_encoder": {
            "status": "SKIPPED",
            "reason": "No compatible public frozen Video-ReID checkpoint and inference stack was already present; adding an unverified implementation would break comparability and scope.",
        },
        "prohibited_or_not_used": [
            "SAM3 inference",
            "MOT association/Hungarian",
            "persistent association integration",
            "TrackEval/full MOT",
            "LoRA or identity decoder training",
            "DanceTrack test split",
            "old PCTIS/N72 caches",
        ],
        "storage_policy": "Only float16 embeddings, metadata, JSONL metric records, and reports are stored; no image crops are written.",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    encoder_manifest = {
        "stage": "N72R17",
        "goal": FINAL_GOAL,
        "central_question": CENTRAL_QUESTION,
        "benchmark_controls": {
            "same_protocol": True,
            "same_anchor_rule": True,
            "same_future_frames": True,
            "same_hard_negative_definition": True,
            "training": False,
            "mot_evaluation": False,
        },
        "encoders": [
            {
                "name": "osnet_x1_0_market1501",
                "role": "reused_N72R16_baseline",
                "embedding_dim": 512,
                "weights_frozen": True,
                "store_train": str(args.osnet_embedding_root.expanduser().resolve()),
                "store_val": str(args.osnet_embedding_root.expanduser().resolve().parent / "embeddings_val"),
            },
            {
                "name": "openai_clip_vit_b32_zero_shot",
                "role": "public_CLIP_visual_representation",
                "model_id": "timm/vit_base_patch32_clip_224.openai",
                "embedding_dim": 512,
                "weights_frozen": True,
                "trained_on_dancetrack": False,
                "identity_reid_finetuning": False,
                "crop_backend": "torchvision_roi_align_gpu",
                "store_train": str(output_dir / "embeddings_clip_train"),
                "store_val": str(output_dir / "embeddings_clip_val"),
                "source_url": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai",
                "original_source_url": "https://github.com/openai/CLIP",
                "reid_reference_url": "https://github.com/OPA067/ReID",
                "reid_reference_note": "No compatible trained ReID checkpoint was used.",
            },
            {
                "name": "video_identity_encoder",
                "role": "planned_optional_encoder",
                "status": "SKIPPED",
                "skipped_reason": "No compatible public frozen Video-ReID checkpoint and inference stack was already present; no unverified implementation was added.",
            },
        ],
    }
    (output_dir / "encoder_manifest.json").write_text(json.dumps(encoder_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "protocol": str(output_dir / "protocol.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
