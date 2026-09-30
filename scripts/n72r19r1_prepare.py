#!/usr/bin/env python
"""Prepare the frozen N72R19R1 protocol, corruption manifest, and data audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import tempfile

import numpy as np
import torch

from sam3_intermot.identity_memory.dataset import FrozenIdentityEpisodeDataset
from sam3_intermot.identity_memory.r1_protocol import (
    R1_GOAL,
    R1_QUESTION,
    build_corruption_manifest,
    make_protocol_document,
)
from sam3_intermot.identity_memory.selective import load_frozen_n72r18_gru


DEFAULT_TRAIN_PROTOCOL = "/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_train.json"
DEFAULT_VAL_PROTOCOL = "/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_val.json"
DEFAULT_TRAIN_STORE = "/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_train"
DEFAULT_VAL_STORE = "/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_val"
DEFAULT_BASE_CHECKPOINT = "outputs/N72R18/checkpoints/identity_memory_gru.pt"
DEFAULT_OUTPUT = Path("outputs/N72R19R1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-protocol", default=DEFAULT_TRAIN_PROTOCOL)
    parser.add_argument("--val-protocol", default=DEFAULT_VAL_PROTOCOL)
    parser.add_argument("--train-store", default=DEFAULT_TRAIN_STORE)
    parser.add_argument("--val-store", default=DEFAULT_VAL_STORE)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=72191)
    parser.add_argument("--manifest-seed", type=int, default=72191)
    return parser.parse_args()


def _percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("mean", "median", "p10", "p25", "p75", "p90")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def _split_stats(
    protocol_path: str,
    *,
    allowed_sequences: set[str] | None,
    split_name: str,
    manifest: dict[str, object],
) -> dict[str, object]:
    document = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    anchors = [
        item for item in document.get("anchors", [])
        if allowed_sequences is None or str(item["sequence"]) in allowed_sequences
    ]
    identities = {(str(item["sequence"]), int(item["track_id"])) for item in anchors}
    observations_per_identity = [
        len(item.get("future", [])) for item in anchors
    ]
    competitor_counts: list[int] = []
    real_observations = 0
    competitive_observations = 0
    for anchor in anchors:
        for future in anchor.get("future", []):
            competitors = future.get("competitors", [])
            competitor_count = len(competitors)
            competitor_counts.append(competitor_count)
            real_observations += 1
            if competitor_count:
                competitive_observations += 1

    rows = manifest.get("rows", {})
    applied_counts: Counter[str] = Counter()
    for row in rows.values() if isinstance(rows, dict) else []:
        if allowed_sequences is not None and str(row["sequence"]) not in allowed_sequences:
            continue
        for slug, condition in row.get("conditions", {}).items():
            if bool(condition.get("applied")):
                applied_counts[slug] += 1

    competitor_histogram = Counter(str(value) for value in competitor_counts)
    return {
        "split": split_name,
        "sequence_count": len({str(item["sequence"]) for item in anchors}),
        "unique_eligible_identity_count": len(identities),
        "episode_count": len(anchors),
        "competitive_frame_count": competitive_observations,
        "real_observations": {
            "same_identity_future_observations": real_observations,
            "competitive_future_observations": competitive_observations,
        },
        "synthetic_corruption_instances": {
            "total_applied_across_conditions": int(sum(applied_counts.values())),
            "applied_by_condition": dict(sorted(applied_counts.items())),
            "note": "Synthetic instances are repeated perturbations of real observations, not independent data.",
        },
        "observations_per_identity": _percentiles(observations_per_identity),
        "competitors_per_frame": {
            "summary": _percentiles(competitor_counts),
            "histogram": dict(sorted(competitor_histogram.items(), key=lambda item: int(item[0]))),
            "frame_count": len(competitor_counts),
        },
    }


def _write_data_audit(output_dir: Path, protocol: dict[str, object], manifest: dict[str, object]) -> None:
    train_split = protocol["train_dev_split"]
    train_sequences = set(train_split["train_sequences"])
    dev_sequences = set(train_split["dev_sequences"])
    val_sequences = set(train_split["val_sequences"])
    train_protocol = protocol["inherited_protocol"]["train_path"]
    val_protocol = protocol["inherited_protocol"]["val_path"]
    stats = {
        "stage": "N72R19R1",
        "goal_reference": "outputs/N72R19R1/FINAL_GOAL.json",
        "goal": R1_GOAL,
        "central_question": R1_QUESTION,
        "protocol_reference": "outputs/N72R19R1/protocol.json",
        "train": _split_stats(train_protocol, allowed_sequences=train_sequences, split_name="train_32_sequences", manifest=manifest),
        "dev": _split_stats(train_protocol, allowed_sequences=dev_sequences, split_name="dev_8_sequences", manifest=manifest),
        "val": _split_stats(val_protocol, allowed_sequences=val_sequences, split_name="val_25_sequences", manifest=manifest),
        "synthetic_corruption_definition": "Each manifest condition is a deterministic replay over the same frozen real observation; it must not be counted as a new identity or new frame.",
    }
    (output_dir / "data_audit.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# N72R19R1 data audit",
        "",
        "FINAL GOAL: Selective Identity Memory Update",
        "",
        "所有统计基于冻结的 N72R17 protocol 和 N72R16/N72R17 embedding store；`synthetic_corruption_instances` 不计入真实数据量。",
        "",
        "| split | sequences | eligible identities | episodes | real observations | competitive frames | mean obs/identity | median | P10/P25/P75/P90 | synthetic corruption instances |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name in ("train", "dev", "val"):
        item = stats[name]
        obs = item["observations_per_identity"]
        lines.append(
            f"| {item['split']} | {item['sequence_count']} | {item['unique_eligible_identity_count']} | {item['episode_count']} | "
            f"{item['real_observations']['same_identity_future_observations']} | {item['competitive_frame_count']} | "
            f"{obs['mean']:.2f} | {obs['median']:.2f} | {obs['p10']:.2f}/{obs['p25']:.2f}/{obs['p75']:.2f}/{obs['p90']:.2f} | "
            f"{item['synthetic_corruption_instances']['total_applied_across_conditions']} |"
        )
    lines.extend(
        [
            "",
            "`competitors_per_frame` 的完整 histogram 和 percentile 位于 `data_audit.json`。",
            "Train-dev 是 sequence-level 32/8；val 的 25 个 sequence 只用于最终冻结评估。",
        ]
    )
    (output_dir / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_manifests(documents: list[dict[str, object]], protocol: dict[str, object]) -> dict[str, object]:
    rows: dict[str, object] = {}
    condition_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()
    train_sequences = set(protocol["train_dev_split"]["train_sequences"]) | set(protocol["train_dev_split"]["dev_sequences"])
    val_sequences = set(protocol["train_dev_split"]["val_sequences"])
    for document in documents:
        for key, row in document["rows"].items():
            row = dict(row)
            row["split"] = "val" if row["sequence"] in val_sequences else "train_dev"
            rows[key] = row
            for slug, condition in row["conditions"].items():
                condition_counts[slug] += 1
                if condition.get("applied"):
                    applied_counts[slug] += 1
    return {
        "stage": "N72R19R1",
        "goal_reference": "outputs/N72R19R1/FINAL_GOAL.json",
        "goal": R1_GOAL,
        "central_question": R1_QUESTION,
        "manifest_version": documents[0]["manifest_version"],
        "seed": documents[0]["seed"],
        "reference_memory": documents[0]["reference_memory"],
        "reference_state_policy": documents[0]["reference_state_policy"],
        "same_manifest_for_all_methods": True,
        "real_observations": len(rows),
        "synthetic_corruption_instances": int(sum(applied_counts.values())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "applied_counts": dict(sorted(applied_counts.items())),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol_document(args.train_protocol, args.val_protocol, split_seed=args.split_seed)
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    device = torch.device(args.device)
    frozen_gru = load_frozen_n72r18_gru(args.base_checkpoint, device)
    train_dataset = FrozenIdentityEpisodeDataset(args.train_protocol, args.train_store, split="train")
    val_dataset = FrozenIdentityEpisodeDataset(args.val_protocol, args.val_store, split="val")
    with tempfile.TemporaryDirectory(prefix="n72r19r1_manifest_") as temporary:
        train_manifest = build_corruption_manifest(
            train_dataset,
            frozen_gru,
            device=device,
            output_path=Path(temporary) / "train.json",
            seed=args.manifest_seed,
            batch_size=args.batch_size,
        )
        val_manifest = build_corruption_manifest(
            val_dataset,
            frozen_gru,
            device=device,
            output_path=Path(temporary) / "val.json",
            seed=args.manifest_seed,
            batch_size=args.batch_size,
        )
    manifest = _merge_manifests([train_manifest, val_manifest], protocol)
    (output_dir / "corruption_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_data_audit(output_dir, protocol, manifest)
    print(
        json.dumps(
            {
                "protocol": str((output_dir / "protocol.json").resolve()),
                "manifest": str((output_dir / "corruption_manifest.json").resolve()),
                "real_observations": manifest["real_observations"],
                "synthetic_corruption_instances": manifest["synthetic_corruption_instances"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
