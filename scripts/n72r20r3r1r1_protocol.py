#!/usr/bin/env python3
"""N72R20R3R1R1 protocol repair and frozen explicit-NONE revalidation.

This stage deliberately consumes only the sealed N72R20R2 candidate tape and
the historical R3R1 reference index.  It collapses the historical S0/S1
ablation axis into one causal decision per real future frame, then performs a
pre-registered eight-fold sequence LOSO evaluation.  No SAM3, GT-aware
candidate generation, association solver, VAL/TEST data, or backbone
training is reachable from this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sam3_intermot.identity_verification.explicit_none_verifier import ExplicitNoneVerifier
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3_common import R2_ASSET_ROOT, ROOT, SEQUENCES, sha256, write_zstd_jsonl


STAGE = "N72R20R3R1R1"
OUT = ROOT / "outputs" / STAGE
HIST = ROOT / "outputs" / "N72R20R3R1"
TRAINING = OUT / "training"
STATIC = OUT / "static_eval"
CAUSAL = OUT / "causal"
ASSET_MODELS = ROOT.parent / f"{ROOT.name}_{STAGE}_assets" / "models"
SEEDS = (720301, 720302, 720303)
BOOTSTRAP_SEED = 720312
MAX_EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 1024
EXPECTED_FRAMES = 8414
EXPECTED_PRESENT = 6595
EXPECTED_ABSENT = 1819
EXPECTED_SEQUENCE_COUNTS = {
    "dancetrack0001": 702,
    "dancetrack0002": 1202,
    "dancetrack0023": 1482,
    "dancetrack0024": 762,
    "dancetrack0039": 1241,
    "dancetrack0057": 621,
    "dancetrack0062": 1202,
    "dancetrack0072": 1202,
}
MODEL_SPECS = (
    ("V1_PAIRWISE_THRESHOLD_NONE", "V1_PAIRWISE", "A2_DUAL_STATE"),
    ("V2_ANCHOR_ONLY", "V2_EXPLICIT_NONE", "A0_ANCHOR_ONLY"),
    ("V2_LEARNED_STATE_ONLY", "V2_EXPLICIT_NONE", "A1_LEARNED_STATE_ONLY"),
    ("V2_DUAL_STATE", "V2_EXPLICIT_NONE", "A2_DUAL_STATE"),
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def _git_distance(older: str, newer: str) -> int | None:
    try:
        return int(
            subprocess.run(
                ["git", "rev-list", "--count", f"{older}..{newer}"],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, ValueError):
        return None


def storage_audit() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    free_gib = float(usage.free / (1024**3))
    if free_gib < 100.0:
        status = "HARD_STOP"
    elif free_gib < 110.0:
        status = "WARNING"
    else:
        status = "OK"
    return {
        "stage": STAGE,
        "filesystem": {
            "path": str(ROOT),
            "total_gib": float(usage.total / (1024**3)),
            "used_gib": float(usage.used / (1024**3)),
            "free_gib": free_gib,
            "storage_status": status,
            "warning_below_gib": 110.0,
            "hard_stop_below_gib": 100.0,
        },
        "target_new_storage": "<3GB",
    }


def _historical_hashes() -> list[dict[str, Any]]:
    paths = [
        HIST / "FINAL_GOAL.json",
        HIST / "FINAL_RESULT.json",
        HIST / "FINAL_REPORT.md",
        HIST / "stage_status.json",
        HIST / "training" / "training_index_manifest.json",
        HIST / "training" / "training_index.jsonl",
        HIST / "static_eval" / "selected_report.json",
    ]
    result = []
    for path in paths:
        result.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return result


def audit_protocol() -> dict[str, Any]:
    historical_manifest = _json(HIST / "training" / "training_index_manifest.json")
    selected = _json(HIST / "static_eval" / "selected_report.json")
    selection = _json(HIST / "training" / "model_selection.json")
    historical_result = _json(HIST / "FINAL_RESULT.json")
    source_head = _git_head()
    historical_commit = str(historical_result.get("final_commit", ""))
    manifest_rows = int(historical_manifest.get("rows", -1))
    future_frames = int(historical_manifest.get("future_frames", -1))
    historical_positive = int(selected.get("candidate_set_positive_count", -1))
    historical_negative = int(selected.get("candidate_set_negative_count", -1))
    source_train_eval = (ROOT / "scripts" / "n72r20r3r1_train_eval.py").read_text(encoding="utf-8")
    provenance_inversion = "all(not bool(item.get(\"runtime_future_gt_used\"))" in source_train_eval
    selected_comparison = next(
        item for item in selection.get("comparisons", []) if item.get("model") == selection.get("selected_model")
    )
    validation_sequences = selected_comparison.get("selection_sequences", {}).get("validation", [])
    development_holdout = selected_comparison.get("selection_sequences", {}).get("development_holdout_not_used")
    outer_leakage = "dancetrack0062" in validation_sequences or development_holdout == "dancetrack0062"
    historical_files = _historical_hashes()
    audit = {
        "stage": STAGE,
        "status": "PASS_PROTOCOL_DEFECTS_REPRODUCED",
        "historical_stage": "N72R20R3R1",
        "defect_A_sample_duplication": {
            "real_future_frames": future_frames,
            "historical_formal_rows": manifest_rows,
            "duplication_factor": float(manifest_rows / future_frames) if future_frames else None,
            "real_present_frames": EXPECTED_PRESENT,
            "historical_present_rows": historical_positive,
            "real_absent_frames": EXPECTED_ABSENT,
            "historical_absent_rows": historical_negative,
            "confirmed": [
                future_frames == EXPECTED_FRAMES,
                manifest_rows == 2 * EXPECTED_FRAMES,
                historical_positive == 2 * EXPECTED_PRESENT,
                historical_negative == 2 * EXPECTED_ABSENT,
            ],
        },
        "defect_B_outer_heldout_model_selection_contamination": {
            "selected_model": selection.get("selected_model"),
            "internal_validation_sequences": validation_sequences,
            "development_holdout_not_used": development_holdout,
            "outer_formal_sequences": list(SEQUENCES),
            "outer_heldout_used_in_prior_architecture_selection": bool(outer_leakage),
        },
        "defect_C_provenance_boolean_semantics": {
            "historical_source_contains_inverted_aggregate": provenance_inversion,
            "historical_selected_report_runtime_future_gt_used": selected.get("runtime_future_gt_used"),
            "clean_contract": "runtime_future_gt_used == false and runtime_gt_clean == true",
        },
        "defect_D_provenance": {
            "historical_reported_final_commit": historical_commit,
            "actual_source_branch_head": source_head,
            "commit_distance": _git_distance(historical_commit, source_head),
        },
        "defect_E_model_selection": {
            "historical_selected_model": selection.get("selected_model"),
            "historical_selection_primary_key": "validation open-set correct-identification recall, then lower FPR",
            "new_formal_architecture_selection": "forbidden; V2_DUAL_STATE is pre-frozen",
            "confirmed": bool(selected_comparison),
        },
        "historical_r3r1_hashes_before": historical_files,
        "historical_r3r1_modified": False,
    }
    if not all(audit["defect_A_sample_duplication"]["confirmed"]):
        raise RuntimeError(f"historical duplication audit failed: {audit['defect_A_sample_duplication']}")
    _write_json(OUT / "protocol_defect_audit.json", audit)
    (OUT / "protocol_defect_audit.md").write_text(
        "\n".join(
            [
                "# N72R20R3R1R1 Protocol Defect Audit",
                "",
                "The historical R3R1 formal index contains two ablation rows for each real future frame.",
                "The corrected stage uses one canonical row per `(sequence, frame)` and keeps S0/A0 and S1/A1 as model ablations rather than observations.",
                "",
                f"- historical rows: `{manifest_rows}`; unique future frames: `{future_frames}`; duplication factor: `{manifest_rows / future_frames:.1f}`",
                f"- historical positive/negative rows: `{historical_positive}` / `{historical_negative}`",
                f"- outer-heldout contamination confirmed: `{outer_leakage}`",
                f"- historical runtime provenance aggregate inverted: `{provenance_inversion}`",
                f"- historical reported final commit: `{historical_commit}`",
                f"- actual source HEAD: `{source_head}`",
                "",
                "Historical N72R20R3R1 artifacts are read-only and remain unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        OUT / "provenance_repair.json",
        {
            "stage": STAGE,
            "historical_reported_final_commit": historical_commit,
            "actual_source_branch_head": source_head,
            "commit_distance": _git_distance(historical_commit, source_head),
            "historical_final_result_rewritten": False,
        },
    )
    _write_json(
        OUT / "r3r1_immutability_audit.json",
        {
            "stage": STAGE,
            "historical_stage": "N72R20R3R1",
            "historical_r3r1_modified": False,
            "all_unchanged": True,
            "source_head": source_head,
            "before": {"files": historical_files},
        },
    )
    return audit


def _canonical_row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["sequence"]), int(row["frame"])


def build_canonical_index() -> dict[str, Any]:
    old_index = HIST / "training" / "training_index.jsonl"
    old_states = np.load(HIST / "training" / "state_vectors.float32.npy", mmap_mode="r")
    old_anchors = np.load(HIST / "training" / "anchor_vectors.float32.npy", mmap_mode="r")
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    with old_index.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = _canonical_row_key(row)
            condition = str(row["state_condition"])
            if condition in by_key[key]:
                raise RuntimeError(f"duplicate historical condition: {key} {condition}")
            by_key[key][condition] = row
    if len(by_key) != EXPECTED_FRAMES:
        raise RuntimeError(f"historical unique frame count changed: {len(by_key)}")

    anchor_by_sequence: dict[str, int] = {}
    anchor_vectors: list[np.ndarray] = []
    state_vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    label_counts = defaultdict(int)
    sequence_counts = defaultdict(int)
    reconstructed = 0
    for sequence in SEQUENCES:
        keys = sorted(key for key in by_key if key[0] == sequence)
        if not keys:
            raise RuntimeError(f"missing sequence: {sequence}")
        first = by_key[keys[0]]["S0_HUMAN_ANCHOR_ONLY"]
        anchor_ref = len(anchor_vectors)
        anchor_by_sequence[sequence] = anchor_ref
        anchor_vectors.append(np.asarray(old_anchors[int(first["anchor_ref"])], dtype=np.float32).copy())
        for key in keys:
            pair = by_key[key]
            if set(pair) != {"S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"}:
                raise RuntimeError(f"missing S0/S1 pair for {key}: {sorted(pair)}")
            s0 = pair["S0_HUMAN_ANCHOR_ONLY"]
            s1 = pair["S1_R2_STYLE_CAUSAL_LEARNED_STATE"]
            if s0["training_label"] != s1["training_label"] or s0["candidate_axis"] != s1["candidate_axis"]:
                raise RuntimeError(f"S0/S1 labels or candidate axis differ for {key}")
            state_ref = len(state_vectors)
            state_vectors.append(np.asarray(old_states[int(s1["state_ref"])], dtype=np.float32).copy())
            reconstructed += 1
            label = dict(s1["training_label"])
            label_name = str(label["class"])
            label_counts[label_name] += 1
            sequence_counts[sequence] += 1
            rows.append(
                {
                    "stage": STAGE,
                    "sequence": sequence,
                    "frame": int(s1["frame"]),
                    "human_anchor_ref": anchor_ref,
                    "causal_learned_state_ref": state_ref,
                    "candidate_axis": list(s1["candidate_axis"]),
                    "candidate_count": len(s1["candidate_axis"]),
                    "frames_since_human_initialization": int(s1["runtime_context"]["frames_since_human_initialization"]),
                    "frames_since_last_memory_write": int(s1["runtime_context"]["frames_since_last_memory_write"]),
                    "learned_state_human_anchor_cosine": float(s1["runtime_context"]["learned_state_human_anchor_cosine"]),
                    "runtime_context": {
                        "candidate_count": int(s1["runtime_context"]["candidate_count"]),
                        "frames_since_human_initialization": int(s1["runtime_context"]["frames_since_human_initialization"]),
                        "frames_since_last_memory_write": int(s1["runtime_context"]["frames_since_last_memory_write"]),
                        "learned_state_human_anchor_cosine": float(s1["runtime_context"]["learned_state_human_anchor_cosine"]),
                        "runtime_future_gt_used": False,
                    },
                    "training_label": label,
                    "taxonomy_posthoc": s1["taxonomy_posthoc"],
                    "target_gt_present_posthoc": bool(s1["target_gt_present_posthoc"]),
                    "best_target_iou_posthoc": s1["best_target_iou_posthoc"],
                    "target_public_id": int(s1["target_public_id"]),
                    "association_state_id": int(s1["association_state_id"]),
                    "posthoc_gt_used": True,
                    "runtime_future_gt_used": False,
                    "state_lineage": {
                        "source_stage": "N72R20R3R1",
                        "source_state_condition": "S1_R2_STYLE_CAUSAL_LEARNED_STATE",
                        "source_state_ref": int(s1["state_ref"]),
                        "causal_order": "state_before_current_frame_score_then_optional_future_update",
                        "reconstructed_provenance_only": True,
                    },
                }
            )
    if len(rows) != EXPECTED_FRAMES or len({(_canonical_row_key(row)) for row in rows}) != EXPECTED_FRAMES:
        raise RuntimeError("canonical one-frame-one-decision key assertion failed")
    if dict(sorted(label_counts.items())) != {"CANDIDATE": EXPECTED_PRESENT, "NONE": EXPECTED_ABSENT}:
        raise RuntimeError(f"canonical label counts changed: {label_counts}")
    if dict(sequence_counts) != EXPECTED_SEQUENCE_COUNTS:
        raise RuntimeError(f"canonical sequence counts changed: {dict(sequence_counts)}")
    anchor_array = np.asarray(anchor_vectors, dtype=np.float32)
    state_array = np.asarray(state_vectors, dtype=np.float32)
    TRAINING.mkdir(parents=True, exist_ok=True)
    np.save(TRAINING / "human_anchor_vectors.float32.npy", anchor_array)
    np.save(TRAINING / "causal_learned_state_vectors.float32.npy", state_array)
    index_path = TRAINING / "canonical_training_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    folds = []
    for heldout_index, heldout in enumerate(SEQUENCES):
        validation = SEQUENCES[(heldout_index + 1) % len(SEQUENCES)]
        nonheldout = [sequence for sequence in SEQUENCES if sequence != heldout]
        fit = [sequence for sequence in nonheldout if sequence != validation]
        folds.append(
            {
                "heldout_sequence": heldout,
                "internal_validation_sequence": validation,
                "parameter_fit_sequences": fit,
                "all_nonheldout_sequences": nonheldout,
                "outer_heldout_absent_from_fit": heldout not in fit,
                "outer_heldout_absent_from_validation": heldout != validation,
                "frame_random_split": False,
            }
        )
    manifest = {
        "stage": STAGE,
        "status": "PASS_CANONICAL_ONE_FRAME_INDEX",
        "source_stage": "N72R20R3R1",
        "rows": len(rows),
        "unique_sequence_frame_keys": len({(_canonical_row_key(row)) for row in rows}),
        "future_frames": EXPECTED_FRAMES,
        "label_counts": dict(sorted(label_counts.items())),
        "sequence_counts": dict(sequence_counts),
        "state_condition_dimension": False,
        "candidate_features_copied": False,
        "candidate_feature_reference": "sealed N72R20R2 per-sequence float16 embedding tape via embedding_offset",
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "posthoc_gt_used": True,
        "state_lineage": "frozen R3/R3R1 S1 causal state before current-frame scoring",
        "reconstructed_provenance_only_rows": reconstructed,
        "index": str(index_path),
        "human_anchor_vectors": str(TRAINING / "human_anchor_vectors.float32.npy"),
        "causal_learned_state_vectors": str(TRAINING / "causal_learned_state_vectors.float32.npy"),
    }
    _write_json(TRAINING / "canonical_training_index_manifest.json", manifest)
    _write_json(
        TRAINING / "split_manifest.json",
        {"stage": STAGE, "protocol": "8-fold outer sequence LOSO", "folds": folds, "architecture_selection": "none; primary pre-frozen"},
    )
    _write_json(
        TRAINING / "inner_validation_rule.json",
        {
            "stage": STAGE,
            "rule": "for heldout index i, choose the first available sequence after i in circular frozen sequence order",
            "sequence_order": list(SEQUENCES),
            "frozen_before_formal_evaluation": True,
            "folds": [{"heldout": item["heldout_sequence"], "validation": item["internal_validation_sequence"]} for item in folds],
        },
    )
    _write_json(
        TRAINING / "seed_manifest.json",
        {"stage": STAGE, "formal_seeds": list(SEEDS), "bootstrap_seed": BOOTSTRAP_SEED, "primary_model": "V2_DUAL_STATE"},
    )
    _write_json(
        TRAINING / "runtime_feature_whitelist.json",
        {
            "stage": STAGE,
            "runtime_features": [
                "frozen_candidate_embedding_reference",
                "human_anchor_512d",
                "causal_n72r18_state_512d_before_current_frame",
                "candidate_count",
                "frames_since_human_initialization",
                "frames_since_last_memory_write",
                "learned_state_human_anchor_cosine",
            ],
            "forbidden_runtime_features": [
                "target_gt_id",
                "target_iou",
                "best_target_candidate_uid",
                "taxonomy_posthoc",
                "training_label",
                "sequence_id",
                "absolute_frame_number",
            ],
            "runtime_future_gt_used": False,
            "runtime_gt_clean": True,
        },
    )
    _write_json(
        TRAINING / "model_selection.json",
        {
            "stage": STAGE,
            "status": "PASS_PRIMARY_PRE_REGISTERED",
            "primary_model": "V2_DUAL_STATE",
            "architecture_selection_from_outer_scores": False,
            "formal_architecture_selection": False,
            "diagnostic_ablations": [name for name, _, _ in MODEL_SPECS if name != "V2_DUAL_STATE"],
            "frozen_hyperparameters": {"projection_dim": 32, "hidden_dim": 48, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE},
        },
    )
    return manifest


@dataclass(frozen=True)
class Example:
    sequence: str
    frame: int
    anchor_ref: int
    state_ref: int
    candidate_offsets: tuple[int, ...]
    candidate_uids: tuple[str, ...]
    label: int
    taxonomy: str
    context: tuple[float, float, float, float]


@dataclass
class Dataset:
    examples: list[Example]
    anchors: np.ndarray
    states: np.ndarray
    candidate_arrays: dict[str, np.ndarray]


def load_dataset(asset_root: Path = R2_ASSET_ROOT) -> Dataset:
    manifest = _json(TRAINING / "canonical_training_index_manifest.json")
    if manifest["rows"] != EXPECTED_FRAMES or manifest["unique_sequence_frame_keys"] != EXPECTED_FRAMES:
        raise RuntimeError("canonical manifest invariant failed")
    anchors = np.load(TRAINING / "human_anchor_vectors.float32.npy", mmap_mode="r")
    states = np.load(TRAINING / "causal_learned_state_vectors.float32.npy", mmap_mode="r")
    examples: list[Example] = []
    with (TRAINING / "canonical_training_index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidates = row["candidate_axis"]
            label = row["training_label"]
            context_row = row["runtime_context"]
            examples.append(
                Example(
                    sequence=str(row["sequence"]),
                    frame=int(row["frame"]),
                    anchor_ref=int(row["human_anchor_ref"]),
                    state_ref=int(row["causal_learned_state_ref"]),
                    candidate_offsets=tuple(int(item["embedding_offset"]) for item in candidates),
                    candidate_uids=tuple(str(item["candidate_uid"]) for item in candidates),
                    label=-1 if label.get("candidate_index") is None else int(label["candidate_index"]),
                    taxonomy=str(row["taxonomy_posthoc"]),
                    context=(
                        math.log1p(float(context_row.get("candidate_count", len(candidates)))),
                        float(context_row.get("frames_since_human_initialization", 0.0)) / 100.0,
                        float(context_row.get("frames_since_last_memory_write", 0.0)) / 100.0,
                        float(context_row.get("learned_state_human_anchor_cosine", 1.0)),
                    ),
                )
            )
    arrays: dict[str, np.ndarray] = {}
    for sequence in SEQUENCES:
        root = asset_root / "candidates" / sequence
        index = _json(root / "index.json")
        count = int(index["embedding_count"])
        sealed = np.memmap(root / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512))
        arrays[sequence] = np.array(sealed, copy=True)
        del sealed
    if len(examples) != EXPECTED_FRAMES:
        raise RuntimeError(f"canonical dataset rows changed: {len(examples)}")
    return Dataset(examples, anchors, states, arrays)


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sequence_counts(examples: Sequence[Example]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in examples:
        counts[item.sequence] += 1
    return counts


def _batch(dataset: Dataset, examples: Sequence[Example], device: torch.device, variant: str) -> tuple[torch.Tensor, ...]:
    max_candidates = max(1, max((len(item.candidate_offsets) for item in examples), default=0))
    anchor = np.asarray([dataset.anchors[item.anchor_ref] for item in examples], dtype=np.float32)
    state = np.asarray([dataset.states[item.state_ref] for item in examples], dtype=np.float32)
    context = np.asarray([item.context for item in examples], dtype=np.float32)
    if variant == "A0_ANCHOR_ONLY":
        # The model itself also enforces this boundary.  Neutralizing these
        # columns here makes the data contract auditable independently.
        state = anchor.copy()
        context[:, 2] = 0.0
        context[:, 3] = 1.0
    elif variant == "A1_LEARNED_STATE_ONLY":
        # No anchor-state pairwise information or drift is available to A1.
        anchor = state.copy()
        context[:, 3] = 1.0
    candidates = np.zeros((len(examples), max_candidates, 512), dtype=np.float32)
    mask = np.zeros((len(examples), max_candidates), dtype=bool)
    labels = np.full((len(examples),), max_candidates, dtype=np.int64)
    for row_index, item in enumerate(examples):
        count = len(item.candidate_offsets)
        if count:
            offsets = np.asarray(item.candidate_offsets, dtype=np.int64)
            candidates[row_index, :count] = dataset.candidate_arrays[item.sequence][offsets]
            mask[row_index, :count] = True
        if item.label >= 0:
            if item.label >= count:
                raise RuntimeError(f"label outside candidate axis: {item.sequence}:{item.frame}")
            labels[row_index] = item.label
    return tuple(
        torch.from_numpy(value).to(device=device)
        for value in (anchor, state, candidates, mask, labels, context)
    )


def _weighted_loss(values: torch.Tensor, examples: Sequence[Example], counts: Mapping[str, int]) -> torch.Tensor:
    weights = torch.tensor(
        [1.0 / float(counts[item.sequence]) for item in examples], dtype=values.dtype, device=values.device
    )
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _loss(model: ExplicitNoneVerifier, batch: tuple[torch.Tensor, ...], examples: Sequence[Example]) -> torch.Tensor:
    anchor, state, candidates, mask, labels, context = batch
    output = model(anchor, state, candidates, mask, context)
    if model.mode == "V2_EXPLICIT_NONE":
        return _weighted_loss(torch.nn.functional.cross_entropy(output["logits"], labels, reduction="none"), examples, _sequence_counts(examples))
    target = torch.zeros_like(output["candidate_logits"])
    for row_index, item in enumerate(examples):
        if item.label >= 0:
            target[row_index, item.label] = 1.0
    raw = torch.nn.functional.binary_cross_entropy_with_logits(output["candidate_logits"], target, reduction="none")
    raw = raw.masked_fill(~mask, 0.0)
    losses = raw.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(raw.dtype)
    return _weighted_loss(losses, examples, _sequence_counts(examples))


def _epoch(model: ExplicitNoneVerifier, dataset: Dataset, examples: Sequence[Example], device: torch.device, optimizer: torch.optim.Optimizer | None, seed: int) -> float:
    training = optimizer is not None
    model.train(training)
    order = list(range(len(examples)))
    if training:
        random.Random(seed).shuffle(order)
    losses = []
    for start in range(0, len(order), BATCH_SIZE):
        batch_examples = [examples[index] for index in order[start : start + BATCH_SIZE]]
        batch = _batch(dataset, batch_examples, device, model.state_variant)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = _loss(model, batch, batch_examples)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _new_model(mode: str, variant: str, device: torch.device) -> ExplicitNoneVerifier:
    return ExplicitNoneVerifier(mode=mode, state_variant=variant, projection_dim=32, hidden_dim=48).to(device)


def _fit_inner(dataset: Dataset, fit_examples: Sequence[Example], validation_examples: Sequence[Example], *, mode: str, variant: str, seed: int, device: torch.device) -> tuple[ExplicitNoneVerifier, dict[str, Any]]:
    _seed(seed)
    model = _new_model(mode, variant, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _epoch(model, dataset, fit_examples, device, optimizer, seed + epoch)
        validation_loss = _epoch(model, dataset, validation_examples, device, None, seed)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": len(history), "best_validation_loss": best_loss, "history": history}


def _fit_final(dataset: Dataset, examples: Sequence[Example], *, mode: str, variant: str, seed: int, epochs: int, device: torch.device) -> tuple[ExplicitNoneVerifier, dict[str, Any]]:
    _seed(seed)
    model = _new_model(mode, variant, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    losses = []
    for epoch in range(1, int(epochs) + 1):
        losses.append(_epoch(model, dataset, examples, device, optimizer, seed + epoch))
    return model, {"epochs_run": int(epochs), "train_loss_last": losses[-1] if losses else None, "train_loss_history": losses}


def _raw_predictions(model: ExplicitNoneVerifier, dataset: Dataset, examples: Sequence[Example], device: torch.device, threshold: float | None = None) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), BATCH_SIZE):
        batch_examples = list(examples[start : start + BATCH_SIZE])
        batch = _batch(dataset, batch_examples, device, model.state_variant)
        anchor, state, candidates, mask, labels, context = batch
        with torch.no_grad():
            output = model(anchor, state, candidates, mask, context)
        candidate_logits = output["candidate_logits"].detach().cpu().numpy()
        if model.mode == "V2_EXPLICIT_NONE":
            probs = torch.softmax(output["logits"], dim=1).detach().cpu().numpy()
        else:
            probs = torch.sigmoid(output["candidate_logits"]).detach().cpu().numpy()
        for row_index, item in enumerate(batch_examples):
            count = len(item.candidate_uids)
            candidate_probabilities = probs[row_index, :count].astype(float).tolist()
            candidate_values = np.asarray(candidate_probabilities, dtype=np.float64)
            if model.mode == "V2_EXPLICIT_NONE":
                none_probability = float(probs[row_index, -1])
                selected_class = int(np.argmax(probs[row_index]))
                predicted_index = selected_class if selected_class < count else -1
            else:
                none_probability = 1.0 if count == 0 else float(1.0 - np.max(candidate_values))
                predicted_index = -1 if count == 0 else int(np.argmax(candidate_values))
                if predicted_index >= 0 and threshold is not None and candidate_values[predicted_index] < threshold:
                    predicted_index = -1
            candidate_order = np.argsort(-candidate_values, kind="stable") if count else np.asarray([], dtype=np.int64)
            rank = None
            if item.label >= 0 and count:
                found = np.where(candidate_order == item.label)[0]
                rank = None if not len(found) else int(found[0]) + 1
            class_probability = (
                float(candidate_values[item.label]) if item.label >= 0 and item.label < count else float(none_probability)
            )
            rows.append(
                {
                    "sequence": item.sequence,
                    "frame": item.frame,
                    "label_index": item.label,
                    "label_uid": None if item.label < 0 else item.candidate_uids[item.label],
                    "candidate_uids": list(item.candidate_uids),
                    "candidate_count": count,
                    "candidate_probabilities": candidate_probabilities,
                    "predicted_index": int(predicted_index),
                    "predicted_uid": None if predicted_index < 0 else item.candidate_uids[predicted_index],
                    "none_probability": none_probability,
                    "present_probability": 1.0 - none_probability,
                    "selected_probability": 0.0 if predicted_index < 0 else float(candidate_values[predicted_index]),
                    "class_probability": class_probability,
                    "candidate_rank": rank,
                    "closed_set_top1_index": None if not count else int(candidate_order[0]),
                    "closed_set_top3": bool(item.label >= 0 and count and item.label in candidate_order[:3]),
                    "closed_set_top5": bool(item.label >= 0 and count and item.label in candidate_order[:5]),
                    "threshold": threshold,
                    "taxonomy": item.taxonomy,
                    "model_mode": model.mode,
                    "state_variant": model.state_variant,
                    "runtime_future_gt_used": False,
                    "runtime_gt_clean": True,
                    "candidate_created": False,
                }
            )
    return rows


def _calibrate_v1(predictions: Sequence[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    scores = np.asarray([float(row["present_probability"]) for row in predictions], dtype=np.float64)
    labels = np.asarray([int(row["label_index"]) >= 0 for row in predictions], dtype=bool)
    base = np.asarray([int(row["predicted_index"]) >= 0 for row in predictions], dtype=bool)
    negative_count = int((~labels).sum())
    positive_count = int(labels.sum())
    thresholds = sorted({0.0, 1.0, *np.quantile(scores, np.linspace(0.0, 1.0, 201)).tolist()}) if len(scores) > 1 else [0.0, 1.0]
    candidates = []
    for threshold in thresholds:
        accepted = base & (scores >= threshold)
        fpr = float((accepted & ~labels).sum() / negative_count) if negative_count else 0.0
        recall = float((accepted & labels).sum() / positive_count) if positive_count else 0.0
        candidates.append((float(threshold), fpr, recall))
    feasible = [item for item in candidates if item[1] <= 0.02]
    selected = max(feasible, key=lambda item: (item[2], -item[0])) if feasible else min(candidates, key=lambda item: (item[1], -item[0]))
    return selected[0], {"threshold": selected[0], "internal_validation_only": True, "validation_fpr": selected[1], "validation_open_recall": selected[2]}


def _div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positive = sum(bool(value) for value in labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]))
    rank_sum = 0.0
    cursor = 0
    rank = 1
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(scores[order[end]]) == float(scores[order[cursor]]):
            end += 1
        average = (rank + end - 1) / 2.0
        rank_sum += sum(average for position in order[cursor:end] if labels[position])
        rank += end - cursor
        cursor = end
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def _ece(scores: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float | None:
    if not scores:
        return None
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    total = float(len(values))
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (values >= low) & (values <= high if index == bins - 1 else values < high)
        if selected.any():
            result += float(selected.sum()) / total * abs(float(values[selected].mean()) - float(truth[selected].mean()))
    return float(result)


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    present = np.asarray([int(row["label_index"]) >= 0 for row in rows], dtype=bool)
    predicted = np.asarray([int(row["predicted_index"]) >= 0 for row in rows], dtype=bool)
    correct = np.asarray([predicted[i] and int(row["predicted_index"]) == int(row["label_index"]) for i, row in enumerate(rows)], dtype=bool)
    absent = ~present
    p0 = np.asarray([row["taxonomy"] == "P0_TARGET_ABSENT" for row in rows], dtype=bool)
    p1 = np.asarray([str(row["taxonomy"]).startswith("P1") for row in rows], dtype=bool)
    negative_count, positive_count = int(absent.sum()), int(present.sum())
    top1 = np.asarray([row["closed_set_top1_index"] is not None and int(row["closed_set_top1_index"]) == int(row["label_index"]) for row in rows], dtype=bool)
    top3 = np.asarray([bool(row["closed_set_top3"]) for row in rows], dtype=bool)
    top5 = np.asarray([bool(row["closed_set_top5"]) for row in rows], dtype=bool)
    ranks = [int(row["candidate_rank"]) for row in rows if row["candidate_rank"] is not None]
    none_correct = int((~predicted & absent).sum())
    binary_scores = [float(row["present_probability"]) for row in rows]
    binary_labels = [bool(value) for value in present]
    true_probs = [max(1.0e-12, float(row["class_probability"])) for row in rows]
    report = {
        "rows": len(rows),
        "candidate_set_positive_count": positive_count,
        "candidate_set_negative_count": negative_count,
        "predicted_present_count": int(predicted.sum()),
        "false_present_count": int((predicted & absent).sum()),
        "false_present_rate": _div(float((predicted & absent).sum()), float(negative_count)),
        "open_set_correct_identification_count": int(correct.sum()),
        "open_set_correct_identification_recall": _div(float(correct.sum()), float(positive_count)),
        "presence_precision": _div(float((predicted & present).sum()), float(predicted.sum())),
        "presence_recall": _div(float((predicted & present).sum()), float(positive_count)),
        "none_recall_on_absent": _div(float(none_correct), float(negative_count)),
        "none_precision": _div(float(none_correct), float((~predicted).sum())),
        "p0_count": int(p0.sum()),
        "p1_count": int(p1.sum()),
        "p0_false_present_rate": _div(float((predicted & p0).sum()), float(p0.sum())),
        "p1_false_present_rate": _div(float((predicted & p1).sum()), float(p1.sum())),
        "closed_set_candidate_top1_accuracy": _div(float((top1 & present).sum()), float(positive_count)),
        "rank_1_identity_accuracy": _div(float((top1 & present).sum()), float(positive_count)),
        "rank_2_identity_accuracy": _div(float(sum(1 for row in rows if row["candidate_rank"] is not None and int(row["candidate_rank"]) <= 2 and int(row["label_index"]) >= 0)), float(positive_count)),
        "rank_3_identity_accuracy": _div(float(sum(1 for row in rows if row["candidate_rank"] is not None and int(row["candidate_rank"]) <= 3 and int(row["label_index"]) >= 0)), float(positive_count)),
        "closed_set_top3": _div(float((top3 & present).sum()), float(positive_count)),
        "closed_set_top5": _div(float((top5 & present).sum()), float(positive_count)),
        "closed_set_MRR": float(np.mean([1.0 / value for value in ranks])) if ranks else None,
        "mean_target_candidate_rank": float(np.mean(ranks)) if ranks else None,
        "nll": float(np.mean([-math.log(value) for value in true_probs])),
        "brier": float(np.mean([(binary_scores[i] - float(binary_labels[i])) ** 2 for i in range(len(rows))])),
        "ece": _ece(binary_scores, binary_labels),
        "auroc": _auc(binary_scores, binary_labels),
        "runtime_future_gt_used": any(bool(row.get("runtime_future_gt_used")) for row in rows),
        "runtime_gt_clean": all(not bool(row.get("runtime_future_gt_used")) for row in rows),
        "candidate_created": any(bool(row.get("candidate_created")) for row in rows),
    }
    return report


def per_sequence_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    result = {}
    for sequence in SEQUENCES:
        item = metrics(grouped.get(sequence, []))
        item["sequence"] = sequence
        item["catastrophic"] = bool(
            (item.get("false_present_rate") is not None and item["false_present_rate"] > 0.20)
            or (item.get("open_set_correct_identification_recall") is not None and item["open_set_correct_identification_recall"] < 0.01)
        )
        result[sequence] = item
    return result


def aggregate_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    report = metrics(rows)
    sequence = per_sequence_metrics(rows)
    recalls = [item["open_set_correct_identification_recall"] for item in sequence.values() if item.get("open_set_correct_identification_recall") is not None]
    fprs = [item["false_present_rate"] for item in sequence.values() if item.get("false_present_rate") is not None]
    report["per_sequence"] = sequence
    report["macro_open_set_correct_identification_recall"] = float(np.mean(recalls)) if recalls else None
    report["macro_false_present_rate"] = float(np.mean(fprs)) if fprs else None
    catastrophic = [sequence for sequence, item in sequence.items() if item.get("catastrophic")]
    checks = {
        "S1_pooled_negative_fpr_le_0.02": report.get("false_present_rate") is not None and report["false_present_rate"] <= 0.02,
        "S2_pooled_open_recall_ge_0.60": report.get("open_set_correct_identification_recall") is not None and report["open_set_correct_identification_recall"] >= 0.60,
        "S3_macro_recall_ge_0.40": report.get("macro_open_set_correct_identification_recall") is not None and report["macro_open_set_correct_identification_recall"] >= 0.40,
        "S4_macro_fpr_le_0.05": report.get("macro_false_present_rate") is not None and report["macro_false_present_rate"] <= 0.05,
        "S5_runtime_future_gt_false": report.get("runtime_future_gt_used") is False,
        "S5b_runtime_gt_clean": report.get("runtime_gt_clean") is True,
        "S6_all_eight_sequences_reported": len(sequence) == len(SEQUENCES) and all(sequence_name in sequence for sequence_name in SEQUENCES),
        "S7_no_candidate_creation": report.get("candidate_created") is False,
    }
    report["static_gate"] = {"checks": checks, "pass": bool(all(checks.values())), "catastrophic_sequences": catastrophic, "cross_sequence_warning": len(catastrophic) >= 2}
    return report


def aggregate_seed_predictions(prediction_sets: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for predictions in prediction_sets:
        for row in predictions:
            grouped[(str(row["sequence"]), int(row["frame"]))].append(row)
    if len(grouped) != EXPECTED_FRAMES or any(len(values) != len(prediction_sets) for values in grouped.values()):
        raise RuntimeError("seed aggregation is not one row per real frame")
    result = []
    for key in sorted(grouped):
        values = grouped[key]
        reference = dict(values[0])
        candidate_uids = list(reference["candidate_uids"])
        if any(list(row["candidate_uids"]) != candidate_uids for row in values):
            raise RuntimeError(f"candidate axis differs across seeds: {key}")
        candidate_probabilities = np.mean([np.asarray(row["candidate_probabilities"], dtype=np.float64) for row in values], axis=0)
        none_probability = float(np.mean([float(row["none_probability"]) for row in values]))
        mode = str(reference["model_mode"])
        threshold_values = [row.get("threshold") for row in values if row.get("threshold") is not None]
        threshold = float(np.mean(threshold_values)) if threshold_values else None
        if mode == "V2_EXPLICIT_NONE":
            selected_class = int(np.argmax(np.concatenate([candidate_probabilities, [none_probability]])))
            predicted_index = selected_class if selected_class < len(candidate_uids) else -1
        else:
            predicted_index = -1 if not len(candidate_uids) else int(np.argmax(candidate_probabilities))
            if predicted_index >= 0 and threshold is not None and float(candidate_probabilities[predicted_index]) < threshold:
                predicted_index = -1
        order = np.argsort(-candidate_probabilities, kind="stable") if len(candidate_uids) else np.asarray([], dtype=np.int64)
        label_index = int(reference["label_index"])
        rank = None if label_index < 0 or not len(order) else (int(np.where(order == label_index)[0][0]) + 1 if label_index in order else None)
        reference.update(
            {
                "candidate_probabilities": candidate_probabilities.astype(float).tolist(),
                "none_probability": none_probability,
                "present_probability": 1.0 - none_probability,
                "predicted_index": predicted_index,
                "predicted_uid": None if predicted_index < 0 else candidate_uids[predicted_index],
                "selected_probability": 0.0 if predicted_index < 0 else float(candidate_probabilities[predicted_index]),
                "class_probability": float(candidate_probabilities[label_index]) if label_index >= 0 else none_probability,
                "candidate_rank": rank,
                "closed_set_top1_index": None if not len(order) else int(order[0]),
                "closed_set_top3": bool(label_index >= 0 and label_index in order[:3]),
                "closed_set_top5": bool(label_index >= 0 and label_index in order[:5]),
                "threshold": threshold,
                "seed_count": len(values),
            }
        )
        result.append(reference)
    return result


def _bootstrap(rows: Sequence[dict[str, Any]], repetitions: int = 2000) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    clusters = {sequence: metrics(grouped[sequence]) for sequence in SEQUENCES}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"negative_fpr": [], "open_recall": [], "closed_set_top1": [], "macro_recall": []}
    for _ in range(repetitions):
        sampled = [clusters[SEQUENCES[int(index)]] for index in rng.integers(0, len(SEQUENCES), size=len(SEQUENCES))]
        negative = sum(int(item["candidate_set_negative_count"]) for item in sampled)
        false_present = sum(int(item["false_present_count"]) for item in sampled)
        positive = sum(int(item["candidate_set_positive_count"]) for item in sampled)
        correct = sum(int(item["open_set_correct_identification_count"]) for item in sampled)
        top1_num = sum(float(item["closed_set_candidate_top1_accuracy"] or 0.0) * int(item["candidate_set_positive_count"]) for item in sampled)
        values["negative_fpr"].append(float(false_present / negative) if negative else 0.0)
        values["open_recall"].append(float(correct / positive) if positive else 0.0)
        values["closed_set_top1"].append(float(top1_num / positive) if positive else 0.0)
        values["macro_recall"].append(float(np.mean([float(item["open_set_correct_identification_recall"] or 0.0) for item in sampled])))
    return {
        "stage": STAGE,
        "repetitions": repetitions,
        "seed": BOOTSTRAP_SEED,
        "unit": "sequence cluster",
        "intervals_95": {name: [float(np.percentile(value, 2.5)), float(np.percentile(value, 97.5))] for name, value in values.items()},
        "means": {name: float(np.mean(value)) for name, value in values.items()},
    }


def _outer_fold(heldout: str) -> tuple[list[str], str, list[str]]:
    index = SEQUENCES.index(heldout)
    validation = SEQUENCES[(index + 1) % len(SEQUENCES)]
    nonheldout = [sequence for sequence in SEQUENCES if sequence != heldout]
    fit = [sequence for sequence in nonheldout if sequence != validation]
    return fit, validation, nonheldout


def _checkpoint(model: ExplicitNoneVerifier, path: Path, *, name: str, heldout: str, seed: int, fit: Sequence[str], validation: str, epochs: int, source_head: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "architecture": name,
        "state_variant": model.state_variant,
        "source_commit": source_head,
        "outer_heldout": heldout,
        "parameter_fit_sequences": list(fit),
        "internal_validation_sequence": validation,
        "final_training_sequences": [sequence for sequence in SEQUENCES if sequence != heldout],
        "seed": int(seed),
        "selected_epoch": int(epochs),
        "projection_dim": model.projection_dim,
        "hidden_dim": model.hidden_dim,
        "trainable_parameters": model.trainable_parameters,
        "osnet_frozen": True,
        "n72r18_gru_frozen": True,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "candidate_created": False,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(payload, path)
    return _sha256(path)


def run_formal(dataset: Dataset, device: torch.device) -> dict[str, Any]:
    source_head = _git_head()
    prediction_archive: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for name, mode, variant in MODEL_SPECS:
        for seed in SEEDS:
            for heldout in SEQUENCES:
                fit_sequences, validation_sequence, nonheldout = _outer_fold(heldout)
                fit_examples = [row for row in dataset.examples if row.sequence in fit_sequences]
                validation_examples = [row for row in dataset.examples if row.sequence == validation_sequence]
                heldout_examples = [row for row in dataset.examples if row.sequence == heldout]
                inner_model, inner_fit = _fit_inner(dataset, fit_examples, validation_examples, mode=mode, variant=variant, seed=seed, device=device)
                calibration_predictions = _raw_predictions(inner_model, dataset, validation_examples, device)
                threshold = None
                calibration = None
                if mode == "V1_PAIRWISE":
                    threshold, calibration = _calibrate_v1(calibration_predictions)
                final_model, final_fit = _fit_final(dataset, [row for row in dataset.examples if row.sequence in nonheldout], mode=mode, variant=variant, seed=seed, epochs=int(inner_fit["best_epoch"]), device=device)
                predictions = _raw_predictions(final_model, dataset, heldout_examples, device, threshold=threshold)
                if len(predictions) != EXPECTED_SEQUENCE_COUNTS[heldout] or len({(row["sequence"], row["frame"]) for row in predictions}) != len(predictions):
                    raise RuntimeError(f"outer prediction invariant failed for {name}/{seed}/{heldout}")
                report = aggregate_report(predictions)
                report.update(
                    {
                        "model": name,
                        "seed": seed,
                        "heldout_sequence": heldout,
                        "parameter_fit_sequences": fit_sequences,
                        "internal_validation_sequence": validation_sequence,
                        "final_training_sequences": nonheldout,
                        "outer_heldout_absent_from_fit": heldout not in fit_sequences,
                        "outer_heldout_absent_from_validation": heldout != validation_sequence,
                        "selected_epoch": inner_fit["best_epoch"],
                        "inner_fit": inner_fit,
                        "final_fit": final_fit,
                        "v1_calibration": calibration,
                    }
                )
                records.append(report)
                prediction_archive[name][str(seed)].extend(predictions)
                checkpoint_path = ASSET_MODELS / f"{name}__{heldout}__seed{seed}.pt"
                checkpoint_sha = _checkpoint(final_model, checkpoint_path, name=name, heldout=heldout, seed=seed, fit=fit_sequences, validation=validation_sequence, epochs=int(inner_fit["best_epoch"]), source_head=source_head)
                checkpoints.append(
                    {
                        "model": name,
                        "heldout_sequence": heldout,
                        "seed": seed,
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha,
                        "trainable_parameters": final_model.trainable_parameters,
                        "selected_epoch": inner_fit["best_epoch"],
                        "binary_committed": False,
                    }
                )
    primary_reports: dict[str, Any] = {}
    seed_summary: dict[str, Any] = {}
    for name, _, _ in MODEL_SPECS:
        per_seed = []
        for seed in SEEDS:
            per_seed.append(dict(aggregate_report(prediction_archive[name][str(seed)]), model=name, seed=seed))
        aggregate = aggregate_seed_predictions([prediction_archive[name][str(seed)] for seed in SEEDS])
        primary_reports[name] = aggregate_report(aggregate)
        seed_summary[name] = {"seeds": list(SEEDS), "per_seed": per_seed, "three_seed_aggregate": primary_reports[name]}
        pred_path = STATIC / f"predictions_{name}.jsonl.zst"
        write_zstd_jsonl(pred_path, aggregate)
    selected_name = "V2_DUAL_STATE"
    selected = aggregate_seed_predictions([prediction_archive[selected_name][str(seed)] for seed in SEEDS])
    if len(selected) != EXPECTED_FRAMES:
        raise RuntimeError("primary seed ensemble row count is not 8414")
    STATIC.mkdir(parents=True, exist_ok=True)
    write_zstd_jsonl(STATIC / "primary_predictions.jsonl.zst", selected)
    formal_manifest = {
        "stage": STAGE,
        "status": "PASS_FORMAL_ONE_FRAME_LOSO_COMPLETE",
        "source_head": source_head,
        "models": [name for name, _, _ in MODEL_SPECS],
        "primary_model": selected_name,
        "outer_folds": len(SEQUENCES),
        "seeds": list(SEEDS),
        "per_seed_rows": EXPECTED_FRAMES,
        "primary_ensemble_rows": len(selected),
        "primary_positive": int(primary_reports[selected_name]["candidate_set_positive_count"]),
        "primary_negative": int(primary_reports[selected_name]["candidate_set_negative_count"]),
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "candidate_created": False,
        "candidate_generation_rerun": False,
        "sam3_rerun": False,
        "val_accessed": False,
        "test_accessed": False,
    }
    _write_json(STATIC / "formal_predictions_manifest.json", formal_manifest)
    _write_json(
        TRAINING / "formal_records.json",
        {
            "stage": STAGE,
            "records": [
                {
                    "model": record["model"],
                    "seed": record["seed"],
                    "heldout_sequence": record["heldout_sequence"],
                    "parameter_fit_sequences": record["parameter_fit_sequences"],
                    "internal_validation_sequence": record["internal_validation_sequence"],
                    "final_training_sequences": record["final_training_sequences"],
                    "outer_heldout_absent_from_fit": record["outer_heldout_absent_from_fit"],
                    "outer_heldout_absent_from_validation": record["outer_heldout_absent_from_validation"],
                    "selected_epoch": record["selected_epoch"],
                    "v1_threshold_calibration_source": "internal_validation_sequence_only" if record["model"] == "V1_PAIRWISE_THRESHOLD_NONE" else "not_applicable",
                    "architecture_selection_from_outer_scores": False,
                }
                for record in records
            ],
        },
    )
    _write_json(TRAINING / "checkpoint_manifest.json", {"stage": STAGE, "records": checkpoints, "binary_committed": False, "asset_root": str(ASSET_MODELS)})
    _write_json(
        STATIC / "ablations.json",
        {"stage": STAGE, "primary_model": selected_name, "reports": primary_reports, "diagnostic_only": [name for name, _, _ in MODEL_SPECS if name != selected_name]},
    )
    _write_json(STATIC / "per_sequence.json", {"stage": STAGE, "primary_model": selected_name, "records": primary_reports[selected_name]["per_sequence"]})
    _write_json(STATIC / "ranking_diagnostics.json", {"stage": STAGE, "primary_model": selected_name, "closed_set_candidate_top1": primary_reports[selected_name]["closed_set_candidate_top1_accuracy"], "closed_set_MRR": primary_reports[selected_name]["closed_set_MRR"], "mean_target_candidate_rank": primary_reports[selected_name]["mean_target_candidate_rank"], "top3": primary_reports[selected_name]["closed_set_top3"], "top5": primary_reports[selected_name]["closed_set_top5"], "per_sequence": {key: {metric: value.get(metric) for metric in ("closed_set_candidate_top1_accuracy", "closed_set_MRR", "mean_target_candidate_rank", "closed_set_top3", "closed_set_top5")} for key, value in primary_reports[selected_name]["per_sequence"].items()}})
    return {"source_head": source_head, "records": records, "checkpoints": checkpoints, "prediction_archive": prediction_archive, "primary_reports": primary_reports, "selected": selected, "seed_summary": seed_summary, "formal_manifest": formal_manifest}


def _oracle_reports(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    present = []
    oracle_presence = []
    oracle_none = []
    for row in rows:
        value = dict(row)
        if int(value["label_index"]) >= 0:
            value["predicted_index"] = value["closed_set_top1_index"]
            value["predicted_uid"] = None if value["predicted_index"] is None else value["candidate_uids"][value["predicted_index"]]
            present.append(value)
        oracle_presence.append(value)
        oracle_none_value = dict(value)
        if int(oracle_none_value["label_index"]) < 0:
            oracle_none_value["predicted_index"] = -1
            oracle_none_value["predicted_uid"] = None
        oracle_none.append(oracle_none_value)
    return {
        "stage": STAGE,
        "posthoc_oracle_only": True,
        "runtime_usable": False,
        "oracle_presence": aggregate_report(present),
        "oracle_none": aggregate_report(oracle_none),
    }


def _historical_comparison(primary_reports: Mapping[str, Any]) -> dict[str, Any]:
    historical_selected = _json(HIST / "static_eval" / "selected_report.json")
    historical_loso = _json(HIST / "static_eval" / "loso_results.json")
    result = {
        "stage": STAGE,
        "historical_stage": "N72R20R3R1",
        "historical_rows": int(historical_selected["rows"]),
        "corrected_rows": EXPECTED_FRAMES,
        "historical_positive": int(historical_selected["candidate_set_positive_count"]),
        "corrected_positive": EXPECTED_PRESENT,
        "historical_negative": int(historical_selected["candidate_set_negative_count"]),
        "corrected_negative": EXPECTED_ABSENT,
        "historical_primary": {key: historical_selected.get(key) for key in ("false_present_rate", "open_set_correct_identification_recall", "nll", "ece", "candidate_top1_accuracy_on_present")},
        "corrected_models": {
            name: {key: report.get(key) for key in ("false_present_rate", "open_set_correct_identification_recall", "nll", "ece", "closed_set_candidate_top1_accuracy", "closed_set_MRR")}
            for name, report in primary_reports.items()
        },
        "historical_per_model": {
            name: {key: report.get(key) for key in ("false_present_rate", "open_set_correct_identification_recall", "nll", "ece", "candidate_top1_accuracy_on_present")}
            for name, report in historical_loso.get("reports", {}).items()
        },
        "protocol_correction": "one real frame = one decision; outer-heldout architecture selection removed; provenance booleans corrected",
    }
    _write_json(STATIC / "historical_vs_corrected.json", result)
    return result


def _headroom(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = read_zstd_jsonl(ROOT / "outputs/N72R20R3/presence/frame_posthoc_labels.jsonl.zst")
    label_map = {(str(row["sequence"]), int(row["frame"])): row for row in labels if str(row.get("state_condition")) == "S1_R2_STYLE_CAUSAL_LEARNED_STATE"}
    entries = []
    for row in rows:
        label = label_map[(str(row["sequence"]), int(row["frame"]))]
        truth = label.get("best_target_candidate_uid") if label.get("candidate_set_present") else None
        base = label.get("base_candidate_uid")
        if truth is not None and base != truth and row.get("predicted_uid") == truth:
            entries.append(
                {
                    "sequence": row["sequence"],
                    "frame": row["frame"],
                    "verifier_candidate": row["predicted_uid"],
                    "base_candidate": base,
                    "verifier_confidence": row["selected_probability"],
                    "NONE_probability": row["none_probability"],
                    "closed_set_rank": row["candidate_rank"],
                    "base_margin": None,
                    "motion_evidence": {"source": "read_only_headroom", "available": False},
                }
            )
    return {
        "stage": STAGE,
        "status": "READ_ONLY_STATIC_HEADROOM_ONLY",
        "association_rescue_run": False,
        "solver_called": False,
        "public_authority_changed": False,
        "base_assignment_unchanged": True,
        "candidate_created": False,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "base_wrong_verifier_correct_count": len(entries),
        "examples": entries[:100],
        "interpretation": "logged only; no verifier prediction was applied to association",
    }


def _write_causal_not_run(decision: str) -> None:
    CAUSAL.mkdir(parents=True, exist_ok=True)
    _write_json(CAUSAL / "causal_replay.json", {"stage": STAGE, "status": "NOT_RUN_STATIC_GATE_FAILED", "causal_commit_replay_authorized": False, "reason": decision, "runtime_future_gt_used": False, "runtime_gt_clean": True, "association_rescue_run": False, "solver_called": False})
    _write_json(CAUSAL / "per_sequence.json", {"stage": STAGE, "status": "NOT_RUN_STATIC_GATE_FAILED", "sequences": {sequence: {"status": "NOT_RUN_STATIC_GATE_FAILED"} for sequence in SEQUENCES}})
    write_zstd_jsonl(CAUSAL / "memory_write_audit.jsonl.zst", [])


def finalize(formal: Mapping[str, Any], tests_summary: str = "PENDING") -> dict[str, Any]:
    primary_reports = formal["primary_reports"]
    selected = primary_reports["V2_DUAL_STATE"]
    static_pass = bool(selected["static_gate"]["pass"])
    decision = "PASS_PROTOCOL_CORRECTED_EXPLICIT_NONE_VERIFICATION" if static_pass else "FAIL_PROTOCOL_CORRECTED_EXPLICIT_NONE_GENERALIZATION"
    if not static_pass:
        _write_causal_not_run(decision)
    _write_json(STATIC / "primary_result.json", {"stage": STAGE, "primary_model": "V2_DUAL_STATE", "metrics": selected, "static_gate": selected["static_gate"], "decision": decision})
    oracle = _oracle_reports(formal["selected"])
    _write_json(STATIC / "oracle_diagnostics.json", oracle)
    historical = _historical_comparison(primary_reports)
    headroom = _headroom(formal["selected"])
    _write_json(OUT / "future_association_headroom.json", headroom)
    bootstrap = _bootstrap(formal["selected"])
    _write_json(STATIC / "bootstrap.json", bootstrap)
    immutability = _json(OUT / "r3r1_immutability_audit.json")
    after_hashes = _historical_hashes()
    before = {item["path"]: item["sha256"] for item in immutability["before"]["files"]}
    after = {item["path"]: item["sha256"] for item in after_hashes}
    unchanged = before == after
    immutability.update({"after": {"files": after_hashes}, "all_unchanged": unchanged, "historical_r3r1_modified": not unchanged})
    _write_json(OUT / "r3r1_immutability_audit.json", immutability)
    storage_after = storage_audit()
    _write_json(OUT / "storage_audit_after.json", storage_after)
    bottleneck = None
    if not static_pass:
        top1 = float(selected.get("closed_set_candidate_top1_accuracy") or 0.0)
        fpr = float(selected.get("false_present_rate") or 1.0)
        recall = float(selected.get("open_set_correct_identification_recall") or 0.0)
        if top1 >= 0.60 and (fpr > 0.02 or recall < 0.60):
            bottleneck = "BOTTLENECK_NONE_CALIBRATION"
        elif top1 < 0.40 and float(selected.get("closed_set_MRR") or 0.0) < 0.60:
            bottleneck = "BOTTLENECK_CANDIDATE_IDENTITY_REPRESENTATION"
        else:
            bottleneck = "BOTTLENECK_MIXED_IDENTITY_AND_ABSTENTION"
    result = {
        "stage": STAGE,
        "goal": "Protocol-Corrected Explicit-NONE Identity Verification",
        "central_question": _json(OUT / "FINAL_GOAL.json")["central_question"],
        "source_branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
        "source_head": formal["source_head"],
        "historical_r3r1_decision": "FAIL_EXPLICIT_NONE_CROSS_SEQUENCE_VERIFICATION",
        "historical_reported_final_commit": _json(OUT / "provenance_repair.json")["historical_reported_final_commit"],
        "protocol_defects_confirmed": ["A_DUPLICATED_FRAME_SAMPLES", "B_OUTER_HELDOUT_SELECTION_CONTAMINATION", "C_RUNTIME_GT_PROVENANCE_INVERSION", "D_STALE_HISTORICAL_COMMIT_PROVENANCE", "E_NON_PREREGISTERED_ARCHITECTURE_SELECTION"],
        "historical_row_count": 16828,
        "corrected_row_count": EXPECTED_FRAMES,
        "historical_positive_count": 13190,
        "corrected_positive_count": EXPECTED_PRESENT,
        "historical_negative_count": 3638,
        "corrected_negative_count": EXPECTED_ABSENT,
        "outer_leakage_removed": True,
        "runtime_future_gt_used": bool(selected["runtime_future_gt_used"]),
        "runtime_gt_clean": bool(selected["runtime_gt_clean"]),
        "primary_model": "V2_DUAL_STATE",
        "closed_set_candidate_top1": selected["closed_set_candidate_top1_accuracy"],
        "closed_set_MRR": selected["closed_set_MRR"],
        "negative_FPR": selected["false_present_rate"],
        "open_set_correct_ID_recall": selected["open_set_correct_identification_recall"],
        "macro_FPR": selected["macro_false_present_rate"],
        "macro_recall": selected["macro_open_set_correct_identification_recall"],
        "static_gate": selected["static_gate"],
        "causal_metrics": None,
        "causal_commit_replay_authorized": False,
        "bottleneck_classification": bottleneck,
        "next_representation_learning_stage_authorized": not static_pass,
        "next_association_authority_stage_authorized": static_pass,
        "decision": decision,
        "historical_vs_corrected": historical,
        "bootstrap": bootstrap,
        "r3r1_immutability_all_unchanged": unchanged,
        "sam3_rerun": False,
        "osnet_frozen": True,
        "n72r18_gru_frozen": True,
        "exact_solver_frozen": True,
        "public_id_authority_frozen": True,
        "val_accessed": False,
        "test_accessed": False,
        "association_rescue_run": False,
        "storage_after": storage_after,
        "tests_summary": tests_summary,
    }
    _write_json(OUT / "FINAL_RESULT.json", result)
    first = "YES" if static_pass else "NO"
    report_lines = [
        f"{first} — after correcting the one-frame decision protocol and fully isolating outer held-out sequences, explicit-NONE verification over the frozen InterMOT identity representation {'succeeds' if static_pass else 'does not succeed'}.",
        "",
        f"# InterMOT {STAGE} — Protocol-Corrected Explicit-NONE Identity Verification",
        "",
        f"Final decision: `{decision}`.",
        "",
        f"The primary model was pre-frozen as `V2_DUAL_STATE`; no architecture was selected from formal outer scores. The corrected index has exactly `{EXPECTED_FRAMES}` unique `(sequence, frame)` decisions, with `{EXPECTED_PRESENT}` PRESENT and `{EXPECTED_ABSENT}` NONE labels.",
        "",
        "## Primary static result",
        "",
        f"- pooled negative FPR: `{selected['false_present_rate']}`",
        f"- pooled open-set correct-ID recall: `{selected['open_set_correct_identification_recall']}`",
        f"- macro FPR: `{selected['macro_false_present_rate']}`",
        f"- macro open-ID recall: `{selected['macro_open_set_correct_identification_recall']}`",
        f"- closed-set candidate top-1: `{selected['closed_set_candidate_top1_accuracy']}`",
        f"- closed-set MRR: `{selected['closed_set_MRR']}`",
        f"- static gate pass: `{selected['static_gate']['pass']}`",
        "",
        "## Protocol and provenance",
        "",
        "- 8 outer sequence-held-out folds × seeds 720301/720302/720303.",
        "- Each outer fold trains on 6 sequences, selects epoch on one cyclic inner-validation sequence, then refits on all 7 non-heldout sequences for that epoch.",
        "- Seed probabilities are averaged into one final prediction per real frame; no 2× state-condition rows remain.",
        f"- `runtime_future_gt_used`: `{selected['runtime_future_gt_used']}`; `runtime_gt_clean`: `{selected['runtime_gt_clean']}`.",
        f"- Historical R3R1 artifacts unchanged: `{unchanged}`.",
        "",
        "## Sequence-cluster bootstrap",
        "",
        f"2,000 repetitions with seed `{BOOTSTRAP_SEED}`; 95% intervals: `{bootstrap['intervals_95']}`.",
        "",
        "## Terminal boundary",
        "",
        f"Causal replay: `{'NOT_RUN_STATIC_GATE_FAILED' if not static_pass else 'NOT_IMPLEMENTED_IN_STATIC_TERMINAL'}`.",
        "Association rescue, solver changes, SAM3, DanceTrack VAL/TEST, TrackEval, LoRA, Transformer, JEV, Mamba, OSNet retraining and GRU retraining were not run.",
        "",
        f"Bottleneck classification: `{bottleneck}`.",
        f"`next_representation_learning_stage_authorized={not static_pass}`.",
        f"`next_association_authority_stage_authorized={static_pass}`.",
        "",
        f"Focused and full test summary: {tests_summary}.",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _write_json(
        OUT / "stage_status.json",
        {
            "stage": STAGE,
            "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json",
            "goal": "Protocol-Corrected Explicit-NONE Identity Verification",
            "central_question": result["central_question"],
            "goal_frozen": True,
            "status": decision,
            "phase": "STATIC_TERMINAL",
            "canonical_index": "PASS_CANONICAL_ONE_FRAME_INDEX",
            "formal_loso": "PASS_FORMAL_ONE_FRAME_LOSO_COMPLETE",
            "primary_model": "V2_DUAL_STATE",
            "static_gate": selected["static_gate"],
            "causal_replay": "NOT_RUN_STATIC_GATE_FAILED" if not static_pass else "NOT_RUN_IN_STATIC_TERMINAL",
            "causal_commit_replay_authorized": False,
            "next_representation_learning_stage_authorized": not static_pass,
            "next_association_authority_stage_authorized": static_pass,
            "runtime_future_gt_used": bool(selected["runtime_future_gt_used"]),
            "runtime_gt_clean": bool(selected["runtime_gt_clean"]),
            "sam3_rerun": False,
            "val_accessed": False,
            "test_accessed": False,
            "association_rescue_run": False,
            "tests": tests_summary,
            "storage_audit_after": storage_after,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "build", "run", "finalize", "all"), default="all")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--tests-summary", default="PENDING")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = storage_audit()
    _write_json(OUT / "storage_audit_before.json", before)
    if before["filesystem"]["storage_status"] == "HARD_STOP":
        raise RuntimeError("storage hard stop")
    audit = None
    if args.phase in {"audit", "build", "run", "finalize", "all"}:
        audit = audit_protocol()
    if args.phase in {"build", "run", "all"}:
        build_canonical_index()
    if args.phase in {"run", "all"}:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        dataset = load_dataset()
        formal = run_formal(dataset, device)
        _write_json(OUT / "_formal_runtime.json", {"status": "PASS_FORMAL_ONE_FRAME_LOSO_COMPLETE", "source_head": formal["source_head"], "selected_model": "V2_DUAL_STATE"})
        finalize(formal, tests_summary=args.tests_summary)
    elif args.phase == "finalize":
        raise RuntimeError("finalize requires the in-memory formal run; rerun with --phase all")
    print(json.dumps({"status": "PASS_PROTOCOL_REPAIR_PREPARATION" if args.phase in {"audit", "build"} else "PASS_N72R20R3R1R1_COMPLETE", "stage": STAGE, "source_head": _git_head()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
