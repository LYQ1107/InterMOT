"""Frozen protocol and corruption-manifest helpers for N72R19R1."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

from sam3_intermot.identity_research.protocol import protocol_sha256

from .encoder import normalize

R1_STAGE = "N72R19R1"
R1_GOAL = "Selective Identity Memory Update"
R1_QUESTION = (
    "Can selective observation updates preserve the strong learned identity memory "
    "while preventing corrupted observations from damaging future identity recognition?"
)
MANIFEST_VERSION = "N72R19R1-fixed-corruption-v1"
MANIFEST_SEED = 72191
SPLIT_SEED = 72191
NOISY_RATES = (0.10, 0.20, 0.30, 0.50)
CONDITION_KINDS = ("wrong_identity", "missing_observation", "hard_negative")


def rate_slug(rate: float) -> str:
    return f"{float(rate):.2f}".replace(".", "p")


def condition_slug(kind: str, rate: float = 0.0) -> str:
    if kind == "clean":
        return "clean_0p00"
    return f"{kind}_{rate_slug(rate)}"


def all_condition_slugs() -> tuple[str, ...]:
    return (condition_slug("clean"),) + tuple(
        condition_slug(kind, rate) for kind in CONDITION_KINDS for rate in NOISY_RATES
    )


def primary_condition_slugs() -> tuple[str, ...]:
    return (
        condition_slug("wrong_identity", 0.20),
        condition_slug("wrong_identity", 0.30),
        condition_slug("hard_negative", 0.20),
        condition_slug("hard_negative", 0.30),
    )


def manifest_key(sequence: str, track_id: int, anchor_frame: int, frame: int) -> str:
    return f"{sequence}|{int(track_id)}|{int(anchor_frame)}|{int(frame)}"


def deterministic_uniform(seed: int, *parts: object) -> float:
    payload = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(2**64)


def _split_sequences(sequences: Iterable[str], seed: int = SPLIT_SEED) -> tuple[list[str], list[str]]:
    values = sorted(set(str(sequence) for sequence in sequences))
    if len(values) < 2:
        raise ValueError("at least two train sequences are required for train-dev splitting")
    generator = random.Random(seed)
    generator.shuffle(values)
    train_count = int(round(len(values) * 0.8))
    train_count = min(max(train_count, 1), len(values) - 1)
    return sorted(values[:train_count]), sorted(values[train_count:])


def make_protocol_document(
    train_protocol_path: str | Path,
    val_protocol_path: str | Path,
    *,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    train_doc = json.loads(Path(train_protocol_path).read_text(encoding="utf-8"))
    val_doc = json.loads(Path(val_protocol_path).read_text(encoding="utf-8"))
    train_sequences = [str(item["sequence"]) for item in train_doc.get("anchors", [])]
    train_split, dev_split = _split_sequences(train_sequences, seed=split_seed)
    val_sequences = sorted({str(item["sequence"]) for item in val_doc.get("anchors", [])})
    return {
        "stage": R1_STAGE,
        "goal_reference": "outputs/N72R19R1/FINAL_GOAL.json",
        "goal": R1_GOAL,
        "central_question": R1_QUESTION,
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "metric_definition": (
            "positive_score > max similarity to all other visible identity embeddings; ties are losses"
        ),
        "tie_policy": "strict positive_score > hard_negative_score; ties are losses",
        "inherited_protocol": {
            "train_path": str(Path(train_protocol_path).resolve()),
            "train_sha256": protocol_sha256(train_protocol_path),
            "train_anchor_count": len(train_doc.get("anchors", [])),
            "train_sequence_count": len(set(train_sequences)),
            "val_path": str(Path(val_protocol_path).resolve()),
            "val_sha256": protocol_sha256(val_protocol_path),
            "val_anchor_count": len(val_doc.get("anchors", [])),
            "val_sequence_count": len(val_sequences),
            "embedding_dimension": 512,
            "encoder": "frozen_osnet_x1_0_market1501",
            "raw_crops_saved": False,
        },
        "train_dev_split": {
            "seed": int(split_seed),
            "unit": "sequence",
            "train_sequence_count": len(train_split),
            "dev_sequence_count": len(dev_split),
            "train_sequences": train_split,
            "dev_sequences": dev_split,
            "val_sequences": val_sequences,
            "identity_cross_sequence_transfer": False,
            "val_used_for_tuning": False,
        },
        "methods": {
            "B0_SINGLE_ANCHOR": "immutable human anchor; no dynamic update",
            "B1_EMA_0.90": "always-accept EMA update with alpha 0.90",
            "B2_FROZEN_GRU": "strictly loaded and frozen N72R18 GRU; always accept",
            "B3_SIMILARITY_THRESHOLD": "frozen GRU with train-dev threshold on cos(z, x)",
            "B4_CORRECTNESS_SELECTOR": "frozen GRU with BCE clean/corrupt selector",
            "B5_FUTURE_UTILITY_SELECTOR": "proposed frozen GRU with future-utility selector",
        },
        "selector": {
            "feature_names": [
                "memory_similarity",
                "anchor_similarity",
                "assigned_score",
                "best_competitor_score",
                "competition_margin",
                "top1_top2_margin",
                "assigned_rank_normalized",
                "competition_entropy",
                "prospective_state_change",
                "anchor_drift_candidate",
                "temporal_gap_normalized",
                "memory_age_normalized",
                "recent_state_stability",
            ],
            "hidden_layers": [64, 32],
            "hard_selection": "selection_score >= threshold",
            "missing_policy": "HOLD without selector",
            "parameter_limit": 100000,
            "human_anchor": "immutable first human-confirmed embedding",
        },
        "corruption_manifest": {
            "path": "outputs/N72R19R1/corruption_manifest.json",
            "version": MANIFEST_VERSION,
            "seed": MANIFEST_SEED,
            "reference": "frozen N72R18 GRU clean causal state",
            "conditions": list(all_condition_slugs()),
            "same_manifest_for_all_methods": True,
            "hard_negative_replacement_is_method_independent": True,
        },
        "training_curriculum": {
            "phase_A": ["clean_0p00", "wrong_identity_0p10", "hard_negative_0p10"],
            "phase_B": ["clean_0p00", "wrong_identity_0p10", "hard_negative_0p10", "wrong_identity_0p20", "hard_negative_0p20"],
            "phase_C": ["clean_0p00", "wrong_identity_0p10", "hard_negative_0p10", "wrong_identity_0p20", "hard_negative_0p20", "wrong_identity_0p30", "hard_negative_0p30"],
            "phase_D_50_percent": "stress test only; not a default training condition",
        },
        "success_rule": {
            "clean_preservation": "B5 clean H100 >= 0.90 or relative regression <= 1.5 percentage points vs B2",
            "robust_primary": "four primary-cell B5 ROBUST_H100_MACRO delta vs B2 >= +0.03 pp and paired sequence bootstrap CI lower > 0",
            "no_catastrophic_cell": "Wrong30 and HardNeg30 B5 must not be below EMA by a catastrophic margin",
            "pass_decision": "PASS_SELECTIVE_MEMORY_UPDATE",
            "clean_failure_decision": "FAIL_CLEAN_PRESERVATION",
            "robust_failure_decision": "FAIL_SELECTIVE_UPDATE_ROBUSTNESS",
        },
        "bootstrap": {"unit": "sequence", "reps": 2000, "seed": 72191},
        "sam3_required": False,
        "mot_required": False,
        "new_encoder_training": False,
        "goal_frozen": True,
        "next_candidate_stream_stage_authorized": False,
        "next_association_stage_authorized": False,
    }


def _valid_competitor_indices(mask: torch.Tensor) -> list[int]:
    return [int(index) for index in torch.nonzero(mask, as_tuple=False).flatten().tolist()]


@torch.no_grad()
def build_corruption_manifest(
    dataset: Any,
    frozen_gru: torch.nn.Module,
    *,
    device: torch.device | str,
    output_path: str | Path,
    seed: int = MANIFEST_SEED,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Create one method-independent corruption decision for every episode/frame."""

    device = torch.device(device)
    frozen_gru.eval()
    rows: dict[str, dict[str, Any]] = {}
    condition_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()

    for indices in dataset.batch_indices(batch_size=batch_size, shuffle=False):
        batch = dataset.make_batch(indices, device=device)
        state = normalize(batch["anchor_embeddings"])
        targets = batch["target_embeddings"]
        target_mask = batch["target_mask"]
        competitors = batch["competitor_embeddings"]
        competitor_mask = batch["competitor_mask"]
        frames = batch["frames"]
        metadata = batch["metadata"]
        for step in range(target_mask.shape[1]):
            valid = target_mask[:, step]
            for batch_index, meta in enumerate(metadata):
                if not bool(valid[batch_index]):
                    continue
                anchor = meta.anchor
                frame = int(frames[batch_index, step].item())
                future = anchor.future[step]
                key = manifest_key(anchor.sequence, anchor.track_id, anchor.anchor_frame, frame)
                valid_indices = _valid_competitor_indices(competitor_mask[batch_index, step])
                hard_index: int | None = None
                hard_score: float | None = None
                if valid_indices:
                    comp = normalize(competitors[batch_index, step, valid_indices])
                    scores = torch.matmul(comp, state[batch_index])
                    local = int(torch.argmax(scores).item())
                    hard_index = valid_indices[local]
                    hard_score = float(scores[local].item())
                if valid_indices:
                    draw = deterministic_uniform(seed, key, step, "wrong_identity")
                    wrong_index = valid_indices[int(draw * len(valid_indices)) % len(valid_indices)]
                else:
                    wrong_index = None

                conditions: dict[str, dict[str, Any]] = {
                    condition_slug("clean"): {
                        "kind": "clean",
                        "rate": 0.0,
                        "applied": False,
                        "present": True,
                        "replacement_index": None,
                        "replacement_track_id": None,
                    }
                }
                for kind in CONDITION_KINDS:
                    for rate in NOISY_RATES:
                        slug = condition_slug(kind, rate)
                        applied = deterministic_uniform(seed, key, step, kind, rate, "apply") < rate
                        replacement_index = None
                        if applied and kind == "wrong_identity":
                            replacement_index = wrong_index
                            if replacement_index is None:
                                applied = False
                        elif applied and kind == "hard_negative":
                            replacement_index = hard_index
                            if replacement_index is None:
                                applied = False
                        present = not (kind == "missing_observation" and applied)
                        replacement_track_id = (
                            int(future.competitors[replacement_index].track_id)
                            if replacement_index is not None
                            else None
                        )
                        conditions[slug] = {
                            "kind": kind,
                            "rate": float(rate),
                            "applied": bool(applied),
                            "present": bool(present),
                            "replacement_index": replacement_index,
                            "replacement_track_id": replacement_track_id,
                        }
                        condition_counts[slug] += 1
                        if applied:
                            applied_counts[slug] += 1
                rows[key] = {
                    "key": key,
                    "sequence": anchor.sequence,
                    "track_id": int(anchor.track_id),
                    "anchor_frame": int(anchor.anchor_frame),
                    "frame": frame,
                    "step": int(step),
                    "competitor_track_ids": [int(box.track_id) for box in future.competitors],
                    "reference_hard_negative_track_id": (
                        int(future.competitors[hard_index].track_id) if hard_index is not None else None
                    ),
                    "reference_hard_negative_score": hard_score,
                    "conditions": conditions,
                }
            new_state, _, _ = frozen_gru(state, targets[:, step])
            state = torch.where(valid.unsqueeze(-1), new_state, state)

    document = {
        "stage": R1_STAGE,
        "goal_reference": "outputs/N72R19R1/FINAL_GOAL.json",
        "goal": R1_GOAL,
        "central_question": R1_QUESTION,
        "manifest_version": MANIFEST_VERSION,
        "seed": int(seed),
        "reference_memory": "outputs/N72R18/checkpoints/identity_memory_gru.pt",
        "reference_state_policy": "clean causal score-before-update frozen N72R18 GRU state",
        "same_manifest_for_all_methods": True,
        "real_observations": len(rows),
        "synthetic_corruption_instances": int(sum(applied_counts.values())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "applied_counts": dict(sorted(applied_counts.items())),
        "rows": rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def load_corruption_manifest(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("corruption manifest version does not match N72R19R1")
    if document.get("goal") != R1_GOAL:
        raise ValueError("corruption manifest Goal does not match N72R19R1")
    return document


__all__ = [
    "CONDITION_KINDS",
    "MANIFEST_SEED",
    "MANIFEST_VERSION",
    "NOISY_RATES",
    "R1_GOAL",
    "R1_QUESTION",
    "R1_STAGE",
    "all_condition_slugs",
    "build_corruption_manifest",
    "condition_slug",
    "deterministic_uniform",
    "load_corruption_manifest",
    "make_protocol_document",
    "manifest_key",
    "primary_condition_slugs",
    "rate_slug",
]
