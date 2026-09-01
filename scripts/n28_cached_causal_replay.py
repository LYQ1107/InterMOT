#!/usr/bin/env python3
"""Run the frozen N28-B cached-feature causal feasibility experiment.

The N27 caches are the only data source in this phase.  They contain one
target parent per row, so this runner evaluates the same target-by-candidate
matrix (including a private NONE column) for each cached parent.  It does not
invent global detection IDs to create a multi-target candidate union.  The
multi-target Hungarian invariant is covered by the N28-A smoke; this phase is
explicitly a cached-feature feasibility result, not a SAM3/FULL_LOOP result.

All challenger updates receive one fixed, chronological B10 feedback stream:
the target is positive and the displayed wrong candidate is an explicit
negative.  The current row is scored before that event is applied, and all
rows sharing a frame are scored before any update from that frame.  Candidate
missingness is excluded from feedback rather than converted to DELETE/NONE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from n27_apcr_model import (  # noqa: E402
    APCRConfig,
    APCRS,
    counterfactual_tensors,
    feature_tensors,
)
from sam3_intermot.adaptation.correction_compiler import (  # noqa: E402
    CorrectionTransaction,
    compile_identity_correction,
)
from sam3_intermot.adaptation.live_identity_lora import (  # noqa: E402
    LiveIdentityLoRA,
    LiveLoRAConfig,
)
from sam3_intermot.adaptation.live_optimizer import (  # noqa: E402
    AtomicChallengerUpdate,
    LiveUpdateEngine,
    SupportExample,
    UpdateConfig,
)
from sam3_intermot.adaptation.update_validator import (  # noqa: E402
    UpdateValidator,
    UpdateValidatorConfig,
    exact_zero,
)
from sam3_intermot.association.relational_challenger import (  # noqa: E402
    CrlsRelationalChallenger,
    LciaRelationalChallenger,
    build_assignment_matrix,
    build_cached_relation_features,
)


NONE_INDEX = 5
FEATURE_DIM = 7
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 28
RESPONSE_HORIZON = 5

ROLE_PATHS = {
    "external_train": ROOT / "outputs/n27/data/external_train_b10_round0.npz",
    "dancetrack_train_real_p2": ROOT / "outputs/n27/data/dance_train_real_b10_round0.npz",
}
APCR_CHECKPOINT = ROOT / "outputs/n27/checkpoints/apcr_s_p2_best.pt"

REQUIRED_FIELDS = (
    "candidate_mask",
    "target",
    "target_present",
    "b10_score",
    "root_similarity",
    "positive_similarity",
    "negative_similarity",
    "hard_similarity",
    "positive_count",
    "negative_count",
    "hard_count",
    "positive_age",
    "negative_age",
    "hard_age",
    "has_positive",
    "has_negative",
    "has_hard",
    "candidate_count",
    "detector_score",
    "selected",
    "correction_event",
    "pair_valid",
    "rejected_index",
    "cf_b10_score",
    "cf_positive_similarity",
    "cf_negative_similarity",
    "cf_has_positive",
    "cf_has_negative",
    "cf_positive_count",
    "cf_negative_count",
    "cf_positive_age",
    "cf_negative_age",
    "sequence",
    "frame",
    "identity",
    "event_hash",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        missing = [field for field in REQUIRED_FIELDS if field not in payload.files]
        if missing:
            raise RuntimeError(f"{path} is missing required N28-B fields: {missing}")
        arrays = {field: payload[field].copy() for field in payload.files}
    if len(arrays["target"]) == 0:
        raise RuntimeError(f"empty cached episode: {path}")
    return arrays


def validate_cache(role: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = len(arrays["target"])
    for field, value in arrays.items():
        if len(value) != rows:
            raise RuntimeError(f"{role}: field {field} has {len(value)} rows, expected {rows}")
    mask = arrays["candidate_mask"].astype(bool)
    target = arrays["target"].astype(np.int64)
    present = arrays["target_present"].astype(bool)
    expected_present = (target >= 0) & (target < NONE_INDEX)
    if not np.array_equal(present, expected_present):
        raise RuntimeError(f"{role}: target_present is not the frozen target<5 semantics")
    target_in_mask = np.zeros(rows, dtype=bool)
    valid_target = expected_present
    target_in_mask[valid_target] = mask[np.flatnonzero(valid_target), target[valid_target]]
    if not np.all(target_in_mask[valid_target]):
        raise RuntimeError(f"{role}: a target_present row has no target candidate in candidate_mask")
    sequence = arrays["sequence"].astype(np.int64)
    frame = arrays["frame"].astype(np.int64)
    # The external cache is materialised in identity-episode order rather
    # than global video-frame order.  Causal validity is therefore checked in
    # each immutable (sequence, identity) stream, which is also the scope of
    # every live state in this experiment.
    identity_values = arrays["identity"].astype(np.int64)
    order_ok = True
    for key in sorted(set(zip(sequence.tolist(), identity_values.tolist()))):
        indices = np.flatnonzero((sequence == key[0]) & (identity_values == key[1]))
        if len(indices) > 1 and not np.all(frame[indices[1:]] >= frame[indices[:-1]]):
            order_ok = False
            break
    if not order_ok:
        raise RuntimeError(f"{role}: cached rows are not chronological by sequence/frame")
    return {
        "rows": rows,
        "sequences": int(len(np.unique(sequence))),
        "identities": int(len(set(zip(sequence.tolist(), arrays["identity"].astype(np.int64).tolist())))),
        "candidate_present": int(present.sum()),
        "candidate_absent": int((~present).sum()),
        "target_in_candidate_mask": int(target_in_mask[valid_target].sum()),
        "chronological_order": True,
        "val25_read": False,
    }


def choose_one(scores: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Use the N28 matrix helper for one cached target parent."""
    values = np.asarray(scores, dtype=np.float32).reshape(1, -1)
    candidate_mask = np.asarray(mask, dtype=bool).reshape(1, -1)
    matrix = build_assignment_matrix(
        values,
        none_scores=np.asarray([0.0], dtype=np.float32),
        candidate_mask=candidate_mask,
    )
    column = int(np.argmax(matrix[0]))
    selected = column if column < values.shape[1] else NONE_INDEX
    return matrix[0, : values.shape[1]].astype(np.float32), selected


def select_rows(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(len(scores), NONE_INDEX, dtype=np.int8)
    for index in range(len(scores)):
        _, result[index] = choose_one(scores[index], mask[index])
    return result


def load_apcr_scores(
    arrays: dict[str, np.ndarray],
    checkpoint_path: Path,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the frozen N27 APCR-S checkpoint to current/cf cached views."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = APCRS(APCRConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    current_chunks: list[np.ndarray] = []
    cf_chunks: list[np.ndarray] = []
    tensor_fields = (
        "candidate_mask",
        "b10_score",
        "positive_similarity",
        "negative_similarity",
        "hard_similarity",
        "detector_score",
        "candidate_count",
        "has_positive",
        "has_negative",
        "has_hard",
        "positive_count",
        "negative_count",
        "hard_count",
        "positive_age",
        "negative_age",
        "hard_age",
    )
    with torch.no_grad():
        for start in range(0, len(arrays["target"]), batch_size):
            stop = min(len(arrays["target"]), start + batch_size)
            batch = {field: torch.from_numpy(arrays[field][start:stop]) for field in tensor_fields}
            current = model(feature_tensors(batch))["scores"].cpu().numpy().astype(np.float32)
            counterfactual_batch = dict(batch)
            for field in (
                "b10_score",
                "positive_similarity",
                "negative_similarity",
                "has_positive",
                "has_negative",
                "positive_count",
                "negative_count",
                "positive_age",
                "negative_age",
            ):
                counterfactual_batch[f"cf_{field}"] = torch.from_numpy(
                    arrays[f"cf_{field}"][start:stop]
                )
            counterfactual = model(counterfactual_tensors(counterfactual_batch))["scores"].cpu().numpy().astype(np.float32)
            current_chunks.append(current)
            cf_chunks.append(counterfactual)
    return (
        np.concatenate(current_chunks, axis=0),
        np.concatenate(cf_chunks, axis=0),
        {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint_path),
            "model_config": checkpoint["model_config"],
            "residual_off": False,
            "val25_read": False,
        },
    )


@dataclass
class StateRecord:
    event_id: int
    row_index: int
    frame: int
    pre: Any
    post: Any


def identity_key(arrays: dict[str, np.ndarray], index: int) -> tuple[int, int]:
    return int(arrays["sequence"][index]), int(arrays["identity"][index])


def rls_delta_from_snapshot(
    snapshot: tuple[np.ndarray, np.ndarray], relation: np.ndarray, residual_scale: float
) -> np.ndarray:
    return (np.asarray(relation, dtype=np.float64) @ snapshot[0] * residual_scale).astype(np.float32)


def lcia_delta_with_counterfactual(
    challenger: LciaRelationalChallenger,
    relation: np.ndarray,
    identity_id: Any,
    record: Optional[StateRecord],
) -> tuple[np.ndarray, np.ndarray]:
    current = challenger.delta(relation, identity_id).astype(np.float32)
    if record is None:
        return current, current.copy()
    challenger.restore({identity_id: record.pre})
    counterfactual = challenger.delta(relation, identity_id).astype(np.float32)
    challenger.restore({identity_id: record.post})
    return current, counterfactual


def lcia_segment_deltas_with_counterfactual(
    challenger: LciaRelationalChallenger,
    relation: np.ndarray,
    identity_id: Any,
    record: Optional[StateRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized LCIA inference while one identity state is fixed."""
    tensor = torch.as_tensor(relation, dtype=torch.float32)
    with torch.no_grad():
        current = challenger.model.delta_tensor(tensor, identity_id).cpu().numpy().astype(np.float32)
        if record is None:
            return current, current.copy()
        challenger.restore({identity_id: record.pre})
        counterfactual = challenger.model.delta_tensor(tensor, identity_id).cpu().numpy().astype(np.float32)
        challenger.restore({identity_id: record.post})
    return current, counterfactual


def rls_delta_with_counterfactual(
    challenger: CrlsRelationalChallenger,
    relation: np.ndarray,
    identity_id: Any,
    record: Optional[StateRecord],
) -> tuple[np.ndarray, np.ndarray]:
    current = challenger.delta(relation, identity_id).astype(np.float32)
    if record is None:
        return current, current.copy()
    counterfactual = rls_delta_from_snapshot(record.pre, relation, challenger.rls.config.residual_scale)
    return current, counterfactual


def rls_segment_deltas_with_counterfactual(
    challenger: CrlsRelationalChallenger,
    relation: np.ndarray,
    identity_id: Any,
    record: Optional[StateRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized C-RLS inference while one identity state is fixed."""
    state = challenger.rls.ensure_identity(identity_id)
    current = (np.asarray(relation, dtype=np.float64) @ state.weight * challenger.rls.config.residual_scale).astype(np.float32)
    if record is None:
        return current, current.copy()
    counterfactual = rls_delta_from_snapshot(
        record.pre, relation, challenger.rls.config.residual_scale
    )
    return current, counterfactual


def build_transaction(
    identity_id: Any,
    target: int,
    displayed: int,
    frame: int,
) -> CorrectionTransaction:
    if displayed == NONE_INDEX:
        return compile_identity_correction(
            identity_id=identity_id,
            positive_candidate=target,
            rejected_candidate=None,
            frame=frame,
            provenance="HUMAN_ADD",
        )
    return compile_identity_correction(
        identity_id=identity_id,
        positive_candidate=target,
        rejected_candidate=displayed,
        frame=frame,
        provenance="HUMAN_REASSIGN",
    )


def apply_control_update(
    challenger: CrlsRelationalChallenger,
    identity_id: Any,
    relation: np.ndarray,
    target: int,
    negative: Optional[int],
) -> None:
    rows = [relation[int(target)]]
    labels = [1.0]
    if negative is not None:
        rows.append(relation[int(negative)])
        labels.append(-1.0)
    challenger.rls.update(identity_id, np.stack(rows), np.asarray(labels, dtype=np.float64))


def ordinary_hard_negative(
    scores: np.ndarray, mask: np.ndarray, target: int, displayed: int
) -> Optional[int]:
    valid = np.flatnonzero(mask)
    candidates = [int(index) for index in valid if int(index) != int(target) and int(index) != int(displayed)]
    if not candidates:
        return None
    return max(candidates, key=lambda index: float(scores[index]))


def choose_random_negative(mask: np.ndarray, target: int, rng: np.random.Generator) -> Optional[int]:
    candidates = np.flatnonzero(mask).astype(int)
    candidates = candidates[candidates != int(target)]
    if not len(candidates):
        return None
    return int(rng.choice(candidates))


def build_frame_groups(
    arrays: dict[str, np.ndarray],
    identity_rows: dict[tuple[int, int], list[int]],
) -> list[np.ndarray]:
    frame = arrays["frame"].astype(np.int64)
    groups: list[np.ndarray] = []
    for identity in sorted(identity_rows):
        indices = sorted(identity_rows[identity], key=lambda index: (int(frame[index]), index))
        start = 0
        for index in range(1, len(indices) + 1):
            if index == len(indices) or frame[indices[index]] != frame[indices[start]]:
                groups.append(np.asarray(indices[start:index], dtype=np.int64))
                start = index
    return groups


def build_identity_rows(arrays: dict[str, np.ndarray]) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for index in range(len(arrays["target"])):
        result.setdefault(identity_key(arrays, index), []).append(index)
    frame = arrays["frame"].astype(np.int64)
    for identity, indices in result.items():
        result[identity] = sorted(indices, key=lambda index: (int(frame[index]), index))
    return result


def finite_snapshot(snapshot: Any) -> bool:
    if isinstance(snapshot, tuple):
        return all(np.isfinite(np.asarray(item, dtype=np.float64)).all() for item in snapshot)
    return all(torch.isfinite(item).all().item() for item in snapshot)


def run_role(
    role: str,
    arrays: dict[str, np.ndarray],
    apcr_current: np.ndarray,
    apcr_cf: np.ndarray,
    *,
    lcia_steps: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    audit = validate_cache(role, arrays)
    rows = audit["rows"]
    mask = arrays["candidate_mask"].astype(bool)
    target = arrays["target"].astype(np.int64)
    sequence = arrays["sequence"].astype(np.int64)
    frame = arrays["frame"].astype(np.int64)
    b10_anchor = arrays["b10_score"].astype(np.float32)
    b10_cf_anchor = arrays["cf_b10_score"].astype(np.float32)
    relations = build_cached_relation_features(
        b10_score=b10_anchor,
        root_similarity=arrays["root_similarity"],
        positive_similarity=arrays["positive_similarity"],
        negative_similarity=arrays["negative_similarity"],
        hard_similarity=arrays["hard_similarity"],
        detector_score=arrays["detector_score"],
        candidate_mask=mask,
    ).astype(np.float32)

    b10_selected = select_rows(b10_anchor, mask)
    frozen_selected = arrays["selected"].astype(np.int64)
    present = arrays["target_present"].astype(bool)
    eligible = present & (target >= 0) & (target < NONE_INDEX)
    feedback_mask = eligible & (b10_selected.astype(np.int64) != target)
    feedback_rows = np.flatnonzero(feedback_mask).astype(int)
    feedback_id_by_row = {int(row): event_id for event_id, row in enumerate(feedback_rows)}
    identity_rows = build_identity_rows(arrays)
    for identity, indices in identity_rows.items():
        identity_frames = [int(frame[index]) for index in indices]
        if len(identity_frames) != len(set(identity_frames)):
            raise RuntimeError(f"{role}: duplicate cached frame for identity {identity}")

    # Four-cell arrays are kept for exact post-hoc audit.  Invalid candidates
    # remain in storage, but every selection/margin helper masks them.
    method_names = (
        "B10",
        "APCR-S",
        "C-RLS",
        "C-RLS_no_update",
        "LCIA_random_live",
        "LCIA_update_disabled",
        "C-RLS_random_negative_control",
        "C-RLS_ordinary_hard_negative_control",
    )
    current_scores: dict[str, np.ndarray] = {
        name: np.full((rows, NONE_INDEX), -1.0e9, dtype=np.float32) for name in method_names
    }
    cf_scores: dict[str, np.ndarray] = {
        name: np.full((rows, NONE_INDEX), -1.0e9, dtype=np.float32) for name in method_names
    }
    selected: dict[str, np.ndarray] = {
        name: np.full(rows, NONE_INDEX, dtype=np.int8) for name in method_names
    }
    cf_selected: dict[str, np.ndarray] = {
        name: np.full(rows, NONE_INDEX, dtype=np.int8) for name in method_names
    }
    current_scores["B10"] = b10_anchor.copy()
    cf_scores["B10"] = b10_cf_anchor.copy()
    selected["B10"] = b10_selected.copy()
    cf_selected["B10"] = select_rows(b10_cf_anchor, mask)
    current_scores["APCR-S"] = apcr_current.copy()
    cf_scores["APCR-S"] = apcr_cf.copy()
    selected["APCR-S"] = select_rows(apcr_current, mask)
    cf_selected["APCR-S"] = select_rows(apcr_cf, mask)
    for name in ("C-RLS_no_update", "LCIA_update_disabled"):
        current_scores[name] = b10_anchor.copy()
        cf_scores[name] = b10_anchor.copy()
        selected[name] = b10_selected.copy()
        cf_selected[name] = b10_selected.copy()

    crls = CrlsRelationalChallenger(FEATURE_DIM, residual_scale=2.0, ridge=1.0)
    crls_random = CrlsRelationalChallenger(FEATURE_DIM, residual_scale=2.0, ridge=1.0)
    crls_hard = CrlsRelationalChallenger(FEATURE_DIM, residual_scale=2.0, ridge=1.0)
    lcia = LciaRelationalChallenger(
        LiveIdentityLoRA(
            LiveLoRAConfig(
                input_dim=FEATURE_DIM,
                d_model=64,
                rank=8,
                blocks=2,
                alpha=8.0,
                seed=28,
            )
        )
    )
    lcia_engine = LiveUpdateEngine(
        lcia,
        config=UpdateConfig(
            learning_rate=0.05,
            max_steps=lcia_steps,
            support_margin=0.01,
        ),
        validator=UpdateValidator(
            UpdateValidatorConfig(
                support_margin=0.01,
                protected_drop=0.0,
                max_update_norm=1000.0,
            )
        ),
    )
    lcia_history: dict[Any, StateRecord] = {}
    crls_history: dict[Any, StateRecord] = {}
    random_history: dict[Any, StateRecord] = {}
    hard_history: dict[Any, StateRecord] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED + (1 if role == "external_train" else 2))

    update_stats: dict[str, dict[str, int]] = {
        name: {"attempts": 0, "accepted": 0, "rejected": 0, "negative_updates": 0, "positive_only_updates": 0}
        for name in (
            "C-RLS",
            "LCIA_random_live",
            "C-RLS_random_negative_control",
            "C-RLS_ordinary_hard_negative_control",
        )
    }
    update_event_acceptance: dict[str, dict[int, bool]] = {name: {} for name in update_stats}
    event_transactions: list[dict[str, Any]] = []
    no_prior_feedback = {name: np.ones(rows, dtype=bool) for name in update_stats}

    # The external cache is grouped by identity episode.  Since candidate
    # columns are local to a parent, identities are independent in this
    # cached replay; vectorizing each fixed-state segment preserves the
    # chronological update rule without pretending that local column 0 is a
    # global detection shared by another parent.
    for identity, identity_indices in sorted(identity_rows.items()):
        indices = list(identity_indices)
        feedback_positions = [
            position for position, row in enumerate(indices) if feedback_mask[row]
        ]
        cursor = 0
        feedback_cursor = 0
        while cursor < len(indices):
            next_feedback_position = (
                feedback_positions[feedback_cursor]
                if feedback_cursor < len(feedback_positions)
                else None
            )
            end = next_feedback_position if next_feedback_position is not None else len(indices)
            segment = np.asarray(
                indices[cursor : end + 1] if next_feedback_position is not None else indices[cursor:],
                dtype=np.int64,
            )
            crls_record = crls_history.get(identity)
            crls_delta, crls_cf_delta = rls_segment_deltas_with_counterfactual(
                crls, relations[segment], identity, crls_record
            )
            current_scores["C-RLS"][segment] = b10_anchor[segment] + crls_delta
            cf_scores["C-RLS"][segment] = b10_anchor[segment] + crls_cf_delta
            selected["C-RLS"][segment] = select_rows(current_scores["C-RLS"][segment], mask[segment])
            cf_selected["C-RLS"][segment] = select_rows(cf_scores["C-RLS"][segment], mask[segment])

            lcia_record = lcia_history.get(identity)
            lcia_delta, lcia_cf_delta = lcia_segment_deltas_with_counterfactual(
                lcia, relations[segment], identity, lcia_record
            )
            current_scores["LCIA_random_live"][segment] = b10_anchor[segment] + lcia_delta
            cf_scores["LCIA_random_live"][segment] = b10_anchor[segment] + lcia_cf_delta
            selected["LCIA_random_live"][segment] = select_rows(
                current_scores["LCIA_random_live"][segment], mask[segment]
            )
            cf_selected["LCIA_random_live"][segment] = select_rows(
                cf_scores["LCIA_random_live"][segment], mask[segment]
            )

            random_record = random_history.get(identity)
            random_delta, random_cf_delta = rls_segment_deltas_with_counterfactual(
                crls_random, relations[segment], identity, random_record
            )
            current_scores["C-RLS_random_negative_control"][segment] = b10_anchor[segment] + random_delta
            cf_scores["C-RLS_random_negative_control"][segment] = b10_anchor[segment] + random_cf_delta
            selected["C-RLS_random_negative_control"][segment] = select_rows(
                current_scores["C-RLS_random_negative_control"][segment], mask[segment]
            )
            cf_selected["C-RLS_random_negative_control"][segment] = select_rows(
                cf_scores["C-RLS_random_negative_control"][segment], mask[segment]
            )

            hard_record = hard_history.get(identity)
            hard_delta, hard_cf_delta = rls_segment_deltas_with_counterfactual(
                crls_hard, relations[segment], identity, hard_record
            )
            current_scores["C-RLS_ordinary_hard_negative_control"][segment] = b10_anchor[segment] + hard_delta
            cf_scores["C-RLS_ordinary_hard_negative_control"][segment] = b10_anchor[segment] + hard_cf_delta
            selected["C-RLS_ordinary_hard_negative_control"][segment] = select_rows(
                current_scores["C-RLS_ordinary_hard_negative_control"][segment], mask[segment]
            )
            cf_selected["C-RLS_ordinary_hard_negative_control"][segment] = select_rows(
                cf_scores["C-RLS_ordinary_hard_negative_control"][segment], mask[segment]
            )

            for method in update_stats:
                no_prior_feedback[method][segment] = crls_record is None

            if next_feedback_position is None:
                break
            row = int(indices[next_feedback_position])
            event_id = int(feedback_id_by_row[row])
            target_index = int(target[row])
            displayed = int(b10_selected[row])
            relation = relations[row]
            transaction = build_transaction(identity, target_index, displayed, int(frame[row]))
            event_transactions.append(
                {
                    "event_id": event_id,
                    "row": row,
                    "sequence": int(sequence[row]),
                    "identity": int(arrays["identity"][row]),
                    "frame": int(frame[row]),
                    "target": target_index,
                    "b10_displayed": displayed,
                    "transaction_id": transaction.transaction_id,
                    "provenance": [constraint.provenance for constraint in transaction.constraints],
                    "constraint_roles": [constraint.role for constraint in transaction.constraints],
                }
            )

            crls_pre = crls.snapshot([identity])[identity]
            crls.update_transaction(transaction, {identity: relation})
            crls_post = crls.snapshot([identity])[identity]
            crls_history[identity] = StateRecord(event_id, row, int(frame[row]), crls_pre, crls_post)
            update_stats["C-RLS"]["attempts"] += 1
            update_stats["C-RLS"]["accepted"] += 1
            if displayed < NONE_INDEX:
                update_stats["C-RLS"]["negative_updates"] += 1
            else:
                update_stats["C-RLS"]["positive_only_updates"] += 1
            update_event_acceptance["C-RLS"][event_id] = True

            example = SupportExample(
                identity_id=identity,
                anchor_scores=b10_anchor[row].copy(),
                relation_features=relation.copy(),
                candidate_mask=mask[row].copy(),
                none_score=0.0,
            )
            lcia_pre = lcia.snapshot([identity])[identity]
            update_stats["LCIA_random_live"]["attempts"] += 1
            with AtomicChallengerUpdate(lcia, None, [identity]) as atomic:
                lcia_update = lcia_engine.apply(transaction, {identity: example})
                if lcia_update.accepted:
                    atomic.commit()
            if lcia_update.accepted:
                lcia_post = lcia.snapshot([identity])[identity]
                lcia_history[identity] = StateRecord(event_id, row, int(frame[row]), lcia_pre, lcia_post)
                update_stats["LCIA_random_live"]["accepted"] += 1
                update_event_acceptance["LCIA_random_live"][event_id] = True
            else:
                update_stats["LCIA_random_live"]["rejected"] += 1
                update_event_acceptance["LCIA_random_live"][event_id] = False

            random_neg = choose_random_negative(mask[row], target_index, rng)
            random_pre = crls_random.snapshot([identity])[identity]
            apply_control_update(crls_random, identity, relation, target_index, random_neg)
            random_post = crls_random.snapshot([identity])[identity]
            random_history[identity] = StateRecord(event_id, row, int(frame[row]), random_pre, random_post)
            update_stats["C-RLS_random_negative_control"]["attempts"] += 1
            update_stats["C-RLS_random_negative_control"]["accepted"] += 1
            update_event_acceptance["C-RLS_random_negative_control"][event_id] = True
            if random_neg is None:
                update_stats["C-RLS_random_negative_control"]["positive_only_updates"] += 1
            else:
                update_stats["C-RLS_random_negative_control"]["negative_updates"] += 1

            hard_neg = ordinary_hard_negative(b10_anchor[row], mask[row], target_index, displayed)
            hard_pre = crls_hard.snapshot([identity])[identity]
            apply_control_update(crls_hard, identity, relation, target_index, hard_neg)
            hard_post = crls_hard.snapshot([identity])[identity]
            hard_history[identity] = StateRecord(event_id, row, int(frame[row]), hard_pre, hard_post)
            update_stats["C-RLS_ordinary_hard_negative_control"]["attempts"] += 1
            update_stats["C-RLS_ordinary_hard_negative_control"]["accepted"] += 1
            update_event_acceptance["C-RLS_ordinary_hard_negative_control"][event_id] = True
            if hard_neg is None:
                update_stats["C-RLS_ordinary_hard_negative_control"]["positive_only_updates"] += 1
            else:
                update_stats["C-RLS_ordinary_hard_negative_control"]["negative_updates"] += 1
            cursor = next_feedback_position + 1
            feedback_cursor += 1

    # Response records use the same fixed event stream for every method.  The
    # B10 pair is the frozen latest-correction-removed view; adaptive cf is the
    # state immediately before the latest accepted identity update.
    response_records: list[dict[str, Any]] = []
    for event in event_transactions:
        row = int(event["row"])
        identity = (int(event["sequence"]), int(event["identity"]))
        rows_for_identity = identity_rows[identity]
        position = rows_for_identity.index(row)
        future: list[int] = []
        for candidate_row in rows_for_identity[position + 1 :]:
            if int(frame[candidate_row]) <= int(frame[row]):
                continue
            future.append(candidate_row)
            if candidate_row in feedback_id_by_row or len(future) >= RESPONSE_HORIZON:
                break
        if not future:
            continue
        event_record = dict(event)
        event_record["future_rows"] = future
        event_record["method_update_accepted"] = {
            name: bool(update_event_acceptance[name].get(int(event["event_id"]), False))
            for name in update_event_acceptance
        }
        response_records.append(event_record)

    def exact_no_update(method: str) -> bool:
        return bool(
            exact_zero(current_scores[method], b10_anchor)
            and exact_zero(cf_scores[method], b10_anchor)
            and np.array_equal(selected[method], b10_selected)
            and np.array_equal(cf_selected[method], b10_selected)
        )

    present_top1: dict[str, Optional[float]] = {}
    correction_counts: dict[str, int] = {}
    for method in method_names:
        correct = selected[method].astype(np.int64) == target
        present_top1[method] = float(correct[eligible].mean()) if eligible.any() else None
        correction_counts[method] = int((eligible & ~correct).sum())

    unaffected_identity_keys = {
        identity
        for identity, identity_indices in identity_rows.items()
        if not any(feedback_mask[index] for index in identity_indices)
    }
    unaffected_rows = np.asarray(
        [index for identity in unaffected_identity_keys for index in identity_rows[identity]],
        dtype=np.int64,
    )
    unaffected_regression: dict[str, Any] = {}
    for method in method_names:
        if not len(unaffected_rows):
            unaffected_regression[method] = None
            continue
        method_errors = (selected[method][unaffected_rows].astype(np.int64) != target[unaffected_rows]).astype(float)
        b10_errors = (b10_selected[unaffected_rows].astype(np.int64) != target[unaffected_rows]).astype(float)
        difference = method_errors - b10_errors
        unaffected_regression[method] = {
            "identities": int(len(unaffected_identity_keys)),
            "rows": int(len(unaffected_rows)),
            "error_difference": float(difference.mean()),
            "error_regression_pp": float(max(0.0, difference.mean()) * 100.0),
            "exact_scores_vs_b10": bool(exact_zero(current_scores[method][unaffected_rows], b10_anchor[unaffected_rows])),
        }

    arrays_out: dict[str, np.ndarray] = {
        "target": target.astype(np.int8),
        "target_present": present,
        "candidate_mask": mask,
        "sequence": sequence.astype(np.int16),
        "identity": arrays["identity"].astype(np.int32),
        "frame": frame.astype(np.int32),
        "feedback_event": feedback_mask,
        "b10_current": b10_anchor,
        "b10_cf": b10_cf_anchor,
    }
    for method in method_names:
        safe = method.lower().replace("-", "_")
        arrays_out[f"current_{safe}"] = current_scores[method].astype(np.float32)
        arrays_out[f"cf_{safe}"] = cf_scores[method].astype(np.float32)
        arrays_out[f"selected_{safe}"] = selected[method].astype(np.int8)
        arrays_out[f"cf_selected_{safe}"] = cf_selected[method].astype(np.int8)

    role_result = {
        "role": role,
        "cache_role": role,
        "cache": str(ROLE_PATHS[role].relative_to(ROOT)),
        "cache_sha256": sha256(ROLE_PATHS[role]),
        "audit": audit,
        "methods": {
            method: {
                "candidate_present_top1": present_top1[method],
                "correction_events_on_present_targets": correction_counts[method],
                "current_cf_four_cell_available": True,
                "unaffected_identity": unaffected_regression[method],
                "val25_read": False,
            }
            for method in method_names
        },
        "frozen_observer": {
            "eligible_feedback_events": int(len(feedback_rows)),
            "candidate_absent_rows_excluded_from_feedback": int((~present).sum()),
            "b10_cache_selected_mismatch_count_under_n28_none_matrix": int(
                (b10_selected.astype(np.int64) != frozen_selected).sum()
            ),
            "b10_present_selected_mismatch_count": int(
                ((b10_selected.astype(np.int64) != frozen_selected) & eligible).sum()
            ),
            "same_frame_updates_deferred": True,
            "current_prediction_rerun_after_feedback": False,
        },
        "update_stats": update_stats,
        "transactions": event_transactions,
        "response_event_count": len(response_records),
        "response_events_with_future_rows": len(response_records),
        "unaffected_identity_count": len(unaffected_identity_keys),
        "no_update_exact_zero": {
            "C-RLS_no_update": exact_no_update("C-RLS_no_update"),
            "LCIA_update_disabled": exact_no_update("LCIA_update_disabled"),
            "LCIA_random_before_first_feedback": bool(
                exact_zero(
                    current_scores["LCIA_random_live"][no_prior_feedback["LCIA_random_live"]],
                    b10_anchor[no_prior_feedback["LCIA_random_live"]],
                )
                if no_prior_feedback["LCIA_random_live"].any()
                else True
            ),
        },
        "lcia": {
            "config": {
                "d_model": 64,
                "rank": 8,
                "blocks": 2,
                "alpha": 8.0,
                "seed": 28,
                "learning_rate": 0.05,
                "max_steps": lcia_steps,
                "support_margin": 0.01,
                "max_update_norm": 1000.0,
            },
            "live_parameters_per_identity": lcia.model.live_parameter_count(
                next(iter(identity_rows), (0, 0))
            ),
            "base_parameters_require_grad": any(
                parameter.requires_grad
                for name, parameter in lcia.model.named_parameters()
                if not name.startswith("fast_states.")
            ),
            "live_state_norm": lcia.state_norm(list(lcia_history)),
            "accepted_state_count": len(lcia_history),
            "rejected_update_count": update_stats["LCIA_random_live"]["rejected"],
        },
        "val25_read": False,
    }
    return role_result, response_records, arrays_out


def masked_margin_loss(scores: np.ndarray, mask: np.ndarray, target: int) -> float:
    if target < 0 or target >= NONE_INDEX or not bool(mask[target]):
        return float("nan")
    valid_scores = np.asarray(scores, dtype=np.float64).copy()
    valid_scores[~mask] = -1.0e9
    target_score = float(valid_scores[target])
    alternatives = np.concatenate([np.delete(valid_scores, target), np.asarray([0.0])])
    return float(max(0.0, float(np.max(alternatives)) - target_score))


def future_frames_to_error(
    rows: list[int],
    scores_selected: np.ndarray,
    target: np.ndarray,
    present: np.ndarray,
    frame: np.ndarray,
    event_frame: int,
) -> float:
    if not rows:
        return float("nan")
    for row in rows:
        if bool(present[row]) and int(scores_selected[row]) != int(target[row]):
            return float(int(frame[row]) - event_frame)
    return float(int(frame[rows[-1]]) - event_frame)


def response_metric_values(
    method: str,
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    replay_arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    future = [int(row) for row in record["future_rows"]]
    target = arrays["target"].astype(np.int64)
    present = arrays["target_present"].astype(bool)
    mask = arrays["candidate_mask"].astype(bool)
    frame = arrays["frame"].astype(np.int64)
    selected_key = method.lower().replace("-", "_")
    current_selected = replay_arrays[f"selected_{selected_key}"].astype(np.int64)
    cf_selected = replay_arrays[f"cf_selected_{selected_key}"].astype(np.int64)
    current_score = replay_arrays[f"current_{selected_key}"]
    cf_score = replay_arrays[f"cf_{selected_key}"]
    b10_selected = replay_arrays["selected_b10"].astype(np.int64)
    b10_cf_selected = replay_arrays["cf_selected_b10"].astype(np.int64)
    b10_current = replay_arrays["current_b10"]
    b10_cf = replay_arrays["cf_b10"]

    valid_future = [row for row in future if bool(present[row])]
    def mean_or_nan(values: Iterable[float]) -> float:
        values = [float(value) for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    current_error = mean_or_nan(float(current_selected[row] != target[row]) for row in valid_future)
    cf_error = mean_or_nan(float(cf_selected[row] != target[row]) for row in valid_future)
    current_recorrection = current_error
    cf_recorrection = cf_error
    current_margin = mean_or_nan(masked_margin_loss(current_score[row], mask[row], int(target[row])) for row in valid_future)
    cf_margin = mean_or_nan(masked_margin_loss(cf_score[row], mask[row], int(target[row])) for row in valid_future)
    b10_current_error = mean_or_nan(float(b10_selected[row] != target[row]) for row in valid_future)
    b10_cf_error = mean_or_nan(float(b10_cf_selected[row] != target[row]) for row in valid_future)
    b10_current_margin = mean_or_nan(masked_margin_loss(b10_current[row], mask[row], int(target[row])) for row in valid_future)
    b10_cf_margin = mean_or_nan(masked_margin_loss(b10_cf[row], mask[row], int(target[row])) for row in valid_future)

    rejected_rows = [
        row
        for row in valid_future
        if int(arrays["rejected_index"][row]) >= 0
        and int(arrays["rejected_index"][row]) < NONE_INDEX
        and int(arrays["rejected_index"][row]) != int(target[row])
    ]
    current_rejected = mean_or_nan(
        float(current_selected[row] == int(arrays["rejected_index"][row])) for row in rejected_rows
    )
    cf_rejected = mean_or_nan(
        float(cf_selected[row] == int(arrays["rejected_index"][row])) for row in rejected_rows
    )
    b10_current_rejected = mean_or_nan(
        float(b10_selected[row] == int(arrays["rejected_index"][row])) for row in rejected_rows
    )
    b10_cf_rejected = mean_or_nan(
        float(b10_cf_selected[row] == int(arrays["rejected_index"][row])) for row in rejected_rows
    )

    event_frame = int(record["frame"])
    current_time = future_frames_to_error(
        future, current_selected, target, present, frame, event_frame
    )
    cf_time = future_frames_to_error(
        future, cf_selected, target, present, frame, event_frame
    )
    b10_current_time = future_frames_to_error(
        future, b10_selected, target, present, frame, event_frame
    )
    b10_cf_time = future_frames_to_error(
        future, b10_cf_selected, target, present, frame, event_frame
    )

    return {
        "future_error_current": current_error,
        "future_error_cf": cf_error,
        "future_error_R_adapted": current_error - cf_error if np.isfinite(current_error) and np.isfinite(cf_error) else float("nan"),
        "future_error_R_B10": b10_current_error - b10_cf_error if np.isfinite(b10_current_error) and np.isfinite(b10_cf_error) else float("nan"),
        "future_error_delta_delta": (current_error - cf_error) - (b10_current_error - b10_cf_error) if all(np.isfinite(value) for value in (current_error, cf_error, b10_current_error, b10_cf_error)) else float("nan"),
        "future_recorrection_current": current_recorrection,
        "future_recorrection_cf": cf_recorrection,
        "future_recorrection_R_adapted": current_recorrection - cf_recorrection if np.isfinite(current_recorrection) and np.isfinite(cf_recorrection) else float("nan"),
        "future_recorrection_R_B10": b10_current_error - b10_cf_error if np.isfinite(b10_current_error) and np.isfinite(b10_cf_error) else float("nan"),
        "future_recorrection_delta_delta": (current_recorrection - cf_recorrection) - (b10_current_error - b10_cf_error) if all(np.isfinite(value) for value in (current_recorrection, cf_recorrection, b10_current_error, b10_cf_error)) else float("nan"),
        "target_margin_loss_current": current_margin,
        "target_margin_loss_cf": cf_margin,
        "target_margin_loss_R_adapted": current_margin - cf_margin if np.isfinite(current_margin) and np.isfinite(cf_margin) else float("nan"),
        "target_margin_loss_R_B10": b10_current_margin - b10_cf_margin if np.isfinite(b10_current_margin) and np.isfinite(b10_cf_margin) else float("nan"),
        "target_margin_loss_delta_delta": (current_margin - cf_margin) - (b10_current_margin - b10_cf_margin) if all(np.isfinite(value) for value in (current_margin, cf_margin, b10_current_margin, b10_cf_margin)) else float("nan"),
        "rejected_identity_selection_proxy_current": current_rejected,
        "rejected_identity_selection_proxy_cf": cf_rejected,
        "rejected_identity_selection_proxy_R_adapted": current_rejected - cf_rejected if np.isfinite(current_rejected) and np.isfinite(cf_rejected) else float("nan"),
        "rejected_identity_selection_proxy_R_B10": b10_current_rejected - b10_cf_rejected if np.isfinite(b10_current_rejected) and np.isfinite(b10_cf_rejected) else float("nan"),
        "rejected_identity_selection_proxy_delta_delta": (current_rejected - cf_rejected) - (b10_current_rejected - b10_cf_rejected) if all(np.isfinite(value) for value in (current_rejected, cf_rejected, b10_current_rejected, b10_cf_rejected)) else float("nan"),
        "frames_to_next_correction_current": current_time,
        "frames_to_next_correction_cf": cf_time,
        "frames_to_next_correction_R_adapted": current_time - cf_time if np.isfinite(current_time) and np.isfinite(cf_time) else float("nan"),
        "frames_to_next_correction_R_B10": b10_current_time - b10_cf_time if np.isfinite(b10_current_time) and np.isfinite(b10_cf_time) else float("nan"),
        "frames_to_next_correction_delta_delta": (current_time - cf_time) - (b10_current_time - b10_cf_time) if all(np.isfinite(value) for value in (current_time, cf_time, b10_current_time, b10_cf_time)) else float("nan"),
        "future_rows": float(len(future)),
        "rejected_proxy_rows": float(len(rejected_rows)),
    }


def bootstrap_summary(values: list[float], groups: list[str]) -> dict[str, Any]:
    finite = [(float(value), str(group)) for value, group in zip(values, groups) if np.isfinite(value)]
    if not finite:
        return {
            "status": "NOT_COMPUTABLE",
            "events": 0,
            "groups": 0,
            "point": None,
            "ci95": None,
            "majority_group_negative": False,
            "group_means": {},
        }
    unique = sorted({group for _, group in finite})
    grouped = {group: np.asarray([value for value, item in finite if item == group], dtype=np.float64) for group in unique}
    group_means = {group: float(chunk.mean()) for group, chunk in grouped.items()}
    if len(unique) < 2:
        ci = None
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
        chunks = [grouped[group] for group in unique]
        for index in range(BOOTSTRAP_REPS):
            sampled = rng.integers(0, len(chunks), size=len(chunks))
            numerator = sum(float(chunks[item].sum()) for item in sampled)
            denominator = sum(int(len(chunks[item])) for item in sampled)
            draws[index] = numerator / max(1, denominator)
        ci = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    return {
        "status": "OK" if len(unique) >= 2 else "UNDERPOWERED_ONE_GROUP",
        "events": len(finite),
        "groups": len(unique),
        "point": float(np.mean([value for value, _ in finite])),
        "ci95": ci,
        "majority_group_negative": bool(sum(value < 0.0 for value in group_means.values()) > len(group_means) / 2),
        "group_means": group_means,
    }


def aggregate_responses(
    response_by_role: dict[str, list[dict[str, Any]]],
    replay_by_role: dict[str, dict[str, np.ndarray]],
    cache_by_role: dict[str, dict[str, np.ndarray]],
    methods: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    method_summaries: dict[str, Any] = {}
    for role, records in response_by_role.items():
        arrays = cache_by_role[role]
        replay = replay_by_role[role]
        for record in records:
            for method in methods:
                metrics = response_metric_values(method, record, arrays, replay)
                enriched = {
                    "role": role,
                    "sequence_group": f"{role}:{int(record['sequence'])}",
                    "method": method,
                    "event_id": int(record["event_id"]),
                    "row": int(record["row"]),
                    "identity": int(record["identity"]),
                    "frame": int(record["frame"]),
                    "update_accepted": bool(record["method_update_accepted"].get(method, True)),
                    **metrics,
                }
                all_rows.append(enriched)

    for method in methods:
        method_rows = [row for row in all_rows if row["method"] == method]
        accepted_rows = [row for row in method_rows if row["update_accepted"]]
        metrics_out: dict[str, Any] = {
            "response_events": len(method_rows),
            "accepted_response_events": len(accepted_rows),
            "statistics": {},
        }
        for metric in (
            "future_error_delta_delta",
            "future_recorrection_delta_delta",
            "target_margin_loss_delta_delta",
            "rejected_identity_selection_proxy_delta_delta",
            "frames_to_next_correction_delta_delta",
        ):
            source = accepted_rows if method == "LCIA_random_live" else method_rows
            values = [float(row[metric]) for row in source]
            groups = [str(row["sequence_group"]) for row in source]
            result = bootstrap_summary(values, groups)
            result["direction"] = "negative_preferred" if "frames_to" not in metric else "positive_preferred"
            metrics_out["statistics"][metric] = result
        method_summaries[method] = metrics_out
    return method_summaries, all_rows


def make_replay_arrays(
    arrays_out: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    # A small named view avoids coupling the response evaluator to its storage
    # names while retaining the full per-method NPZ audit artifact.
    view = dict(arrays_out)
    for method in (
        "B10",
        "APCR-S",
        "C-RLS",
        "C-RLS_no_update",
        "LCIA_random_live",
        "LCIA_update_disabled",
        "C-RLS_random_negative_control",
        "C-RLS_ordinary_hard_negative_control",
    ):
        safe = method.lower().replace("-", "_")
        view[f"current_{safe}"] = arrays_out[f"current_{safe}"]
        view[f"cf_{safe}"] = arrays_out[f"cf_{safe}"]
        view[f"selected_{safe}"] = arrays_out[f"selected_{safe}"]
        view[f"cf_selected_{safe}"] = arrays_out[f"cf_selected_{safe}"]
    return view


def run_n28b(
    *,
    output: Path = ROOT / "outputs/n28/n28b_result.json",
    artifact_dir: Path = ROOT / "outputs/n28/data",
    lcia_steps: int = 5,
) -> dict[str, Any]:
    started = time.monotonic()
    # Cached parents are small (five candidates).  A single CPU BLAS thread
    # avoids oversubscription overhead and keeps the frozen CPU feasibility
    # run deterministic; no GPU is requested by N28-B.
    torch.set_num_threads(1)
    cache_by_role = {role: load_cache(path) for role, path in ROLE_PATHS.items()}
    apcr_by_role: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for role, arrays in cache_by_role.items():
        apcr_by_role[role] = load_apcr_scores(arrays, APCR_CHECKPOINT)

    role_results: dict[str, Any] = {}
    response_by_role: dict[str, list[dict[str, Any]]] = {}
    replay_by_role: dict[str, dict[str, np.ndarray]] = {}
    for role, arrays in cache_by_role.items():
        apcr_current, apcr_cf, _ = apcr_by_role[role]
        result, responses, arrays_out = run_role(
            role,
            arrays,
            apcr_current,
            apcr_cf,
            lcia_steps=lcia_steps,
        )
        role_results[role] = result
        response_by_role[role] = responses
        atomic_npz(artifact_dir / f"n28b_{role}.npz", arrays_out)
        replay_by_role[role] = make_replay_arrays(arrays_out)

    methods = (
        "B10",
        "APCR-S",
        "C-RLS",
        "C-RLS_no_update",
        "LCIA_random_live",
        "LCIA_update_disabled",
        "C-RLS_random_negative_control",
        "C-RLS_ordinary_hard_negative_control",
    )
    response_summaries, response_rows = aggregate_responses(
        response_by_role,
        replay_by_role,
        cache_by_role,
        methods,
    )
    response_csv_rows = []
    for row in response_rows:
        response_csv_rows.append(
            {
                key: ("" if not np.isfinite(value) else value) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    atomic_csv(ROOT / "outputs/n28/n28b_response_events.csv", response_csv_rows)

    lcia_summary = response_summaries["LCIA_random_live"]
    future_error = lcia_summary["statistics"]["future_error_delta_delta"]
    future_recorrection = lcia_summary["statistics"]["future_recorrection_delta_delta"]
    lcia_unaffected = [
        role_results[role]["methods"]["LCIA_random_live"]["unaffected_identity"]
        for role in role_results
        if role_results[role]["methods"]["LCIA_random_live"]["unaffected_identity"] is not None
    ]
    max_unaffected_regression_pp = max(
        (float(item["error_regression_pp"]) for item in lcia_unaffected),
        default=0.0,
    )
    no_update_exact = all(
        role_results[role]["no_update_exact_zero"]["LCIA_update_disabled"]
        and role_results[role]["no_update_exact_zero"]["C-RLS_no_update"]
        for role in role_results
    )
    criteria = {
        "future_error_delta_delta_point_negative": bool(
            future_error["point"] is not None and future_error["point"] < 0.0
        ),
        "future_error_delta_delta_ci95_upper_negative": bool(
            future_error["ci95"] is not None and future_error["ci95"][1] < 0.0
        ),
        "future_recorrection_delta_delta_ci95_upper_negative": bool(
            future_recorrection["ci95"] is not None and future_recorrection["ci95"][1] < 0.0
        ),
        "majority_response_sequence_negative": bool(
            future_error["majority_group_negative"]
            and future_recorrection["majority_group_negative"]
        ),
        "unaffected_identity_error_regression_pp_within_1": bool(max_unaffected_regression_pp <= 1.0),
        "no_update_exact_zero_reference": bool(no_update_exact),
        "response_groups_at_least_2": bool(future_error["groups"] >= 2),
    }
    lcia_gate_pass = bool(all(criteria.values()))
    if lcia_gate_pass:
        next_phase = "N28-C_AUTHORIZED_AUTO_START"
    else:
        next_phase = "N28-C_NOT_AUTHORIZED_STOP"

    result = {
        "phase": "N28-B",
        "status": "SCIENTIFIC_GATE_PASS" if lcia_gate_pass else "SCIENTIFIC_GATE_FAIL",
        "method_status": "CACHED_FEATURE_FEASIBILITY_ONLY",
        "val25_read": False,
        "started_unix": started,
        "elapsed_seconds": time.monotonic() - started,
        "protocol": "outputs/n28/n28b_frozen_protocol.json",
        "protocol_sha256": sha256(ROOT / "outputs/n28/n28b_frozen_protocol.json"),
        "inputs": {
            "roles": list(cache_by_role),
            "apcr_checkpoint": apcr_by_role["external_train"][2],
            "all_roles_read_without_val25": True,
        },
        "roles": role_results,
        "response_attribution": {
            "formula": "R_B10=G(B10-current)-G(B10-cf); R_adapted=G(adapted-current)-G(adapted-cf); delta_delta=R_adapted-R_B10",
            "horizon_rows": RESPONSE_HORIZON,
            "sequence_group_bootstrap_repetitions": BOOTSTRAP_REPS,
            "sequence_group_bootstrap_seed": BOOTSTRAP_SEED,
            "methods": response_summaries,
            "response_csv": "outputs/n28/n28b_response_events.csv",
            "rejected_identity_selection_note": "cached local candidate-column proxy; no global detection identity was invented",
        },
        "primary_lcia_gate": {
            "criteria": criteria,
            "max_unaffected_identity_error_regression_pp": max_unaffected_regression_pp,
            "passed": lcia_gate_pass,
            "future_error_delta_delta": future_error,
            "future_recorrection_delta_delta": future_recorrection,
        },
        "transition": {
            "next_phase": next_phase,
            "n28c_started": False,
            "n28c_authorized": lcia_gate_pass,
        },
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n28/n28b_result.json")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "outputs/n28/data")
    parser.add_argument("--lcia-steps", type=int, default=5)
    args = parser.parse_args()
    if args.lcia_steps <= 0:
        raise ValueError("--lcia-steps must be positive")
    result = run_n28b(output=args.output, artifact_dir=args.artifact_dir, lcia_steps=args.lcia_steps)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("N28B_CACHED_CAUSAL_REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
