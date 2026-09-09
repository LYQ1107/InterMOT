"""Learned, causal gate for the N72R12 counterfactual intervention.

The gate is deliberately a small decision module around the already frozen
base/proposal solvers.  It does not score candidates, infer public IDs, or
change solver semantics.  All inputs are runtime-causal audit values; labels
are attached only by the offline corpus builder after the runtime row has
been sealed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sam3_intermot.association.counterfactual_safe_intervention import (
    APPLY_PCTIS,
    KEEP_BASELINE,
)


SAFE_GATE_FEATURE_SCHEMA = (
    "selected_score",
    "selected_margin",
    "best_score",
    "second_score",
    "tanh_none_logit",
    "base_assignment_margin",
    "base_target_assigned",
    "proposal_target_changed",
    "proposal_target_is_selected",
    "tanh_selected_base_target_edge",
    "tanh_selected_proposed_target_edge",
    "tanh_selected_best_other_edge",
    "tanh_proposed_target_minus_best_other",
    "tanh_target_edge_gain",
    "selected_incumbent_is_target",
    "selected_incumbent_is_other",
    "non_target_changed_public_fraction",
    "total_changed_public_fraction",
    "selected_confidence",
    "selected_presence",
    "selected_motion_iou",
    "previous_assignment_uncertainty",
    "trusted_age_over_100",
    "previous_score_clipped",
    "causal_top_agrees",
    "candidate_count_over_16",
    "base_target_same_selected",
    "base_target_unassigned",
    "source_MAIN_B0",
    "source_CURRENT_RAW",
    "source_STATIC_REQUERY",
    "source_FUTURE_REQUERY",
    "source_UNKNOWN",
    "action_ADD",
    "action_ATOMIC",
    "action_AUTHORITATIVE",
    "action_RECOVER",
)
SAFE_GATE_FEATURE_DIM = 37
SAFE_GATE_THRESHOLD = 0.5

if len(SAFE_GATE_FEATURE_SCHEMA) != SAFE_GATE_FEATURE_DIM:  # pragma: no cover - import invariant
    raise RuntimeError("N72R12 safe-gate feature schema must contain exactly 37 fields")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _tanh(value: Any) -> float:
    return float(np.tanh(_finite(value)))


def _flag(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _selected_incumbent(
    audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]] | None,
) -> tuple[Any, Any]:
    index = audit.get("selected_candidate_index")
    uid = audit.get("selected_candidate_uid")
    if index is None or uid in (None, "", "None") or pool is None:
        return None, None
    try:
        item = pool[int(index)]
    except (IndexError, TypeError, ValueError):
        return None, None
    return item, uid


def _source_one_hot(source: Any) -> list[float]:
    text = str(source or "")
    aliases = {
        "MAIN_B0_CANDIDATE": "MAIN_B0",
        "TARGET_SESSION_CURRENT_RAW": "CURRENT_RAW",
        "TARGET_SESSION_REQUERY": "STATIC_REQUERY",
        "STATIC_EVENT_REQUERY": "STATIC_REQUERY",
        "FUTURE_FRAME_REQUERY": "FUTURE_REQUERY",
    }
    name = aliases.get(text, "UNKNOWN")
    return [float(name == value) for value in ("MAIN_B0", "CURRENT_RAW", "STATIC_REQUERY", "FUTURE_REQUERY", "UNKNOWN")]


def _action_one_hot(action: Any) -> list[float]:
    text = str(action or "")
    if text == "ADD_NEW_IDENTITY":
        name = "ADD"
    elif text == "ATOMIC_ID_SWAP":
        name = "ATOMIC"
    elif text == "AUTHORITATIVE_REASSIGN":
        name = "AUTHORITATIVE"
    elif text == "RECOVER_IDENTITY":
        name = "RECOVER"
    else:
        name = ""
    return [float(name == value) for value in ("ADD", "ATOMIC", "AUTHORITATIVE", "RECOVER")]


def build_safe_gate_feature(
    *,
    audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    state_audit: Mapping[str, Any] | None,
    action: str,
    source: str | None = None,
    candidate_count: int | None = None,
    pool: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Build the frozen causal 37-D feature vector.

    ``audit`` is the CSI audit made before the decision.  ``state_audit`` is
    the state immediately before the current frame.  Neither argument may
    contain posthoc labels; this function does not inspect or accept them.
    """

    state = state_audit or {}
    selected, selected_uid = _selected_incumbent(audit, selection, pool)
    target_public = audit.get("target_public_id")
    incumbent = None if selected is None else selected.get("incumbent_public_id_if_any")
    public_axis = audit.get("public_axis", [])
    public_count = max(len(public_axis), 1)
    non_target_count = _finite(audit.get("non_target_changed_public_count"))
    changed_count = _finite(audit.get("changed_public_count"))
    trusted_age = _finite(state.get("trusted_age"))
    previous_score = np.clip(_finite(state.get("previous_score")), -1.0, 1.0)
    count = max(int(candidate_count if candidate_count is not None else len(pool or [])), 0)
    values = [
        _finite(selection.get("selected_score")),
        _finite(selection.get("best_minus_second_margin")),
        _finite(selection.get("best_score")),
        _finite(selection.get("second_score")),
        _tanh(selection.get("none_logit")),
        _finite(audit.get("base_assignment_margin")),
        _flag(audit.get("base_target_uid") not in (None, "", "None")),
        _flag(audit.get("target_assignment_changed")),
        _flag(audit.get("proposal_target_is_selected")),
        _tanh(audit.get("selected_base_target_edge")),
        _tanh(audit.get("selected_proposed_target_edge")),
        _tanh(audit.get("selected_best_other_edge")),
        _tanh(audit.get("selected_target_minus_best_other")),
        _tanh(audit.get("target_edge_gain")),
        _flag(incumbent is not None and target_public is not None and int(incumbent) == int(target_public)),
        _flag(incumbent is not None and (target_public is None or int(incumbent) != int(target_public))),
        float(np.clip(non_target_count / max(public_count - 1, 1), 0.0, 1.0)),
        float(np.clip(changed_count / public_count, 0.0, 1.0)),
        _finite(audit.get("selected_confidence")),
        _finite(audit.get("selected_presence")),
        _finite(audit.get("selected_motion_iou")),
        float(np.clip(_finite(state.get("previous_uncertainty")), 0.0, 1.0)),
        float(trusted_age > 100.0),
        float(previous_score),
        _flag(audit.get("causal_top_agrees_with_selected")),
        float(np.clip(count / 16.0, 0.0, 1.0)),
        _flag(audit.get("base_target_uid") not in (None, "", "None") and audit.get("base_target_uid") == selected_uid),
        _flag(audit.get("base_target_uid") in (None, "", "None")),
        *_source_one_hot(source if source is not None else audit.get("selected_candidate_source")),
        *_action_one_hot(action),
    ]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (SAFE_GATE_FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("safe-gate feature is not finite 37-D")
    return result


class SafeInterventionMLP(nn.Module):
    """The fixed N72R12 37 -> 64 -> 32 -> 1 safe-gate MLP."""

    def __init__(self, input_dim: int = SAFE_GATE_FEATURE_DIM):
        super().__init__()
        if int(input_dim) != SAFE_GATE_FEATURE_DIM:
            raise ValueError(f"safe-gate input dimension must be {SAFE_GATE_FEATURE_DIM}")
        self.net = nn.Sequential(
            nn.Linear(SAFE_GATE_FEATURE_DIM, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != SAFE_GATE_FEATURE_DIM:
            raise ValueError(f"safe-gate tensor must end in {SAFE_GATE_FEATURE_DIM}")
        return self.net(features).squeeze(-1)


def safe_gate_probability(model: SafeInterventionMLP, feature: np.ndarray | torch.Tensor) -> float:
    """Return a finite sigmoid probability without mutating runtime state."""

    if isinstance(feature, np.ndarray):
        tensor = torch.as_tensor(feature, dtype=torch.float32)
    else:
        tensor = feature.detach().to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != SAFE_GATE_FEATURE_DIM or not torch.isfinite(tensor).all():
        raise ValueError("safe-gate input is invalid")
    device = next(model.parameters()).device
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor.to(device))).reshape(-1)[0].item()
    if not math.isfinite(float(probability)):
        raise ValueError("safe-gate probability is non-finite")
    return float(probability)


def safe_gate_decision(probability: float) -> str:
    """Apply the single preregistered 0.5 decision threshold."""

    value = _finite(probability, default=-1.0)
    return APPLY_PCTIS if value >= SAFE_GATE_THRESHOLD else KEEP_BASELINE
