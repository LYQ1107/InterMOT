"""Compact pairwise open-set identity verifier for N72R20R3R1.

The module deliberately operates on already-produced 512-D vectors.  It does
not import SAM3, OSNet, the N72R18 GRU, the association solver, or GT.  The
``predict_open_set_identity`` function is the runtime boundary: it only
selects NONE or an existing candidate UID and fails closed on malformed input.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FEATURE_DIMENSION = 512
VARIANTS = ("A0_ANCHOR_ONLY", "A1_LEARNED_STATE_ONLY", "A2_DUAL_STATE")
FORBIDDEN_RUNTIME_KEYS = {
    "target_gt_id",
    "target_iou",
    "target_candidate_iou",
    "taxonomy",
    "taxonomy_posthoc",
    "training_label",
    "best_target_candidate_uid",
    "best_target_candidate_iou",
    "target_gt_present",
    "candidate_set_present",
}


def _vector(value: Any, *, name: str, device: torch.device | None = None) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
    if tensor.shape != (FEATURE_DIMENSION,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite 512-D vector")
    norm = torch.linalg.vector_norm(tensor)
    if not bool(norm > 1.0e-8):
        raise ValueError(f"{name} is zero")
    return tensor / norm


def _masked_top2(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return top1, top2, margin, mean, std for a padded candidate matrix."""

    safe = values.masked_fill(~mask, -1.0e9)
    top = torch.topk(safe, k=min(2, values.shape[1]), dim=1).values
    top1 = top[:, 0]
    top2 = top[:, 1] if top.shape[1] > 1 else torch.zeros_like(top1)
    raw_count = mask.sum(dim=1)
    no_candidates = raw_count == 0
    one_candidate = raw_count == 1
    top1 = top1.masked_fill(no_candidates, 0.0)
    # A missing second candidate is not a -1e9 evidence value.  Treat the
    # singleton set as zero-margin and the empty set as an all-zero statistic.
    top2 = torch.where(one_candidate, top1, top2)
    top2 = top2.masked_fill(no_candidates, 0.0)
    count = raw_count.clamp_min(1)
    mean = (values.masked_fill(~mask, 0.0).sum(dim=1) / count)
    centered = values.masked_fill(~mask, 0.0) - mean.unsqueeze(1)
    std = torch.sqrt((centered.pow(2).masked_fill(~mask, 0.0).sum(dim=1) / count).clamp_min(1.0e-8))
    return top1, top2, top1 - top2, mean, std


class ExplicitNoneVerifier(nn.Module):
    """A small verifier with a true candidate-set NONE logit.

    ``mode`` is ``V1`` (candidate logits only) or ``V2`` (candidate logits
    plus an explicit NONE logit).  ``state_variant`` selects the frozen target
    representation.  The two linear projections, pairwise MLP, and NONE head
    are the only trainable parameters.
    """

    def __init__(
        self,
        *,
        mode: str = "V2_EXPLICIT_NONE",
        state_variant: str = "A2_DUAL_STATE",
        projection_dim: int = 32,
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        if mode not in {"V1_PAIRWISE", "V2_EXPLICIT_NONE"}:
            raise ValueError(f"unknown verifier mode: {mode}")
        if state_variant not in VARIANTS:
            raise ValueError(f"unknown state variant: {state_variant}")
        if not 1 <= int(projection_dim) <= 64 or not 1 <= int(hidden_dim) <= 64:
            raise ValueError("projection and hidden dimensions must be in 1..64")
        self.mode = str(mode)
        self.state_variant = str(state_variant)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)

        self.target_projection = nn.Linear(FEATURE_DIMENSION, self.projection_dim)
        self.candidate_projection = nn.Linear(FEATURE_DIMENSION, self.projection_dim)
        relation_dim = 2 * self.projection_dim + 2
        if self.state_variant == "A2_DUAL_STATE":
            relation_dim = 2 * relation_dim + 1
        self.pairwise = nn.Sequential(
            nn.Linear(relation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.none_head = nn.Sequential(
            nn.Linear(9, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _project(self, vectors: Tensor) -> Tensor:
        return F.normalize(self.target_projection(vectors), dim=-1, eps=1.0e-8)

    def _candidate_project(self, vectors: Tensor) -> Tensor:
        return F.normalize(self.candidate_projection(vectors), dim=-1, eps=1.0e-8)

    def _relation(self, target: Tensor, candidates: Tensor, candidate_projected: Tensor | None = None) -> Tensor:
        target_projected = self._project(target).unsqueeze(1)
        projected = candidate_projected if candidate_projected is not None else self._candidate_project(candidates)
        raw_cos = F.cosine_similarity(target.unsqueeze(1), candidates, dim=-1, eps=1.0e-8).unsqueeze(-1)
        projected_cos = (target_projected * projected).sum(dim=-1, keepdim=True)
        return torch.cat(
            [target_projected * projected, torch.abs(target_projected - projected), raw_cos, projected_cos], dim=-1
        )

    def _target_relations(self, human_anchor: Tensor, learned_state: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor | None]:
        candidate_projected = self._candidate_project(candidates)
        if self.state_variant == "A0_ANCHOR_ONLY":
            return self._relation(human_anchor, candidates, candidate_projected), None
        if self.state_variant == "A1_LEARNED_STATE_ONLY":
            return self._relation(learned_state, candidates, candidate_projected), None
        anchor_relation = self._relation(human_anchor, candidates, candidate_projected)
        learned_relation = self._relation(learned_state, candidates, candidate_projected)
        drift = F.cosine_similarity(human_anchor, learned_state, dim=-1, eps=1.0e-8).unsqueeze(1)
        drift = drift.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        return torch.cat([anchor_relation, learned_relation, drift], dim=-1), drift[:, 0, 0]

    def forward(
        self,
        human_anchor: Tensor,
        learned_state: Tensor,
        candidates: Tensor,
        candidate_mask: Tensor,
        runtime_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return padded candidate logits and, for V2, a NONE logit.

        Shapes are ``[B,512]``, ``[B,512]``, ``[B,N,512]``, and ``[B,N]``.
        ``runtime_context`` has four causal, non-GT columns: candidate count,
        frames since anchor, frames since last memory write, and state/anchor
        cosine.  Values are normalized by the caller and are not sequence IDs.
        """

        if candidates.ndim != 3 or candidate_mask.ndim != 2 or candidates.shape[:2] != candidate_mask.shape:
            raise ValueError("candidate tensors must have shapes [B,N,512] and [B,N]")
        if human_anchor.ndim != 2 or learned_state.ndim != 2 or human_anchor.shape != learned_state.shape:
            raise ValueError("state tensors must both have shape [B,512]")
        if candidates.shape[0] != human_anchor.shape[0] or candidates.shape[-1] != FEATURE_DIMENSION:
            raise ValueError("batch or feature dimension mismatch")
        if not torch.isfinite(human_anchor).all() or not torch.isfinite(learned_state).all() or not torch.isfinite(candidates).all():
            raise ValueError("non-finite verifier input")
        human_anchor = F.normalize(human_anchor, dim=-1, eps=1.0e-8)
        learned_state = F.normalize(learned_state, dim=-1, eps=1.0e-8)
        candidates = F.normalize(candidates, dim=-1, eps=1.0e-8)
        relation, _ = self._target_relations(human_anchor, learned_state, candidates)
        candidate_logits = self.pairwise(relation).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1.0e9)
        output: dict[str, Tensor] = {"candidate_logits": candidate_logits}
        if self.mode == "V2_EXPLICIT_NONE":
            top1, top2, margin, mean, std = _masked_top2(candidate_logits, candidate_mask)
            count = candidate_mask.sum(dim=1).to(candidate_logits.dtype)
            drift = 1.0 - F.cosine_similarity(human_anchor, learned_state, dim=-1, eps=1.0e-8)
            if runtime_context is None:
                context = torch.zeros((candidates.shape[0], 4), dtype=candidate_logits.dtype, device=candidates.device)
                context[:, 0] = torch.log1p(count)
                context[:, 3] = 1.0 - drift
            else:
                if runtime_context.shape != (candidates.shape[0], 4):
                    raise ValueError("runtime_context must have shape [B,4]")
                context = runtime_context.to(dtype=candidate_logits.dtype, device=candidates.device)
            none_features = torch.stack(
                [top1, top2, margin, mean, std, torch.log1p(count), context[:, 1], context[:, 2], 1.0 - context[:, 3]],
                dim=1,
            )
            # Empty candidate sets have no valid candidate class.  A finite
            # NONE logit keeps CE stable and the runtime API is forced NONE.
            output["none_logit"] = self.none_head(none_features).squeeze(-1)
            output["logits"] = torch.cat([candidate_logits, output["none_logit"].unsqueeze(1)], dim=1)
        return output


@torch.no_grad()
def predict_open_set_identity(
    *,
    human_anchor: Any,
    learned_state: Any,
    candidate_rows: Sequence[Mapping[str, Any]],
    runtime_context: Mapping[str, Any] | None,
    verifier: ExplicitNoneVerifier,
    device: torch.device | str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict NONE or an existing candidate UID without GT or mutation."""

    for row in candidate_rows:
        if FORBIDDEN_RUNTIME_KEYS.intersection(row):
            raise ValueError("posthoc/GT candidate key crossed runtime verifier boundary")
        if row.get("candidate_uid") in (None, "", "None"):
            raise ValueError("candidate UID is missing")
    uids = [str(row["candidate_uid"]) for row in candidate_rows]
    if len(set(uids)) != len(uids):
        raise ValueError("candidate UID collision")
    if runtime_context is not None and FORBIDDEN_RUNTIME_KEYS.intersection(runtime_context):
        raise ValueError("posthoc/GT runtime key crossed verifier boundary")
    target_device = torch.device(device) if device is not None else next(verifier.parameters()).device
    anchor = _vector(human_anchor, name="human_anchor", device=target_device).unsqueeze(0)
    state = _vector(learned_state, name="learned_state", device=target_device).unsqueeze(0)
    if candidate_rows:
        candidate_tensor = torch.stack([_vector(row.get("feature", row.get("embedding")), name="candidate", device=target_device) for row in candidate_rows]).unsqueeze(0)
        mask = torch.ones((1, len(candidate_rows)), dtype=torch.bool, device=target_device)
    else:
        candidate_tensor = torch.zeros((1, 1, FEATURE_DIMENSION), dtype=torch.float32, device=target_device)
        mask = torch.zeros((1, 1), dtype=torch.bool, device=target_device)
    context_values = runtime_context or {}
    context = torch.tensor(
        [[
            float(torch.log1p(torch.tensor(float(len(candidate_rows))))),
            float(context_values.get("frames_since_human_initialization", 0.0)) / 100.0,
            float(context_values.get("frames_since_last_memory_write", 0.0)) / 100.0,
            float(context_values.get("learned_state_human_anchor_cosine", 1.0)),
        ]],
        dtype=torch.float32,
        device=target_device,
    )
    output = verifier(anchor, state, candidate_tensor, mask, context)
    candidate_logits = output["candidate_logits"][0]
    if verifier.mode == "V2_EXPLICIT_NONE":
        probabilities = torch.softmax(output["logits"][0], dim=0)
        selected = int(torch.argmax(output["logits"][0]).item())
        none_probability = float(probabilities[-1].item())
        if selected >= len(candidate_rows):
            selected_uid = None
            selected_probability = none_probability
        else:
            selected_uid = uids[selected]
            selected_probability = float(probabilities[selected].item())
    else:
        probabilities = torch.sigmoid(candidate_logits)
        if not candidate_rows:
            selected = -1
        else:
            selected = int(torch.argmax(candidate_logits).item())
        selected_probability = 0.0 if selected < 0 else float(probabilities[selected].item())
        accepted = selected >= 0 and (threshold is None or selected_probability >= float(threshold))
        selected_uid = uids[selected] if accepted else None
        none_probability = 1.0 - selected_probability if selected_uid is not None else 1.0
    return {
        "selected_candidate_uid": selected_uid,
        "none_probability": none_probability,
        "selected_probability": selected_probability,
        "candidate_probabilities": probabilities[: len(candidate_rows)].detach().cpu().tolist(),
        "candidate_logits": candidate_logits[: len(candidate_rows)].detach().cpu().tolist(),
        "runtime_future_gt_used": False,
        "candidate_created": False,
    }


__all__ = ["ExplicitNoneVerifier", "FEATURE_DIMENSION", "FORBIDDEN_RUNTIME_KEYS", "VARIANTS", "predict_open_set_identity"]
