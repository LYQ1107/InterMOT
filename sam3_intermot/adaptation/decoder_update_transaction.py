"""Atomic correction transaction for N29 decoder and B10 state."""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from sam3_intermot.adaptation.corrected_mask_teacher import (
    CompiledMaskSupervision,
    MaskTeacherConfig,
    compile_mask_supervision,
    decoder_loss,
)
from sam3_intermot.adaptation.sam3_decoder_lit import (
    AdapterSnapshot,
    IdentityAdapterState,
    SAM3DecoderLITAdapter,
)
from sam3_intermot.adaptation.correction_compiler import CorrectionTransaction


@dataclass(frozen=True)
class DecoderCorrectionEvent:
    """A legal spatial correction after the current output was recorded."""

    video_id: Any
    public_id: Any
    frame_idx: int
    provenance: str
    box_xyxy: Optional[Sequence[float]] = None
    corrected_mask: Optional[object] = None
    image_size: Optional[tuple[int, int]] = None
    identity_transaction: Optional[CorrectionTransaction] = None
    current_output_recorded: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)

    def compile(self, *, allow_oracle: bool = False) -> CompiledMaskSupervision:
        if not self.current_output_recorded:
            raise ValueError("the current frame must be recorded before an update")
        return compile_mask_supervision(
            provenance=self.provenance,
            corrected_mask=self.corrected_mask,
            box_xyxy=self.box_xyxy,
            image_size=self.image_size,
            allow_oracle=allow_oracle,
        )


@dataclass(frozen=True)
class DecoderUpdateConfig:
    inner_steps: int = 5
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    teacher: MaskTeacherConfig = MaskTeacherConfig()
    optimizer_enabled: bool = True
    require_loss_decrease: bool = False
    require_observable_update: bool = False
    parameter_delta_l2_tol: float = 1.0e-8
    logit_delta_linf_tol: float = 1.0e-8
    support_loss_decrease_tol: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.parameter_delta_l2_tol < 0 or self.logit_delta_linf_tol < 0:
            raise ValueError("observable-update tolerances must be non-negative")
        if self.support_loss_decrease_tol < 0:
            raise ValueError("support loss tolerance must be non-negative")


@dataclass(frozen=True)
class DecoderUpdateResult:
    transaction_id: str
    status: str
    committed: bool
    frame_idx: int
    public_id: Any
    adapter_version: int
    loss_history: tuple[float, ...]
    gradient_parameter_count: int
    rollback_reason: Optional[str] = None
    provenance: str = ""
    exception_traceback: Optional[str] = None
    optimization_diagnostic: Mapping[str, Any] = field(default_factory=dict)


ForwardFn = Callable[[CompiledMaskSupervision, int], Tensor]
ValidatorFn = Callable[
    [DecoderCorrectionEvent, IdentityAdapterState, CompiledMaskSupervision, tuple[float, ...]],
    tuple[bool, str],
]
DeterministicForwardFn = Callable[[CompiledMaskSupervision], Tensor]


class DecoderUpdateTransaction:
    """Update decoder A/B and B10 as one commit-or-rollback operation.

    ``forward_fn`` must call the official propagation decoder (or a declared
    offline episodic equivalent) under ``adapter.activate(state)``.  It is
    deliberately injected so that the transaction can be tested without
    loading the 3.5 GB checkpoint.  The caller is responsible for ensuring
    that any current-frame observation has already been delivered.
    """

    def __init__(
        self,
        adapter: SAM3DecoderLITAdapter,
        config: Optional[DecoderUpdateConfig] = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or DecoderUpdateConfig()
        self.ledger: list[dict[str, Any]] = []

    def apply(
        self,
        event: DecoderCorrectionEvent,
        state: IdentityAdapterState,
        *,
        forward_fn: ForwardFn,
        b10_snapshot_fn: Optional[Callable[[], Any]] = None,
        b10_update_fn: Optional[Callable[[DecoderCorrectionEvent], None]] = None,
        b10_restore_fn: Optional[Callable[[Any], None]] = None,
        validator_fn: Optional[ValidatorFn] = None,
        reference_logits: Optional[Tensor] = None,
        deterministic_forward_fn: Optional[DeterministicForwardFn] = None,
    ) -> DecoderUpdateResult:
        supervision = event.compile()
        if not supervision.online_legal:
            raise ValueError("an evaluation-only mask cannot enter an online transaction")
        snapshot = self.adapter.snapshot(state)
        b10_snapshot = b10_snapshot_fn() if b10_snapshot_fn else None
        optimizer_params = state.parameters_for_update()
        if not optimizer_params:
            raise ValueError("the adapter has no trainable A/B parameters")
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        losses: list[float] = []
        gradient_l2_by_step: list[float] = []
        status = "COMMIT"
        reason: Optional[str] = None
        exception_traceback: Optional[str] = None
        diagnostic_error: Optional[str] = None

        pre_logits: Optional[Tensor] = None
        if deterministic_forward_fn is not None:
            try:
                with self.adapter.activate(state):
                    pre_logits = deterministic_forward_fn(supervision).detach().clone()
            except Exception as exc:
                diagnostic_error = f"deterministic_before_failed: {type(exc).__name__}: {exc}"
        try:
            step_count = self.config.inner_steps if self.config.optimizer_enabled else 1
            for step in range(step_count):
                optimizer.zero_grad(set_to_none=True)
                with self.adapter.activate(state):
                    logits = forward_fn(supervision, step)
                if not torch.is_tensor(logits) or not logits.requires_grad:
                    raise ValueError("decoder forward must return differentiable logits")
                terms = decoder_loss(
                    logits,
                    supervision,
                    config=self.config.teacher,
                    reference_logits=reference_logits,
                    weight_delta=self._weight_deltas(state, snapshot),
                )
                loss = terms["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite decoder loss")
                if self.config.optimizer_enabled:
                    loss.backward()
                    gradient_l2_by_step.append(self._gradient_l2(optimizer_params))
                    if self.config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            optimizer_params,
                            self.config.grad_clip_norm,
                        )
                    optimizer.step()
                else:
                    gradient_l2_by_step.append(0.0)
                losses.append(float(loss.detach().cpu()))
            post_logits: Optional[Tensor] = None
            if deterministic_forward_fn is not None:
                try:
                    with self.adapter.activate(state):
                        post_logits = deterministic_forward_fn(supervision).detach().clone()
                except Exception as exc:
                    diagnostic_error = (
                        diagnostic_error
                        or f"deterministic_after_failed: {type(exc).__name__}: {exc}"
                    )
            diagnostic = self._optimization_diagnostic(
                state,
                snapshot,
                supervision,
                pre_logits,
                post_logits,
                gradient_l2_by_step,
                diagnostic_error,
            )
            diagnostic["train_loss_history"] = [float(value) for value in losses]
            diagnostic["optimizer_enabled"] = bool(self.config.optimizer_enabled)
            if b10_update_fn is not None:
                b10_update_fn(event)
            if self.config.require_observable_update:
                accepted, reason = self._observable_update_gate(diagnostic)
                if not accepted:
                    status = "ROLLBACK"
                    reason = reason or "observable_update_gate_rejected"
            if status == "COMMIT" and validator_fn is not None:
                accepted, reason = validator_fn(event, state, supervision, tuple(losses))
                if not accepted:
                    status = "ROLLBACK"
                    reason = reason or "validator_rejected"
            elif status == "COMMIT" and self.config.require_loss_decrease and (
                len(losses) > 1 and losses[-1] >= losses[0]
            ):
                status = "ROLLBACK"
                reason = "support_loss_did_not_decrease"
        except Exception as exc:
            status = "ROLLBACK"
            reason = f"{type(exc).__name__}: {exc}"
            exception_traceback = traceback.format_exc(limit=8)
            diagnostic = self._optimization_diagnostic(
                state,
                snapshot,
                supervision,
                pre_logits,
                None,
                gradient_l2_by_step,
                diagnostic_error,
            )
            diagnostic["train_loss_history"] = [float(value) for value in losses]
            diagnostic["optimizer_enabled"] = bool(self.config.optimizer_enabled)

        if status == "ROLLBACK":
            self.adapter.restore(state, snapshot)
            if b10_restore_fn is not None:
                b10_restore_fn(b10_snapshot)
            result = DecoderUpdateResult(
                transaction_id=self._transaction_id(event),
                status=status,
                committed=False,
                frame_idx=int(event.frame_idx),
                public_id=event.public_id,
                adapter_version=state.adapter_version,
                loss_history=tuple(losses),
                gradient_parameter_count=len(optimizer_params),
                rollback_reason=reason,
                provenance=event.provenance,
                exception_traceback=exception_traceback,
                optimization_diagnostic=diagnostic,
            )
        else:
            state.mark_update(event.frame_idx, event.provenance)
            result = DecoderUpdateResult(
                transaction_id=self._transaction_id(event),
                status="COMMIT",
                committed=True,
                frame_idx=int(event.frame_idx),
                public_id=event.public_id,
                adapter_version=state.adapter_version,
                loss_history=tuple(losses),
                gradient_parameter_count=len(optimizer_params),
                provenance=event.provenance,
                exception_traceback=None,
                optimization_diagnostic=diagnostic,
            )
        self.ledger.append(
            {
                "transaction_id": result.transaction_id,
                "video_id": event.video_id,
                "public_id": event.public_id,
                "frame_idx": int(event.frame_idx),
                "provenance": event.provenance,
                "status": result.status,
                "committed": result.committed,
                "adapter_version": result.adapter_version,
                "loss_history": list(result.loss_history),
                "rollback_reason": result.rollback_reason,
                "exception_traceback": result.exception_traceback,
                "optimization_diagnostic": dict(result.optimization_diagnostic),
                "current_output_recorded": event.current_output_recorded,
            }
        )
        return result

    @staticmethod
    def _gradient_l2(parameters: Sequence[Tensor]) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().pow(2).sum().cpu())
        return math.sqrt(total)

    def _optimization_diagnostic(
        self,
        state: IdentityAdapterState,
        snapshot: AdapterSnapshot,
        supervision: CompiledMaskSupervision,
        pre_logits: Optional[Tensor],
        post_logits: Optional[Tensor],
        gradient_l2_by_step: Sequence[float],
        diagnostic_error: Optional[str],
    ) -> dict[str, Any]:
        per_layer: dict[str, float] = {}
        a_l2 = 0.0
        b_l2 = 0.0
        delta_linf = 0.0
        finite = diagnostic_error is None
        for path in self.adapter.target_paths:
            spec = state.specs[path]
            old_a, old_b = snapshot.tensors[path]
            da = (state.lora_a[spec.key].detach() - old_a.to(state.lora_a[spec.key])).float()
            db = (state.lora_b[spec.key].detach() - old_b.to(state.lora_b[spec.key])).float()
            a_sq = float(da.pow(2).sum().cpu())
            b_sq = float(db.pow(2).sum().cpu())
            a_l2 += a_sq
            b_l2 += b_sq
            per_layer[path] = math.sqrt(a_sq + b_sq)
            if da.numel():
                delta_linf = max(delta_linf, float(da.abs().max().cpu()))
            if db.numel():
                delta_linf = max(delta_linf, float(db.abs().max().cpu()))
        parameter_l2 = math.sqrt(a_l2 + b_l2)
        finite = finite and math.isfinite(parameter_l2) and math.isfinite(delta_linf)
        diagnostic: dict[str, Any] = {
            "train_loss_history": [],
            "gradient_l2_by_step": [float(value) for value in gradient_l2_by_step],
            "parameter_delta_l2_total": parameter_l2,
            "parameter_delta_linf": delta_linf,
            "parameter_delta_l2_a": math.sqrt(a_l2),
            "parameter_delta_l2_b": math.sqrt(b_l2),
            "per_layer_parameter_delta_l2": per_layer,
            "deterministic_support_loss_before": None,
            "deterministic_support_loss_after": None,
            "deterministic_support_loss_delta": None,
            "logit_delta_l1_mean": None,
            "logit_delta_linf": None,
            "binary_mask_changed_pixel_count": None,
            "support_soft_dice_before": None,
            "support_soft_dice_after": None,
            "support_binary_iou_before": None,
            "support_binary_iou_after": None,
            "finite": finite,
            "error": diagnostic_error,
        }
        if pre_logits is None or post_logits is None:
            diagnostic["finite"] = False
            diagnostic["error"] = diagnostic["error"] or "deterministic_logits_unavailable"
            return diagnostic
        if not torch.isfinite(pre_logits).all() or not torch.isfinite(post_logits).all():
            diagnostic["finite"] = False
            diagnostic["error"] = diagnostic["error"] or "non_finite_deterministic_logits"
            return diagnostic
        before_loss = self._support_loss(pre_logits, supervision)
        after_loss = self._support_loss(post_logits, supervision)
        difference = (post_logits.float() - pre_logits.float()).detach()
        diagnostic.update(
            {
                "deterministic_support_loss_before": before_loss,
                "deterministic_support_loss_after": after_loss,
                "deterministic_support_loss_delta": after_loss - before_loss,
                "logit_delta_l1_mean": float(difference.abs().mean().cpu()),
                "logit_delta_linf": float(difference.abs().max().cpu()),
                "binary_mask_changed_pixel_count": self._binary_changed_pixels(
                    pre_logits, post_logits
                ),
                "support_soft_dice_before": self._soft_dice(pre_logits, supervision),
                "support_soft_dice_after": self._soft_dice(post_logits, supervision),
                "support_binary_iou_before": self._binary_iou(pre_logits, supervision),
                "support_binary_iou_after": self._binary_iou(post_logits, supervision),
            }
        )
        scalar_values = [
            diagnostic["deterministic_support_loss_before"],
            diagnostic["deterministic_support_loss_after"],
            diagnostic["deterministic_support_loss_delta"],
            diagnostic["logit_delta_l1_mean"],
            diagnostic["logit_delta_linf"],
            diagnostic["support_soft_dice_before"],
            diagnostic["support_soft_dice_after"],
            diagnostic["support_binary_iou_before"],
            diagnostic["support_binary_iou_after"],
        ]
        diagnostic["finite"] = bool(
            diagnostic["finite"]
            and all(value is None or math.isfinite(float(value)) for value in scalar_values)
            and all(math.isfinite(float(value)) for value in diagnostic["gradient_l2_by_step"])
            and all(math.isfinite(float(value)) for value in diagnostic["per_layer_parameter_delta_l2"].values())
        )
        return diagnostic

    def _support_loss(
        self,
        logits: Tensor,
        supervision: CompiledMaskSupervision,
    ) -> float:
        terms = decoder_loss(
            logits,
            supervision,
            config=self.config.teacher,
            weight_delta=None,
        )
        return float(terms["total"].detach().float().cpu())

    @staticmethod
    def _align_target(logits: Tensor, supervision: CompiledMaskSupervision) -> Optional[Tensor]:
        if supervision.mask_target is None:
            return None
        target = supervision.mask_target.to(device=logits.device, dtype=logits.dtype)
        result = logits
        while result.ndim > 2 and result.shape[0] == 1:
            result = result[0]
        while result.ndim > 2 and result.shape[1] == 1:
            result = result[:, 0]
        while target.ndim > 2 and target.shape[0] == 1:
            target = target[0]
        if target.shape != result.shape:
            target = F.interpolate(target[None, None], size=result.shape, mode="nearest")[0, 0]
        return target

    def _soft_dice(
        self,
        logits: Tensor,
        supervision: CompiledMaskSupervision,
    ) -> Optional[float]:
        target = self._align_target(logits, supervision)
        if target is None:
            return None
        result = logits
        while result.ndim > 2 and result.shape[0] == 1:
            result = result[0]
        probability = torch.sigmoid(result).float()
        target = target.float()
        score = (2.0 * (probability * target).sum() + 1.0e-6) / (
            probability.sum() + target.sum() + 1.0e-6
        )
        return float(score.detach().cpu())

    def _binary_iou(
        self,
        logits: Tensor,
        supervision: CompiledMaskSupervision,
    ) -> Optional[float]:
        target = self._align_target(logits, supervision)
        if target is None:
            return None
        result = logits
        while result.ndim > 2 and result.shape[0] == 1:
            result = result[0]
        prediction = result.detach().float().cpu().numpy() > self.config.teacher.mask_threshold
        truth = target.detach().float().cpu().numpy() > 0.5
        union = np.logical_or(prediction, truth).sum()
        intersection = np.logical_and(prediction, truth).sum()
        return float(intersection / union) if union else 1.0

    @staticmethod
    def _binary_changed_pixels(before: Tensor, after: Tensor) -> Optional[int]:
        left = before.detach()
        right = after.detach()
        while left.ndim > 2 and left.shape[0] == 1:
            left = left[0]
        while right.ndim > 2 and right.shape[0] == 1:
            right = right[0]
        if left.shape != right.shape:
            return None
        return int(
            torch.logical_xor(left > 0.0, right > 0.0).sum().detach().cpu()
        )

    def _observable_update_gate(self, diagnostic: Mapping[str, Any]) -> tuple[bool, str]:
        if not diagnostic.get("finite", False):
            return False, f"observable_update_nonfinite:{diagnostic.get('error', 'unknown')}"
        parameter_delta = diagnostic.get("parameter_delta_l2_total")
        if parameter_delta is None or parameter_delta <= self.config.parameter_delta_l2_tol:
            return False, "parameter_delta_below_tolerance"
        logit_delta = diagnostic.get("logit_delta_linf")
        if logit_delta is None or logit_delta <= self.config.logit_delta_linf_tol:
            return False, "logit_delta_below_tolerance"
        before = diagnostic.get("deterministic_support_loss_before")
        after = diagnostic.get("deterministic_support_loss_after")
        if before is None or after is None:
            return False, "deterministic_support_loss_unavailable"
        if after >= before - self.config.support_loss_decrease_tol:
            return False, "deterministic_support_loss_did_not_decrease"
        return True, ""

    def _weight_deltas(
        self,
        state: IdentityAdapterState,
        snapshot: AdapterSnapshot,
    ) -> list[Tensor]:
        deltas: list[Tensor] = []
        for path in self.adapter.target_paths:
            spec = state.specs[path]
            old_a, old_b = snapshot.tensors[path]
            deltas.extend(
                [
                    state.lora_a[spec.key] - old_a.to(state.lora_a[spec.key]),
                    state.lora_b[spec.key] - old_b.to(state.lora_b[spec.key]),
                ]
            )
        return deltas

    @staticmethod
    def _transaction_id(event: DecoderCorrectionEvent) -> str:
        return f"N29_DECODER:{event.video_id}:{event.public_id}:{int(event.frame_idx)}"
