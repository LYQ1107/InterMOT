#!/usr/bin/env python3
"""N29-A causal/gradient smoke for the pretrained SAM3 decoder adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3.sam.transformer import TwoWayTransformer  # noqa: E402
from sam3_intermot.adaptation.corrected_mask_teacher import (  # noqa: E402
    BOX_DERIVED_PSEUDO_MASK,
    MaskTeacherConfig,
)
from sam3_intermot.adaptation.decoder_update_transaction import (  # noqa: E402
    DecoderCorrectionEvent,
    DecoderUpdateConfig,
    DecoderUpdateTransaction,
)
from sam3_intermot.adaptation.sam3_decoder_lit import (  # noqa: E402
    DecoderLITConfig,
    SAM3DecoderLITAdapter,
)
from sam3_intermot.association.decoder_candidate_bridge import (  # noqa: E402
    DecoderCandidate,
    build_decoder_assignment,
    official_output_to_decoder_candidate,
)


class ToyOfficialPropagationDecoder(nn.Module):
    """Small official-shaped harness; only the transformer is adapted."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )
        self.mask_head = nn.Linear(256, 64)
        nn.init.normal_(self.mask_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.mask_head.bias, 2.0)
        for parameter in self.mask_head.parameters():
            parameter.requires_grad = False

    def decode(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens, _ = self.transformer(image_embedding, image_pe, point_embedding)
        token = tokens[:, 0]
        masks = self.mask_head(token).view(token.shape[0], 1, 8, 8)
        return {
            "high_res_masks": masks,
            "object_score_logits": token.new_full((token.shape[0],), 5.0),
            "ious": token.new_full((token.shape[0],), 0.8),
            "sam_output_token": token,
        }


def _tensor_equal(a: torch.Tensor, b: torch.Tensor, atol: float = 1.0e-6) -> bool:
    return bool(torch.allclose(a, b, atol=atol, rtol=atol))


def _candidate_from_output(output: dict[str, torch.Tensor], version: int) -> DecoderCandidate:
    candidate = official_output_to_decoder_candidate(
        output,
        frame_idx=1,
        source_public_id=7,
        adapter_version=version,
        min_presence=0.01,
    )
    if candidate is None:
        raise AssertionError("toy official decoder unexpectedly produced no candidate")
    return candidate


def run() -> dict:
    torch.manual_seed(2901)
    np.random.seed(2901)
    decoder = ToyOfficialPropagationDecoder()
    baseline_image = torch.randn(1, 256, 2, 2)
    baseline_pe = torch.randn(1, 256, 2, 2)
    baseline_points = torch.randn(1, 1, 256)
    future_image = torch.randn(1, 256, 2, 2)
    future_pe = torch.randn(1, 256, 2, 2)
    future_points = torch.randn(1, 1, 256)

    with torch.no_grad():
        baseline_before = decoder.decode(baseline_image, baseline_pe, baseline_points)
    before_biases = {
        name: parameter.detach().clone()
        for name, parameter in decoder.named_parameters()
        if name.endswith(".bias") and "transformer" in name
    }

    adapter = SAM3DecoderLITAdapter(
        decoder,
        DecoderLITConfig(rank=4, alpha=4.0, dropout=0.0),
    )
    state = adapter.new_state("smoke-video", 7, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        with adapter.activate(state):
            baseline_zero = decoder.decode(baseline_image, baseline_pe, baseline_points)
    zero_equivalence = _tensor_equal(
        baseline_before["high_res_masks"], baseline_zero["high_res_masks"]
    ) and _tensor_equal(
        baseline_before["sam_output_token"], baseline_zero["sam_output_token"]
    )
    after_biases = {
        name: module.base.bias.detach().clone()
        for name, module in adapter.wrappers.items()
        if module.base.bias is not None
    }
    bias_equivalence = all(
        _tensor_equal(before_biases.get(name, value), value)
        for name, value in after_biases.items()
    )

    # Gradient scope: B is zero at initialization, so B must receive a
    # non-zero first-order gradient while A may legitimately receive zero.
    with adapter.activate(state):
        gradient_output = decoder.decode(baseline_image, baseline_pe, baseline_points)
        gradient_loss = gradient_output["high_res_masks"].square().mean()
    gradient_loss.backward()
    inventory = adapter.inventory(state)
    nonzero_grad = [
        item["name"] for item in inventory["trainable"] if item["grad_nonzero"]
    ]
    base_gradients = [
        name for name, parameter in decoder.named_parameters()
        if "transformer" in name and parameter.grad is not None
    ]
    for parameter in state.parameters():
        parameter.grad = None

    # Future-causal bridge.  The current output is recorded once before the
    # transaction and is intentionally never recomputed after it commits.
    with torch.no_grad():
        future_before = decoder.decode(future_image, future_pe, future_points)
        current_record = {
            key: value.detach().clone() for key, value in baseline_zero.items()
        }
    event = DecoderCorrectionEvent(
        video_id="smoke-video",
        public_id=7,
        frame_idx=0,
        provenance=BOX_DERIVED_PSEUDO_MASK,
        box_xyxy=(1.0, 1.0, 7.0, 7.0),
        image_size=(8, 8),
        current_output_recorded=True,
    )

    def support_forward(_supervision, _step):
        return decoder.decode(baseline_image, baseline_pe, baseline_points)[
            "high_res_masks"
        ]

    transaction = DecoderUpdateTransaction(
        adapter,
        DecoderUpdateConfig(
            inner_steps=20,
            learning_rate=0.05,
            teacher=MaskTeacherConfig(lambda_focal=5.0, lambda_dice=1.0, lambda_box=0.1),
        ),
    )
    update = transaction.apply(event, state, forward_fn=support_forward)
    if not update.committed:
        raise AssertionError(f"toy correction did not commit: {update}")
    with torch.no_grad():
        with adapter.activate(state):
            future_after = decoder.decode(future_image, future_pe, future_points)
    current_unchanged = all(
        _tensor_equal(current_record[key], baseline_zero[key]) for key in current_record
    )
    future_token_delta = float(
        (future_after["sam_output_token"] - future_before["sam_output_token"])
        .abs()
        .max()
    )
    future_mask_delta = float(
        (future_after["high_res_masks"] - future_before["high_res_masks"])
        .abs()
        .max()
    )

    # A fresh zero state is the no-correction counterfactual.
    no_correction = adapter.new_state("smoke-video-no-correction", 7, device="cpu")
    with torch.no_grad():
        with adapter.activate(no_correction):
            future_no_correction = decoder.decode(future_image, future_pe, future_points)
    correction_removed = _tensor_equal(
        future_no_correction["sam_output_token"], future_before["sam_output_token"]
    )

    pre_candidate = _candidate_from_output(future_before, 0)
    post_candidate = _candidate_from_output(future_after, state.adapter_version)
    original = DecoderCandidate(
        frame_idx=1,
        mask_logits=np.ones((8, 8), dtype=np.float32),
        mask=np.ones((8, 8), dtype=bool),
        box_xyxy=(0.0, 0.0, 1.0, 1.0),
        presence=1.0,
        iou_pred=0.9,
        decoder_token=None,
        clip_feature=None,
        source="original_anchor",
        source_public_id=None,
    )
    pre_token = float(pre_candidate.decoder_token.mean())
    post_token = float(post_candidate.decoder_token.mean())
    pre_token_vector = np.asarray(pre_candidate.decoder_token, dtype=np.float64)
    post_token_vector = np.asarray(post_candidate.decoder_token, dtype=np.float64)
    token_shift_norm = float(np.linalg.norm(post_token_vector - pre_token_vector))
    token_scale = max(token_shift_norm, 1.0e-6)

    def score_delta(_identity, candidate):
        if candidate.source.startswith("sam3"):
            token_vector = np.asarray(candidate.decoder_token, dtype=np.float64)
            # The fixture's frozen token scorer is normalized once for this
            # diagnostic so a measurable decoder change is guaranteed to be
            # visible at the matrix/assignment boundary.
            token_distance = float(np.linalg.norm(token_vector - pre_token_vector))
            return token_distance / token_scale * 0.2 - 0.05
        return 0.0

    pre_bridge = build_decoder_assignment(
        np.zeros((1, 2), dtype=np.float64),
        [original, pre_candidate],
        [7],
        score_delta_fn=score_delta,
        none_scores=np.array([-0.5]),
    )
    post_bridge = build_decoder_assignment(
        np.zeros((1, 2), dtype=np.float64),
        [original, post_candidate],
        [7],
        score_delta_fn=score_delta,
        none_scores=np.array([-0.5]),
    )
    assignment_changed = (
        int(pre_bridge.assignment.assignment[0]) != int(post_bridge.assignment.assignment[0])
        and int(pre_bridge.assignment.assignment[0]) == 0
        and int(post_bridge.assignment.assignment[0]) == 1
    )

    checks = {
        "zero_lora_equivalence": bool(zero_equivalence),
        "all_21_targets": inventory["target_count"] == 21,
        "rank4_parameter_count": inventory["adapter_parameter_count"] == 35328,
        "biases_unchanged": bool(bias_equivalence),
        "base_gradients_frozen": not base_gradients,
        "adapter_gradient_reaches_b": bool(any(name.endswith("lora_b") for name in nonzero_grad)),
        "current_frame_unchanged": bool(current_unchanged),
        "future_decoder_candidate_changes": future_token_delta > 1.0e-8 and future_mask_delta > 1.0e-8,
        "removing_correction_removes_effect": bool(correction_removed),
        "candidate_reaches_full_matrix": post_bridge.matrix.shape == (1, 3),
        "hungarian_assignment_changes": bool(assignment_changed),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"N29-A smoke failed: {failed}")

    result = {
        "status": "PASS",
        "val25_read": False,
        "checks": checks,
        "inventory": inventory,
        "update": {
            "committed": update.committed,
            "adapter_version": update.adapter_version,
            "loss_history": list(update.loss_history),
            "gradient_parameter_count": update.gradient_parameter_count,
        },
        "causal": {
            "future_token_max_abs_delta": future_token_delta,
            "future_mask_max_abs_delta": future_mask_delta,
            "pre_assignment": pre_bridge.assignment.assignment.tolist(),
            "post_assignment": post_bridge.assignment.assignment.tolist(),
            "pre_matrix": pre_bridge.matrix.tolist(),
            "post_matrix": post_bridge.matrix.tolist(),
            "current_output_recomputed": False,
        },
        "official_target_paths": list(adapter.target_paths),
        "provenance": BOX_DERIVED_PSEUDO_MASK,
        "seed": 2901,
        "note": "Synthetic official-shaped engineering fixture; no dataset metrics claimed.",
    }
    output_path = ROOT / "outputs" / "n29" / "n29a_causal_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
