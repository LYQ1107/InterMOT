"""Differentiable inner update on the official SAM3.1 full pipeline.

The update uses only the current human correction (box at frame t):
1. detector backbone features for frame t (frozen),
2. ``tracker.model.track_step`` with the box-derived point prompt under grad,
3. mask-in-box + objectness loss,
4. one or a few Adam steps on LoRA parameters only.

No future GT enters the update.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TFf

from sam3_intermot.adaptation.cfa_runner import box_to_points


def _box_grid(
    human_box: np.ndarray,
    ih: int,
    iw: int,
    h: int,
    w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = np.asarray(human_box, dtype=float)
    gx1, gy1 = max(0, int(x1 / iw * w)), max(0, int(y1 / ih * h))
    gx2, gy2 = min(w, int(np.ceil(x2 / iw * w))), min(h, int(np.ceil(y2 / ih * h)))
    inside = torch.zeros(1, 1, h, w, device=device)
    inside[..., gy1:gy2, gx1:gx2] = 1.0
    return inside, 1.0 - inside


def load_frame_tensor(
    video_dir: str, frame_idx: int, image_size: int = 1008
) -> Tuple[torch.Tensor, int, int]:
    """Load one normalized frame as [1,3,H,W] float32 (same preprocessing as official)."""
    from pathlib import Path

    names = sorted(
        p for p in Path(video_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    img = Image.open(names[frame_idx]).convert("RGB")
    orig_w, orig_h = img.size
    t = TFf.to_tensor(TFf.resize(img, (image_size, image_size))).float()
    mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    t = (t - mean) / std
    return t.unsqueeze(0), orig_w, orig_h


def inner_update_backend(
    backend,
    frame_idx: int,
    human_box: np.ndarray,
    lora_params: List[torch.nn.Parameter],
    steps: int = 2,
    lr: float = 1e-3,
    image: Optional[torch.Tensor] = None,
    video_h: int = 1080,
    video_w: int = 1920,
) -> Tuple[float, float]:
    """Run the inner update and return (loss, wall-seconds)."""
    demo = backend._predictor.model
    tracker = demo.tracker.model
    state = backend._predictor._all_inference_states[backend._session_id]
    device = next(tracker.parameters()).device
    num_frames = state.get("num_frames", 1000)

    if image is None:
        raise ValueError("image tensor required for inner update")
    image = image.to(device)
    from sam3.model.data_misc import NestedTensor

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            backbone_out = demo.detector.backbone.forward_image(
                NestedTensor(tensors=image, mask=None),
                need_sam3_out=False,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            features = tracker._prepare_backbone_features(backbone_out)

    pts = box_to_points(np.asarray(human_box, dtype=float))
    point_inputs = {
        "point_coords": (pts * tracker.image_size).unsqueeze(0).to(device),
        "point_labels": torch.ones(1, pts.shape[0], dtype=torch.int32, device=device),
    }
    mux = tracker.multiplex_controller.get_state(
        num_valid_entries=1,
        device=device,
        dtype=torch.float32,
        random=False,
        object_ids=[1],
    )
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    ih, iw = video_h, video_w

    opt = torch.optim.Adam(lora_params, lr=lr)
    t0 = __import__("time").time()
    total_loss = 0.0
    for _ in range(steps):
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.enable_grad():
                out = tracker.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=True,
                    backbone_features_interactive=features["interactive"],
                    backbone_features_propagation=features["sam2_backbone_out"],
                    image=image,
                    point_inputs=point_inputs,
                    mask_inputs=None,
                    gt_masks=None,
                    frames_to_add_correction_pt=[],
                    output_dict=output_dict,
                    num_frames=num_frames,
                    run_mem_encoder=False,
                    prev_sam_mask_logits=None,
                    multiplex_state=mux,
                    objects_to_interact=None,
                )
                pred = out["pred_masks"]
                if pred.dim() == 4:
                    pred = pred[0]
                if pred.dim() == 3 and pred.shape[0] == 1:
                    pred = pred[0]
                h, w = pred.shape[-2], pred.shape[-1]
                inside, outside = _box_grid(
                    np.asarray(human_box, dtype=float), ih, iw, h, w, pred.device
                )
                prob = torch.sigmoid(pred)
                loss_outside = (prob * outside).sum() / outside.sum().clamp(min=1)
                loss_coverage = 1.0 - (
                    (prob * inside).sum() / inside.sum().clamp(min=1)
                )
                obj_logits = out.get("object_score_logits")
                loss_obj = torch.tensor(0.0, device=pred.device)
                if obj_logits is not None:
                    o = obj_logits.reshape(-1)
                    loss_obj = torch.nn.functional.binary_cross_entropy_with_logits(
                        o, torch.ones_like(o)
                    )
                loss = loss_outside + 0.5 * loss_coverage + 0.1 * loss_obj
                loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        total_loss += float(loss.detach().cpu())
    elapsed = __import__("time").time() - t0
    return total_loss / max(1, steps), elapsed


def inner_update_shadow(
    shadow_model,
    image: torch.Tensor,
    human_box: np.ndarray,
    lora_params: List[torch.nn.Parameter],
    video_h: int,
    video_w: int,
    steps: int = 2,
    lr: float = 1e-3,
    num_frames: int = 1000,
) -> Tuple[float, float]:
    """Differentiable inner update on a standalone tracker copy (proven path)."""
    from sam3.model.data_misc import NestedTensor

    device = next(shadow_model.parameters()).device
    image = image.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            backbone_out = shadow_model.forward_image(
                NestedTensor(tensors=image, mask=None),
                need_sam3_out=False,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            features = shadow_model._prepare_backbone_features(backbone_out)

    pts = box_to_points(np.asarray(human_box, dtype=float))
    point_inputs = {
        "point_coords": (pts * shadow_model.image_size).unsqueeze(0).to(device),
        "point_labels": torch.ones(
            1, pts.shape[0], dtype=torch.int32, device=device
        ),
    }
    mux = shadow_model.multiplex_controller.get_state(
        num_valid_entries=1,
        device=device,
        dtype=torch.float32,
        random=False,
        object_ids=[1],
    )
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    opt = torch.optim.Adam(lora_params, lr=lr)
    t0 = __import__("time").time()
    total_loss = 0.0
    for _ in range(steps):
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.enable_grad():
                out = shadow_model.track_step(
                    frame_idx=0,
                    is_init_cond_frame=True,
                    backbone_features_interactive=features["interactive"],
                    backbone_features_propagation=features["sam2_backbone_out"],
                    image=image,
                    point_inputs=point_inputs,
                    mask_inputs=None,
                    gt_masks=None,
                    frames_to_add_correction_pt=[],
                    output_dict=output_dict,
                    num_frames=num_frames,
                    run_mem_encoder=False,
                    prev_sam_mask_logits=None,
                    multiplex_state=mux,
                    objects_to_interact=None,
                )
                pred = out["pred_masks"]
                if pred.dim() == 4:
                    pred = pred[0]
                if pred.dim() == 3 and pred.shape[0] == 1:
                    pred = pred[0]
                h, w = pred.shape[-2], pred.shape[-1]
                inside, outside = _box_grid(
                    np.asarray(human_box, dtype=float),
                    video_h,
                    video_w,
                    h,
                    w,
                    pred.device,
                )
                prob = torch.sigmoid(pred)
                loss_outside = (prob * outside).sum() / outside.sum().clamp(min=1)
                obj_logits = out.get("object_score_logits")
                loss_obj = torch.tensor(0.0, device=pred.device)
                if obj_logits is not None:
                    o = obj_logits.reshape(-1)
                    loss_obj = torch.nn.functional.binary_cross_entropy_with_logits(
                        o, torch.ones_like(o)
                    )
                loss = loss_outside + 0.1 * loss_obj
                loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        total_loss += float(loss.detach().cpu())
    elapsed = __import__("time").time() - t0
    return total_loss / max(1, steps), elapsed


def copy_lora_params(src_model, dst_model) -> int:
    """Copy LoRA A/B params from src (trained shadow) to dst (official pipeline)."""
    src = dict(src_model.named_parameters())
    dst = dict(dst_model.named_parameters())
    copied = 0
    for name, p in src.items():
        if "lora_a" not in name and "lora_b" not in name:
            continue
        if name in dst:
            dst[name].data.copy_(p.data)
            copied += 1
    return copied
