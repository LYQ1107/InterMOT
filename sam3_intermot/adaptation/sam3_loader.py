"""Build the standalone SAM3.1 tracker with correctly remapped checkpoint keys."""

from pathlib import Path
from typing import Optional

import torch


ROOT = Path(".")
CKPT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"


def remap_checkpoint(ckpt: dict) -> dict:
    """Map full-predictor keys to standalone tracker-model keys.

    - ``tracker.model.*`` -> ``*`` (tracker state)
    - ``detector.backbone.vision_backbone.*`` -> ``backbone.*`` (visual encoder)
    Detector-only keys are dropped (not needed for single-object tracking).
    """
    out = {}
    dropped = 0
    for k, v in ckpt.items():
        parts = k.split(".")
        if parts[:2] == ["tracker", "model"]:
            out[".".join(parts[2:])] = v
        elif parts[:3] == ["detector", "backbone", "vision_backbone"]:
            out["backbone.vision_backbone." + ".".join(parts[3:])] = v
        else:
            dropped += 1
    return out, dropped


def build_tracker_model(
    checkpoint_path: Optional[Path] = None,
    multiplex_count: int = 16,
    device: str = "cuda",
):
    from sam3_intermot.backend import sam3_compat  # noqa: F401
    from sam3.model_builder import build_sam3_multiplex_video_model

    ckpt_path = Path(checkpoint_path) if checkpoint_path else CKPT
    model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=multiplex_count,
        use_fa3=False,
        use_rope_real=True,
        strict_state_dict_loading=False,
        device="cpu",
    )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state, dropped = remap_checkpoint(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    report = {
        "checkpoint": str(ckpt_path),
        "remapped_keys": len(state),
        "dropped_keys": dropped,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_sample": missing[:8],
        "unexpected_sample": unexpected[:8],
    }
    model.to(device=device)
    model.eval()
    # Runtime shim: TriHeadVisionOnly flattens the SAM3 image-head outputs into
    # top-level keys; the tracker forward only consumes "interactive" and
    # "sam2_backbone_out".  Drop the flattened keys so the demo path works.
    _orig_backbone_fi = model.backbone.forward_image

    def _backbone_forward_image(samples, **kwargs):
        out = _orig_backbone_fi(samples, **kwargs)
        for k in ("vision_features", "vision_mask", "vision_pos_enc", "backbone_fpn"):
            out.pop(k, None)
        return out

    model.backbone.forward_image = _backbone_forward_image
    return model, report
