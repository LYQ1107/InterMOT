#!/usr/bin/env python
"""N12 SAM3.1 trainable-surface audit.

Builds the standalone multiplex tracker model (backbone kept) from the pinned
checkpoint and records the module tree / parameter counts / candidate LoRA
targets.  CPU-side audit plus one GPU load; no heavy inference.
"""

import json
import time
from pathlib import Path

import torch

ROOT = Path(".")
OUT = ROOT / "outputs/n12"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"


def count_params(mod: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in mod.parameters())
    trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def main() -> None:
    from sam3_intermot.backend import sam3_compat  # noqa: F401
    from sam3.model_builder import build_sam3_multiplex_video_model

    t0 = time.time()
    model = build_sam3_multiplex_video_model(
        checkpoint_path=str(CKPT),
        load_from_HF=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        strict_state_dict_loading=False,
        device="cuda",
    )
    model.eval()
    load_s = round(time.time() - t0, 1)

    tree = {}
    for name, child in model.named_children():
        tree[name] = {
            "class": child.__class__.__name__,
            "params": count_params(child),
        }
    # candidate surfaces: expand one level for the important modules
    for top in ("backbone", "maskmem_backbone", "sam_mask_decoder",
                "multiplex_controller", "transformer"):
        child = getattr(model, top, None)
        if child is None:
            continue
        subs = {}
        for n, c in child.named_children():
            subs[n] = {"class": c.__class__.__name__, "params": count_params(c)}
        tree[f"{top}__sub"] = subs

    mem = {}
    if torch.cuda.is_available():
        mem = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
            "device": torch.cuda.get_device_name(0),
        }

    audit = {
        "stage": "N12-SAM3-ADAPTER-AUDIT",
        "date": "2026-08-09",
        "pinned_commit": "4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b",
        "checkpoint": str(CKPT),
        "model_class": model.__class__.__name__,
        "load_seconds": load_s,
        "module_tree": tree,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "gpu_memory": mem,
        "candidate_surfaces": [
            {
                "name": "maskmem_backbone",
                "class": model.maskmem_backbone.__class__.__name__,
                "params": count_params(model.maskmem_backbone),
                "role": "memory encoder (per-frame, target-conditioned mask memory)",
            },
            {
                "name": "sam_mask_decoder",
                "class": model.sam_mask_decoder.__class__.__name__,
                "params": count_params(model.sam_mask_decoder),
                "role": "mask decoder (output mask / object score)",
            },
            {
                "name": "backbone",
                "class": model.backbone.__class__.__name__,
                "params": count_params(model.backbone),
                "role": "image encoder (global; highest interference)",
            },
        ],
    }
    (OUT / "sam3_adapter_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"load_s": load_s, "total_params": audit["total_params"],
                      "tree_top": {k: v["params"]["total"] for k, v in tree.items()
                                   if isinstance(v, dict) and "params" in v}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
