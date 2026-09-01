"""Test decoder-only grad forward with synthetic (fully normal) inputs."""

import os
import torch

from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner


ROOT = "."
CKPT = f"{ROOT}/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
VIDEO = "/path/to/dancetrack/train/dancetrack0075/img1"


def main():
    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 0)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend.start_video(VIDEO)
    model = backend._predictor.model
    d = model.detector
    mem = torch.randn(5184, 1, 256, device="cuda")
    pos = torch.randn(5184, 1, 256, device="cuda")
    pad = torch.zeros(1, 5184, device="cuda", dtype=torch.bool)
    prompt = torch.randn(33, 1, 256, device="cuda")
    pmask = torch.zeros(1, 33, device="cuda", dtype=torch.bool)
    enc = {
        "level_start_index": torch.zeros(1, dtype=torch.long, device="cuda"),
        "spatial_shapes": torch.tensor([[36, 36]], dtype=torch.long, device="cuda"),
        "valid_ratios": torch.ones(1, 1, 2, device="cuda"),
        "vis_feat_sizes": [(36, 36)],
        "prompt_before_enc": prompt,
        "prompt_after_enc": prompt,
        "prompt_mask": pmask,
    }
    out = {"encoder_hidden_states": mem}
    try:
        with torch.enable_grad():
            out, hs = d._run_decoder(
                pos_embed=pos, memory=mem, src_mask=pad, out=out,
                prompt=prompt, prompt_mask=pmask, encoder_out=enc,
            )
        print("SYNTH_OK", sorted(out.keys()), flush=True)
    except RuntimeError as e:
        print("SYNTH_FAIL", e, flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
