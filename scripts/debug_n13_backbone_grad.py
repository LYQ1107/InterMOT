"""Isolate: does a grad-enabled vision-backbone forward fail with normal input?"""

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
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    img_batch = state["input_batch"].img_batch
    img_t = (
        img_batch.tensors[1].unsqueeze(0).clone().to("cuda")
        if hasattr(img_batch, "tensors")
        else img_batch[1].unsqueeze(0).clone().to("cuda")
    )
    print("img clone inf?", torch.is_inference(img_t), flush=True)
    model = backend._predictor.model
    try:
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.detector.backbone.forward_image(img_t)
        print("BACKBONE_OK", sorted(out.keys())[:8], flush=True)
    except RuntimeError as e:
        print("BACKBONE_FAIL", e, flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
