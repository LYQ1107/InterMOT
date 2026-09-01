"""Runtime probe: does the detector encoder see the injected per-frame prompt?"""

import json
import numpy as np
import torch

from sam3_intermot.adaptation.cfa_backend_runner import (
    CFABackendRunner,
    parse_raw_outputs,
)
from sam3_intermot.detection_query.prompt_replay import set_frame_geometric_prompt
from sam3_intermot.detection_query.prompt_replay import invalidate_detector_prefetch


ROOT = "."
CKPT = f"{ROOT}/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
VIDEO = (
    "/path/to/dancetrack/train/dancetrack0075/img1"
)


def main():
    import os
    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 3)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend.start_video(VIDEO)
    backend._predictor.model.use_batched_grounding = False
    iw, ih = backend._frame_w, backend._frame_h
    box = [629.0, 335.0, 713.0, 613.0]
    x1, y1, x2, y2 = box
    req_prompt = dict(
        type="add_prompt",
        session_id=backend._session_id,
        frame_index=1,
        text="person",
        bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]],
        bounding_box_labels=[1],
        clear_old_boxes=True,
    )
    backend._predictor.handle_request(req_prompt)
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"].clear()

    # Inject oracle GT box for frame 3 and a far-away box for frame 4.
    gt3 = [629.0, 343.0, 719.0, 620.0]
    far4 = [10.0, 10.0, 60.0, 100.0]
    set_frame_geometric_prompt(runner, 3, np.asarray(gt3))
    set_frame_geometric_prompt(runner, 4, np.asarray(far4))
    print("prompt3 set:", state["per_frame_geometric_prompt"][3] is not None,
          "prompt4 set:", state["per_frame_geometric_prompt"][4] is not None,
          flush=True)
    p3 = state["per_frame_geometric_prompt"][3]
    p4 = state["per_frame_geometric_prompt"][4]
    print("p3 boxes:", p3.box_embeddings.detach().cpu().tolist(),
          "p4 boxes:", p4.box_embeddings.detach().cpu().tolist(), flush=True)

    model = backend._predictor.model
    print("world_size", model.detector.world_size, "rank", model.detector.rank,
          "det_ws", model.detector.world_size, flush=True)
    orig = model.detector._encode_prompt
    seen = {}

    def probe(backbone_out, find_input, geometric_prompt, *a, **kw):
        img_ids = find_input.img_ids.detach().cpu().tolist()
        img = int(min(img_ids))
        nbox = 0
        emb = None
        if geometric_prompt is not None and getattr(geometric_prompt, "box_embeddings", None) is not None:
            nbox = int(geometric_prompt.box_embeddings.shape[0])
            if nbox:
                emb = geometric_prompt.box_embeddings.detach().float().cpu().numpy()
        seen[img] = {
            "img_ids": img_ids,
            "nbox": nbox,
            "emb_first": None if emb is None else emb[0, 0].tolist(),
        }
        return orig(backbone_out, find_input, geometric_prompt, *a, **kw)

    model.detector._encode_prompt = probe
    print("buf keys before:", sorted(
        state["feature_cache"].get("multigpu_buffer", {}).keys()), flush=True)
    invalidate_detector_prefetch(runner, 2)
    print("buf keys after invalidate(2):", sorted(
        state["feature_cache"].get("multigpu_buffer", {}).keys()), flush=True)
    req = dict(
        type="propagate_in_video",
        session_id=backend._session_id,
        propagation_direction="forward",
        start_frame_index=3,
        max_frame_num_to_track=None,
    )
    out = {}
    for response in backend._predictor.handle_stream_request(request=req):
        f = int(response["frame_index"])
        cands = parse_raw_outputs(response, frame_size=(iw, ih))
        out[f] = len(cands)
        if f >= 5:
            break
    print(json.dumps({"seen_nboxes": seen, "output_counts": out}, ensure_ascii=False),
          flush=True)
    backend.close()


if __name__ == "__main__":
    main()
