"""Probe raw detector candidates (pre-association) per frame with and
without injected per-frame box prompts."""

import json
import numpy as np
import torch

from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.detection_query.prompt_replay import (
    set_frame_geometric_prompt,
    invalidate_detector_prefetch,
)


ROOT = "."
CKPT = f"{ROOT}/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
VIDEO = "/path/to/dancetrack/train/dancetrack0075/img1"


def run_branch(runner, prompt_frames):
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
    model = backend._predictor.model

    raw = {}
    orig = model.run_backbone_and_detection

    def wrap(frame_idx, num_frames, input_batch, geometric_prompt,
             feature_cache, reverse, use_batched_grounding=False,
             batched_grounding_batch_size=16):
        det_out, pos = orig(
            frame_idx, num_frames, input_batch, geometric_prompt,
            feature_cache, reverse, use_batched_grounding,
            batched_grounding_batch_size,
        )
        if det_out is not None:
            keys = sorted(det_out.keys())
            box_key = "bbox" if "bbox" in det_out else (
                "pred_boxes_xyxy" if "pred_boxes_xyxy" in det_out else "pred_boxes"
            )
            boxes = det_out[box_key][0].detach().float().cpu().numpy().tolist()
            score_key = "scores" if "scores" in det_out else "pred_logits"
            scores = det_out[score_key][0].detach().float().cpu().numpy().tolist()
            raw[int(frame_idx)] = {
                "n": len(boxes),
                "boxes": boxes,
                "scores": [float(x) if isinstance(x, (int, float)) else x for x in scores],
                "keys": keys,
                "box_key": box_key,
                "score_key": score_key,
            }
        else:
            raw[int(frame_idx)] = {"n": -1, "keys": []}
        return det_out, pos

    model.run_backbone_and_detection = wrap

    for f in prompt_frames:
        set_frame_geometric_prompt(runner, f, np.asarray(prompt_frames[f]))
    req = dict(
        type="propagate_in_video",
        session_id=backend._session_id,
        propagation_direction="forward",
        start_frame_index=3,
        max_frame_num_to_track=None,
    )
    for response in backend._predictor.handle_stream_request(request=req):
        f = int(response["frame_index"])
        if f >= 5:
            break
        invalidate_detector_prefetch(runner, f)
        nf = f + 1
        if nf in prompt_frames:
            set_frame_geometric_prompt(runner, nf, np.asarray(prompt_frames[nf]))
        elif nf <= 5:
            set_frame_geometric_prompt(runner, nf, None)
    backend.close()
    return raw


def main():
    import os
    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 3)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    gt = DanceTrackDataset(
        "/path/to/dancetrack", sequences=[], split="train"
    ).load_gt("dancetrack0075")
    gid = 4
    oracle = {}
    for f in (3, 4, 5):
        entry = gt.get(f)
        oracle[f] = entry.boxes[entry.gt_ids.index(gid)]
    raw_a = run_branch(runner, {})
    raw_b = run_branch(runner, oracle)
    print(json.dumps({"one_shot_raw": raw_a, "oracle_raw": raw_b},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
