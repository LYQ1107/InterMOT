"""Probe: at the prompted frame and future frames, do box prompts change raw
detector candidates when the text prompt is present?"""

import json
import os
import numpy as np
import torch

from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
from sam3_intermot.detection_query.prompt_replay import (
    set_frame_geometric_prompt,
    invalidate_detector_prefetch,
)


ROOT = "."
CKPT = f"{ROOT}/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
VIDEO = "/path/to/dancetrack/train/dancetrack0075/img1"


def run_branch(runner, video, human_box):
    backend = runner._ensure_backend()
    backend.start_video(video)
    backend._predictor.model.use_batched_grounding = False
    iw, ih = backend._frame_w, backend._frame_h
    req_prompt = dict(
        type="add_prompt",
        session_id=backend._session_id,
        frame_index=1,
        text="person",
        bounding_boxes=None,
        bounding_box_labels=None,
        clear_old_boxes=True,
    )
    if human_box is not None:
        x1, y1, x2, y2 = human_box
        req_prompt["bounding_boxes"] = [
            [x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]
        ]
        req_prompt["bounding_box_labels"] = [1]
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
            boxes = det_out["bbox"][0].detach().float().cpu().numpy().tolist()
            scores = det_out["scores"][0].detach().float().cpu().numpy().tolist()
            raw[int(frame_idx)] = {"boxes": boxes, "scores": scores}
        return det_out, pos

    model.run_backbone_and_detection = wrap
    req = dict(
        type="propagate_in_video",
        session_id=backend._session_id,
        propagation_direction="forward",
        start_frame_index=1,
        max_frame_num_to_track=None,
    )
    for response in backend._predictor.handle_stream_request(request=req):
        f = int(response["frame_index"])
        if f >= 3:
            break
        invalidate_detector_prefetch(runner, f)
        set_frame_geometric_prompt(runner, f + 1, None)
    backend.close()
    return raw


def main():
    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 3)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    events = [
        ("dancetrack0075", [629.0, 335.0, 713.0, 613.0]),
        ("dancetrack0082", None),
        ("dancetrack0096", None),
    ]
    summary = {}
    for seq, box in events:
        video = f"/path/to/dancetrack/train/{seq}/img1"
        if box is None:
            import json as _json
            evs = [
                _json.loads(l) for l in open(
                    f"./outputs/n10/real/human_b8/{seq}/interaction_events.jsonl"
                )
            ]
            box = next(e["gt_box"] for e in evs
                       if e.get("accepted") and e.get("event_type") == "TRUE_MISS_NEW")
        raw_text = run_branch(runner, video, None)
        raw_box = run_branch(runner, video, box)
        diff = {}
        for f in sorted(raw_text, key=int):
            a = np.array(raw_text[f]["boxes"])
            b = np.array(raw_box[f]["boxes"])
            sa = np.array(raw_text[f]["scores"])
            sb = np.array(raw_box[f]["scores"])
            diff[int(f)] = {
                "max_box_diff": float(np.abs(a - b).max()) if a.shape == b.shape else -1,
                "max_score_diff": float(np.abs(sa - sb).max()) if sa.shape == sb.shape else -1,
            }
        summary[seq] = diff
        print(json.dumps({seq: diff}, ensure_ascii=False), flush=True)
    with open("./outputs/n13/text_vs_box_probe.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
