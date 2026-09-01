"""Probe raw detector candidates before/after online detector LoRA training."""

import copy
import json
import os
import numpy as np
import torch

from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
from sam3_intermot.adaptation.lora import inject_lora


ROOT = "."
CKPT = f"{ROOT}/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
VIDEO = "/path/to/dancetrack/train/dancetrack0075/img1"


def clone_find_input(fin, img_id):
    out = copy.copy(fin)
    for field in fin.__dataclass_fields__:
        v = getattr(out, field)
        if isinstance(v, torch.Tensor):
            setattr(out, field, v.clone())
        elif isinstance(v, list) and v and all(isinstance(x, torch.Tensor) for x in v):
            setattr(out, field, [x.clone() for x in v])
    out.img_ids = torch.tensor([img_id], dtype=torch.long, device="cuda")
    return out


def raw_candidates(runner, model):
    backend = runner.backend
    backend.start_video(VIDEO)
    backend._predictor.model.use_batched_grounding = False
    iw, ih = backend._frame_w, backend._frame_h
    backend._predictor.handle_request(dict(
        type="add_prompt", session_id=backend._session_id, frame_index=1,
        text="person", bounding_boxes=None, bounding_box_labels=None,
        clear_old_boxes=True,
    ))
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"].clear()
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
        type="propagate_in_video", session_id=backend._session_id,
        propagation_direction="forward", start_frame_index=1,
        max_frame_num_to_track=None,
    )
    for response in backend._predictor.handle_stream_request(request=req):
        f = int(response["frame_index"])
        if f >= 3:
            break
    backend.close()
    return raw


def train_adapter(model, runner, human_box):
    backend = runner.backend
    backend.start_video(VIDEO)
    for m in model.modules():
        for k in list(vars(m)):
            v = getattr(m, k, None)
            if k == "cache" and isinstance(v, dict):
                setattr(m, k, {})
            elif k == "coord_cache" and isinstance(v, dict):
                setattr(m, k, {})
            elif k == "compilable_cord_cache":
                setattr(m, k, None)
            if isinstance(v, dict):
                for kk, vv in list(v.items()):
                    if isinstance(vv, torch.Tensor) and torch.is_inference(vv):
                        v[kk] = vv.clone()
    iw, ih = backend._frame_w, backend._frame_h
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    fin = clone_find_input(state["input_batch"].find_inputs[1], 0)
    img_batch = state["input_batch"].img_batch
    img_t = (
        img_batch.tensors[1].unsqueeze(0).clone().to("cuda")
        if hasattr(img_batch, "tensors") else img_batch[1].unsqueeze(0).clone().to("cuda")
    )
    with torch.no_grad():
        text_out = model.detector.backbone.forward_text(["person"], device="cuda")
    backbone_out = {
        "img_batch_all_stages": img_t,
        "language_features": text_out["language_features"].clone(),
        "language_mask": text_out["language_mask"].clone(),
    }
    from sam3.model.geometry_encoders import Prompt
    empty_geo = Prompt(
        box_embeddings=torch.zeros(0, 1, 4, device="cuda"),
        box_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
        box_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
        point_embeddings=torch.zeros(0, 1, 2, device="cuda"),
        point_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
        point_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
    )
    with torch.no_grad():
        prompt, pmask, bo2 = model.detector._encode_prompt(backbone_out, fin, empty_geo)
        bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)
    def cl(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        if isinstance(x, (list, tuple)):
            return [cl(v) for v in x]
        if isinstance(x, dict):
            return {k: cl(v) for k, v in x.items()}
        return x
    enc2 = cl(enc)
    prompt2, pmask2 = prompt.clone(), pmask.clone()
    dec = model.detector.transformer.decoder
    dec.compilable_cord_cache = None
    dec.coord_cache = {}
    gt = np.asarray(human_box, dtype=float)
    cxcy = torch.tensor(
        [(gt[0]+gt[2])/2/iw, (gt[1]+gt[3])/2/ih, (gt[2]-gt[0])/iw, (gt[3]-gt[1])/ih],
        device="cuda",
    )
    params = [p for m in model.modules() for p in getattr(m, "lora_a", [])]
    lora_params = [
        p for m in model.modules()
        if hasattr(m, "lora_a") for p in (m.lora_a, m.lora_b)
    ]
    opt = torch.optim.AdamW(lora_params, lr=1e-3)
    for step in range(10):
        opt.zero_grad()
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = {"encoder_hidden_states": enc2["encoder_hidden_states"]}
            out2, _ = model.detector._run_decoder(
                pos_embed=enc2["pos_embed"], memory=enc2["encoder_hidden_states"],
                src_mask=enc2["padding_mask"], out=out2,
                prompt=prompt2, prompt_mask=pmask2, encoder_out=enc2,
            )
            boxes = out2["pred_boxes"][0].float()
            scores = out2["pred_logits"][0].float()[:, 0]
            bx1, by1 = boxes[:, 0]-boxes[:, 2]/2, boxes[:, 1]-boxes[:, 3]/2
            bx2, by2 = boxes[:, 0]+boxes[:, 2]/2, boxes[:, 1]+boxes[:, 3]/2
            gx1, gy1, gx2, gy2 = cxcy.tolist()
            ix1 = torch.maximum(bx1, torch.tensor(gx1, device="cuda"))
            iy1 = torch.maximum(by1, torch.tensor(gy1, device="cuda"))
            ix2 = torch.minimum(bx2, torch.tensor(gx2, device="cuda"))
            iy2 = torch.minimum(by2, torch.tensor(gy2, device="cuda"))
            inter = (ix2-ix1).clamp(min=0)*(iy2-iy1).clamp(min=0)
            ua = (bx2-bx1).clamp(min=0)*(by2-by1).clamp(min=0)+(gx2-gx1)*(gy2-gy1)-inter+1e-9
            tidx = int((inter/ua).argmax().item())
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scores[tidx].unsqueeze(0), torch.ones(1, device="cuda")
            ) + 0.1*torch.nn.functional.l1_loss(boxes[tidx], cxcy)
        loss.backward()
        opt.step()
    backend.close()


def main():
    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 0)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend.start_video(VIDEO)
    backend.close()
    model = backend._predictor.model
    model.use_batched_grounding = False
    inject_lora(model, (
        "detector.transformer.decoder.layers",
        "detector.transformer.decoder.bbox_embed",
        "detector.dot_prod_scoring",
    ), r=8, alpha=4.0)
    raw_before = raw_candidates(runner, model)
    # reset LoRA to init
    lora_params = [
        p for m in model.modules()
        if hasattr(m, "lora_a") for p in (m.lora_a, m.lora_b)
    ]
    init_lora = [p.detach().clone() for p in lora_params]
    def reset():
        with torch.no_grad():
            for p, v in zip(lora_params, init_lora):
                p.copy_(v)
    reset()
    train_adapter(model, runner, [629.0, 335.0, 713.0, 613.0])
    raw_after = raw_candidates(runner, model)
    diff = {}
    for f in sorted(raw_before, key=int):
        a = np.array(raw_before[f]["boxes"]); b = np.array(raw_after[f]["boxes"])
        sa = np.array(raw_before[f]["scores"]); sb = np.array(raw_after[f]["scores"])
        diff[int(f)] = {
            "max_box_diff": float(np.abs(a-b).max()) if a.shape == b.shape else -1,
            "max_score_diff": float(np.abs(sa-sb).max()) if sa.shape == sb.shape else -1,
            "n_above04_before": int((sa > 0.4).sum()),
            "n_above04_after": int((sb > 0.4).sum()),
        }
    print(json.dumps({"diff": diff}, ensure_ascii=False), flush=True)
    with open(f"{ROOT}/outputs/n13/lora_raw_probe.json", "w") as f:
        json.dump({"diff": diff}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
