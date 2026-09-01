"""Isolated autograd check for the dynamic-query -> decoder -> slot head path."""

import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(".")
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")


def main():
    torch.cuda.set_device(0)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.persistent_identity import (
        HumanWriteEncoder,
        SlotHeadAdapter,
        build_ref_boxes_with_queries,
        build_tgt_with_queries,
        roi_pool_feature,
        run_decoder_with_tgt,
    )
    from sam3.model.geometry_encoders import Prompt

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    image = model.detector
    d_model = image.transformer.decoder.query_embed.weight.shape[1]
    encoder = HumanWriteEncoder(d_model=d_model).cuda()
    adapter = SlotHeadAdapter(d_model=d_model).cuda()

    backend.start_video(str(DT / "train" / "dancetrack0001" / "img1"))
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    ib = state["input_batch"]

    import copy

    def clone_fin(fin):
        out = copy.copy(fin)
        for field in fin.__dataclass_fields__:
            v = getattr(out, field)
            if isinstance(v, torch.Tensor):
                setattr(out, field, v.clone())
            elif isinstance(v, list) and v and all(
                isinstance(x, torch.Tensor) for x in v
            ):
                setattr(out, field, [x.clone() for x in v])
        out.img_ids = torch.tensor([0], dtype=torch.long, device="cuda")
        return out

    def enc_for(f):
        fin = clone_fin(ib.find_inputs[f])
        img_t = ib.img_batch.tensors[f].unsqueeze(0).clone().to("cuda")
        with torch.no_grad():
            text = model.detector.backbone.forward_text(["person"], device="cuda")
            bo = {
                "img_batch_all_stages": img_t,
                "language_features": text["language_features"].clone(),
                "language_mask": text["language_mask"].clone(),
            }
            geo = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device="cuda"),
                box_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                box_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
                point_embeddings=torch.zeros(0, 1, 2, device="cuda"),
                point_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                point_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
            )
            prompt, pmask, bo2 = model.detector._encode_prompt(bo, fin, geo)
            bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)

        def cl(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            if isinstance(x, dict):
                return {k: cl(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [cl(v) for v in x]
            return x

        return {
            "enc": cl(enc),
            "prompt": prompt.clone(),
            "pmask": pmask.clone(),
        }

    ft = enc_for(1)
    ff = enc_for(2)
    box = np.asarray([229.0, 378.0, 346.0, 570.0])
    box_norm = np.asarray(
        [box[0] / 1920, box[1] / 1080, box[2] / 1920, box[3] / 1080]
    )
    roi = roi_pool_feature(
        ft["enc"]["encoder_hidden_states"], ft["enc"], box_norm
    )
    q = encoder(roi.float()).to(torch.float32)
    ref = torch.as_tensor(
        [0.1497, 0.4389, 0.0609, 0.1778], dtype=torch.float32, device="cuda"
    )
    tgt = build_tgt_with_queries(image, [q], [199], bs=1)
    refs = build_ref_boxes_with_queries(image, [ref], [199], bs=1)
    out2 = {"encoder_hidden_states": ff["enc"]["encoder_hidden_states"]}
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out2, hs = run_decoder_with_tgt(
            image, tgt,
            pos_embed=ff["enc"]["pos_embed"],
            memory=ff["enc"]["encoder_hidden_states"],
            src_mask=ff["enc"]["padding_mask"],
            out=out2,
            prompt=ff["prompt"],
            prompt_mask=ff["pmask"],
            encoder_out=ff["enc"],
            reference_boxes=refs,
            slot_adapter=adapter,
            active_slots=[199],
            dyn_queries=[q],
        )
        loss = (
            out2["pred_logits"][0, 199].float().pow(2)
            + out2["pred_boxes"][0, 199].float().pow(2).sum()
        )
    print("hs", tuple(hs.shape), "hs[-1] reqgrad", hs[-1].requires_grad,
          flush=True)
    print("tgt reqgrad", tgt.requires_grad, "q reqgrad", q.requires_grad,
          flush=True)
    gq = torch.autograd.grad(loss, q, retain_graph=True, allow_unused=True)[0]
    print("autograd.grad(loss,q) norm", None if gq is None else gq.norm().item(),
          flush=True)
    ghs = torch.autograd.grad(
        hs[-1].sum(), tgt, retain_graph=True, allow_unused=True
    )[0]
    gt_row = torch.autograd.grad(
        hs[-1].sum(), tgt, retain_graph=True, allow_unused=True
    )[0]
    gl_tgt = torch.autograd.grad(
        loss, tgt, retain_graph=True, allow_unused=True
    )[0]
    gl_hs = torch.autograd.grad(
        loss, hs[-1], retain_graph=True, allow_unused=True
    )[0]
    print(
        "autograd.grad(hs[-1],tgt) norm",
        None if ghs is None else ghs.norm().item(),
        "tgt.grad_fn", tgt.grad_fn,
        "hs[-1].grad_fn", hs[-1].grad_fn,
        "row199", None if gt_row is None else gt_row[199].norm().item(),
        "loss->tgt", None if gl_tgt is None else gl_tgt.norm().item(),
        "loss->hs[-1]", None if gl_hs is None else gl_hs.norm().item(),
        flush=True,
    )
    loss.backward()
    print("q.grad norm", q.grad.norm().item() if q.grad is not None else None,
          flush=True)
    print("encoder grad sum", sum(
        float(p.grad.abs().sum().item())
        for p in encoder.parameters() if p.grad is not None
    ), flush=True)
    print("adapter grad sum", sum(
        float(p.grad.abs().sum().item())
        for p in adapter.parameters() if p.grad is not None
    ), flush=True)
    runner.close()


if __name__ == "__main__":
    main()
