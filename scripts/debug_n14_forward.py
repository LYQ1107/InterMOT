"""Forward a trained N14 query head on manifest samples and print slot
logits/boxes, to verify the loss variable matches the adapter output."""

import csv
from pathlib import Path

import numpy as np
import torch


ROOT = Path(".")
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="outputs/n14/models/hwe_f0_simscore.pt")
    ap.add_argument("--rows", default="0,1")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

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

    ck = torch.load(ROOT / args.model, map_location="cuda", weights_only=False)
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
    encoder.load_state_dict(ck["encoder_state"])
    adapter.load_state_dict(ck["adapter_state"])
    encoder.eval()
    adapter.eval()

    rows = list(csv.DictReader(open(ROOT / "outputs/n14/episode_manifest.csv")))
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

    cache = {}

    def enc_for(seq, f):
        key = (seq, f)
        if key in cache:
            return cache[key]
        if not cache:
            backend.start_video(str(DT / "train" / seq / "img1"))
        state = backend._predictor._all_inference_states[
            backend._session_id
        ]["state"]
        ib = state["input_batch"]
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

        feat = {"enc": cl(enc), "prompt": prompt.clone(), "pmask": pmask.clone()}
        cache[key] = feat
        return feat

    for idx in (int(i) for i in args.rows.split(",")):
        r = rows[idx]
        seq = r["sequence"]
        t, f = int(r["human_frame"]), int(r["future_frame"])
        vis = int(r["target_visible"])
        hb = np.asarray(
            [r["human_box_x1"], r["human_box_y1"],
             r["human_box_x2"], r["human_box_y2"]], dtype=float
        )
        ft = enc_for(seq, t)
        ff = enc_for(seq, f)
        box_norm = np.asarray(
            [hb[0] / 1920, hb[1] / 1080, hb[2] / 1920, hb[3] / 1080]
        )
        roi = roi_pool_feature(
            ft["enc"]["encoder_hidden_states"], ft["enc"], box_norm
        )
        with torch.no_grad():
            q = encoder(roi.float()).to(torch.float32)
            ref = torch.as_tensor(
                [
                    (hb[0] + hb[2]) / 2 / 1920,
                    (hb[1] + hb[3]) / 2 / 1080,
                    (hb[2] - hb[0]) / 1920,
                    (hb[3] - hb[1]) / 1080,
                ],
                dtype=torch.float32, device="cuda",
            )
            tgt = build_tgt_with_queries(image, [q], [199], bs=1)
            refs = build_ref_boxes_with_queries(image, [ref], [199], bs=1)
            out2 = {"encoder_hidden_states": ff["enc"]["encoder_hidden_states"]}
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
            logit = out2["pred_logits"][0, 199].float().item()
            box = out2["pred_boxes"][0, 199].float().cpu().numpy()
            cx, cy, w, h = ref.unbind(-1)
            roi_norm = torch.stack(
                [
                    (cx - w / 2).clamp(0.0, 1.0),
                    (cy - h / 2).clamp(0.0, 1.0),
                    (cx + w / 2).clamp(0.0, 1.0),
                    (cy + h / 2).clamp(0.0, 1.0),
                ],
                dim=-1,
            ).detach().cpu().numpy()
            from sam3_intermot.persistent_identity import roi_pool_feature

            roi_feat = roi_pool_feature(
                ff["enc"]["encoder_hidden_states"],
                ff["enc"],
                roi_norm[0],
            ).unsqueeze(0)
            dbox, dscore = adapter(
                q.unsqueeze(0), roi_feat, roi_feat, ref.unsqueeze(0)
            )
            print(
                f"row{idx} {seq} t{t} f{f} vis={vis} logit={logit:.4f} "
                f"sig={torch.sigmoid(torch.tensor(logit)).item():.4f} "
                f"dscore={dscore.item():.4f} box={np.round(box, 4).tolist()}",
                flush=True,
            )
    runner.close()


if __name__ == "__main__":
    main()
