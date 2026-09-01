"""Identity-specificity check: two human writes (q_A, q_B) matched against
future ROIs of A and B.  Expect q_A >> q_B on A's ROI and vice versa."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch


ROOT = Path(".")
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="outputs/n14/models/human_write_encoder_f0_v3.pt")
    ap.add_argument("--seq", default="dancetrack0001")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--gids", default="0,1")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.persistent_identity import (
        HumanWriteEncoder,
        SlotHeadAdapter,
        roi_pool_feature,
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
    hidden = int(ck["args"].get("hidden", 512))
    encoder = HumanWriteEncoder(d_model=d_model, hidden=hidden).cuda().eval()
    adapter = SlotHeadAdapter(
        d_model=d_model, hidden=hidden // 4
    ).cuda().eval()
    encoder.load_state_dict(ck["encoder_state"])
    adapter.load_state_dict(ck["adapter_state"])

    ds = DanceTrackDataset(str(DT), sequences=None, split="train")
    gt = ds.load_gt(args.seq)
    gids = [int(x) for x in args.gids.split(",")]
    t = args.frame

    backend.start_video(str(DT / "train" / args.seq / "img1"))
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    ib = state["input_batch"]

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
        return enc

    def roi_of(enc, f, gid):
        entry = gt[f]
        box = np.asarray(entry.boxes[entry.gt_ids.index(gid)], dtype=float)
        x1, y1, x2, y2 = box
        bn = np.asarray([x1 / 1920, y1 / 1080, x2 / 1920, y2 / 1080])
        return roi_pool_feature(
            enc["encoder_hidden_states"], enc, bn
        )

    enc_t = enc_for(t)
    enc_f = enc_for(t + 2)
    qs = {
        g: encoder(roi_of(enc_t, t, g).float()).to(torch.float32)
        for g in gids
    }
    rois = {g: roi_of(enc_f, t + 2, g).float() for g in gids}
    refs = {}
    for g in gids:
        entry = gt[t + 2]
        box = np.asarray(entry.boxes[entry.gt_ids.index(g)], dtype=float)
        x1, y1, x2, y2 = box
        refs[g] = torch.as_tensor(
            [
                (x1 + x2) / 2 / 1920, (y1 + y2) / 2 / 1080,
                (x2 - x1) / 1920, (y2 - y1) / 1080,
            ],
            dtype=torch.float32, device="cuda",
        )
    with torch.no_grad():
        print("query_gid | roi_gid | score | sigmoid")
        for qg in gids:
            for rg in gids:
                _, s = adapter(
                    qs[qg].unsqueeze(0), rois[rg].unsqueeze(0),
                    rois[rg].unsqueeze(0), refs[rg].unsqueeze(0),
                )
                print(
                    f"  q{qg}  |  r{rg}  | {s.item(): .3f} | "
                    f"{torch.sigmoid(s).item():.3f}"
                )
    runner.close()


if __name__ == "__main__":
    main()
