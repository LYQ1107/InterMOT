"""Debug the N14.6 shadow candidate computation on one event."""

import copy
from pathlib import Path

import numpy as np
import torch


ROOT = Path(".")
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")


def main():
    torch.cuda.set_device(0)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.persistent_identity import (
        HumanWriteEncoder,
        SlotHeadAdapter,
        roi_pool_feature,
    )
    from sam3.model.geometry_encoders import Prompt

    ck = torch.load(
        ROOT / "outputs/n14/models/human_write_encoder_f0_v10.pt",
        map_location="cuda", weights_only=False,
    )
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.detector.transformer.decoder.query_embed.weight.shape[1]
    hidden = int(ck["args"]["hidden"])
    encoder = HumanWriteEncoder(d_model=d_model, hidden=hidden).cuda().eval()
    adapter = SlotHeadAdapter(d_model=d_model, hidden=hidden // 4).cuda().eval()
    encoder.load_state_dict(ck["encoder_state"])
    adapter.load_state_dict(ck["adapter_state"])

    ds = DanceTrackDataset(str(DT), sequences=None, split="train")
    seq, t, gid = "dancetrack0074", 6, 1
    gt = ds.load_gt(seq)
    hb = np.asarray(gt[t].boxes[gt[t].gt_ids.index(gid)], dtype=float)
    backend.start_video(str(DT / "train" / seq / "img1"))
    iw, ih = backend._frame_w, backend._frame_h
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

    box_norm = np.asarray(
        [hb[0] / iw, hb[1] / ih, hb[2] / iw, hb[3] / ih], dtype=float
    )
    enc_t = enc_for(t)
    roi_t = roi_pool_feature(enc_t["encoder_hidden_states"], enc_t, box_norm)
    print("roi_t shape", tuple(roi_t.shape), "norm", float(roi_t.norm().item()))
    q = encoder(roi_t.float()).to(torch.float32)
    print("q shape", tuple(q.shape), "norm", float(q.norm().item()),
          "minmax", float(q.min().item()), float(q.max().item()))
    ref = torch.as_tensor(
        [
            (hb[0] + hb[2]) / 2 / iw, (hb[1] + hb[3]) / 2 / ih,
            (hb[2] - hb[0]) / iw, (hb[3] - hb[1]) / ih,
        ],
        dtype=torch.float32, device="cuda",
    )
    print("ref", ref.tolist())
    for f in (7, 8, 10):
        enc_f = enc_for(f)
        roi_f = roi_pool_feature(
            enc_f["encoder_hidden_states"], enc_f, box_norm
        )
        print("roi_f", f, "norm", float(roi_f.norm().item()))
        with torch.no_grad():
            dbox, dscore = adapter(
                q.unsqueeze(0), roi_f.unsqueeze(0),
                roi_f.unsqueeze(0), ref.unsqueeze(0),
            )
        print("f", f, "dbox", dbox.detach().cpu().tolist(),
              "logit", float(dscore.item()),
              "score", float(torch.sigmoid(dscore).item()))
    runner.close()


if __name__ == "__main__":
    main()
