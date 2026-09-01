#!/usr/bin/env python
"""N14 Dynamic Query Causal Smoke.

Question: can a per-identity dynamic embedding injected into the detector
decoder query table (reserved slot of ``query_embed``) change raw detector
candidates and official output rows?

Branches per event:
  O  official pipeline, no patch
  I  patched decoder, slot replaced with its own value (identity control)
  R  patched decoder, slot replaced with a fixed random vector
  C  patched decoder, slot replaced with ROI-pooled detector feature written
     at the human frame (persistent human-frame query)

Expected: O and I byte-identical; R/C different from O if the injection is on
the causal path.  Empty bank => modified pipeline is byte-identical to B0.
"""

import argparse
import csv
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(".")
OUT = ROOT / "outputs/n14"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")

EVENTS = [
    # (sequence, human_frame, gid, gt_box)
    ("dancetrack0074", 6, 1, [692.0, 512.0, 864.0, 945.0]),
    ("dancetrack0083", 1, 1, [257.0, 341.0, 324.0, 527.0]),
    ("dancetrack0096", 1, 2, [484.0, 670.0, 579.0, 848.0]),
]
BRANCHES = ("O", "I", "R", "C")
FUTURE_FRAMES = 3
SLOT = -1  # reserved slot index in query_embed (200 queries)


def blob_hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype="<f4").tobytes())
    return h.hexdigest()[:16]


def row_hash(rows) -> str:
    h = hashlib.sha256()
    for oid, box in rows:
        h.update(np.asarray(oid, dtype="<i8").tobytes())
        h.update(np.asarray(box, dtype="<f4").tobytes())
    return h.hexdigest()[:16]


class DecoderPatch:
    """Instance-level replacement of Sam3Image._run_decoder with dynamic tgt."""

    def __init__(self, image, mode: str, slot: int, human_box_norm=None,
                 dyn_query=None):
        self.image = image
        self.mode = mode
        self.slot = slot
        self.human_box_norm = human_box_norm
        self.dyn_query = dyn_query
        self.saved_query = None

    def _roi_query(self, memory, encoder_out):
        mem = memory[:, 0, :]  # bs=1
        ls = encoder_out["level_start_index"].cpu().numpy()
        ss = encoder_out["spatial_shapes"].cpu().numpy()
        h0, w0 = int(ss[0, 0]), int(ss[0, 1])
        end0 = int(ls[1]) if len(ls) > 1 else mem.shape[0]
        feats = mem[:end0]
        x0, y0, x1, y1 = self.human_box_norm
        yy = torch.arange(h0, device=mem.device)
        xx = torch.arange(w0, device=mem.device)
        ym = (yy[:, None] >= y0 * h0) & (yy[:, None] <= y1 * h0)
        xm = (xx[None, :] >= x0 * w0) & (xx[None, :] <= x1 * w0)
        sel = (ym & xm).reshape(-1)
        if int(sel.sum().item()) == 0:
            return None
        feat = feats[sel].mean(0)
        qe = self.image.transformer.decoder.query_embed.weight
        scale = qe.detach().norm(dim=-1).mean()
        return feat * (scale / (feat.norm() + 1e-9))

    def install(self):
        orig = self.image.__class__._run_decoder

        def patched(self, pos_embed, memory, src_mask, out, prompt, prompt_mask,
                    encoder_out):
            bs = memory.shape[1]
            qe = self.transformer.decoder.query_embed.weight
            tgt = qe.unsqueeze(1).repeat(1, bs, 1)
            if self._n14.mode == "save_roi":
                self._n14.saved_query = self._n14._roi_query(memory, encoder_out)
            elif self._n14.mode == "identity":
                tgt = tgt.clone()
            elif self._n14.mode in ("random", "roi"):
                tgt = tgt.clone()
                q = self._n14.dyn_query
                if self._n14.mode == "roi":
                    q = self._n14.saved_query
                if q is not None:
                    tgt[self._n14.slot] = q.unsqueeze(0).repeat(bs, 1)
            apply_dac = self.transformer.decoder.dac and self.training
            hs, reference_boxes, dec_presence_out, dec_presence_feats = (
                self.transformer.decoder(
                    tgt=tgt,
                    memory=memory,
                    memory_key_padding_mask=src_mask,
                    pos=pos_embed,
                    reference_boxes=None,
                    level_start_index=encoder_out["level_start_index"],
                    spatial_shapes=encoder_out["spatial_shapes"],
                    valid_ratios=encoder_out["valid_ratios"],
                    tgt_mask=None,
                    memory_text=prompt,
                    text_attention_mask=prompt_mask,
                    apply_dac=apply_dac,
                )
            )
            hs = hs.transpose(1, 2)
            reference_boxes = reference_boxes.transpose(1, 2)
            if dec_presence_out is not None:
                dec_presence_out = dec_presence_out.transpose(1, 2)
            out["presence_feats"] = dec_presence_feats
            self._update_scores_and_boxes(
                out, hs, reference_boxes, prompt, prompt_mask,
                dec_presence_out=dec_presence_out,
            )
            return out, hs

        self.image._n14 = self
        bound = types.MethodType(patched, self.image)
        self.image._run_decoder = bound
        return bound

    def set_mode(self, mode: str):
        self.mode = mode


def run_branch(runner, seq, human_frame, gid, box, branch, patch_state):
    """Run one branch and return per-frame raw + official records."""
    from sam3_intermot.adaptation.cfa_backend_runner import parse_raw_outputs
    from sam3_intermot.detection_query.prompt_replay import invalidate_detector_prefetch

    backend = runner._ensure_backend()
    video = str(DT / "train" / seq / "img1")
    backend.start_video(video)
    model = backend._predictor.model
    model.use_batched_grounding = False
    iw, ih = backend._frame_w, backend._frame_h

    x1, y1, x2, y2 = (float(v) for v in box)
    req_prompt = dict(
        type="add_prompt",
        session_id=backend._session_id,
        frame_index=human_frame,
        text="person",
        bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]],
        bounding_box_labels=[1],
        clear_old_boxes=True,
    )

    records = []
    raw_by_frame = {}

    orig_rbd = model.run_backbone_and_detection

    def wrap(frame_idx, num_frames, input_batch, geometric_prompt,
             feature_cache, reverse, use_batched_grounding=False,
             batched_grounding_batch_size=16):
        det_out, pos = orig_rbd(
            frame_idx, num_frames, input_batch, geometric_prompt,
            feature_cache, reverse, use_batched_grounding,
            batched_grounding_batch_size,
        )
        if det_out is not None:
            boxes = det_out["bbox"][0].detach().float().cpu().numpy()
            scores = det_out["scores"][0].detach().float().cpu().numpy()
            raw_by_frame[int(frame_idx)] = (boxes, scores)
        return det_out, pos

    model.run_backbone_and_detection = wrap

    if branch == "I":
        patch_state.set_mode("identity")
    elif branch == "R":
        patch_state.set_mode("random")
    elif branch == "C":
        patch_state.set_mode("save_roi")

    backend._predictor.handle_request(req_prompt)
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"].clear()

    if branch == "C":
        patch_state.set_mode("roi")

    req = dict(
        type="propagate_in_video",
        session_id=backend._session_id,
        propagation_direction="forward",
        start_frame_index=human_frame,
        max_frame_num_to_track=None,
    )
    for response in backend._predictor.handle_stream_request(request=req):
        f = int(response["frame_index"])
        if f > human_frame + FUTURE_FRAMES:
            break
        invalidate_detector_prefetch(runner, f)
        if f >= human_frame + 1:
            rows = parse_raw_outputs(response, frame_size=(iw, ih))
            records.append(
                {
                    "frame": f,
                    "n_official": len(rows),
                    "official_hash": row_hash(rows),
                }
            )
        if f >= human_frame + FUTURE_FRAMES:
            break

    for rec in records:
        f = rec["frame"]
        if f in raw_by_frame:
            boxes, scores = raw_by_frame[f]
            rec["n_raw"] = int(boxes.shape[0])
            rec["n_raw_pos"] = int((scores > 0.4).sum().item())
            rec["raw_max_score"] = float(scores.max().item())
            rec["raw_hash"] = blob_hash(boxes, scores)
        else:
            rec["n_raw"] = -1
            rec["n_raw_pos"] = -1
            rec["raw_max_score"] = -1.0
            rec["raw_hash"] = ""
        rec["branch"] = branch
        rec["sequence"] = seq
        rec["event_frame"] = human_frame
        rec["gid"] = gid

    model.run_backbone_and_detection = orig_rbd
    backend.close()
    return records, raw_by_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--events", default="0,1,2")
    ap.add_argument("--branches", default="O,I,R,C")
    args = ap.parse_args()
    branches = tuple(args.branches.split(","))

    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend0 = runner._ensure_backend()
    backend0._ensure_model()
    model = backend0._predictor.model
    image = model.detector
    orig_decoder_bound = image._run_decoder
    slot = SLOT
    d_model = image.transformer.decoder.query_embed.weight.shape[1]
    static_scale = (
        image.transformer.decoder.query_embed.weight.detach().norm(dim=-1).mean()
    )

    out_dir = OUT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    official = {}
    raw_o_dir = OUT / "raw_o"
    raw_o_dir.mkdir(parents=True, exist_ok=True)

    for idx in (int(i) for i in args.events.split(",")):
        seq, hf, gid, box = EVENTS[idx]
        x1, y1, x2, y2 = (float(v) for v in box)
        iw, ih = 1920, 1080  # filled after first start; DanceTrack images are 1920x1080
        box_norm = np.asarray(
            [x1 / iw, y1 / ih, x2 / iw, y2 / ih], dtype=float
        )
        patch = DecoderPatch(image, mode="identity", slot=slot,
                             human_box_norm=box_norm)
        patch.install()
        gen = torch.Generator(device="cuda").manual_seed(20260809 + idx)
        rand_q = torch.randn(d_model, generator=gen, device="cuda")
        rand_q = rand_q * (static_scale / (rand_q.norm() + 1e-9))
        patch.dyn_query = rand_q
        patched_bound = patch.install()

        for branch in branches:
            if branch == "O":
                image._run_decoder = orig_decoder_bound
            else:
                image._run_decoder = patched_bound
            if branch != "C":
                patch.saved_query = None
            recs, raw = run_branch(
                runner, seq, hf, gid, box, branch, patch
            )
            if branch == "O":
                official[(seq, hf)] = raw
                np.savez(
                    raw_o_dir / f"{seq}_{hf}.npz",
                    **{
                        f"f{f}": np.concatenate([b.reshape(-1), s], axis=0)
                        for f, (b, s) in raw.items()
                    },
                )
            for r in recs:
                if branch == "O":
                    ob = os_ = None
                else:
                    o = official.get((seq, hf), {}).get(r["frame"])
                    if o is not None:
                        ob, os_ = o
                    else:
                        saved = np.load(raw_o_dir / f"{seq}_{hf}.npz")
                        key = f"f{r['frame']}"
                        if key in saved:
                            arr = saved[key]
                            ob = arr[:4 * 200].reshape(200, 4)
                            os_ = arr[4 * 200:]
                        else:
                            ob = os_ = None
                if ob is not None and r["frame"] in raw:
                    rb, rs = raw[r["frame"]]
                    r["raw_diff_bbox_max"] = float(
                        np.abs(rb - ob).max().item()
                    )
                    r["raw_diff_score_max"] = float(
                        np.abs(rs - os_).max().item()
                    )
                else:
                    r["raw_diff_bbox_max"] = -1.0
                    r["raw_diff_score_max"] = -1.0
                rows.append(r)
            print(
                f"{seq} f{hf} gid{gid} branch={branch}: "
                + ", ".join(
                    f"f{r['frame']} raw={r['raw_hash']} off={r['official_hash']}"
                    f" db={r['raw_diff_bbox_max']:.3g} ds={r['raw_diff_score_max']:.3g}"
                    for r in recs
                ),
                flush=True,
            )

    csv_path = OUT / "query_causal_smoke.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sequence", "event_frame", "gid", "branch", "frame",
                "n_raw", "n_raw_pos", "raw_max_score", "raw_hash",
                "n_official", "official_hash",
                "raw_diff_bbox_max", "raw_diff_score_max",
            ],
        )
        w.writeheader()
        for r in sorted(
            rows,
            key=lambda r: (r["sequence"], r["event_frame"], r["branch"], r["frame"]),
        ):
            w.writerow(r)
    print(f"WROTE {csv_path}", flush=True)


if __name__ == "__main__":
    main()
