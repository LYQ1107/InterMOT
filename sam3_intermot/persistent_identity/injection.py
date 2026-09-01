"""Dynamic detector-query injection (instance-level override, no third-party
source edits).  Empty query bank => byte-identical to the official pipeline
(verified in N14 query causal smoke)."""

import types
import os
from typing import Callable, List, Optional, Sequence

import torch

from .human_write_encoder import roi_pool_feature


def build_tgt_with_queries(
    image,
    dyn_queries: Sequence[Optional[torch.Tensor]],
    slot_indices: Sequence[int],
    bs: int = 1,
) -> torch.Tensor:
    qe = image.transformer.decoder.query_embed.weight
    tgt = qe.unsqueeze(1).repeat(1, bs, 1)
    parts = []
    prev = 0
    for slot, q in sorted(zip(slot_indices, dyn_queries)):
        if q is None:
            continue
        parts.append(tgt[prev:slot])
        parts.append(q.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1))
        prev = slot + 1
    if prev == 0:
        return tgt
    parts.append(tgt[prev:])
    return torch.cat(parts, dim=0)


def build_ref_boxes_with_queries(
    image,
    dyn_refs: Sequence[Optional[torch.Tensor]],
    slot_indices: Sequence[int],
    bs: int = 1,
) -> torch.Tensor:
    """Clone the static reference-point table and replace reserved slots with
    dynamic per-identity reference boxes (normalized cxcywh, after sigmoid)."""
    rp = image.transformer.decoder.reference_points.weight
    refs = rp.sigmoid().unsqueeze(1).repeat(1, bs, 1)
    parts = []
    prev = 0
    for slot, r in sorted(zip(slot_indices, dyn_refs)):
        if r is None:
            continue
        parts.append(refs[prev:slot])
        parts.append(r.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1).to(refs.dtype))
        prev = slot + 1
    if prev == 0:
        return refs
    parts.append(refs[prev:])
    return torch.cat(parts, dim=0)


def run_decoder_with_tgt(
    image,
    tgt: torch.Tensor,
    pos_embed,
    memory,
    src_mask,
    out: dict,
    prompt,
    prompt_mask,
    encoder_out: dict,
    reference_boxes=None,
    slot_adapter=None,
    active_slots=(),
    dyn_queries=None,
    gate_score=None,
    gate_iou=None,
):
    """Official Sam3Image._run_decoder body with a caller-supplied tgt."""
    bs = memory.shape[1]
    apply_dac = image.transformer.decoder.dac and image.training
    initial_refs = reference_boxes
    hs, reference_boxes, dec_presence_out, dec_presence_feats = (
        image.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=reference_boxes,
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
    image._update_scores_and_boxes(
        out,
        hs,
        reference_boxes,
        prompt,
        prompt_mask,
        dec_presence_out=dec_presence_out,
    )
    if slot_adapter is not None and active_slots:
        if initial_refs is None:
            initial_refs = (
                image.transformer.decoder.reference_points.weight
                .sigmoid().unsqueeze(1).repeat(1, bs, 1)
            )
        pb, px, pl = out["pred_boxes"], out["pred_boxes_xyxy"], out["pred_logits"]
        pb_parts = [pb[:, :active_slots[0]]]
        px_parts = [px[:, :active_slots[0]]]
        pl_parts = [pl[:, :active_slots[0]]]
        prev = active_slots[0]
        for i, slot in enumerate(active_slots):
            if os.environ.get("N14_DEBUG_ADAPTER"):
                print(
                    "ADAPTER_REQ", "hs", hs.requires_grad,
                    "feat", feat.requires_grad,
                    flush=True,
            )
            ref = initial_refs[slot].float()
            q = (
                dyn_queries[i].unsqueeze(0).repeat(bs, 1).float()
                if dyn_queries is not None else torch.zeros(
                    bs, image.transformer.decoder.query_embed.weight.shape[1],
                    device=memory.device,
                ).float()
            )
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
            roi_feat = roi_pool_feature(
                memory, encoder_out, roi_norm[0]
            ).unsqueeze(0)
            dbox, dscore = slot_adapter(q, roi_feat, roi_feat, ref)
            if os.environ.get("N14_DEBUG_ADAPTER"):
                print(
                    "ADAPTER_RAW",
                    [round(float(x), 4) for x in pb[:, slot].detach().cpu().tolist()[0]],
                    [round(float(x), 4) for x in pl[:, slot].detach().cpu().tolist()[0]],
                    "DBOX_NORM", round(float(dbox.detach().norm().item()), 5),
                    flush=True,
                )
            raw_box = ref + dbox
            cx, cy, w, h = raw_box.unbind(-1)
            new_box = torch.stack(
                [
                    cx.clamp(0.0, 1.0),
                    cy.clamp(0.0, 1.0),
                    w.clamp(0.01, 1.0),
                    h.clamp(0.01, 1.0),
                ],
                dim=-1,
            )
            if gate_score is not None or gate_iou is not None:
                fire = torch.ones(
                    (bs,), dtype=torch.bool, device=new_box.device
                )
                if gate_score is not None:
                    fire = fire & (
                        torch.sigmoid(dscore).squeeze(-1) >= gate_score
                    )
                if gate_iou is not None:
                    a = new_box
                    b = ref
                    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
                    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
                    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
                    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
                    ix1 = torch.maximum(ax1, bx1)
                    iy1 = torch.maximum(ay1, by1)
                    ix2 = torch.minimum(ax2, bx2)
                    iy2 = torch.minimum(ay2, by2)
                    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                    ua = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
                    ub = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
                    iou = inter / (ua + ub - inter + 1e-9)
                    fire = fire & (iou >= gate_iou)
                if not bool(fire.all().item()):
                    new_box = pb[:, slot].float()
                    new_logit = pl[:, slot].float()
            if os.environ.get("N14_DEBUG_ADAPTER"):
                print(
                    "ADAPTER_NEW",
                    [round(float(x), 4) for x in new_box.detach().cpu().tolist()[0]],
                    flush=True,
                )
            cx, cy, w, h = new_box.unbind(-1)
            new_xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            )
            new_logit = dscore
            if os.environ.get("N14_DEBUG_ADAPTER"):
                print(
                    "ADAPTER_LOGIT",
                    [round(float(x), 4) for x in new_logit.detach().cpu().tolist()[0]],
                    flush=True,
                )
            pb_parts.append(new_box.unsqueeze(1))
            px_parts.append(new_xyxy.unsqueeze(1))
            pl_parts.append(new_logit.unsqueeze(1))
            if i + 1 < len(active_slots):
                nxt = active_slots[i + 1]
                pb_parts.append(pb[:, slot + 1:nxt])
                px_parts.append(px[:, slot + 1:nxt])
                pl_parts.append(pl[:, slot + 1:nxt])
            prev = slot
        pb_parts.append(pb[:, prev + 1:])
        px_parts.append(px[:, prev + 1:])
        pl_parts.append(pl[:, prev + 1:])
        out["pred_boxes"] = torch.cat(pb_parts, dim=1).to(pb.dtype)
        out["pred_boxes_xyxy"] = torch.cat(px_parts, dim=1).to(px.dtype)
        out["pred_logits"] = torch.cat(pl_parts, dim=1).to(pl.dtype)
    return out, hs


def install_query_patch(
    image,
    bank_fn: Callable[[], tuple],
    slot_indices: Sequence[int],
    slot_adapter=None,
    gate_score=None,
    gate_iou=None,
):
    """Patch Sam3Image._run_decoder so active persistent queries (and their
    reference boxes) are injected into reserved slots before the decoder.
    bank_fn() -> (queries, refs).  Returns uninstall()."""
    orig = image.__class__._run_decoder
    image._n14_adapter = slot_adapter

    def patched(self, pos_embed, memory, src_mask, out, prompt, prompt_mask,
                encoder_out):
        queries, refs = bank_fn()
        active = (
            slot_indices
            if any(q is not None for q in queries)
            else ()
        )
        tgt = build_tgt_with_queries(
            self, queries, slot_indices, bs=memory.shape[1]
        )
        ref_boxes = build_ref_boxes_with_queries(
            self, refs, slot_indices, bs=memory.shape[1]
        )
        return run_decoder_with_tgt(
            self, tgt, pos_embed, memory, src_mask, out, prompt,
            prompt_mask, encoder_out, reference_boxes=ref_boxes,
            slot_adapter=getattr(self, "_n14_adapter", None),
            active_slots=active,
            dyn_queries=queries,
            gate_score=gate_score,
            gate_iou=gate_iou,
        )

    bound = types.MethodType(patched, image)
    image._run_decoder = bound

    def uninstall():
        image._run_decoder = types.MethodType(orig, image)

    return uninstall
