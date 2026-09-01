"""Fast debug: which tensor is still an inference tensor in _run_decoder?"""

import copy
import os
import numpy as np
import torch

from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
from sam3_intermot.adaptation.lora import inject_lora


ROOT = "."
CKPT = f"{ROOT}/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
VIDEO = "/path/to/dancetrack/train/dancetrack0075/img1"


def clone_find_input(fin):
    out = copy.copy(fin)
    for field in fin.__dataclass_fields__:
        v = getattr(out, field)
        if isinstance(v, torch.Tensor):
            setattr(out, field, v.clone())
        elif isinstance(v, list) and v and all(isinstance(x, torch.Tensor) for x in v):
            setattr(out, field, [x.clone() for x in v])
    out.img_ids = torch.tensor([0], dtype=torch.long, device="cuda")
    return out


def main():
    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 3)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend.start_video(VIDEO)
    backend._predictor.model.use_batched_grounding = False
    backend._predictor.handle_request(dict(
        type="add_prompt", session_id=backend._session_id, frame_index=1,
        text="person", bounding_boxes=None, bounding_box_labels=None,
        clear_old_boxes=True,
    ))
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"].clear()
    model = backend._predictor.model

    def unfreeze_inference_tensors(module):
        n = 0
        for m in module.modules():
            for k, v in list(vars(m).items()):
                if isinstance(v, torch.Tensor) and torch.is_inference(v):
                    setattr(m, k, v.clone())
                    n += 1
            for k, v in list(m._buffers.items()):
                if isinstance(v, torch.Tensor) and torch.is_inference(v):
                    m._buffers[k] = v.clone()
                    n += 1
        return n

    n_fixed = unfreeze_inference_tensors(model)
    print("unfrozen_inference_tensors", n_fixed, flush=True)
    if not os.environ.get("N13_NO_LORA"):
        inject_lora(model, (
            "detector.transformer.decoder.layers",
            "detector.transformer.decoder.bbox_embed",
            "detector.dot_prod_scoring",
        ), r=8, alpha=4.0)

    fin = clone_find_input(state["input_batch"].find_inputs[1])
    img_batch = state["input_batch"].img_batch
    img_t = (
        img_batch.tensors[1].unsqueeze(0).clone().to("cuda")
        if hasattr(img_batch, "tensors")
        else img_batch[1].unsqueeze(0).clone().to("cuda")
    )
    backbone_out = {
        "img_batch_all_stages": img_t,
        "language_features": state["backbone_out"]["language_features"].clone(),
        "language_mask": state["backbone_out"]["language_mask"].clone(),
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
    def cl(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        if isinstance(x, (list, tuple)):
            return [cl(v) for v in x]
        if isinstance(x, dict):
            return {k: cl(v) for k, v in x.items()}
        return x

    import sam3.model.decoder as D
    orig_gen = D.gen_sineembed_for_position
    calls = []
    def gen_probe(pos_tensor, num_feats=256):
        calls.append((torch.is_inference(pos_tensor), tuple(pos_tensor.shape)))
        return orig_gen(pos_tensor, num_feats)
    D.gen_sineembed_for_position = gen_probe

    flin_fail = []
    lin_calls = []
    if os.environ.get("N13_PROBES"):
        orig_linear_fwd = torch.nn.Linear.forward
        import torch.nn.functional as _F
        orig_flinear = _F.linear
        def probe_flinear(input, weight, bias=None):
            try:
                return orig_flinear(input, weight, bias)
            except RuntimeError as e:
                if "Inference tensors" in str(e):
                    try:
                        out = orig_flinear(input.clone(), weight, bias)
                        flin_fail.append(("CLONE_FIXES", tuple(input.shape)))
                        return out
                    except RuntimeError as e2:
                        flin_fail.append((
                            torch.is_inference(input), tuple(input.shape),
                            input.dtype, str(input.device),
                            input._base is not None,
                            torch.is_inference(weight),
                            None if bias is None else torch.is_inference(bias),
                            "clone_fail",
                        ))
                        raise e2
                raise
        _F.linear = probe_flinear
        def probe_linear(self, input):
            if torch.is_inference(input):
                lin_calls.append((type(self).__name__, tuple(input.shape)))
            return orig_linear_fwd(self, input)
        torch.nn.Linear.forward = probe_linear
    with torch.inference_mode():
        prompt, prompt_mask, bo2 = model.detector._encode_prompt(
            backbone_out, fin, empty_geo
        )
        bo2, encoder_out, _ = model.detector._run_encoder(
            bo2, fin, prompt, prompt_mask
        )
    enc2 = cl(encoder_out)
    prompt2 = prompt.clone()
    pmask2 = prompt_mask.clone()
    synth = {
        "encoder_hidden_states": torch.randn_like(enc2["encoder_hidden_states"]),
        "pos_embed": torch.randn_like(enc2["pos_embed"]),
        "padding_mask": None if enc2["padding_mask"] is None
                        else torch.zeros_like(enc2["padding_mask"]),
        "prompt": torch.randn_like(prompt2),
        "prompt_mask": torch.zeros_like(pmask2),
        "valid_ratios": torch.ones_like(enc2["valid_ratios"]),
        "level_start_index": torch.zeros_like(enc2["level_start_index"]),
        "spatial_shapes": torch.tensor([[72, 72]], dtype=torch.long, device="cuda"),
    }
    variants = {
        "all_synth": {k: v for k, v in synth.items()},
        "all_real": {
            "encoder_hidden_states": enc2["encoder_hidden_states"],
            "pos_embed": enc2["pos_embed"],
            "padding_mask": enc2["padding_mask"],
            "prompt": prompt2,
            "prompt_mask": pmask2,
            "valid_ratios": enc2["valid_ratios"],
            "level_start_index": enc2["level_start_index"],
            "spatial_shapes": enc2["spatial_shapes"],
        },
    }
    for key in synth:
        v = {
            "encoder_hidden_states": enc2["encoder_hidden_states"],
            "pos_embed": enc2["pos_embed"],
            "padding_mask": enc2["padding_mask"],
            "prompt": prompt2,
            "prompt_mask": pmask2,
            "valid_ratios": enc2["valid_ratios"],
            "level_start_index": enc2["level_start_index"],
            "spatial_shapes": enc2["spatial_shapes"],
        }
        v[key] = synth[key]
        variants[f"real_except_{key}"] = v

    import contextlib
    for vname, v in variants.items():
        enc_v = dict(enc2)
        enc_v["valid_ratios"] = v["valid_ratios"]
        enc_v["level_start_index"] = v["level_start_index"]
        enc_v["spatial_shapes"] = v["spatial_shapes"]
        try:
            def pack_hook(t):
                return t.clone() if torch.is_floating_point(t) else t
            def unpack_hook(t):
                return t
            with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(
                pack_hook, unpack_hook
            ):
                out2 = {"encoder_hidden_states": v["encoder_hidden_states"]}
                out2, hs = model.detector._run_decoder(
                    pos_embed=v["pos_embed"],
                    memory=v["encoder_hidden_states"],
                    src_mask=v["padding_mask"],
                    out=out2,
                    prompt=v["prompt"],
                    prompt_mask=v["prompt_mask"],
                    encoder_out=enc_v,
                )
            print("VARIANT_OK", vname, flush=True)
        except RuntimeError as e:
            print("VARIANT_FAIL", vname, str(e)[:80], flush=True)


if __name__ == "__main__":
    main()
