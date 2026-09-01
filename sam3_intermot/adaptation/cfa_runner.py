"""Correction-to-Future (CFA) single-object episode runner.

Each episode:
  - frame t gets a human box for one target identity;
  - branch A (no update) runs the frozen tracker from t;
  - branch B (update) runs 1-3 LoRA steps on the current-frame box/mask loss,
    then runs the tracker from t with the updated weights;
  - both branches are evaluated against GT at t+1..t+H.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TFf

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.observations.mask_to_box import mask_to_box


DT = Path("/path/to/dancetrack")


def box_to_points(box_xyxy: np.ndarray) -> torch.Tensor:
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=float)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pts = np.asarray(
        [[cx, cy], [x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=float
    )
    return torch.tensor(pts, dtype=torch.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class EpisodeResult:
    sequence: str
    frame: int
    event_type: str
    gid: int
    human_box: List[float]
    no_update: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)
    update: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)
    update_seconds: float = 0.0

    def recall(self, boxes: Dict[int, Optional[np.ndarray]], gt: Dict[int, np.ndarray]) -> float:
        hits = sum(
            1 for f, b in boxes.items()
            if b is not None and f in gt and iou(b, gt[f]) >= 0.5
        )
        return hits / max(1, len(boxes))


class CFARunner:
    def __init__(self, model, split: str = "train"):
        self.model = model
        self.split = split
        self.dataset = DanceTrackDataset(str(DT), sequences=[], split=split)
        self.image_size = int(model.image_size)

    def _load_window(self, sequence: str, start: int, length: int):
        """Load frames [start, start+length) resized/normalized like the official loader."""
        img_dir = DT / self.split / sequence / "img1"
        names = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        names = names[start : start + length]
        img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16)[:, None, None]
        img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16)[:, None, None]
        imgs = torch.zeros(len(names), 3, self.image_size, self.image_size,
                           dtype=torch.float16)
        vh = vw = None
        for i, p in enumerate(names):
            img = Image.open(p).convert("RGB")
            vh, vw = img.height, img.width
            t = TFf.to_tensor(TFf.resize(img, (self.image_size, self.image_size)))
            imgs[i] = t.to(torch.float16)
        imgs = imgs.cuda()
        imgs = (imgs - img_mean.cuda()) / img_std.cuda()
        return imgs, vh, vw

    def _new_state(self, sequence: str, num_frames: int, start_frame: int = 0):
        images, video_height, video_width = self._load_window(
            sequence, start_frame, num_frames
        )
        state = self.model.init_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
        state["images"] = images
        return state

    def _prompt(self, state, frame_idx: int, box_xyxy: np.ndarray):
        pts = box_to_points(box_xyxy)
        labels = torch.ones(pts.shape[0], dtype=torch.int32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Keep a point-based prompt (absolute pixel coords in the 1008 input)
            # for the differentiable inner update; the box mask below is the
            # actual correction-frame output (mask-as-output path).
            state["n12_point_inputs"] = {
                "point_coords": (pts * self.image_size).unsqueeze(0).cuda(),
                "point_labels": labels.unsqueeze(0).cuda(),
            }
            x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=float)
            vh, vw = state["video_height"], state["video_width"]
            mask = torch.zeros(1, vh, vw, device=state["device"])
            mask[..., int(y1) : int(y2), int(x1) : int(x2)] = 1.0
            self.model.add_new_masks(
                inference_state=state,
                frame_idx=frame_idx,
                obj_ids=[1],
                masks=mask,
            )

    def _run_branch(
        self, state, start: int, horizon: int, base: int
    ) -> Dict[int, Optional[np.ndarray]]:
        boxes: Dict[int, Optional[np.ndarray]] = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.model.propagate_in_video_preflight(
                state, run_mem_encoder=True
            )
            for frame_idx, _obj_ids, _low, video_res_masks, _obj_scores in self.model.propagate_in_video(
                inference_state=state,
                start_frame_idx=start,
                max_frame_num_to_track=start + horizon,
                reverse=False,
                tqdm_disable=True,
                run_mem_encoder=True,
            ):
                masks = video_res_masks
                if isinstance(masks, (list, tuple)):
                    if len(masks) == 0:
                        boxes[base + frame_idx] = None
                        continue
                    mask = np.asarray(
                        masks[0].cpu() if torch.is_tensor(masks[0]) else masks[0]
                    )
                else:
                    mask = np.asarray(
                        masks.cpu() if torch.is_tensor(masks) else masks
                    )
                while mask.ndim > 2:
                    mask = mask[0]
                boxes[base + frame_idx] = mask_to_box(mask > 0)
        return boxes

    def _inner_update(
        self,
        state,
        frame_idx: int,
        human_box: np.ndarray,
        lora_params: List[torch.nn.Parameter],
        steps: int = 2,
        lr: float = 1e-3,
    ) -> float:
        point_inputs = state.get("n12_point_inputs")
        if point_inputs is None:
            point_inputs = state["point_inputs_per_obj"][0][frame_idx]
        image, backbone_out = state["cached_features"][frame_idx]
        features = self.model._prepare_backbone_features(backbone_out)
        mux = self.model.multiplex_controller.get_state(
            num_valid_entries=1,
            device=state["device"],
            dtype=torch.float32,
            random=False,
            object_ids=[1],
        )
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        x1, y1, x2, y2 = np.asarray(human_box, dtype=float)
        ih, iw = state["video_height"], state["video_width"]

        opt = torch.optim.Adam(lora_params, lr=lr)
        t0 = __import__("time").time()
        for _ in range(steps):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.enable_grad():
                    out = self.model.track_step(
                        frame_idx=frame_idx,
                        is_init_cond_frame=True,
                        backbone_features_interactive=features["interactive"],
                        backbone_features_propagation=features["sam2_backbone_out"],
                        image=image,
                        point_inputs=point_inputs,
                        mask_inputs=None,
                        gt_masks=None,
                        frames_to_add_correction_pt=[],
                        output_dict=output_dict,
                        num_frames=state["num_frames"],
                        run_mem_encoder=False,
                        prev_sam_mask_logits=None,
                        multiplex_state=mux,
                        objects_to_interact=None,
                    )
                pred = out["pred_masks"]
                if pred.dim() == 4:
                    pred = pred[0]
                if pred.dim() == 3 and pred.shape[0] == 1:
                    pred = pred[0]
                H, W = pred.shape[-2], pred.shape[-1]
                gx1, gy1 = max(0, int(x1 / iw * W)), max(0, int(y1 / ih * H))
                gx2, gy2 = min(W, int(np.ceil(x2 / iw * W))), min(H, int(np.ceil(y2 / ih * H)))
                inside = torch.zeros(1, 1, H, W, device=pred.device)
                inside[..., gy1:gy2, gx1:gx2] = 1.0
                outside = 1.0 - inside
                prob = torch.sigmoid(pred)
                loss_outside = (prob * outside).sum() / outside.sum().clamp(min=1)
                obj_logits = out.get("object_score_logits")
                loss_obj = torch.tensor(0.0, device=pred.device)
                if obj_logits is not None:
                    o = obj_logits.reshape(-1)
                    loss_obj = torch.nn.functional.binary_cross_entropy_with_logits(
                        o, torch.ones_like(o)
                    )
                loss = loss_outside + 0.1 * loss_obj
                loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
        elapsed = __import__("time").time() - t0
        return elapsed

    def run_episode(
        self,
        sequence: str,
        frame_idx: int,
        event_type: str,
        gid: int,
        human_box: np.ndarray,
        horizon: int = 30,
        lora_params: Optional[List[torch.nn.Parameter]] = None,
        update_steps: int = 2,
        lr: float = 1e-3,
    ) -> EpisodeResult:
        num_frames = self.dataset.num_frames(sequence)
        gt = self.dataset.load_gt(sequence)

        res = EpisodeResult(
            sequence=sequence,
            frame=frame_idx,
            event_type=event_type,
            gid=gid,
            human_box=list(np.asarray(human_box, dtype=float)),
        )

        # Branch A: no update (window starts at correction frame)
        state_a = self._new_state(sequence, horizon + 1, start_frame=frame_idx)
        self._prompt(state_a, 0, human_box)
        res.no_update = self._run_branch(state_a, 0, horizon, base=frame_idx)
        del state_a

        # Branch B: LoRA update
        if lora_params is not None and len(lora_params) > 0:
            state_b = self._new_state(sequence, horizon + 1, start_frame=frame_idx)
            self._prompt(state_b, 0, human_box)
            res.update_seconds = self._inner_update(
                state_b, 0, human_box, lora_params, steps=update_steps, lr=lr
            )
            del state_b
            state_b2 = self._new_state(sequence, horizon + 1, start_frame=frame_idx)
            self._prompt(state_b2, 0, human_box)
            res.update = self._run_branch(state_b2, 0, horizon, base=frame_idx)
            del state_b2
        else:
            res.update = dict(res.no_update)
        return res


def target_gt_boxes(
    gt: Dict[int, object], gid: int, frames: List[int]
) -> Dict[int, np.ndarray]:
    out = {}
    for f in frames:
        entry = gt.get(f)
        if entry is None or gid not in entry.gt_ids:
            continue
        out[f] = entry.boxes[entry.gt_ids.index(gid)]
    return out
