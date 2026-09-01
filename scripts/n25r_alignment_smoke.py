#!/usr/bin/env python
"""Candidate -> returned obj_id -> multiplex slot alignment smoke for N25-R.

The script replays the N20 isolated shadow contract on a fixed, deterministic
subset of 0074/0083/0096.  It retains the selected SAM3 object ID and extracts
only causally observed internal tensors through the official mappings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(".")
DATA = Path("/path/to/dancetrack")
CKPT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
EPISODES = ROOT / "outputs/n25/dataset/episodes_cal10.jsonl"
GROUP_AUDIT = ROOT / "outputs/n25r/protocol_audit/group_audit_cal10.csv"
DEFAULT_OUT = ROOT / "outputs/n25r/alignment_smoke"
SEQUENCES = ("dancetrack0074", "dancetrack0083", "dancetrack0096")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner  # noqa: E402
from sam3_intermot.detection_query.prompt_replay import (  # noqa: E402
    invalidate_detector_prefetch,
    set_frame_geometric_prompt,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def normalize(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all():
        return None
    norm = float(np.linalg.norm(x))
    return x / norm if norm > 1e-8 else None


def response_candidates(response: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    raw = response.get("outputs") or {}
    if not isinstance(raw, dict) or "out_obj_ids" not in raw or "out_boxes_xywh" not in raw:
        return []
    obj_ids = np.asarray(raw["out_obj_ids"]).reshape(-1)
    boxes = np.asarray(raw["out_boxes_xywh"], dtype=np.float32).reshape(-1, 4)
    probs = np.asarray(raw.get("out_probs", np.full(len(obj_ids), np.nan)), dtype=np.float32).reshape(-1)
    sam2_probs = np.asarray(raw.get("out_sam2_probs", np.full(len(obj_ids), np.nan)), dtype=np.float32).reshape(-1)
    out = []
    for index, obj_id in enumerate(obj_ids):
        if index >= len(boxes):
            break
        x, y, w, h = boxes[index]
        out.append(
            {
                "obj_id": int(obj_id),
                "box": np.asarray([x * width, y * height, (x + w) * width, (y + h) * height], dtype=np.float32),
                "score": float(probs[index]) if index < len(probs) and np.isfinite(probs[index]) else None,
                "sam2_score": float(sam2_probs[index]) if index < len(sam2_probs) and np.isfinite(sam2_probs[index]) else None,
            }
        )
    return out


def select_candidate(previous_box: np.ndarray, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    ranked = sorted(
        [(iou(previous_box, candidate["box"]), candidate) for candidate in candidates],
        key=lambda item: (-item[0], item[1]["obj_id"]),
    )
    best_iou = ranked[0][0] if ranked else None
    second_iou = ranked[1][0] if len(ranked) > 1 else None
    selected = ranked[0][1] if ranked and ranked[0][0] >= 0.3 else None
    return selected, {
        "best_iou": best_iou,
        "second_iou": second_iou,
        "match_margin": None if best_iou is None else best_iou - (second_iou or 0.0),
        "ambiguous": bool(best_iou is not None and second_iou is not None and best_iou - second_iou < 0.10),
    }


def find_slot(assignments: list[list[int]], object_index: int) -> tuple[int | None, int | None]:
    for bucket, values in enumerate(assignments):
        for slot, value in enumerate(values):
            if int(value) == int(object_index):
                return bucket, slot
    return None, None


def frame_output(state: dict[str, Any], obj_id: int, frame: int) -> tuple[int, dict[str, Any], dict[str, Any], int] | None:
    fallbacks = []
    for state_index, tracker_state in enumerate(state.get("sam2_inference_states", [])):
        if obj_id not in tracker_state.get("obj_ids", []):
            continue
        current_index = tracker_state.get("obj_id_to_idx", {}).get(obj_id)
        if current_index is None:
            continue
        for storage in ("cond_frame_outputs", "non_cond_frame_outputs"):
            out = tracker_state.get("output_dict", {}).get(storage, {}).get(frame)
            if out is None:
                continue
            local = out.get("local_obj_id_to_idx") or {}
            if obj_id in local:
                return state_index, tracker_state, out, int(local[obj_id])
            fallbacks.append((state_index, tracker_state, out, int(current_index)))
    return fallbacks[0] if fallbacks else None


def image_grid(frame_out: dict[str, Any], bucket: int) -> torch.Tensor | None:
    value = frame_out.get("image_features")
    if value is None or not torch.is_tensor(value):
        return None
    value = value.detach().float()
    if value.ndim == 3 and value.shape[0] == 72 * 72:
        if bucket >= value.shape[1]:
            return None
        return value[:, bucket, :].reshape(72, 72, -1).permute(2, 0, 1).contiguous()
    if value.ndim == 4 and value.shape[-2:] == (72, 72):
        if bucket >= value.shape[0]:
            return None
        return value[bucket]
    return None


def pool_box(grid: torch.Tensor | None, box: np.ndarray, width: int, height: int) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if grid is None:
        return None, None, 0
    gh, gw = int(grid.shape[-2]), int(grid.shape[-1])
    x1, y1, x2, y2 = map(float, box)
    gx1 = max(0, min(gw - 1, int(math.floor(x1 / width * gw))))
    gy1 = max(0, min(gh - 1, int(math.floor(y1 / height * gh))))
    gx2 = max(gx1 + 1, min(gw, int(math.ceil(x2 / width * gw))))
    gy2 = max(gy1 + 1, min(gh, int(math.ceil(y2 / height * gh))))
    roi = grid[:, gy1:gy2, gx1:gx2]
    if roi.numel() == 0:
        return None, None, 0
    mean = normalize(roi.mean(dim=(-2, -1)).cpu().numpy())
    maximum = normalize(roi.amax(dim=(-2, -1)).cpu().numpy())
    return mean, maximum, int(roi.shape[-2] * roi.shape[-1])


def pool_mask(grid: torch.Tensor | None, frame_out: dict[str, Any], local_index: int) -> tuple[np.ndarray | None, float | None, int]:
    masks = frame_out.get("pred_masks")
    if grid is None or masks is None or not torch.is_tensor(masks) or local_index >= masks.shape[0]:
        return None, None, 0
    mask = masks[local_index : local_index + 1].detach().float()
    while mask.ndim > 4:
        mask = mask.squeeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate(mask, size=grid.shape[-2:], mode="bilinear", align_corners=False)[0, 0] > 0
    count = int(mask.sum().item())
    if count == 0:
        return None, 0.0, 0
    pooled = normalize(grid[:, mask].mean(dim=1).cpu().numpy())
    return pooled, count / float(mask.numel()), count


def pointer_feature(frame_out: dict[str, Any], tracker_state: dict[str, Any], local_index: int) -> np.ndarray | None:
    value = frame_out.get("obj_ptr")
    multiplex = tracker_state.get("multiplex_state")
    if value is None or multiplex is None or not torch.is_tensor(value):
        return None
    try:
        data = multiplex.demux(value)
    except (AssertionError, RuntimeError, IndexError):
        return None
    if local_index >= data.shape[0]:
        return None
    return normalize(data[local_index].detach().float().cpu().numpy())


def memory_grid(frame_out: dict[str, Any], tracker_state: dict[str, Any], bucket: int, local_index: int) -> tuple[torch.Tensor | None, str]:
    value = frame_out.get("maskmem_features")
    if value is None or not torch.is_tensor(value):
        return None, "invalid"
    value = value.detach().float()
    multiplex = tracker_state.get("multiplex_state")
    if value.ndim >= 5 and multiplex is not None:
        try:
            data = multiplex.demux(value)
            if local_index < data.shape[0]:
                return data[local_index], "official_demuxed_per_object"
        except (AssertionError, RuntimeError, IndexError):
            pass
    if value.ndim == 4 and value.shape[-2:] == (72, 72) and bucket < value.shape[0]:
        return value[bucket], "candidate_pooled_consolidated_memory"
    return None, "invalid"


def extract_selected(state: dict[str, Any], obj_id: int, frame: int, box: np.ndarray, width: int, height: int) -> tuple[dict[str, Any], dict[str, np.ndarray | None]]:
    located = frame_output(state, obj_id, frame)
    if located is None:
        return {"binding_valid": False, "reason": "selected_obj_id_not_in_frame_state"}, {}
    state_index, tracker_state, out, local_index = located
    multiplex = tracker_state.get("multiplex_state")
    assignments = multiplex.assignments if multiplex is not None else []
    bucket, slot = find_slot(assignments, local_index)
    if bucket is None:
        return {"binding_valid": False, "reason": "object_index_not_in_multiplex_assignments", "state_index": state_index, "local_index": local_index}, {}
    grid = image_grid(out, bucket)
    box_mean, box_max, roi_pixels = pool_box(grid, box, width, height)
    mask_mean, mask_coverage, mask_pixels = pool_mask(grid, out, local_index)
    pointer = pointer_feature(out, tracker_state, local_index)
    mem_grid, mem_name = memory_grid(out, tracker_state, bucket, local_index)
    mem_box_mean, _, mem_pixels = pool_box(mem_grid, box, width, height)
    metadata = {
        "binding_valid": True,
        "state_index": state_index,
        "selected_obj_id": obj_id,
        "local_object_index": local_index,
        "obj_id_to_idx": {str(k): int(v) for k, v in tracker_state.get("obj_id_to_idx", {}).items()},
        "local_obj_id_to_idx": {str(k): int(v) for k, v in (out.get("local_obj_id_to_idx") or {}).items()},
        "multiplex_assignments": [[int(x) for x in values] for values in assignments],
        "multiplex_bucket": bucket,
        "multiplex_slot": slot,
        "multiplex_total_valid_entries": int(multiplex.total_valid_entries),
        "multiplex_capacity": int(multiplex.num_buckets * multiplex.multiplex_count),
        "capacity_reached": bool(multiplex.total_valid_entries >= multiplex.num_buckets * multiplex.multiplex_count),
        "roi_valid": box_mean is not None,
        "roi_pixels": roi_pixels,
        "mask_pool_valid": mask_mean is not None,
        "mask_coverage": mask_coverage,
        "mask_pixels": mask_pixels,
        "pointer_valid": pointer is not None,
        "pointer_norm_before_l2": None if pointer is None else float(np.linalg.norm(pointer)),
        "memory_pool_valid": mem_box_mean is not None,
        "memory_feature_name": mem_name,
        "memory_roi_pixels": mem_pixels,
        "image_feature_shape": list(out["image_features"].shape) if torch.is_tensor(out.get("image_features")) else None,
        "obj_ptr_shape": list(out["obj_ptr"].shape) if torch.is_tensor(out.get("obj_ptr")) else None,
        "maskmem_shape": list(out["maskmem_features"].shape) if torch.is_tensor(out.get("maskmem_features")) else None,
    }
    return metadata, {
        "roi_mean": box_mean,
        "roi_max": box_max,
        "mask_mean": mask_mean,
        "obj_ptr": pointer,
        "memory_box_mean": mem_box_mean,
    }


def extract_unbound_roi(
    state: dict[str, Any], frame: int, box: np.ndarray, width: int, height: int
) -> tuple[dict[str, Any], dict[str, np.ndarray | None]]:
    """Pool the frame backbone when the prompted candidate missed the 16-object cap.

    The image grid is candidate-independent, so this remains a valid F1 feature.
    Object-conditioned mask/pointer/memory outputs remain invalid by construction.
    """
    for state_index, tracker_state in enumerate(state.get("sam2_inference_states", [])):
        for storage in ("cond_frame_outputs", "non_cond_frame_outputs"):
            out = tracker_state.get("output_dict", {}).get(storage, {}).get(frame)
            if out is None:
                continue
            grid = image_grid(out, 0)
            box_mean, box_max, roi_pixels = pool_box(grid, box, width, height)
            if box_mean is None:
                continue
            multiplex = tracker_state.get("multiplex_state")
            return {
                "binding_valid": False,
                "reason": "prompt_candidate_not_returned_but_frame_backbone_is_candidate_independent",
                "state_index": state_index,
                "selected_obj_id": None,
                "local_object_index": None,
                "multiplex_bucket": None,
                "multiplex_slot": None,
                "capacity_reached": bool(
                    multiplex is not None
                    and multiplex.total_valid_entries
                    >= multiplex.num_buckets * multiplex.multiplex_count
                ),
                "roi_valid": True,
                "roi_pixels": roi_pixels,
                "mask_pool_valid": False,
                "pointer_valid": False,
                "memory_pool_valid": False,
                "image_feature_shape": list(out["image_features"].shape)
                if torch.is_tensor(out.get("image_features"))
                else None,
            }, {
                "roi_mean": box_mean,
                "roi_max": box_max,
                "mask_mean": None,
                "obj_ptr": None,
                "memory_box_mean": None,
            }
    return {
        "binding_valid": False,
        "reason": "prompt_candidate_not_returned_and_no_frame_backbone_found",
        "roi_valid": False,
        "mask_pool_valid": False,
        "pointer_valid": False,
        "memory_pool_valid": False,
    }, {}


def reset_and_check(backend) -> dict[str, Any]:
    backend._predictor.handle_request({"type": "reset_session", "session_id": backend._session_id})
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    return {
        "sam2_state_count": len(state.get("sam2_inference_states", [])),
        "tracker_metadata_count": len(state.get("tracker_metadata", {})),
        "cached_frame_output_count": len(state.get("cached_frame_outputs", {})),
        "action_history_count": len(state.get("action_history", [])),
        "feature_cache_frame_keys": sorted(int(key) for key in state.get("feature_cache", {}) if isinstance(key, int)),
        "clean": not state.get("sam2_inference_states") and not state.get("tracker_metadata") and not state.get("cached_frame_outputs") and not state.get("action_history"),
    }


def prompt_request(backend, frame: int, box: np.ndarray) -> dict[str, Any]:
    width, height = backend._frame_w, backend._frame_h
    x1, y1, x2, y2 = map(float, box)
    return backend._predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": backend._session_id,
            "frame_index": frame,
            "text": "person",
            "bounding_boxes": [[x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]],
            "bounding_box_labels": [1],
            "clear_old_boxes": True,
        }
    )


def choose_groups() -> list[list[dict[str, Any]]]:
    rows = [row for row in load_jsonl(EPISODES) if row["sequence"] in SEQUENCES]
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sequence"], int(row["decision_frame"]), int(row["gid"]))].append(row)
    chosen = []
    for sequence in SEQUENCES:
        groups = sorted((members for key, members in grouped.items() if key[0] == sequence), key=lambda members: (members[0]["decision_frame"], members[0]["gid"]))
        present = [members for members in groups if any(bool(row["positive"]) for row in members)]
        absent = [members for members in groups if not any(bool(row["positive"]) for row in members)]
        if not present:
            raise RuntimeError(f"no positive smoke group for {sequence}")
        chosen.append(present[0])
        if absent:
            chosen.append(absent[0])
        else:
            chosen.append(present[len(present) // 2])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = choose_groups()
    selection = []
    for members in groups:
        first = members[0]
        selection.append(
            {
                "sequence": first["sequence"],
                "decision_frame": int(first["decision_frame"]),
                "gid": int(first["gid"]),
                "query_frame": int(first["correction_frame"]),
                "candidate_count": len(members),
                "positive_ranks": [int(row["candidate_rank"]) for row in members if bool(row["positive"])],
                "label": "VISIBLE_AND_CANDIDATE_PRESENT" if any(bool(row["positive"]) for row in members) else "CANDIDATE_SET_POSITIVE_ABSENT",
            }
        )
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    runner = CFABackendRunner(checkpoint_path=str(CKPT), split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    backend._predictor.model.eval()
    for parameter in backend._predictor.model.parameters():
        parameter.requires_grad_(False)

    records: list[dict[str, Any]] = []
    feature_rows: list[dict[str, np.ndarray | None]] = []
    query_cache: dict[tuple[str, int, int], tuple[dict[str, Any], dict[str, np.ndarray | None]]] = {}
    reset_checks = []
    propagation_errors: list[dict[str, Any]] = []
    previous_sequence = None
    started = time.time()
    for group_index, members in enumerate(groups):
        first = members[0]
        sequence = str(first["sequence"])
        if sequence != previous_sequence:
            if backend._session_id is not None:
                backend.close()
            backend.start_video(str(DATA / "train" / sequence / "img1"))
            previous_sequence = sequence
        width, height = backend._frame_w, backend._frame_h
        query_key = (sequence, int(first["correction_frame"]), int(first["gid"]))
        if query_key not in query_cache:
            clean_before = reset_and_check(backend)
            clean_before.update({"kind": "query_pre", "key": f"{query_key[0]}:{query_key[1]}:{query_key[2]}"})
            reset_checks.append(clean_before)
            query_box = np.asarray(first["legal_human_positive"]["box"], dtype=np.float32)
            response = prompt_request(backend, query_key[1], query_box)
            candidates = response_candidates(response, width, height)
            selected, match = select_candidate(query_box, candidates)
            if selected is None:
                query_meta, query_features = {"binding_valid": False, "reason": "query_prompt_no_matched_output"}, {}
            else:
                state = backend._predictor._all_inference_states[backend._session_id]["state"]
                query_meta, query_features = extract_selected(state, selected["obj_id"], query_key[1], selected["box"], width, height)
                query_meta.update(match)
                query_meta["all_returned"] = [{"obj_id": candidate["obj_id"], "box": candidate["box"].tolist(), "score": candidate["score"], "sam2_score": candidate["sam2_score"]} for candidate in candidates]
            query_cache[query_key] = (query_meta, query_features)
            records.append(
                {
                    "kind": "query",
                    "sequence": sequence,
                    "gid": query_key[2],
                    "frame": query_key[1],
                    "box": query_box.tolist(),
                    "selected_obj_id": None if selected is None else selected["obj_id"],
                    "metadata": query_meta,
                    "feature_row": len(feature_rows),
                }
            )
            feature_rows.append(query_features)

        for candidate in sorted(members, key=lambda row: int(row["candidate_rank"])):
            event_key = f"{sequence}:{int(first['decision_frame'])}:{int(first['gid'])}:r{int(candidate['candidate_rank'])}"
            clean_before = reset_and_check(backend)
            clean_before.update({"kind": "candidate_pre", "key": event_key})
            reset_checks.append(clean_before)
            start_frame = int(candidate["decision_frame"])
            start_box = np.asarray(candidate["candidate_start_box"], dtype=np.float32)
            prompt_response = prompt_request(backend, start_frame, start_box)
            state = backend._predictor._all_inference_states[backend._session_id]["state"]
            root_candidates = response_candidates(prompt_response, width, height)
            root_selected, root_match = select_candidate(start_box, root_candidates)
            if root_selected is None:
                root_box = start_box.copy()
                root_obj_id = None
                root_metadata, root_features = extract_unbound_roi(
                    state, start_frame, root_box, width, height
                )
                root_metadata["prompt_binding_failure"] = True
            else:
                root_box = root_selected["box"]
                root_obj_id = root_selected["obj_id"]
                root_metadata, root_features = extract_selected(
                    state, root_obj_id, start_frame, root_box, width, height
                )
                root_metadata["prompt_binding_failure"] = False
            root_metadata.update(root_match)
            root_metadata["obj_id_switch"] = False
            root_metadata["observation_source"] = "add_prompt_root"
            root_metadata["all_returned"] = [
                {
                    "obj_id": item["obj_id"],
                    "box": item["box"].tolist(),
                    "score": item["score"],
                    "sam2_score": item["sam2_score"],
                }
                for item in root_candidates
            ]
            records.append(
                {
                    "kind": "candidate",
                    "event_key": event_key,
                    "sequence": sequence,
                    "gid": int(first["gid"]),
                    "decision_frame": start_frame,
                    "frame": start_frame,
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "positive": bool(candidate["positive"]),
                    "group_label": "VISIBLE_AND_CANDIDATE_PRESENT" if any(bool(row["positive"]) for row in members) else "CANDIDATE_SET_POSITIVE_ABSENT",
                    "selected_obj_id": root_obj_id,
                    "selected_box": root_box.tolist(),
                    "metadata": root_metadata,
                    "feature_row": len(feature_rows),
                }
            )
            feature_rows.append(root_features)
            if start_frame + 1 < state["num_frames"]:
                set_frame_geometric_prompt(runner, start_frame + 1, None)
            request = {
                "type": "propagate_in_video",
                "session_id": backend._session_id,
                "propagation_direction": "forward",
                "start_frame_index": start_frame,
                # SAM3.1 multiplex needs the historical N20 eight-frame
                # confirmation window.  A four-frame request creates a
                # zero-object propagation batch before any response is emitted.
                "max_frame_num_to_track": args.horizon,
            }
            previous_box = root_box.copy()
            previous_obj_id = root_obj_id
            stream_response_frames: list[int] = []
            try:
                for response in backend._predictor.handle_stream_request(request=request):
                    stream_response_frames.append(int(response["frame_index"]))
            except RuntimeError as exc:
                # The official tracker can terminate an individual shadow after
                # all multiplex objects are dropped.  N20 silently discarded
                # this condition; N25-R keeps every preceding observation and
                # records the exact termination instead of aborting the audit.
                propagation_errors.append(
                    {
                        "event_key": event_key,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "observed_candidate_frames": sum(
                            record.get("event_key") == event_key for record in records
                        ),
                    }
                )
            state = backend._predictor._all_inference_states[backend._session_id]["state"]
            root_metadata["stream_response_frames"] = stream_response_frames
            # With the official hot-start/batched postprocessor, stream delivery
            # is delayed and may not be chronological for a short window.  The
            # official frame-fetch API reads the same cached masks in exact frame
            # order after propagation has completed.
            fetch_stop = min(start_frame + args.horizon, int(state["num_frames"]))
            for frame in range(start_frame + 1, fetch_stop):
                try:
                    _, outputs = backend._predictor.model.fetch_and_process_single_frame_results(
                        state, frame
                    )
                except (KeyError, RuntimeError, IndexError) as exc:
                    propagation_errors.append(
                        {
                            "event_key": event_key,
                            "error_type": f"FrameFetch{type(exc).__name__}",
                            "message": str(exc),
                            "frame": frame,
                            "observed_candidate_frames": sum(
                                record.get("event_key") == event_key for record in records
                            ),
                        }
                    )
                    continue
                candidates = response_candidates({"outputs": outputs}, width, height)
                selected, match = select_candidate(previous_box, candidates)
                if selected is None:
                    metadata, features = {
                        "binding_valid": False,
                        "reason": "delivery_match_failed",
                        "roi_valid": False,
                        "mask_pool_valid": False,
                        "pointer_valid": False,
                        "memory_pool_valid": False,
                        "capacity_reached": len(candidates) >= 16,
                    }, {}
                    selected_box = None
                    selected_obj_id = None
                else:
                    selected_box = selected["box"]
                    selected_obj_id = selected["obj_id"]
                    metadata, features = extract_selected(
                        state, selected_obj_id, frame, selected_box, width, height
                    )
                    previous_box = selected_box.copy()
                metadata.update(match)
                metadata["observation_source"] = "official_cached_frame_fetch"
                metadata["obj_id_switch"] = bool(
                    previous_obj_id is not None
                    and selected_obj_id is not None
                    and selected_obj_id != previous_obj_id
                )
                metadata["all_returned"] = [
                    {
                        "obj_id": item["obj_id"],
                        "box": item["box"].tolist(),
                        "score": item["score"],
                        "sam2_score": item["sam2_score"],
                    }
                    for item in candidates
                ]
                records.append(
                    {
                        "kind": "candidate",
                        "event_key": event_key,
                        "sequence": sequence,
                        "gid": int(first["gid"]),
                        "decision_frame": start_frame,
                        "frame": frame,
                        "candidate_rank": int(candidate["candidate_rank"]),
                        "positive": bool(candidate["positive"]),
                        "group_label": "VISIBLE_AND_CANDIDATE_PRESENT" if any(bool(row["positive"]) for row in members) else "CANDIDATE_SET_POSITIVE_ABSENT",
                        "selected_obj_id": selected_obj_id,
                        "selected_box": None if selected_box is None else selected_box.tolist(),
                        "metadata": metadata,
                        "feature_row": len(feature_rows),
                    }
                )
                feature_rows.append(features)
                if selected_obj_id is not None:
                    previous_obj_id = selected_obj_id
            clean_after = reset_and_check(backend)
            clean_after.update({"kind": "candidate_post", "key": event_key})
            reset_checks.append(clean_after)
        print(f"SMOKE_PROGRESS groups={group_index + 1}/{len(groups)} elapsed_s={time.time() - started:.1f}", flush=True)

    if backend._session_id is not None:
        backend.close()

    dims = {"roi_mean": 256, "roi_max": 256, "mask_mean": 256, "obj_ptr": 256, "memory_box_mean": 256}
    arrays: dict[str, np.ndarray] = {}
    for name, dim in dims.items():
        array = np.full((len(feature_rows), dim), np.nan, dtype=np.float32)
        valid = np.zeros(len(feature_rows), dtype=np.uint8)
        for index, feature in enumerate(feature_rows):
            value = feature.get(name)
            if value is not None and np.asarray(value).size == dim:
                array[index] = np.asarray(value, dtype=np.float32).reshape(dim)
                valid[index] = 1
        arrays[name] = array
        arrays[f"{name}_valid"] = valid
    arrays["keys"] = np.asarray(
        [
            f"{record['kind']}:{record['sequence']}:{record.get('gid')}:{record['frame']}:{record.get('candidate_rank', 0)}"
            for record in records
        ]
    )
    np.savez_compressed(out_dir / "features.npz", **arrays)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (out_dir / "reset_checks.json").write_text(json.dumps(reset_checks, indent=2) + "\n", encoding="utf-8")

    candidate_records = [record for record in records if record["kind"] == "candidate"]
    query_records = [record for record in records if record["kind"] == "query"]
    binding = [bool(record["metadata"].get("binding_valid")) for record in candidate_records]
    roi = [bool(record["metadata"].get("roi_valid")) for record in candidate_records]
    pointer = [bool(record["metadata"].get("pointer_valid")) for record in candidate_records]
    positive_records = [record for record in candidate_records if record["positive"]]
    negative_records = [record for record in candidate_records if not record["positive"]]
    positive_missing = 1.0 - np.mean([bool(record["metadata"].get("roi_valid")) for record in positive_records]) if positive_records else None
    negative_missing = 1.0 - np.mean([bool(record["metadata"].get("roi_valid")) for record in negative_records]) if negative_records else None
    groups_seen = {(record["sequence"], record["decision_frame"], record["gid"]) for record in candidate_records}
    event_frame_counts = Counter(record["event_key"] for record in candidate_records)
    expected_events = sum(len(members) for members in groups)
    summary = {
        "status": "COMPLETE_WITH_RECORDED_TRACK_TERMINATIONS" if propagation_errors else "COMPLETE",
        "gpu": args.gpu,
        "horizon": args.horizon,
        "runtime_s": time.time() - started,
        "selection": selection,
        "candidate_frame_records": len(candidate_records),
        "query_records": len(query_records),
        "groups_seen": len(groups_seen),
        "expected_candidate_events": expected_events,
        "candidate_events_with_root": len(event_frame_counts),
        "candidate_event_h1_coverage": len(event_frame_counts) / expected_events if expected_events else 0.0,
        "candidate_event_h5_coverage": sum(count >= 5 for count in event_frame_counts.values()) / expected_events if expected_events else 0.0,
        "sequences_seen": sorted({record["sequence"] for record in candidate_records}),
        "selected_box_obj_id_trace_coverage": float(np.mean(binding)) if binding else 0.0,
        "roi_feature_coverage": float(np.mean(roi)) if roi else 0.0,
        "query_roi_feature_coverage": float(np.mean([bool(record["metadata"].get("roi_valid")) for record in query_records])) if query_records else 0.0,
        "pointer_binding_coverage": float(np.mean(pointer)) if pointer else 0.0,
        "mask_pool_coverage": float(np.mean([bool(record["metadata"].get("mask_pool_valid")) for record in candidate_records])) if candidate_records else 0.0,
        "memory_pool_coverage": float(np.mean([bool(record["metadata"].get("memory_pool_valid")) for record in candidate_records])) if candidate_records else 0.0,
        "positive_roi_missing_fraction": positive_missing,
        "negative_roi_missing_fraction": negative_missing,
        "positive_negative_missing_gap": None if positive_missing is None or negative_missing is None else abs(positive_missing - negative_missing),
        "ambiguous_binding_fraction": float(np.mean([bool(record["metadata"].get("ambiguous")) for record in candidate_records])) if candidate_records else 0.0,
        "obj_id_switches": sum(bool(record["metadata"].get("obj_id_switch")) for record in candidate_records),
        "capacity_reached_records": sum(bool(record["metadata"].get("capacity_reached")) for record in candidate_records),
        "reset_checks": len(reset_checks),
        "clean_reset_fraction": float(np.mean([bool(check["clean"]) for check in reset_checks])) if reset_checks else 0.0,
        "propagation_error_count": len(propagation_errors),
        "propagation_errors": propagation_errors,
        "query_candidate_path": "IDENTICAL_SAM3_PERSON_PLUS_BOX_ISOLATED_SHADOW_AND_OFFICIAL_MAPPING",
        "gt_feature_use": False,
        "future_gt_use": False,
        "observation_cutoff": "decision_frame through decision_frame+H-1 only; no future labels enter features",
    }
    summary["alignment_gate"] = {
        "roi_coverage_ge_0_95": summary["roi_feature_coverage"] >= 0.95,
        "same_query_candidate_path": True,
        "selected_box_obj_id_traceable": summary["selected_box_obj_id_trace_coverage"] >= 0.95,
        "pointer_coverage_ge_0_80": summary["pointer_binding_coverage"] >= 0.80,
        "missing_gap_le_0_05": bool(
            summary["positive_negative_missing_gap"] is not None
            and summary["positive_negative_missing_gap"] <= 0.05
        ),
        "clean_resets": summary["clean_reset_fraction"] == 1.0,
        "three_sequences_have_groups": set(summary["sequences_seen"]) == set(SEQUENCES),
        "h5_event_coverage_ge_0_90": summary["candidate_event_h5_coverage"] >= 0.90,
    }
    summary["alignment_gate"]["pass"] = bool(all(summary["alignment_gate"].values()))
    (out_dir / "feature_alignment.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("N25R_ALIGNMENT_SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
