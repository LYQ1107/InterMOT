#!/usr/bin/env python3
"""N31-F: verify whether a correction masklet can receive future loss gradients."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.correction_state_candidates import BOX_RECTANGLE_MASKLET, rectangle_mask, write_target_mask  # noqa: E402
from scripts.n29_lit_online_replay import _image_files, _install_official_box_singleton, _make_backend, _read_gt, _session  # noqa: E402
from scripts.n29r_paired_replay import _load_manifest  # noqa: E402


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
MANIFEST = ROOT / "outputs/n29r/hard_episode_manifest.json"
OUTPUT = ROOT / "outputs/n31/future_gradient_gate.json"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(type(value).__name__)


def run(*, manifest: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    data, _ = _load_manifest(manifest)
    episode = data["episodes"][0]
    sequence = Path(episode["sequence_path"])
    gt = _read_gt(sequence)
    images = _image_files(sequence)
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    public_id = int(episode["public_id"])
    backend = _make_backend(checkpoint)
    result: dict[str, Any]
    try:
        _session(backend, sequence)
        init_box = np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float)
        backend.add_box(init, public_id, init_box)
        _install_official_box_singleton(backend, frame_idx=init, public_id=public_id, box_xyxy=init_box)
        backend.propagate(init, correction, start_frame_index=init)
        state = backend._predictor._all_inference_states[backend._session_id]["state"]
        # This parameter is intentionally not trained; it tests whether the
        # official writer/future path preserves a graph from a correction mask.
        writer_parameter = torch.nn.Parameter(torch.zeros((int(state["orig_height"]), int(state["orig_width"])), device=state["device"]))
        soft_mask = writer_parameter.sigmoid()
        writer = write_target_mask(
            backend,
            frame_idx=correction,
            public_id=public_id,
            mask=soft_mask,
            provenance=BOX_RECTANGLE_MASKLET,
        )
        future = backend.propagate(correction + 1, min(correction + 5, len(images) - 1), start_frame_index=correction + 1)
        # Official outputs are PromptObjectObservation numpy arrays and the
        # add_new_masks/propagation path is inference_mode.  Constructing a
        # tensor from that output makes the break explicit rather than
        # inventing a surrogate differentiable objective.
        output_area = sum(float(np.asarray(getattr(obs, "mask", np.zeros((1, 1))), dtype=bool).sum()) for values in future.values() for obs in values if int(getattr(obs, "sam_object_id", -1)) == public_id)
        future_loss = torch.as_tensor(output_area, device=writer_parameter.device, dtype=torch.float32)
        backward_error = None
        try:
            future_loss.backward()
        except Exception as exc:
            backward_error = f"{type(exc).__name__}: {exc}"
        grad_count = int(writer_parameter.grad is not None and torch.isfinite(writer_parameter.grad).all().item())
        result = {
            "protocol": "N31-F-FUTURE-GRADIENT-SMOKE",
            "status": "PASS" if grad_count > 0 and future_loss.requires_grad else "FAIL",
            "episode_id": str(episode["episode_id"]),
            "writer": writer,
            "future_frame_count": len(future),
            "future_loss_requires_grad": bool(future_loss.requires_grad),
            "writer_gradient_present": bool(grad_count),
            "writer_gradient_count": grad_count,
            "backward_error": backward_error,
            "reason_if_failed": "official add_new_masks and future propagation are inference_mode/detached; use selector path B" if grad_count == 0 else None,
            "future_gt_used_for_selection": False,
            "val25_read": False,
            "test_labels_used": False,
        }
    except Exception as exc:
        result = {
            "protocol": "N31-F-FUTURE-GRADIENT-SMOKE",
            "status": "NOT_RUN",
            "episode_id": str(episode.get("episode_id")),
            "failure": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(limit=20),
            "future_gt_used_for_selection": False,
            "val25_read": False,
            "test_labels_used": False,
        }
    finally:
        backend.close()
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(manifest=args.manifest, checkpoint=args.checkpoint, output=args.output)
    print(json.dumps({key: result.get(key) for key in ("protocol", "status", "future_loss_requires_grad", "writer_gradient_present")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
