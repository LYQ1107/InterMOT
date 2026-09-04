#!/usr/bin/env python3
"""One-event learned N72R7 closed-loop replay worker.

The worker reuses the sealed N72R7 D1/D2 runtime path and swaps only the
target selector implementation.  The exact global public-ID solver, frozen
B0 stream, event-frame boundary, and public-ID authority remain unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.models.target_id_decoder import HumanConditionedTargetIDDecoder  # noqa: E402
from sam3_intermot.reacquisition.hypothesis_beam import HypothesisBeamTargetCandidateSelector  # noqa: E402
from sam3_intermot.reacquisition.progressive_concept import ProgressiveConceptTargetCandidateSelector  # noqa: E402
from sam3_intermot.reacquisition.target_id_features import (  # noqa: E402
    candidate_feature_vector,
    context_feature_vector,
)
from scripts.n72r7_dev_replay import (  # noqa: E402
    DEV_PROTOCOL,
    TargetSelectionContext,
    atomic_json,
    run_event,
    validate_inputs,
    read_json,
)


DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
DEFAULT_CHECKPOINT = ROOT / "outputs/N72R7/training/HumanConditionedTargetIDDecoder_v1.pt"
DEFAULT_PROTOCOL = ROOT / "outputs/N72R7/training/learned_replay_protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/N72R7/learned_replay/worker"


class LearnedSelectorConfig:
    """Frozen runtime interpretation of decoder logits.

    These values are protocol constants, not validation-selected thresholds.
    The model's best candidate must beat both NONE and the second candidate;
    only a 0.5 logit advantage and 0.2 margin admit trusted memory.
    """

    none_score = 0.0
    admission_score = 0.5
    admission_margin = 0.2


class LearnedTargetCandidateSelector:
    def __init__(self, checkpoint: Path, *, device: torch.device, protocol: Mapping[str, Any]) -> None:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = dict(payload.get("model_config", {}))
        required = {
            "candidate_feature_dim", "context_feature_dim", "hidden_dim", "layers", "heads", "dropout"
        }
        if not required.issubset(config):
            raise RuntimeError("learned decoder checkpoint lacks complete model config")
        self.model = HumanConditionedTargetIDDecoder(
            candidate_feature_dim=int(config["candidate_feature_dim"]),
            context_feature_dim=int(config["context_feature_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            dropout=float(config["dropout"]),
        ).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = device
        self.config = LearnedSelectorConfig()
        self.protocol = protocol
        self._dimensions: dict[str, tuple[int, int]] = {}
        self._anchor_box: list[float] | None = None

    def _image_dimensions(self, sequence: str, frame: int) -> tuple[int, int]:
        if sequence in self._dimensions:
            return self._dimensions[sequence]
        image_path = DATA_ROOT / "train" / sequence / "img1" / f"{int(frame) + 1:08d}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            dimensions = (int(image.width), int(image.height))
        self._dimensions[sequence] = dimensions
        return dimensions

    def select(
        self,
        candidates: list[Mapping[str, Any]],
        *,
        context: TargetSelectionContext,
        base_target_scores: Mapping[str, float | None],
    ) -> dict[str, Any]:
        if context.frame <= context.event_frame:
            raise ValueError("learned future selector may only run after event frame")
        if self._anchor_box is None:
            if context.predicted_box is None:
                raise RuntimeError("learned selector has no causal anchor box")
            self._anchor_box = [float(value) for value in context.predicted_box]
        sequence = str(candidates[0].get("sequence")) if candidates else ""
        if not sequence:
            raise RuntimeError("learned selector cannot resolve candidate sequence")
        width, height = self._image_dimensions(sequence, int(context.frame))
        candidate_values = np.stack(
            [
                candidate_feature_vector(
                    candidate,
                    anchor_feature=context.human_anchor,
                    anchor_box=self._anchor_box,
                    predicted_box=context.predicted_box,
                    previous_raw_sam_id=context.previous_raw_sam_id,
                    previous_native_scope=context.previous_native_scope,
                    image_width=width,
                    image_height=height,
                    candidate_count=len(candidates),
                    base_target_score=base_target_scores.get(str(candidate["candidate_uid"])),
                )
                for candidate in candidates
            ],
            axis=0,
        ) if candidates else np.zeros((1, 530), dtype=np.float32)
        context_values = context_feature_vector(
            anchor_feature=context.human_anchor,
            predicted_box=context.predicted_box,
            anchor_box=self._anchor_box,
            velocity=(0.0, 0.0),
            previous_raw_sam_id=context.previous_raw_sam_id,
            frame=int(context.frame),
            event_frame=int(context.event_frame),
            trusted_count=len(context.trusted_features),
            image_width=width,
            image_height=height,
        )
        candidate_tensor = torch.as_tensor(candidate_values[None], dtype=torch.float32, device=self.device)
        mask_tensor = torch.ones((1, len(candidates)), dtype=torch.bool, device=self.device) if candidates else torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        context_tensor = torch.as_tensor(context_values[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(candidate_tensor, mask_tensor, context_tensor)[0].detach().float().cpu().numpy()
        none_logit = float(logits[len(candidates)])
        scores = [float(value - none_logit) for value in logits[: len(candidates)]]
        ranked_indices = sorted(range(len(candidates)), key=lambda index: (-scores[index], str(candidates[index]["candidate_uid"])))
        ranked: list[dict[str, Any]] = []
        for index in ranked_indices:
            candidate = candidates[index]
            ranked.append({
                "candidate_uid": str(candidate["candidate_uid"]),
                "candidate_source": str(candidate.get("candidate_source")),
                "raw_sam_id": candidate.get("official_raw_sam_id"),
                "model_logit": float(logits[index]),
                "none_logit": none_logit,
                "selector_score": scores[index],
                "runtime_future_gt_used": False,
            })
        best_index = ranked_indices[0] if ranked_indices else None
        second_index = ranked_indices[1] if len(ranked_indices) > 1 else None
        best_score = None if best_index is None else scores[best_index]
        second_score = None if second_index is None else scores[second_index]
        margin = None if best_score is None else best_score - max(0.0, second_score if second_score is not None else 0.0)
        selected_uid = None if best_index is None or best_score is None or best_score <= 0.0 or (margin is not None and margin < 0.0) else str(candidates[best_index]["candidate_uid"])
        reliable = bool(selected_uid is not None and best_score is not None and best_score >= self.config.admission_score and margin is not None and margin >= self.config.admission_margin)
        return {
            "schema_version": "N72R7_LEARNED_TARGET_CANDIDATE_SELECTION_V1",
            "frame": int(context.frame),
            "event_frame": int(context.event_frame),
            "selected_candidate_uid": selected_uid,
            "selected_score": best_score,
            "second_candidate_uid": None if second_index is None else str(candidates[second_index]["candidate_uid"]),
            "second_score": second_score,
            "best_minus_second_margin": margin,
            "none_score": 0.0,
            "none_logit": none_logit,
            "none_selected": selected_uid is None,
            "reliable_for_memory_admission": reliable,
            "ranked_candidates": ranked,
            "candidate_count": len(candidates),
            "memory_read": bool(context.memory_read),
            "event_frame_memory_read": False,
            "runtime_future_gt_used": False,
            "public_id_inference": False,
            "learned_decoder_checkpoint_sha256": str(self.protocol["checkpoint_sha256"]),
            "learned_decoder_protocol_sha256": str(self.protocol["protocol_sha256"]),
        }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("D1", "D2"), required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--replay-protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--selector-mode", choices=("greedy", "beam", "concept"), default="greedy")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    result: dict[str, Any] = {"status": "FAIL", "event_id": args.event_id, "variant": args.variant, "started_at_utc": now_utc()}
    try:
        _, policy, frozen = validate_inputs()
        event_id = str(args.event_id)
        if event_id not in frozen:
            raise ValueError(f"event is not in frozen N72R6 set: {event_id}")
        checkpoint = Path(args.checkpoint)
        replay_protocol = read_json(Path(args.replay_protocol))
        if not checkpoint.is_file() or replay_protocol.get("checkpoint_sha256") != __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest():
            raise RuntimeError("learned checkpoint/protocol hash mismatch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.selector_mode == "beam":
            selector = HypothesisBeamTargetCandidateSelector(checkpoint, device=device, protocol=replay_protocol)
        elif args.selector_mode == "concept":
            selector = ProgressiveConceptTargetCandidateSelector(checkpoint, device=device, protocol=replay_protocol)
        else:
            selector = LearnedTargetCandidateSelector(checkpoint, device=device, protocol=replay_protocol)
        output = run_event(
            policy["events"][next(index for index, item in enumerate(policy["events"]) if str(item["event_id"]) == event_id)],
            frozen[event_id],
            variant=args.variant,
            output_root=output_root,
            selector=selector,
            protocol=read_json(DEV_PROTOCOL),
        )
        result.update({
            "status": output["status"],
            "event_manifest": str(output_root / event_id / "event_manifest.json"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": replay_protocol["checkpoint_sha256"],
            "device": str(device),
            "selector_mode": args.selector_mode,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "finished_at_utc": now_utc(),
        })
        print(json.dumps({"status": result["status"], "event_id": event_id, "variant": args.variant}, sort_keys=True))
        return 0
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "runtime_future_gt_used": False, "posthoc_gt_used": False, "finished_at_utc": now_utc()})
        failure = output_root / "attempts" / f"{args.event_id}.{args.variant}.failure.json"
        atomic_json(failure, result)
        print(json.dumps({"status": "FAIL", "event_id": args.event_id, "variant": args.variant, "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
