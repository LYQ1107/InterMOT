"""Causal K-path target selector for the N72R7 R3 route.

This module changes only target-candidate selection.  It keeps multiple
candidate histories over the same sealed pool, while the exact public-ID
solver remains the caller's authority.  No public ID, GT label, or future
metric enters the beam state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .models.target_id_decoder import HumanConditionedTargetIDDecoder
from .target_candidate_selector import TargetSelectionContext, box_iou
from .target_id_features import candidate_feature_vector, context_feature_vector


DATA_ROOT = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"


@dataclass
class _BeamPath:
    cumulative_log_score: float
    last_candidate_uid: str | None
    last_box: list[float] | None
    predicted_box: list[float] | None
    velocity: np.ndarray
    previous_raw_sam_id: int | None
    previous_native_scope: str | None
    appearance_anchor: np.ndarray
    recent_features: list[np.ndarray]
    last_selector_score: float | None
    last_margin: float | None
    last_none_logit: float | None


@dataclass(frozen=True)
class _BeamSelectorConfig:
    none_score: float = 0.0


class HypothesisBeamTargetCandidateSelector:
    """A fixed K=3 causal candidate-history beam.

    The beam expands each path with its top three decoder candidates plus an
    explicit NONE action, scores each extension by the decoder log-probability,
    and retains diverse paths using both candidate UID and box IoU.  Trusted
    appearance updates are admitted only under the same fixed score/margin
    rule as the greedy decoder; they are path-local and bounded.
    """

    beam_size = 3
    expansion_k = 3
    diversity_box_iou = 0.70
    admission_score = 0.5
    admission_margin = 0.2

    def __init__(self, checkpoint: Any, *, device: torch.device, protocol: Mapping[str, Any]) -> None:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = dict(payload.get("model_config", {}))
        required = {"candidate_feature_dim", "context_feature_dim", "hidden_dim", "layers", "heads", "dropout"}
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
        self.protocol = protocol
        # The frozen replay runner reads only this compatibility field when
        # adding target-public evidence; the beam never owns public authority.
        self.config = _BeamSelectorConfig()
        self._dimensions: dict[str, tuple[int, int]] = {}
        self._anchor_box: list[float] | None = None
        self._event_frame: int | None = None
        self._paths: list[_BeamPath] = []

    def _image_dimensions(self, sequence: str, frame: int) -> tuple[int, int]:
        if sequence in self._dimensions:
            return self._dimensions[sequence]
        image_path = (
            f"{DATA_ROOT}/train/{sequence}/img1/{int(frame) + 1:08d}.jpg"
        )
        with Image.open(image_path) as image:
            dimensions = (int(image.width), int(image.height))
        self._dimensions[sequence] = dimensions
        return dimensions

    @staticmethod
    def _centre(box: Sequence[float] | None) -> np.ndarray | None:
        if box is None:
            return None
        return np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)

    def _initial_path(self, context: TargetSelectionContext) -> _BeamPath:
        if context.predicted_box is None or context.human_anchor is None:
            raise RuntimeError("beam selector requires causal anchor box and feature")
        anchor = np.asarray(context.human_anchor, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(anchor))
        if anchor.size != 512 or norm <= 1.0e-6 or not np.all(np.isfinite(anchor)):
            raise RuntimeError("beam selector received invalid human anchor")
        anchor = anchor / norm
        return _BeamPath(
            cumulative_log_score=0.0,
            last_candidate_uid=None,
            last_box=None,
            predicted_box=[float(value) for value in context.predicted_box],
            velocity=np.zeros(2, dtype=np.float64),
            previous_raw_sam_id=context.previous_raw_sam_id,
            previous_native_scope=context.previous_native_scope,
            appearance_anchor=anchor.astype(np.float32),
            recent_features=[],
            last_selector_score=None,
            last_margin=None,
            last_none_logit=None,
        )

    def _path_logits(
        self,
        path: _BeamPath,
        candidates: Sequence[Mapping[str, Any]],
        *,
        sequence: str,
        frame: int,
        event_frame: int,
        base_target_scores: Mapping[str, float | None],
    ) -> tuple[np.ndarray, list[dict[str, Any]], float]:
        width, height = self._image_dimensions(sequence, frame)
        candidate_values = np.stack(
            [
                candidate_feature_vector(
                    candidate,
                    anchor_feature=path.appearance_anchor,
                    anchor_box=self._anchor_box or path.predicted_box or [0.0] * 4,
                    predicted_box=path.predicted_box,
                    previous_raw_sam_id=path.previous_raw_sam_id,
                    previous_native_scope=path.previous_native_scope,
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
            anchor_feature=path.appearance_anchor,
            predicted_box=path.predicted_box,
            anchor_box=self._anchor_box or path.predicted_box or [0.0] * 4,
            velocity=path.velocity,
            previous_raw_sam_id=path.previous_raw_sam_id,
            frame=frame,
            event_frame=event_frame,
            trusted_count=len(path.recent_features),
            image_width=width,
            image_height=height,
        )
        candidate_tensor = torch.as_tensor(candidate_values[None], dtype=torch.float32, device=self.device)
        mask_tensor = torch.ones((1, len(candidates)), dtype=torch.bool, device=self.device) if candidates else torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        context_tensor = torch.as_tensor(context_values[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(candidate_tensor, mask_tensor, context_tensor)[0].detach().float().cpu().numpy()
        candidate_logits = logits[: len(candidates)]
        none_index = len(candidates) if candidates else 1
        none_logit = float(logits[none_index])
        if not np.isfinite(logits).all():
            raise RuntimeError("beam decoder produced non-finite logits")
        scores = candidate_logits - none_logit
        order = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), str(candidates[index]["candidate_uid"])))
        ranked = [
            {
                "candidate_uid": str(candidates[index]["candidate_uid"]),
                "candidate_source": str(candidates[index].get("candidate_source")),
                "raw_sam_id": candidates[index].get("official_raw_sam_id"),
                "model_logit": float(candidate_logits[index]),
                "none_logit": none_logit,
                "selector_score": float(scores[index]),
                "runtime_future_gt_used": False,
            }
            for index in order
        ]
        normalized_logits = logits if candidates else np.asarray([none_logit], dtype=np.float32)
        return normalized_logits, ranked, none_logit

    def _extend(
        self,
        path: _BeamPath,
        candidate: Mapping[str, Any] | None,
        *,
        candidate_score: float,
        none_logit: float,
        log_probability: float,
        margin: float | None,
    ) -> _BeamPath:
        if candidate is None:
            predicted = None if path.predicted_box is None else [
                float(path.predicted_box[0] + path.velocity[0]),
                float(path.predicted_box[1] + path.velocity[1]),
                float(path.predicted_box[2] + path.velocity[0]),
                float(path.predicted_box[3] + path.velocity[1]),
            ]
            return _BeamPath(
                cumulative_log_score=path.cumulative_log_score + float(log_probability),
                last_candidate_uid=None,
                last_box=None,
                predicted_box=predicted,
                velocity=0.5 * path.velocity,
                previous_raw_sam_id=path.previous_raw_sam_id,
                previous_native_scope=path.previous_native_scope,
                appearance_anchor=path.appearance_anchor.copy(),
                recent_features=list(path.recent_features[-3:]),
                last_selector_score=None,
                last_margin=margin,
                last_none_logit=none_logit,
            )
        box = [float(value) for value in candidate["box_xyxy"]]
        old_centre = self._centre(path.predicted_box)
        new_centre = self._centre(box)
        velocity = path.velocity.copy()
        if old_centre is not None and new_centre is not None:
            velocity = 0.5 * velocity + 0.5 * (new_centre - old_centre)
        feature = candidate.get("feature")
        appearance = path.appearance_anchor.copy()
        recent = list(path.recent_features[-3:])
        if feature is not None and candidate_score >= self.admission_score and (margin is None or margin >= self.admission_margin):
            value = np.asarray(feature, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(value))
            if value.size == 512 and norm > 1.0e-6 and np.all(np.isfinite(value)):
                value = value / norm
                appearance = (0.75 * appearance + 0.25 * value).astype(np.float32)
                appearance /= max(float(np.linalg.norm(appearance)), 1.0e-6)
                recent.append(value)
        raw = candidate.get("official_raw_sam_id")
        return _BeamPath(
            cumulative_log_score=path.cumulative_log_score + float(log_probability),
            last_candidate_uid=str(candidate["candidate_uid"]),
            last_box=box,
            predicted_box=box,
            velocity=velocity,
            previous_raw_sam_id=None if raw is None else int(raw),
            previous_native_scope=None if candidate.get("native_scope") is None else str(candidate.get("native_scope")),
            appearance_anchor=appearance,
            recent_features=recent[-3:],
            last_selector_score=float(candidate_score),
            last_margin=margin,
            last_none_logit=none_logit,
        )

    def _retain_diverse(self, paths: Sequence[_BeamPath]) -> tuple[list[_BeamPath], dict[str, Any]]:
        ordered = sorted(
            paths,
            key=lambda item: (-float(item.cumulative_log_score), str(item.last_candidate_uid)),
        )
        selected: list[_BeamPath] = []
        rejected_for_duplicate = 0
        rejected_for_box_overlap = 0
        for path in ordered:
            if len(selected) >= self.beam_size:
                break
            if path.last_candidate_uid is not None and any(
                path.last_candidate_uid == other.last_candidate_uid for other in selected
            ):
                rejected_for_duplicate += 1
                continue
            if path.last_box is not None and any(
                other.last_box is not None and box_iou(path.last_box, other.last_box) >= self.diversity_box_iou
                for other in selected
            ):
                rejected_for_box_overlap += 1
                continue
            selected.append(path)
        # If the candidate set is smaller than K, retain the best duplicate
        # histories rather than silently shrinking the causal beam.
        if len(selected) < self.beam_size:
            selected_ids = {id(item) for item in selected}
            for path in ordered:
                if id(path) in selected_ids:
                    continue
                selected.append(path)
                selected_ids.add(id(path))
                if len(selected) == self.beam_size:
                    break
        return selected, {
            "beam_size": self.beam_size,
            "path_count_before_retention": len(paths),
            "path_count_after_retention": len(selected),
            "rejected_duplicate_last_candidate": rejected_for_duplicate,
            "rejected_high_box_overlap": rejected_for_box_overlap,
            "retained_last_candidate_uids": [path.last_candidate_uid for path in selected],
            "retained_cumulative_log_scores": [float(path.cumulative_log_score) for path in selected],
            "diversity_box_iou_threshold": self.diversity_box_iou,
        }

    def select(
        self,
        candidates: list[Mapping[str, Any]],
        *,
        context: TargetSelectionContext,
        base_target_scores: Mapping[str, float | None],
    ) -> dict[str, Any]:
        if context.frame <= context.event_frame:
            raise ValueError("beam future selector may only run after event frame")
        if self._event_frame is None:
            self._event_frame = int(context.event_frame)
            if context.predicted_box is None:
                raise RuntimeError("beam selector has no causal anchor box")
            self._anchor_box = [float(value) for value in context.predicted_box]
            self._paths = [self._initial_path(context)]
        if not self._paths:
            self._paths = [self._initial_path(context)]
        sequence = str(candidates[0].get("sequence")) if candidates else ""
        if not sequence:
            raise RuntimeError("beam selector cannot resolve candidate sequence")
        frame = int(context.frame)
        event_frame = int(context.event_frame)
        expansions: list[_BeamPath] = []
        base_ranked: list[dict[str, Any]] = []
        base_logits: np.ndarray | None = None
        base_none_logit: float | None = None
        for path in self._paths:
            logits, ranked, none_logit = self._path_logits(
                path,
                candidates,
                sequence=sequence,
                frame=frame,
                event_frame=event_frame,
                base_target_scores=base_target_scores,
            )
            if base_logits is None or path is self._paths[0]:
                base_logits, base_ranked, base_none_logit = logits, ranked, none_logit
            scores = logits[: len(candidates)] - none_logit
            valid_logits = logits[: len(candidates) + 1]
            log_probs = valid_logits - (float(np.max(valid_logits)) + float(np.log(np.exp(valid_logits - np.max(valid_logits)).sum())))
            order = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), str(candidates[index]["candidate_uid"])))
            top_indices = order[: self.expansion_k]
            second_score = None if len(order) < 2 else float(scores[order[1]])
            best_score = None if not order else float(scores[order[0]])
            margin = None if best_score is None else best_score - max(0.0, second_score if second_score is not None else 0.0)
            for index in top_indices:
                expansions.append(
                    self._extend(
                        path,
                        candidates[index],
                        candidate_score=float(scores[index]),
                        none_logit=none_logit,
                        log_probability=float(log_probs[index]),
                        margin=margin,
                    )
                )
            expansions.append(
                self._extend(
                    path,
                    None,
                    candidate_score=0.0,
                    none_logit=none_logit,
                    log_probability=float(log_probs[len(candidates)]),
                    margin=margin,
                )
            )
            for item in ranked:
                item["path_source_cumulative_log_score"] = float(path.cumulative_log_score)
            if path is self._paths[0]:
                base_ranked = ranked
        self._paths, diversity = self._retain_diverse(expansions)
        best_path = self._paths[0]
        best_score = best_path.last_selector_score
        second_score = None
        if base_ranked and len(base_ranked) > 1:
            second_score = float(base_ranked[1]["selector_score"])
        margin = best_path.last_margin
        selected_uid = None
        if best_path.last_candidate_uid is not None and best_score is not None and best_score > 0.0 and (margin is None or margin >= 0.0):
            selected_uid = best_path.last_candidate_uid
        reliable = bool(selected_uid is not None and best_score is not None and best_score >= self.admission_score and margin is not None and margin >= self.admission_margin)
        return {
            "schema_version": "N72R7_HYPOTHESIS_BEAM_TARGET_CANDIDATE_SELECTION_V1",
            "frame": frame,
            "event_frame": event_frame,
            "selected_candidate_uid": selected_uid,
            "selected_score": best_score,
            "second_candidate_uid": None if len(base_ranked) < 2 else base_ranked[1]["candidate_uid"],
            "second_score": second_score,
            "best_minus_second_margin": margin,
            "none_score": 0.0,
            "none_logit": base_none_logit,
            "none_selected": selected_uid is None,
            "reliable_for_memory_admission": reliable,
            "ranked_candidates": base_ranked,
            "candidate_count": len(candidates),
            "memory_read": bool(context.memory_read),
            "event_frame_memory_read": False,
            "runtime_future_gt_used": False,
            "public_id_inference": False,
            "beam_size": self.beam_size,
            "beam_expansion_k": self.expansion_k,
            "beam_diversity": diversity,
            "beam_hypotheses": [
                {
                    "last_candidate_uid": path.last_candidate_uid,
                    "last_box": path.last_box,
                    "cumulative_log_score": float(path.cumulative_log_score),
                    "previous_raw_sam_id": path.previous_raw_sam_id,
                    "previous_native_scope": path.previous_native_scope,
                    "recent_memory_count": len(path.recent_features),
                }
                for path in self._paths
            ],
            "learned_decoder_checkpoint_sha256": str(self.protocol["checkpoint_sha256"]),
            "learned_decoder_protocol_sha256": str(self.protocol["protocol_sha256"]),
        }


__all__ = ["HypothesisBeamTargetCandidateSelector"]
