"""Frozen R4 progressive target-concept selector.

This is a small, deterministic SeC-inspired adapter over the already trained
N72R7 decoder: the initial human ROI, recent trusted candidate appearance and
long-term stable appearance form a causal concept.  No semantic model or new
checkpoint is introduced, and distractor/solver/public-ID state remains
outside this selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .models.target_id_decoder import HumanConditionedTargetIDDecoder
from .target_candidate_selector import TargetSelectionContext
from .target_id_features import candidate_feature_vector, context_feature_vector


DATA_ROOT = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"


@dataclass(frozen=True)
class _ConceptSelectorConfig:
    none_score: float = 0.0
    admission_score: float = 0.5
    admission_margin: float = 0.2
    concept_score_weight: float = 0.25
    short_term_update: float = 0.25
    long_term_update: float = 0.10
    min_concept_cosine_for_update: float = 0.20


class ProgressiveConceptTargetCandidateSelector:
    """Causal initial/recent/long-term target concept over a frozen decoder."""

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
        self.config = _ConceptSelectorConfig()
        self._dimensions: dict[str, tuple[int, int]] = {}
        self._initial: np.ndarray | None = None
        self._recent: np.ndarray | None = None
        self._long_term: np.ndarray | None = None
        self._predicted_box: list[float] | None = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._previous_raw: int | None = None
        self._previous_scope: str | None = None
        self._trusted_count = 0

    def _image_dimensions(self, sequence: str, frame: int) -> tuple[int, int]:
        if sequence in self._dimensions:
            return self._dimensions[sequence]
        path = f"{DATA_ROOT}/train/{sequence}/img1/{int(frame) + 1:08d}.jpg"
        with Image.open(path) as image:
            value = (int(image.width), int(image.height))
        self._dimensions[sequence] = value
        return value

    @staticmethod
    def _unit(value: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != 512 or not np.all(np.isfinite(array)):
            raise ValueError("concept feature must be finite 512-D")
        norm = float(np.linalg.norm(array))
        if norm <= 1.0e-6:
            raise ValueError("concept feature has zero norm")
        return (array / norm).astype(np.float32)

    def _concept(self) -> np.ndarray:
        if self._initial is None:
            raise RuntimeError("concept has not been initialized")
        recent = self._recent if self._recent is not None else self._initial
        long_term = self._long_term if self._long_term is not None else self._initial
        concept = 0.50 * self._initial + 0.30 * recent + 0.20 * long_term
        return self._unit(concept)

    @staticmethod
    def _centre(box: Sequence[float] | None) -> np.ndarray | None:
        if box is None:
            return None
        return np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)

    def _update_state(self, candidate: Mapping[str, Any], score: float, margin: float | None) -> bool:
        feature = candidate.get("feature")
        if feature is None or score < self.config.admission_score or margin is None or margin < self.config.admission_margin:
            return False
        value = self._unit(feature)
        concept = self._concept()
        cosine = float(np.dot(value, concept))
        if cosine < self.config.min_concept_cosine_for_update:
            return False
        if self._recent is None:
            self._recent = value
        else:
            self._recent = self._unit((1.0 - self.config.short_term_update) * self._recent + self.config.short_term_update * value)
        if self._long_term is None:
            self._long_term = value
        else:
            self._long_term = self._unit((1.0 - self.config.long_term_update) * self._long_term + self.config.long_term_update * value)
        self._trusted_count = min(self._trusted_count + 1, 8)
        return True

    def select(
        self,
        candidates: list[Mapping[str, Any]],
        *,
        context: TargetSelectionContext,
        base_target_scores: Mapping[str, float | None],
    ) -> dict[str, Any]:
        if context.frame <= context.event_frame:
            raise ValueError("concept future selector may only run after event frame")
        if self._initial is None:
            if context.human_anchor is None or context.predicted_box is None:
                raise RuntimeError("concept selector requires causal human anchor and box")
            self._initial = self._unit(context.human_anchor)
            self._predicted_box = [float(value) for value in context.predicted_box]
            self._previous_raw = context.previous_raw_sam_id
            self._previous_scope = context.previous_native_scope
        sequence = str(candidates[0].get("sequence")) if candidates else ""
        if not sequence:
            raise RuntimeError("concept selector cannot resolve candidate sequence")
        concept = self._concept()
        width, height = self._image_dimensions(sequence, int(context.frame))
        candidate_values = np.stack(
            [
                candidate_feature_vector(
                    candidate,
                    anchor_feature=concept,
                    anchor_box=self._predicted_box or context.predicted_box or [0.0] * 4,
                    predicted_box=self._predicted_box,
                    previous_raw_sam_id=self._previous_raw,
                    previous_native_scope=self._previous_scope,
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
            anchor_feature=concept,
            predicted_box=self._predicted_box,
            anchor_box=self._predicted_box or context.predicted_box or [0.0] * 4,
            velocity=self._velocity,
            previous_raw_sam_id=self._previous_raw,
            frame=int(context.frame),
            event_frame=int(context.event_frame),
            trusted_count=self._trusted_count,
            image_width=width,
            image_height=height,
        )
        candidate_tensor = torch.as_tensor(candidate_values[None], dtype=torch.float32, device=self.device)
        mask_tensor = torch.ones((1, len(candidates)), dtype=torch.bool, device=self.device) if candidates else torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        context_tensor = torch.as_tensor(context_values[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(candidate_tensor, mask_tensor, context_tensor)[0].detach().float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise RuntimeError("concept decoder produced non-finite logits")
        none_index = len(candidates) if candidates else 1
        none_logit = float(logits[none_index])
        raw_scores = logits[: len(candidates)] - none_logit
        concept_cosines = []
        for candidate in candidates:
            value = self._unit(candidate.get("feature")) if candidate.get("feature") is not None else np.zeros(512, dtype=np.float32)
            concept_cosines.append(float(np.dot(value, concept)) if np.linalg.norm(value) > 0 else -1.0)
        guided_scores = raw_scores + self.config.concept_score_weight * np.asarray(concept_cosines, dtype=np.float32)
        order = sorted(range(len(candidates)), key=lambda index: (-float(guided_scores[index]), str(candidates[index]["candidate_uid"])))
        ranked = [
            {
                "candidate_uid": str(candidates[index]["candidate_uid"]),
                "candidate_source": str(candidates[index].get("candidate_source")),
                "raw_sam_id": candidates[index].get("official_raw_sam_id"),
                "model_logit": float(logits[index]),
                "none_logit": none_logit,
                "raw_decoder_score": float(raw_scores[index]),
                "concept_similarity": float(concept_cosines[index]),
                "selector_score": float(guided_scores[index]),
                "runtime_future_gt_used": False,
            }
            for index in order
        ]
        best_index = order[0] if order else None
        second_index = order[1] if len(order) > 1 else None
        best_score = None if best_index is None else float(guided_scores[best_index])
        second_score = None if second_index is None else float(guided_scores[second_index])
        margin = None if best_score is None else float(best_score - max(self.config.none_score, second_score if second_score is not None else self.config.none_score))
        selected_uid = None if best_index is None or best_score is None or best_score <= self.config.none_score or (margin is not None and margin < 0.0) else str(candidates[best_index]["candidate_uid"])
        reliable = bool(selected_uid is not None and best_score is not None and best_score >= self.config.admission_score and margin is not None and margin >= self.config.admission_margin)
        state_update = False
        if selected_uid is not None and best_index is not None:
            chosen = candidates[best_index]
            box = [float(value) for value in chosen["box_xyxy"]]
            old_centre = self._centre(self._predicted_box)
            new_centre = self._centre(box)
            if old_centre is not None and new_centre is not None:
                self._velocity = 0.5 * self._velocity + 0.5 * (new_centre - old_centre)
            self._predicted_box = box
            raw = chosen.get("official_raw_sam_id")
            self._previous_raw = None if raw is None else int(raw)
            self._previous_scope = None if chosen.get("native_scope") is None else str(chosen.get("native_scope"))
            state_update = self._update_state(chosen, float(best_score), margin)
        elif self._predicted_box is not None:
            self._predicted_box = [
                float(self._predicted_box[0] + self._velocity[0]),
                float(self._predicted_box[1] + self._velocity[1]),
                float(self._predicted_box[2] + self._velocity[0]),
                float(self._predicted_box[3] + self._velocity[1]),
            ]
            self._velocity *= 0.5
        return {
            "schema_version": "N72R7_PROGRESSIVE_CONCEPT_TARGET_CANDIDATE_SELECTION_V1",
            "frame": int(context.frame),
            "event_frame": int(context.event_frame),
            "selected_candidate_uid": selected_uid,
            "selected_score": best_score,
            "second_candidate_uid": None if second_index is None else str(candidates[second_index]["candidate_uid"]),
            "second_score": second_score,
            "best_minus_second_margin": margin,
            "none_score": float(self.config.none_score),
            "none_logit": none_logit,
            "none_selected": selected_uid is None,
            "reliable_for_memory_admission": reliable,
            "ranked_candidates": ranked,
            "candidate_count": len(candidates),
            "memory_read": bool(context.memory_read),
            "event_frame_memory_read": False,
            "runtime_future_gt_used": False,
            "public_id_inference": False,
            "concept_state": {
                "initial_weight": 0.50,
                "recent_weight": 0.30,
                "long_term_weight": 0.20,
                "concept_score_weight": self.config.concept_score_weight,
                "recent_memory_count": self._trusted_count,
                "state_updated": state_update,
                "distractors_update_concept": False,
            },
            "learned_decoder_checkpoint_sha256": str(self.protocol["checkpoint_sha256"]),
            "learned_decoder_protocol_sha256": str(self.protocol["protocol_sha256"]),
        }


__all__ = ["ProgressiveConceptTargetCandidateSelector"]
