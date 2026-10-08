"""Causal identity evidence and small, competing authority hypotheses."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn

from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter
from .learned_identity_memory import sha256_file


FEATURE_NAMES = ("identity_top1", "identity_margin", "base_margin", "motion_consistency", "quality", "state_reliability", "track_age", "trusted_gap", "candidate_count", "competitor_cost")


@dataclass(frozen=True)
class AuthorityConfig:
    mode: str = "off"
    strength: float = 0.0
    source: str = "adapter"
    memory: str = "P0"
    identity_mean: float = 0.5
    identity_std: float = 0.2
    base_scale: float = 1.0
    low_margin: float = 0.35
    write_score: float = 0.7
    write_margin: float = 0.05
    write_quality: float = 0.5
    conflict_guard: bool = False
    persistence_guard: bool = False
    lifecycle: str = "dynamic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdapterEnsemble:
    """Strict checkpoint loading and actual shared-axis tower inference."""
    def __init__(self, paths: Sequence[Path], *, expected_shas: Sequence[str] | None = None, forbidden_sequences: Sequence[str] = ()) -> None:
        self.models = []
        self.manifest = []
        self.calls = 0
        for i, path in enumerate(paths):
            digest = sha256_file(path)
            if expected_shas is not None and digest != expected_shas[i]:
                raise ValueError(f"adapter checkpoint SHA mismatch: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("architecture") != "CrossSceneIdentityAdapter" or checkpoint.get("feature_dim") != 512:
                raise ValueError("invalid adapter architecture/dimension")
            # Historic checkpoint metadata omitted its final refit validation
            # sequence. The caller must supply audited actual training lineage.
            actual_fit = checkpoint.get("actual_training_sequences", checkpoint.get("parameter_fit_sequences", []))
            if set(actual_fit) & set(forbidden_sequences):
                raise ValueError("adapter supervision overlaps heldout/inner sequence")
            model = CrossSceneIdentityAdapter(512, int(checkpoint["bottleneck_dim"]))
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            if model.trainable_parameters != int(checkpoint["trainable_parameters"]):
                raise ValueError("adapter parameter count mismatch")
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
            self.manifest.append({"path": str(path), "sha256": digest, "seed": checkpoint["seed"], "actual_training_sequences": list(actual_fit), "parameters": model.trainable_parameters})
        if not self.models:
            raise ValueError("adapter ensemble must contain actual models")

    def encode_candidates(self, features: np.ndarray) -> list[np.ndarray]:
        with torch.inference_mode():
            tensor = torch.as_tensor(features, dtype=torch.float32)
            return [model.encode_candidates(tensor).numpy() for model in self.models]

    def scores(self, state: np.ndarray, features: np.ndarray, encoded: Sequence[np.ndarray] | None = None) -> np.ndarray:
        self.calls += 1
        if len(features) == 0:
            return np.zeros(0, dtype=np.float64)
        with torch.inference_mode():
            query = torch.as_tensor(np.array(state, copy=True), dtype=torch.float32).reshape(1, 512)
            candidates = self.encode_candidates(features) if encoded is None else encoded
            if any(c.shape != features.shape for c in candidates) or len(candidates) != len(self.models):
                raise ValueError("adapter candidate axis mismatch")
            return np.mean([c @ model.encode_query(query).numpy()[0] for model, c in zip(self.models, candidates)], axis=0).astype(np.float64)


class AuthorityController(nn.Module):
    """Scalar/logistic/MLP/structured comparators; < 50k parameters."""
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode == "scalar":
            self.bias = nn.Parameter(torch.zeros(()))
        elif mode == "logistic":
            self.net = nn.Linear(len(FEATURE_NAMES), 1)
        elif mode in {"mlp", "structured"}:
            self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 16), nn.Tanh(), nn.Linear(16, 1))
        else:
            raise ValueError(f"unknown learned authority: {mode}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.mode == "scalar":
            return self.bias.expand(values.shape[0]).sigmoid()
        probability = self.net(values).squeeze(-1).sigmoid()
        if self.mode == "structured":
            # Monotone causal safety factors: identity margin, observation
            # quality, and global opportunity cost. No GT or sequence inputs.
            probability = probability * torch.sigmoid(8 * values[:, 1]) * values[:, 4].clamp(0, 1) * torch.sigmoid(-values[:, 9])
        return probability

    def probability(self, values: np.ndarray) -> float:
        with torch.inference_mode():
            return float(self(torch.as_tensor(values, dtype=torch.float32)[None])[0])


class ControllerEnsemble:
    def __init__(self, models: Sequence[AuthorityController]) -> None:
        if not models or len({m.mode for m in models}) != 1:
            raise ValueError("controller ensemble mode mismatch")
        self.models = list(models)
        self.mode = models[0].mode

    def probability(self, values: np.ndarray) -> float:
        return float(np.mean([m.probability(values) for m in self.models]))


def calibrated_residual(scores: np.ndarray, config: AuthorityConfig) -> np.ndarray:
    if config.identity_std <= 0 or config.base_scale <= 0:
        raise ValueError("invalid training-side identity calibration")
    return np.clip((scores - config.identity_mean) / config.identity_std, -3, 3) * config.base_scale


def authority_value(config: AuthorityConfig, features: np.ndarray, *, missing: bool, controller: AuthorityController | None = None, previous_challenger_native: int | None = None, challenger_native: int | None = None) -> float:
    if config.mode == "off" or config.strength == 0:
        return 0.0
    if config.mode == "fixed":
        value = 1.0
    elif config.mode == "low_confidence":
        value = float(features[2] < config.low_margin)
    elif config.mode == "recovery":
        value = float(missing or features[2] < config.low_margin)
    else:
        if controller is None or controller.mode != config.mode:
            raise ValueError("learned authority missing/mismatched controller")
        value = controller.probability(features)
    if config.conflict_guard and features[9] > max(0.0, features[1] * config.base_scale):
        value = 0.0
    if config.persistence_guard and previous_challenger_native != challenger_native:
        value = 0.0
    return float(value)
