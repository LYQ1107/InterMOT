"""Deterministic observation corruption for the N72R19 memory probe.

The simulator operates on already-frozen 512-D embeddings.  It never reads an
image, creates a crop, or changes the inherited protocol.  Corruption choices
are keyed by episode and frame so paired methods see identical Noise A/B draws.
Noise C chooses its replacement from the current method's pre-update state by
definition; that choice is therefore recorded per method.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import torch

from .encoder import normalize

NoiseKind = Literal[
    "clean",
    "wrong_identity_injection",
    "missing_observation",
    "hard_negative_replacement",
]

NOISE_KINDS: tuple[str, ...] = (
    "clean",
    "wrong_identity_injection",
    "missing_observation",
    "hard_negative_replacement",
)
NOISY_RATES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class NoiseSpec:
    """One frozen corruption condition."""

    kind: NoiseKind
    rate: float
    seed: int = 7219

    def __post_init__(self) -> None:
        if self.kind not in NOISE_KINDS:
            raise ValueError(f"unknown N72R19 noise kind: {self.kind}")
        if not 0.0 <= float(self.rate) <= 1.0:
            raise ValueError("noise rate must be in [0, 1]")
        if self.kind == "clean" and float(self.rate) != 0.0:
            raise ValueError("clean condition must use rate 0")


@dataclass(frozen=True)
class ObservationCorruption:
    """Result of corrupting one current same-identity observation."""

    observation: torch.Tensor
    present: bool
    applied: bool
    replacement_index: int | None
    source: str


def _uniform(seed: int, *parts: object) -> float:
    payload = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64)


def _replacement_index(
    previous_state: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
    spec: NoiseSpec,
    episode_key: str,
    step: int,
) -> int | None:
    valid = torch.nonzero(competitor_mask, as_tuple=False).flatten()
    if valid.numel() == 0:
        return None
    if spec.kind == "wrong_identity_injection":
        draw = _uniform(spec.seed, episode_key, step, "wrong_identity")
        return int(valid[int(draw * valid.numel()) % valid.numel()].item())
    if spec.kind == "hard_negative_replacement":
        state = normalize(previous_state.detach())
        candidates = normalize(competitors[valid].detach())
        scores = torch.matmul(candidates, state)
        # Stable track ordering is supplied by the protocol tensor order; ties
        # are resolved by argmax's first occurrence and remain deterministic.
        return int(valid[int(torch.argmax(scores).item())].item())
    return None


def corrupt_observation(
    previous_state: torch.Tensor,
    clean_observation: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
    spec: NoiseSpec,
    episode_key: str,
    step: int,
) -> ObservationCorruption:
    """Apply one frozen N72R19 corruption to a current observation.

    ``previous_state`` is used only to choose Noise C's hard competitor.  It
    is detached for that selection, so the simulator cannot create a gradient
    path through the corruption decision.
    """

    if spec.kind == "clean" or spec.rate == 0.0:
        return ObservationCorruption(clean_observation, True, False, None, "same_identity")

    applied = _uniform(spec.seed, episode_key, step, "apply") < spec.rate
    if not applied:
        return ObservationCorruption(clean_observation, True, False, None, "same_identity")

    if spec.kind == "missing_observation":
        return ObservationCorruption(clean_observation, False, True, None, "missing")

    replacement = _replacement_index(
        previous_state,
        competitors,
        competitor_mask,
        spec,
        episode_key,
        step,
    )
    if replacement is None:
        return ObservationCorruption(clean_observation, True, False, None, "same_identity_no_competitor")
    source = "wrong_visible_identity" if spec.kind == "wrong_identity_injection" else "hard_negative_visible_identity"
    return ObservationCorruption(competitors[replacement], True, True, replacement, source)


def training_noise_spec(episode_key: str, step: int, seed: int = 7219) -> NoiseSpec:
    """Return the frozen mixed-noise training condition for one step."""

    draw = _uniform(seed, episode_key, step, "training_kind")
    if draw < 0.20:
        return NoiseSpec("clean", 0.0, seed)
    if draw < 0.50:
        kind: NoiseKind = "wrong_identity_injection"
    elif draw < 0.70:
        kind = "missing_observation"
    else:
        kind = "hard_negative_replacement"
    rate_index = int(_uniform(seed, episode_key, step, "training_rate") * len(NOISY_RATES))
    rate = NOISY_RATES[min(rate_index, len(NOISY_RATES) - 1)]
    return NoiseSpec(kind, rate, seed)


__all__ = [
    "NOISE_KINDS",
    "NOISY_RATES",
    "NoiseKind",
    "NoiseSpec",
    "ObservationCorruption",
    "corrupt_observation",
    "training_noise_spec",
]
