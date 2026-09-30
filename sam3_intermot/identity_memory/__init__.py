"""N72R18/N72R19 Human Identity Memory Learning components.

This package operates only on the frozen N72R17 512-D embedding stores.  It
does not encode images, run SAM3, perform detection, or associate tracks.
"""

from .encoder import FEATURE_DIMENSION, FROZEN_ENCODER_NAME, normalize
from .memory import GRUMemoryUpdater, ReliabilityGate
from .noise import NoiseSpec, corrupt_observation, training_noise_spec
from .updater import HIIM, build_updater

__all__ = [
    "FEATURE_DIMENSION",
    "FROZEN_ENCODER_NAME",
    "GRUMemoryUpdater",
    "ReliabilityGate",
    "NoiseSpec",
    "corrupt_observation",
    "training_noise_spec",
    "HIIM",
    "build_updater",
    "normalize",
]
