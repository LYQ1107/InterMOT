"""Small frozen-backbone identity verifiers for N72R20R3R1."""

from .explicit_none_verifier import ExplicitNoneVerifier, predict_open_set_identity
from .cross_scene_adapter import CrossSceneIdentityAdapter
from .frozen_none_head import FrozenMetricNoneHead

__all__ = ["CrossSceneIdentityAdapter", "ExplicitNoneVerifier", "FrozenMetricNoneHead", "predict_open_set_identity"]
