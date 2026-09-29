"""Human identity representation probe for N72R16.

This package intentionally stops at identity-signal measurement.  It does not
implement tracking, association, SAM3 inference, or MOT evaluation.
"""

from .dataset import GTBox, DanceTrackSequence, discover_sequences
from .encoders import OSNetEncoder

__all__ = ["GTBox", "DanceTrackSequence", "discover_sequences", "OSNetEncoder"]
