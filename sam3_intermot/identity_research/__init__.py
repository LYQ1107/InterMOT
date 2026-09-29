"""N72R17 identity-representation benchmark.

This package deliberately stays outside the SAM3/MOT runtime.  It reuses the
frozen N72R16 protocol and evaluates only visual identity representations and
causal diagnostic memory baselines.
"""

from .encoder import OpenAIClipEncoder

__all__ = ["OpenAIClipEncoder"]
