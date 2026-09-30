"""Record-only correction update interface reserved for N72R21.

N72R20 must not learn or mutate identity memory.  This adapter makes the
future input/output boundary explicit while returning the memory unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class CorrectionMemoryUpdater:
    """Dummy updater that records a correction without changing state."""

    def update_from_correction(
        self,
        memory_state: Any,
        wrong_observation: Any,
        corrected_observation: Any,
    ) -> dict[str, Any]:
        """Return a trace of the future update contract; never mutate memory."""

        preserved_state = deepcopy(memory_state)
        return {
            "updated": False,
            "memory_state_before": preserved_state,
            "memory_state_after": deepcopy(preserved_state),
            "wrong_observation": deepcopy(wrong_observation),
            "corrected_observation": deepcopy(corrected_observation),
            "reason": "N72R20_record_only; learned_correction_update_reserved_for_N72R21",
        }


__all__ = ["CorrectionMemoryUpdater"]
