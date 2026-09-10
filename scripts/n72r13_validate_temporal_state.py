#!/usr/bin/env python3
"""Small non-scientific contract test for N72R13 state cloning."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.temporal_intervention_value import (
    clone_temporal_state,
    temporal_state_digest,
)
from sam3_intermot.reacquisition.temporal_state_policy import TemporalIdentityState


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    rng = np.random.default_rng(7213)
    anchor = rng.normal(size=512).astype(np.float32)
    state = TemporalIdentityState(
        predicted_box=[1.0, 2.0, 10.0, 20.0],
        previous_raw_sam_id=7,
        previous_native_scope="toy",
        previous_score=0.5,
        previous_uncertainty=0.25,
        trusted_age=3,
        recent_trusted=[anchor],
        long_term_trusted=[anchor.copy()],
        distractors=[-anchor],
    )
    clone = clone_temporal_state(state)
    before = temporal_state_digest(state)
    clone.predicted_box[0] += 100.0
    clone.recent_trusted[0][0] += 100.0
    clone.long_term_trusted.append(anchor.copy())
    clone.distractors.clear()
    after = temporal_state_digest(state)
    passed = before == after and state.predicted_box[0] == 1.0 and len(state.long_term_trusted) == 1 and len(state.distractors) == 1
    payload = {
        "schema_version": "N72R13_STAGE_STATUS_V1",
        "stage": "N72R13-01-STATE-CLONE-PASSIVE-ROLLOUT-CONTRACT",
        "status": "PASS_N72R13_STATE_CLONE_CONTRACT" if passed else "FAIL_N72R13_STATE_CLONE_CONTRACT",
        "test_fixture": "toy_non_scientific",
        "state_before_sha256": before,
        "state_after_original_sha256": after,
        "branch_memory_is_independent": bool(passed),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "real_human_evidence": False,
        "historical_outputs_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(ROOT / "outputs/N72R13/stage_01_status.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
