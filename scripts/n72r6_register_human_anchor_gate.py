#!/usr/bin/env python3
"""Register the single N72R6 human-ROI verification probe.

The protocol is intentionally a one-value development probe, not a threshold
scan or a production promotion.  It is written before the gated replay and
does not inspect posthoc labels or future-effect metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import atomic_json  # noqa: E402


PROTOCOL = ROOT / "outputs/N72R6/human_anchor_gate_protocol.json"
STATUS = ROOT / "outputs/N72R6/stage_08_human_anchor_gate_protocol_status.json"
THRESHOLD = 0.85


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    payload = {
        "schema_version": "N72R6_HUMAN_ANCHOR_GATE_PROTOCOL_V1",
        "status": "PASS_N72R6_HUMAN_ANCHOR_GATE_PROTOCOL_REGISTERED",
        "mechanism": "HUMAN_ROI_VERIFICATION_GATE",
        "threshold": THRESHOLD,
        "formula": "dot(l2_normalize(target_session_feature), l2_normalize(human_anchor)) >= 0.85",
        "applies_to": "future target-session candidate frames strictly after event_frame",
        "event_frame_rule": "event-frame authoritative target binding is never rejected by this gate",
        "rejection_semantics": "omit target-session row from solver input; target public may only take explicit NONE/LOST",
        "main_candidate_fallback": False,
        "candidate_generation_changed": False,
        "checkpoint_changed": False,
        "hungarian_solver_changed": False,
        "future_window_changed": False,
        "selection_basis": "fixed cosine acceptance angle (approximately 31.8 degrees); one development probe, no posthoc-label or future-effect selection",
        "selection_context": "registered after C1 recovery root-cause diagnosis; development-only, not confirmation or production calibration",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "git_head_at_registration": git_head(),
        "created_at_utc": now_utc(),
    }
    atomic_json(PROTOCOL, payload)
    atomic_json(
        STATUS,
        {
            "schema_version": "N72R6_STAGE_STATUS_V1",
            "stage": "N72R6-08_HUMAN_ANCHOR_GATE_PROTOCOL",
            "status": payload["status"],
            "protocol": str(PROTOCOL),
            "threshold": THRESHOLD,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "development_only": True,
            "created_at_utc": now_utc(),
        },
    )
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
