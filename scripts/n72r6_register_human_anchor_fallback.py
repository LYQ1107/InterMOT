#!/usr/bin/env python3
"""Register the single, target-scoped main-row fallback probe for N72R6.

This protocol is deliberately separate from the verification-only protocol.
When the target-session row is rejected or absent, it may expose only the
frozen B0 main row that already carried the human-supplied target public ID.
The row is accepted only when its feature passes the same fixed anchor gate;
otherwise the target state remains explicit NONE.  No other main row may be
relabelled or used as a fallback.
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


PROTOCOL = ROOT / "outputs/N72R6/human_anchor_fallback_protocol.json"
STATUS = ROOT / "outputs/N72R6/stage_08_human_anchor_fallback_protocol_status.json"
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
        "schema_version": "N72R6_HUMAN_ANCHOR_FALLBACK_PROTOCOL_V1",
        "status": "PASS_N72R6_HUMAN_ANCHOR_FALLBACK_PROTOCOL_REGISTERED",
        "mechanism": "HUMAN_ROI_VERIFICATION_GATE_WITH_FROZEN_B0_TARGET_MAIN_FALLBACK",
        "threshold": THRESHOLD,
        "formula": "dot(l2_normalize(candidate_feature), l2_normalize(human_anchor)) >= 0.85",
        "applies_to": "future frames strictly after event_frame only",
        "event_frame_rule": "event-frame authoritative target binding is never gated or replaced",
        "fallback_precondition": "target-session candidate is absent or rejected, and the main UID was assigned to target_public_id in the same frozen B0 frame",
        "fallback_selection": "deterministic highest anchor cosine among shadowed B0 target-main UIDs; lexicographic UID tie-break",
        "fallback_acceptance": "selected frozen B0 target-main row may claim only target_public_id or explicit NONE",
        "fallback_rejection": "if no shadowed row passes the fixed gate, expose explicit NONE; never use another protected main row",
        "main_candidate_fallback": True,
        "main_candidates_relabelled": False,
        "other_main_candidates_can_claim_target_public": False,
        "candidate_generation_changed": False,
        "checkpoint_changed": False,
        "hungarian_solver_changed": False,
        "future_window_changed": False,
        "selection_basis": "fixed cosine acceptance angle (approximately 31.8 degrees); one development probe registered before fallback replay, no posthoc-label or future-effect selection",
        "selection_context": "registered after the fixed human-anchor gate showed target coverage loss; development-only, not confirmation or production calibration",
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
            "stage": "N72R6-08_HUMAN_ANCHOR_FALLBACK_PROTOCOL",
            "status": payload["status"],
            "protocol": str(PROTOCOL),
            "threshold": THRESHOLD,
            "main_candidate_fallback": True,
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
