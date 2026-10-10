"""Changing identity bank, zero permission to alter any MOT assignment."""
from .committed_identity_memory import MemoryCommitPolicy, CommittedIdentityMemory
from .committed_memory_bridge import CommittedMemoryMOTBridge
from .intervention_gate import GatePolicy

FIXED_MEMORY_CASES = ("FROZEN", "MEAN", "DELAYED", "MULTICUE", "PENDING_TRUSTED", "ROLLBACK")


def fixed_memory_policy(case):
    if case not in FIXED_MEMORY_CASES:
        raise ValueError("Risk-learned writer requires a separately fitted fresh checkpoint")
    family = {"FROZEN": "frozen", "MEAN": "unsafe", "DELAYED": "delayed", "MULTICUE": "consensus",
              "PENDING_TRUSTED": "diverse", "ROLLBACK": "rollback"}[case]
    return MemoryCommitPolicy(family=family, capacity=8, aggregation="mean" if case == "MEAN" else "attention",
                              delay_frames=3, probability_min=.7, anchor_min=.6, quality_min=.5, motion_min=.2,
                              diversity_gap=5, novelty_cosine_max=.98, risk_max=.02, rollback_confirmations=2)


def memory_actor(original, case):
    actor = CommittedIdentityMemory(original.model, original.anchor, original.token, write_policy=fixed_memory_policy(case))
    actor.start_recording(original.recording, fps=original.fps, width=original.width, height=original.height,
                          camera=original.camera, initial_frame=original.initial_frame, initial_box=original.last_box)
    return actor


class ZeroAuthorityMemoryBridge(CommittedMemoryMOTBridge):
    """M-A runtime boundary: score/write actual commits, never reassign."""
    def __init__(self, event, identity, *, frames):
        super().__init__(event, identity, policy=GatePolicy(family="shadow"), frames=frames)

    def step(self, frame, rows):
        if self.policy.family != "shadow":
            raise ValueError("M-A has zero association authority")
        actual = super().step(frame, rows)
        action = actual["selected_action"]
        # The original unforced pre-click C0 solve exports action=None. Its
        # matcher decision is still complete; do not invent a forced action.
        illegal_action = (action is None and frame > self.event_frame) or (action is not None and action["family"] != "KEEP")
        if illegal_action or actual["authority"].get("approved") or actual["authority"].get("effective_assignment_change"):
            raise RuntimeError("Identity memory escaped zero-authority boundary")
        return actual
