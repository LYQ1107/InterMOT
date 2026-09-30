# InterMOT N72R20 — Learned Identity Memory Integration into InterMOT

## Frozen Final Goal

**Goal:** Learned Identity Memory Integration into InterMOT.

**Central question:** “在候选生成、assignment solver、public-ID authority 和人工交互协议保持不变的情况下，一次人工确认初始化的 N72R18 learned identity memory，能否真正改善 InterMOT 后续的 public-ID association？”

The only scientific claim allowed in this stage is whether the frozen N72R18
learned identity memory changes future public-ID association in the existing
runtime for the better. This stage is not a SAM3 rebuild, a new tracker, a
Hungarian redesign, a generic ReID study, or a training stage.

## Current status

The code and asset audits are complete, the minimal integration seam is
implemented, and its 34 targeted regression tests pass. The feature-level
shadow on the two existing real-SAM3 train tapes detects non-zero learned
identity signal. No formal association replay result exists yet: the
assignment shadow is explicitly blocked because this host does not contain the
frozen N72R15 runtime/base-score tape needed to preserve the original public
and candidate axes. The current
runtime already provides official SAM3 candidate rows, persistent identity
state, public-ID authority, explicit NONE columns, and the exact assignment
solver used by the N72R15 replay. The feature-lineage audit finds the N72R18
GRU input contract compatible with the frozen 512-D OSNet candidate stream.

The full repository suite currently reports `236 passed, 4 failed`; the four
failures are pre-existing TrackEval test infrastructure failures: three invoke
an unavailable `python` executable, and one reaches the pinned TrackEval
`SEQMAP_FILE` list/scalar incompatibility. They are not counted as N72R20
scientific failures.

Available development tapes are the real-SAM3 train sequences
`dancetrack0001` and `dancetrack0002` only. DanceTrack val remains frozen and
has not been used for tuning or selection. No new data, checkpoint, or
training run is authorized by this report.

## Required result before a final decision

The replay must compare a frozen baseline and treatment while preserving
candidate rows, solver, public-ID authority, and interaction timing. It must
report H20/H50/H100 association results, candidate coverage, learned-memory
decision activity, memory writes, wrong-memory writes, and sequence-cluster
confidence intervals. The final decision must be exactly one of the six values
in `outputs/N72R20/FINAL_GOAL.json`.

The historical B2 trace used by the feature shadow contains post-hoc wrong
updates; it is therefore a diagnostic gate only and is not evidence for the
formal E2 treatment. No base score is fabricated from GT or embeddings, and
no TrackEval conclusion is emitted. Until the assignment/causal replay is
complete, `next_full_interactive_mot_stage_authorized` is **false**.

## Frozen N72R15 tape recovery status

The old-session handoff identifies the historical source as
`/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R15/formal_attempt_04`,
with 32 events, 3 variants, and 9,696 runtime rows. That path is not present on
the current host. A read-only search of the available local roots found no
`formal_manifest.json` or `runtime_frames.jsonl`; the recorded NAS candidates
were also not reachable from this host. The evidence is preserved in
`outputs/N72R20/frozen_tape_recovery_audit.json`. This is an asset-recovery
blocker, not a scientific PASS/FAIL result.
