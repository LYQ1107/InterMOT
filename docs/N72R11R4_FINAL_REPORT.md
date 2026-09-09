# InterMOT N72R11R4 Final Report

**Date:** 2026-09-09 (Asia/Shanghai artifacts)
**Repository:** `https://github.com/LYQ1107/InterMOT`
**Branch:** `codex/n72r11r4-public-competition`
**Code commit:** `f1bc28e6d52b0b01462f52a78c04f414d7c85d78`
**Disposition:** `REVISE`; research evidence retained, production not authorized.

## Conclusion

N72R11R4 repaired the N72R11R3 stale-feature/scorer-UID mismatch and completed
the exact solver-on-policy V3 route. V3 failed the preregistered Stage-A rule.
The same-capacity Public-Competition-Aware Temporal Identity Scorer (PCTIS)
was then initialized from V3, trained, rebuilt on its own on-policy corpus, and
evaluated in a complete 32-event E1B replay.

E1B improved the descriptive result over E1A, but all sequence-cluster 95% CI
lower bounds still included zero and PCTIS readiness failed. The prescribed E2
live replay remained incomplete after the allowed resource retries: 31/32
events passed and one event remained CUDA-OOM blocked. No E2 aggregate, Oracle,
calibration head, selector, decoder LoRA, or production promotion is authorized.

All events are `simulated_from_gt`; there is still no provenance-complete real
human event tape. Simulated events are not presented as historical human clicks.

## Frozen protocol and invariants

- Reused the frozen N72R9 32-event / 18-independent-sequence development
  protocol, checkpoint, candidate stream, public-ID semantics, Hungarian solver,
  H20/H50/H100 windows, posthoc metrics, seed `7211`, and 2,000 bootstrap
  repetitions.
- Runtime future GT remained disabled (`runtime_future_gt_used=false`); GT was
  used only by sealed posthoc scoring.
- Event-frame correction precedes memory write; the event frame cannot read the
  new memory, and the first possible memory effect is event+1.
- No historical N36--N72R11R3 evidence or `third_party/sam3` file was changed.
- `git diff --name-only -- third_party/sam3` was empty. MOT/OVMOT shared
  configuration and production checkpoints were not modified.

## Exact V3 repair and PCTIS implementation

The exact corpus builder now runs the BASE public solver before model scoring and
recomputes candidate, causal, motion, neighbor and temporal features from the
same rollout state. The V3 corpus has 24,690 train examples (419 events/12
sequences) and 5,700 validation examples (97 events/6 sequences); the exact
contract and formal E1A replay completed.

PCTIS preserves V3 capacity and changes the candidate input from 535 to 541
dimensions by adding exactly six competition values: tanh legacy target score,
tanh best-other score, tanh target-minus-best-other, incumbent-is-target,
incumbent-is-other, and exact BASE-solver-assigns-target. The first 535 columns
were copied from V3, six new columns were zeroed, and same-shape parameters were
strictly copied. Initialization smoke found 51 copied keys, one adapted key,
zero missing/unexpected keys, finite loss, and recorded zero-input agreement.

The fixed objective was cross-entropy + 0.15 pairwise + 0.25 positive-boundary
+ 0.25 protected-boundary + 0.01 delta-L2. Initial training used two epochs,
learning rate `2.5e-4`, batch 64, AdamW, and weight decay `1e-4`; the rebuilt
on-policy finetune used one epoch at `1.25e-4`.

## PCTIS readiness

| Measure | Observed | Required | Result |
|---|---:|---:|---|
| target accuracy | 0.50046 | >= 0.70 | fail |
| NONE accuracy | 0.25592 | >= 0.50 | fail |
| positive-boundary accuracy | 0.62351 | >= 0.70 | fail |
| protected-safe rate | 0.99847 | >= 0.90 | pass |

The future requery readiness check had only 42 labels and was correctly marked
`FUTURE_SOURCE_COVERAGE_LIMITED`; it was not used to claim success.

## Complete E1A/E1B results

`identity_error_reduction > 0` is beneficial. Both replays are complete (32/32
events, 18 sequence clusters).

| Variant | H20 reduction / CI | H50 reduction / CI | H100 reduction / CI |
|---|---|---|---|
| E1A | -0.019576 / [-0.106251, 0.181495] | -0.081081 / [-0.157491, 0.065641] | -0.068690 / [-0.121157, 0.072562] |
| E1B | +0.042414 / [-0.041032, 0.201799] | -0.000644 / [-0.066751, 0.132664] | -0.008307 / [-0.046482, 0.094342] |

E1A assignment-change rates were 0.5351/0.4170/0.3703 (H20/H50/H100), with
correct/incorrect crossings 64/76, 92/218, 176/391 and protected regressions
8/19/25. E1B rates were 0.2741/0.1763/0.1399, crossings 56/30, 80/81,
114/140, and protected regressions 7/8/14. E1B minus E1A reductions were
`+0.061990`, `+0.080438`, and `+0.060383` at H20/H50/H100.

The H20 E1B action pattern is mixed: ADD `13/1`, ATOMIC `17/0`, AUTHORITATIVE
`8/3`, but RECOVER `18/26` correct/incorrect. The point-estimate improvement
and fewer harmful crossings are mechanism evidence, not a strict future-effect
pass, because no E1B CI lower bound is above zero.

## E2 live replay: preserved failures and strict gate

E2 was PCTIS + legacy injection + the existing official SAM3 live requery path.
Every event ran in an independent serialized child.

| Attempt | Scope | Result |
|---|---|---|
| initial attempt-1 | 32 events | 25 PASS, 7 FAIL_CHILD |
| fresh attempt-2 | 7 initial failures | 6 PASS, one CUDA OOM |
| final attempt-3 | remaining event, outer `cuda:1` | CUDA OOM again |
| lossless merge | 32 unique event IDs | 31 PASS, one unresolved |

The six initial `-9` children have no process exception traceback; their logs
contain only the environment `.pth` warning. They are classified as
`SIGKILL_NO_TRACEBACK_RESOURCE_OR_EXTERNAL_KILL`, not converted to PASS. The
initial audit is preserved; the corrected cross-attempt audit is
`outputs/N72R11R4/e2_failure_audit_attempt_03.json`.

The unresolved event is
`n72r5-pool-n37-dancetrack0002-0329-authoritative_reassign-042`. Attempt-2
requested 82 MiB with 75.12 MiB free. The final GPU1 attempt requested 42 MiB
with 35.12 MiB free. Both tracebacks reach official SAM3 position encoding
during `add_box`/future propagation. The final failure artifact records outer
`cuda:1` but an internal GPU0 allocation message, showing that the outer device
argument did not relocate the internal official path.

The host `dmesg --ctime` audit contains global OOM kills, but its PIDs/times do
not prove attribution to each earlier `-9` child; it is recorded as resource
context only. Existing live lifecycle cleanup and supported parameters
`async_loading_frames=false` and `trim_past_non_cond_mem_for_eval=true` were
retained. No unsupported SAM3 API, shortened window, precision change,
candidate substitution, or further retry was used.

Strict merge numbers are: required `32`, records `32`, unique IDs `32`,
duplicates `0`, missing IDs `0`, PASS `31`, unresolved/non-PASS `1`,
`strict_complete=false`, Oracle authorized `false`. Because this gate failed,
no E2 aggregate was generated and no Oracle/selector branch was run.

## Evidence and reproducibility

Key machine-readable outputs are:

- `outputs/N72R11R4/formal_e1a_metrics.json`
- `outputs/N72R11R4/formal_e1b_metrics_attempt_02.json`
- `outputs/N72R11R4/e1a_e1b_comparison.json`
- `outputs/N72R11R4/e2_failure_audit_attempt_03.json`
- `outputs/N72R11R4/formal_e2_pctis_merged_manifest_attempt_03.json`
- `outputs/N72R11R4/formal_e2_merge_gate.json`
- `outputs/N72R11R4/stage_17_final_e2_status.json`
- `outputs/N72R11R4/CONTROLLER_STATUS.json`
- `outputs/N72R11R4/n72r11r4_final_gate.json`

Important hashes: frozen protocol
`e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9`;
PCTIS finetuned checkpoint
`77b41dbe2fc0f03c48ca4b8cda22e6b219eb58e266566eb96fcf0df3fb8c2c16`;
E1B metrics `41802fef72a5bdde662ad21cdbd4b702ae5cffaffd246789c990b6817669436e`;
E2 merge gate
`7ed1ad5438c11060f220b1088fa11045a81c875100f76382c8cd44677f5903f7`.

The final machine gate is `BLOCKED_RESOURCE_INCOMPLETE_E2_FAIL_FUTURE_EFFECT`;
research gate `FAIL_FUTURE_EFFECT`; production, Oracle, calibration head,
selector and decoder LoRA authorization are all false.

## Decision and next step

**Keep:** persistent public-ID semantics, exact global solver boundary, causal
event-frame/t+1 semantics, exact on-policy construction, and PCTIS as isolated
research code.

**Revise:** official SAM3 internal device placement/streaming memory and PCTIS
evidence quality. The one unresolved event must not receive an imputed metric.

The only defensible technical continuation is to verify/fix the official
internal device path or use a genuinely supported larger-memory/state-streaming
mechanism, then rerun only that frozen event under the same H100 protocol. A
production claim additionally requires real provenance-complete human events;
the current simulated-from-GT corpus cannot substitute for them.

## ICLR 2027 schedule

The controlling dates are the user-specified abstract deadline **2026-09-18 AoE**
and full-paper deadline **2026-09-25 AoE**. The remaining work should prioritize
reproducible resource verification and the frozen missing-event audit; if that
would require changing the protocol, document the negative/blocked result rather
than adding ungrounded training.
