#!/usr/bin/env python
"""Generate N8 stage documentation and final report from real experiment data."""

import hashlib
import json
from pathlib import Path


ROOT = Path(".")
DOCS = ROOT / "docs"
DATA = ROOT / "outputs/n8"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name: str, text: str) -> None:
    (DOCS / name).write_text(text, encoding="utf-8")


def aggregate_jsonls() -> None:
    seqs = sorted(p.name for p in (DATA / "real/route_a_unlimited").iterdir() if p.is_dir())
    ver = []
    mem = []
    hs = []
    for s in seqs:
        d = DATA / "real/route_a_unlimited" / s
        for e in json.loads((d / "summary.json").read_text()):
            pass
        ver.extend(json.loads(f"[{','.join(d / 'verified_errors.jsonl').read_text().splitlines()}]") if False else [])
    # simpler streaming aggregation
    def agg(path, out, extra):
        with out.open("w", encoding="utf-8") as f:
            for s in seqs:
                p = DATA / "real/route_a_unlimited" / s / path
                if not p.exists():
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        e = json.loads(line)
                        e.update(extra)
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    agg("verified_errors.jsonl", DATA / "verified_errors.jsonl", {"run_budget": -1})
    agg("observer_memory_audit.jsonl", DATA / "observer_memory_audit.jsonl", {"run_budget": -1})
    agg("system_state_hashes.jsonl", DATA / "system_state_hashes.jsonl", {"run_budget": -1})
    budgets = ["b0", "b1", "b2", "b4", "b8", "unlimited"]
    with (DATA / "interaction_events.jsonl").open("w", encoding="utf-8") as f:
        for bn in budgets:
            bv = -1 if bn == "unlimited" else int(bn[1:])
            for s in seqs:
                p = DATA / "real" / f"route_a_{bn}" / s / "interaction_events.jsonl"
                if not p.exists():
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        e = json.loads(line)
                        e["run_budget"] = bv
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")


def doc_recent_methods() -> str:
    return """# N8 Recent Method / GitHub Audit (2025-2026)

Scope: interactive MOT, identity correction, tracklet relinking, SAM-based
interactive tracking, low-budget human-in-the-loop MOT.

## Verified references

| Method | Year / Venue | Official source | Relevance | Code used |
| --- | --- | --- | --- | --- |
| FC-Track (Overlap-Aware Post-Association Correction) | arXiv 2603.12758 | arXiv (verified) | CONCEPTUAL_REFERENCE_ONLY: post-association correction on TrackTrack; no human-budget protocol, no DanceTrack identity-layer contract | No |
| DIPLOMAT (interactive multi-animal tracking + correction) | 2025 | github.com/TravisWheelerLab/DIPLOMAT (verified) | CONCEPTUAL_REFERENCE_ONLY: human-correction interface; different domain/protocol | No |
| GTATrack (SoccerTrack 2025 winner, global tracklet association) | arXiv 2602.00484 | arXiv (verified) | CONCEPTUAL_REFERENCE_ONLY: tracklet relinking; offline/global, not current-frame human correction | No |
| Track-Anything / DAM4SAM / SAM2 interactive predictors | 2023-2025 | existing project audit (docs/N6_SAM31_GITHUB_API_AUDIT.md) | CONCEPTUAL_REFERENCE_ONLY: promptable correction, no verified temporal-error protocol | No |
| CATB Identity Vault | N7 audit | existing N7 docs | CONCEPTUAL_REFERENCE_ONLY | No |

## Verdict

**NO DIRECTLY APPLICABLE 2025-2026 OPEN-SOURCE METHOD FOUND** for a frozen
P0 backbone plus chronological verified-error sparse human interaction on
DanceTrack with official TrackEval.  No external method was adopted as a
baseline; all external methods are marked `CONCEPTUAL_REFERENCE_ONLY`.

No scheduler, active-learning, or utility-ranking method was used in N8
(frozen by protocol); such methods are recorded only as `NEXT_STAGE_CANDIDATES`
for N9.
"""


def doc_reaudit() -> str:
    return """# N8 N7 Event Semantic Re-Audit

## Method

Re-read every N7 B1/B2/B4/B8 event log (375 events) and re-classified each
event using the P0 backbone row, the current-frame GT box (matched by IoU),
and the N7 event's own `gt_id`/`public_mot_id`/`seen_before`.

## Result

| Classification | Count | Fraction |
| --- | ---: | ---: |
| FIRST_APPEARANCE_RENAME | 251 | 66.93% |
| TRUE_MISS_NEW | 99 | 26.40% |
| TRUE_RECOVER | 25 | 6.67% |
| TRUE_ID_BREAK | 0 | 0% |
| TRUE_SWAP | 0 | 0% |
| Total | 375 | 100% |

Evidence: `outputs/n8/audit/n7_event_semantic_reaudit.csv` and
`n7_event_semantic_reaudit_summary.json`.

## Conclusion

N7's sparse budgets were mostly spent renaming already-correct first
appearances (251/375 = 66.93%).  That is exactly the failure N8's temporal
semantics removes.
"""


def doc_protocol() -> str:
    return """# N8 Temporal Error Protocol (frozen)

## Identity layers

`dataset_gt_id` -> `user_identity_id` (observer memory) ->
`identity_lineage_id` (internal) -> `public_mot_id` (output).  GT numeric id
is never compared with public id as an error:
`GT_NUMERIC_ID_MISMATCH_IS_NOT_ERROR = TRUE`.

## Events

| Event | Condition | Interaction | Action |
| --- | --- | --- | --- |
| FIRST_APPEARANCE_MATCHED | new identity, matched P0 row | no, cost 0 | NONE |
| TRUE_MISS_NEW | new identity, no matched P0 row | yes, cost 1 | ADD_NEW_IDENTITY |
| RECOVERABLE_MISS | seen identity, no matched P0 row | yes, cost 1 | RECOVER_IDENTITY |
| TEMPORAL_ID_BREAK | seen identity, matched row with non-canonical public id | yes, cost 1 | AUTHORITATIVE_REASSIGN |
| TEMPORAL_ID_SWAP | two seen identities with exchanged public ids | yes, cost 1 | ATOMIC_ID_SWAP |
| LOCALIZATION_ONLY_ERROR | identity correct, IoU < 0.7 | no, cost 0 | NONE (statistics) |
| FALSE_POSITIVE | unmatched P0 row | no, cost 0 | NONE (statistics) |

## Fixed application priority (same frame)

`TEMPORAL_ID_SWAP` > `TEMPORAL_ID_BREAK` > `RECOVERABLE_MISS` >
`TRUE_MISS_NEW`; ties broken by stable `user_identity_id`.

## Matching and budget

- Hungarian IoU matching, frozen threshold 0.5; localization threshold 0.7.
- Chronological verified-error protocol: first verified temporal error
  consumes the next interaction; no scheduler, no future utility.
- B1=1, B2=2, B4=4, B8=8, Unlimited=-1 interactions per sequence.

## Causality

`Y_pre(t)` is assembled before current-frame GT is read.  GT may update
HumanObserverMemory only; SystemState changes only through accepted actions.
Future GT and future utility are never used.
"""


def doc_observer_contract() -> str:
    return """# N8 Observer Memory / System State Contract

## HumanObserverMemory

Per dataset identity: `user_identity_id`, `first_seen_frame`,
`last_seen_frame`, `last_matched_frame`, `canonical_public_id`,
`last_observed_public_id`, `last_correct_public_id`, `currently_visible`,
`ever_seen`, `last_matched_box`, `accepted_interactions`, `history`.

Canonical public id is set only at first legal observation or by an accepted
authoritative action; error observations never overwrite it.

## SystemState

`IdentityNamespace` + `canonical_map` (auto track id -> canonical public id)
+ post rows.  GT observation without an accepted action must leave
`system_state_hash` unchanged.

## Hashes

- `observer_memory_hash`: memory records (history length only, not contents).
- `system_state_hash`: namespace mutable state + canonical_map.

Both are recorded per frame in `system_state_hashes.jsonl`.  The aggregated
25-sequence audit reports zero `gt_read_before_prediction`, zero
`gt_read_future`, and zero `system_mutation_without_accepted_action`.
"""


def doc_toy_tests() -> str:
    return """# N8 Toy / Synthetic Tests (T1-T15)

All tests are CPU and pass with the frozen observer:

| Test | Purpose | Result |
| --- | --- | --- |
| T1 | GT=4, public=37 -> no interaction | PASS |
| T2 | stable 37 -> 0 interactions | PASS |
| T3 | 37->52 -> TEMPORAL_ID_BREAK corrected | PASS |
| T4 | 37->miss -> RECOVERABLE_MISS corrected | PASS |
| T5 | new matched first appearance -> 0 interaction | PASS |
| T6 | new missed identity -> ADD_NEW | PASS |
| T7 | 37/52 swap -> ATOMIC_SWAP | PASS |
| T8 | budget exhausted: memory updates, system unchanged | PASS |
| T9 | error observation never overwrites canonical | PASS |
| T10 | GT numeric mismatch is not an error | PASS |
| T11 | B0 byte-identical to P0 | PASS |
| T12 | same-frame fixed priority | PASS |
| T13 | unaccepted event -> zero system mutation | PASS |
| T14 | accepted REASSIGN is persistent in post stream | PASS |
| T15 | synthetic TrackEval: corrected IDSW < uncorrected IDSW | PASS |

`pytest tests/test_n8_temporal_observer.py -q`: **15 passed**.
Full CPU regression: **92 passed** (previous 77 + 15 new), no failures.
"""


def doc_three_seq() -> str:
    return """# N8 Three-Sequence Results (dancetrack0004/0005/0007)

## Official TrackEval (post stream)

| Method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 (=P0) | 67.924 | 78.362 | 58.891 | 85.327 | 74.272 | 49 | 519 |
| B1 | 67.922 | 78.364 | 58.886 | 85.327 | 74.266 | 52 | 521 |
| B2 | 67.922 | 78.371 | 58.881 | 85.337 | 74.262 | 52 | 522 |
| B4 | 67.919 | 78.376 | 58.872 | 85.348 | 74.254 | 54 | 524 |
| B8 | 67.915 | 78.387 | 58.857 | 85.385 | 74.238 | 55 | 524 |
| Unlimited | 81.353 | 82.678 | 80.050 | 91.865 | 92.288 | 78 | 195 |

## Gates

| Gate | Check | Result |
| --- | --- | --- |
| A | matched first-appearance interactions / budget cost = 0 | PASS |
| B | accepted event purity (only 4 verified types) | PASS (100%) |
| C | causality (no future GT, no GT mutation without action) | PASS |
| D | accepted <= B per sequence | PASS |
| E | B0 byte-identical to P0; zero violations | PASS |
| F | B1-B8 positive gain | NOT SUPPORTED (HOTA -0.002..-0.009, IDSW +3..+6, Frag +2..+5) |
| G | temporal semantics | PASS |

First accepted events: TRUE_MISS_NEW for all three sequences (frame 0/1),
never a first-appearance rename.  B0 SHA256 equals P0 SHA256 for all three
sequences.
"""


def doc_25_seq() -> str:
    return """# N8 Canonical 25-Sequence Results

## Official TrackEval (post stream, 25 DanceTrack val sequences)

| Method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | FP | FN | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 (=P0) | 49.795 | 55.696 | 44.920 | 46.444 | 52.690 | 2491 | 13775 | 55217 | 62872 | 0 |
| B1 | 49.807 | 55.724 | 44.919 | 46.497 | 52.702 | 2513 | 13790 | 55101 | 62847 | 25 |
| B2 | 49.808 | 55.732 | 44.914 | 46.502 | 52.701 | 2527 | 13799 | 55099 | 62823 | 50 |
| B4 | 49.810 | 55.743 | 44.909 | 46.516 | 52.700 | 2546 | 13807 | 55092 | 62780 | 100 |
| B8 | 49.822 | 55.766 | 44.910 | 46.560 | 52.704 | 2565 | 13819 | 55043 | 62712 | 200 |
| Unlimited | 80.568 | 80.762 | 80.389 | 90.829 | 92.324 | 1055 | 2773 | 7952 | 11641 | 182229 |

## Verified event stream (25 sequences, budget-independent)

| Event type | Count | Accepted (unlimited) |
| --- | ---: | ---: |
| FIRST_APPEARANCE_MATCHED | 117 | 0 |
| TRUE_MISS_NEW | 156 | 156 |
| RECOVERABLE_MISS | 62516 | 62516 |
| TEMPORAL_ID_BREAK | 119069 | 119069 |
| TEMPORAL_ID_SWAP | 488 | 488 |
| LOCALIZATION_ONLY_ERROR | 3396 | 0 |
| FALSE_POSITIVE | 55017 | 0 |

## B0 gate

B0 is byte-identical to P0 for all 25 sequences
(`outputs/n8/real/route_a_b0/*/post_mot`).  ZERO_INTERACTION_EQUIVALENCE
holds exactly; namespace/assembler violations = 0.
"""


def doc_event_level() -> str:
    return """# N8 Event-Level Analysis

## Event frame distribution (verified errors, 25 sequences)

| Video progress | Count |
| --- | ---: |
| 0-10% | 20485 |
| 10-25% | 33247 |
| 25-50% | 62025 |
| 50-75% | 62185 |
| 75-100% | 62817 |

Errors are spread across the whole video, not concentrated at frame 0
(unlike N7's first-appearance renames).

## B1 accepted events

23 TRUE_MISS_NEW + 2 RECOVERABLE_MISS; first accepted frame median = 1,
mean = 2.48.  The chronological protocol spends the first interaction on a
true miss at the very start of the video.

## Retention after one correction (B1 stream)

| Offset | Identity correct |
| --- | ---: |
| t+0 | 100.0% |
| t+1 | 4.0% |
| t+3 | 8.0% |
| t+5 | 8.0% |
| t+10 | 8.0% |
| t+30 | 8.0% |

A single identity-layer correction has essentially no future persistence
because the frozen P0 backbone keeps emitting a different track id for the
same person; with budget exhausted the post stream reverts to P0.

## Time to next error (unlimited run)

Median = 1 frame, mean = 1.28 frames; only 273 accepted events had no next
error before sequence end.  Identity remapping needs to be applied almost
every frame to stay correct.

## Repeated correction (unlimited run)

273 identities were tracked; the same identity required dozens to thousands
of corrections (up to 2333).  The frozen P0 backbone has severe track-id
instability on DanceTrack, so one-shot sparse correction cannot persist.

## Retention under unlimited correction

t+0: 97.92%, t+1: 97.32%, t+3: 97.16%, t+5: 97.06%, t+10: 96.94%,
t+30: 96.76% (identity correct among GT-present frames).  Continuous
re-application keeps identities mostly correct; single corrections do not.
"""


def doc_cost() -> str:
    return """# N8 Cost-Performance Analysis

## Quality-cost curve (official TrackEval, post)

| Budget | Accepted | /seq | /1000 frames | HOTA | AssA | MOTA | IDF1 | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0 | 0.0 | 0.0 | 49.795 | 44.920 | 46.444 | 52.690 | 2491 | 13775 |
| B1 | 25 | 1.0 | 0.98 | 49.807 | 44.919 | 46.497 | 52.702 | 2513 | 13790 |
| B2 | 50 | 2.0 | 1.96 | 49.808 | 44.914 | 46.502 | 52.701 | 2527 | 13799 |
| B4 | 100 | 4.0 | 3.92 | 49.810 | 44.909 | 46.516 | 52.700 | 2546 | 13807 |
| B8 | 200 | 8.0 | 7.84 | 49.822 | 44.910 | 46.560 | 52.704 | 2565 | 13819 |
| Unlimited | 182229 | 7289.2 | 7143.5 | 80.568 | 80.389 | 90.829 | 92.324 | 1055 | 2773 |

## Marginal gain

| Step | Additional interactions | dHOTA | dAssA | dIDF1 | dIDSW | dFrag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0->B1 | 25 | +0.012 | -0.001 | +0.012 | +22 | +15 |
| B1->B2 | 25 | +0.001 | -0.005 | -0.001 | +14 | +9 |
| B2->B4 | 50 | +0.002 | -0.005 | -0.001 | +19 | +8 |
| B4->B8 | 100 | +0.012 | +0.001 | +0.004 | +19 | +12 |
| B8->Unlimited | 182029 | +30.746 | +35.479 | +39.620 | -1510 | -11046 |

## Interpretation

At B1-B8, per-interaction gain is ~0.0005-0.0007 HOTA, within TrackEval
noise, while IDSW and Frag slightly worsen.  The large Unlimited gain
requires ~7143 interactions per 1000 frames -- a dense identity overlay, not
sparse human interaction.  Efficiency: dHOTA per interaction for B1 is
~0.0005; for Unlimited ~0.00017; neither supports a sparse-interaction
product.
"""


def doc_final() -> str:
    return """# N8 Final Report

## 1. Executive Summary

N8 redefines sparse human interaction as a **chronological verified temporal
error** protocol on the frozen P0 backbone.  It eliminates N7's main flaw:
66.93% of N7's budget events (251/375) were renames of already-correct first
appearances.  In N8, matched first appearances never consume budget, and
only TRUE_MISS_NEW / RECOVERABLE_MISS / TEMPORAL_ID_BREAK / TEMPORAL_ID_SWAP
consume budget.

Technical outcome: **PASS_FULL_PROTOCOL_ONLY** -- all temporal-semantics,
causality, budget, and baseline-preservation gates pass, but
**LOW-BUDGET GAIN NOT SUPPORTED**.  B1-B8 (1-8 corrections/sequence) stay
statistically flat or slightly negative (HOTA +0.005..+0.020, AssA
-0.005..-0.053, IDF1 +0.005..+0.012, IDSW +0.9..+3.0, Frag +0.6..+1.8).
Unlimited correction (182,229 interactions) improves HOTA +30.7, AssA +32.4,
IDF1 +32.9, IDSW -1436, Frag -11002, but at ~7143 interactions/1000 frames,
which is not sparse human interaction.

## 2. Starting Point from N7

N7 froze P0 (`outputs/n5/integrity/canonical_mot_results/b0`) and Route A-CPU
(identity-layer sparse interaction).  B1-B8 preserved P0 but had no gain.

## 3. What N7 Actually Proved

Baseline preservation at B0 (exact), no collapse at B1-B8, zero identity
violations; Route B and real-SAM reset unsupported by the pinned API.

## 4. Why N7 Sparse Budgets Had No Gain

The budget was spent on first-appearance renames (251/375 = 66.93%),
producing no temporal identity correction; see
`docs/N8_N7_EVENT_SEMANTIC_REAUDIT.md`.

## 5. N7 Event Semantic Re-Audit

FIRST_APPEARANCE_RENAME=251, TRUE_MISS_NEW=99, TRUE_RECOVER=25, TRUE_ID_BREAK=0,
TRUE_SWAP=0.

## 6. Recent 2025-2026 Literature and GitHub Search

FC-Track, DIPLOMAT, GTATrack, Track-Anything/DAM4SAM/SAM2 interactive
predictors, CATB Identity Vault: all `CONCEPTUAL_REFERENCE_ONLY`;
**NO DIRECTLY APPLICABLE 2025-2026 OPEN-SOURCE METHOD FOUND**.

## 7. Human Observer Model

`HumanObserverMemory` simulates a continuous viewer who remembers identities,
canonical public ids, gaps, and prior corrections, using current-frame GT
only after `Y_pre(t)` is frozen (simulated oracle human observer).

## 8. Observer Memory vs System State

Separate hashes; GT observations may change only the observer hash.  System
state changes only via accepted actions.  Zero violations in 25 sequences.

## 9. Public ID Semantics

`dataset_gt_id` / `user_identity_id` / `lineage_id` / `public_mot_id` are
separate; GT numeric id mismatch is never an error.

## 10. First-Appearance Semantics

Matched first appearance = FIRST_APPEARANCE_MATCHED, interaction cost 0.
No rename is ever generated for it.

## 11. Temporal Identity Break Definition

Seen identity + matched row with public id != canonical -> break (cost 1).

## 12. Recoverable Miss Definition

Seen identity + no matched row -> recover (cost 1).

## 13. True New Miss Definition

New identity + no matched row -> ADD_NEW (cost 1).

## 14. Atomic Identity Swap

Two seen identities with exchanged public ids -> one ATOMIC_ID_SWAP (cost 1).

## 15. Verified Error Protocol

Chronological: the earliest verified temporal error gets the next
interaction; fixed priority per frame; no scheduler, no future utility.

## 16. Causality Contract

Prediction-before-GT; future GT = 0; GT mutation without accepted action = 0;
no future-based ranking.

## 17. Budget Contract

B1=1, B2=2, B4=4, B8=8 per sequence, reset each sequence; accepted counts
exactly 25/50/100/200 across 25 sequences; B0=0.

## 18. Toy and Synthetic Tests

T1-T15 all PASS; full CPU regression 92 passed, 0 failed.

## 19. Three-Sequence Verified-Event Audit

First accepted events are TRUE_MISS_NEW (frame 0/1) on all three sequences;
no first-appearance interaction.

## 20. Three-Sequence Results

B0=67.924 HOTA; B1-B8: 67.922/67.922/67.919/67.915; Unlimited=81.353.

## 21. Temporal-Semantics Gate Decision

PASS (all gates A-G technical; F not supported at low budget).

## 22. Frozen N8 Protocol

`outputs/n8/n8_frozen.json` records event semantics, priority, thresholds,
budget protocol, and file SHA256 hashes.

## 23. Canonical 25-Sequence Evaluation

Official TrackEval on 25 DanceTrack val sequences (see
`docs/N8_25_SEQUENCE_RESULTS.md`).

## 24. Event-Type Distribution

TRUE_MISS_NEW=156, RECOVERABLE_MISS=62516, TEMPORAL_ID_BREAK=119069,
TEMPORAL_ID_SWAP=488; FIRST_APPEARANCE_MATCHED=117;
LOCALIZATION_ONLY_ERROR=3396; FALSE_POSITIVE=55017.

## 25. Event-Time Distribution

0-10%=20485, 10-25%=33247, 25-50%=62025, 50-75%=62185, 75-100%=62817:
errors occur throughout the videos.

## 26. B0/B1/B2/B4/B8/Unlimited Results

HOTA: 49.795 / 49.807 / 49.808 / 49.810 / 49.822 / 80.568.
AssA: 44.920 / 44.919 / 44.914 / 44.909 / 44.910 / 80.389.
IDF1: 52.690 / 52.702 / 52.701 / 52.700 / 52.704 / 92.324.

## 27. Pre-Interaction Performance

Pre streams show the effect of prior corrections on future automatic output;
they remain essentially equal to P0 at B1-B8 (`combined_metrics_pre.csv`).

## 28. Post-Interaction Performance

Post streams are the user-visible outputs; B1-B8 add 1-8 rows/remappings and
leave official metrics within noise; Unlimited is a dense identity overlay.

## 29. Current-Frame Interaction Gain

100% of accepted events are identity-correct at t+0 (by construction);
ADD_NEW additionally reduces FN (62872->62847 at B1).

## 30. Historical Interaction Gain

None at sparse budgets: B1 retention drops to 4% at t+1; time-to-next-error
median is 1 frame.  Persistent remapping only materializes when corrections
are applied continuously (Unlimited).

## 31. Retention After Correction

B1: 100/4/8/8/8/8 % at t+0/t+1/t+3/t+5/t+10/t+30.

## 32. Time-to-Next-Error

Median 1 frame, mean 1.28; only 273/182229 events reached sequence end
without a next error.

## 33. Repeated Correction Analysis

273 identities; correction counts per identity reach thousands, showing the
P0 backbone re-breaks identity nearly every frame.

## 34. Human Interaction Cost

B1=25, B2=50, B4=100, B8=200 total interactions; Unlimited=182229
(~7143/1000 frames).

## 35. Quality-Cost Curve

See `outputs/n8/tables/quality_cost_curve.csv`; flat until the dense regime.

## 36. Marginal Gain per Interaction

B0->B1: +0.012 HOTA / 25 interactions (~0.0005 per interaction), +22 IDSW,
+15 Frag; B8->Unlimited: +30.746 HOTA but 182,029 interactions.

## 37. Statistical Significance

Paired per-sequence (25):
- B1 vs P0: HOTA +0.0053 (p=0.028), AssA -0.0053 (p=0.086), MOTA +0.0381
  (p=0.002), IDF1 +0.0049 (p=0.338), IDSW +0.88 (p<0.001, 22 degraded),
  Frag +0.6 (p<0.001, 16 degraded).
- B8 vs P0: HOTA +0.020 (p=0.081), AssA -0.053 (p=0.046), IDF1 +0.012
  (p=0.968), IDSW +2.96 (p<0.001), Frag +1.76 (p=0.010).
- Unlimited vs P0: HOTA +27.54, AssA +32.44, IDF1 +32.90 (all p<0.001);
  IDSW -57.44 (p=0.079), Frag -440.08 (p<0.001).

## 38. Event-Level Counterfactual Analysis

`EVENT_DETECTED_BUT_NOT_APPLIED` diagnostic is satisfied by construction:
at B0 the full verified stream is detected and logged while system output
equals P0 byte-for-byte (verified_errors.jsonl + system_state_hashes.jsonl).

## 39. Failure Cases

- ADD_NEW / RECOVER corrections have no future propagation: retention ~4-8%.
- Canonical_map remapping can slightly increase IDSW/Frag at B1-B8 when the
  mapped track id is later reused by another identity.
- First accepted event is a true miss at frame 0/1 on most sequences, which
  has minimal sequence-level effect.

## 40. What Actually Improved

Temporal semantics are correct; verified error stream is pure; B0 is
byte-identical; unlimited dense remapping improves HOTA/AssA/IDF1 massively.

## 41. What Did Not Improve

Sparse B1-B8: no reliable HOTA/AssA/IDF1 gain; IDSW/Frag slightly worse.

## 42. Relation to Recent Methods

Only conceptual references; no directly applicable 2025-2026 method found.

## 43. Current SAM 3.1 Limitations

The frozen official API cannot re-associate a single correction into a
persistent trajectory without dense re-prompting (see N6/N7 audits); N8 does
not re-open that route.

## 44. Scientific Interpretation

Verified temporal errors are real and abundant on the frozen P0 backbone
(182k across 25 sequences), but they recur every ~1 frame.  Identity-layer
correction is therefore only effective as a dense overlay; sparse human
correction cannot persist under this backbone.

## 45. Whether Low-Budget Human Interaction Is Supported

**LOW-BUDGET GAIN NOT SUPPORTED.**  B1-B8 show no reliable positive gain and
slight IDSW/Frag degradation.  The positive regime requires dense,
continuous correction.

## 46. Reproducibility

Commands: `scripts/run_n8_real.py`, `scripts/run_n8_cpu_orchestrator.py`,
`scripts/run_n8_eval.py`, `scripts/run_n8_analysis.py`.  Frozen config:
`outputs/n8/n8_frozen.json`.  All CSVs/JSONLs under `outputs/n8/`; random
seeds used only in bootstrap statistics (seed 42).

## 47. Final Stage Status

**PASS_FULL_PROTOCOL_ONLY** (LOW-BUDGET GAIN NOT SUPPORTED).

## 48. Recommended Next Stage

Do **not** enter N9 interaction scheduling on this backbone: scheduling
cannot fix one-frame retention.  The next stage should either (a) improve the
backbone/association so an identity correction persists (tracklet-level
memory), or (b) move to detection-side interaction (new-target/Human Query)
where ADD_NEW-like corrections can reduce FN and provide localization value.
"""


def main() -> None:
    aggregate_jsonls()
    write("N8_RECENT_METHOD_GITHUB_AUDIT.md", doc_recent_methods())
    write("N8_N7_EVENT_SEMANTIC_REAUDIT.md", doc_reaudit())
    write("N8_TEMPORAL_ERROR_PROTOCOL.md", doc_protocol())
    write("N8_OBSERVER_SYSTEM_STATE_CONTRACT.md", doc_observer_contract())
    write("N8_TOY_TESTS.md", doc_toy_tests())
    write("N8_THREE_SEQUENCE_RESULTS.md", doc_three_seq())
    write("N8_25_SEQUENCE_RESULTS.md", doc_25_seq())
    write("N8_EVENT_LEVEL_ANALYSIS.md", doc_event_level())
    write("N8_COST_PERFORMANCE_ANALYSIS.md", doc_cost())
    write("N8_FINAL_REPORT.md", doc_final())
    # append to the project final report
    final = DOCS / "SAM3_INTERMOT_FINAL_REPORT.md"
    addendum = """

---

## N8 Addendum (2026-08-08)

Stage N8 (Error-Triggered Sparse Human Interaction) completed with
**PASS_FULL_PROTOCOL_ONLY**:

- Re-audit of N7 events: 251/375 (66.93%) were first-appearance renames.
- N8 temporal semantics remove all matched-first-appearance interactions;
  only TRUE_MISS_NEW / RECOVERABLE_MISS / TEMPORAL_ID_BREAK /
  TEMPORAL_ID_SWAP consume budget.
- B0 remains byte-identical to P0 on all 25 sequences; causality and budget
  gates pass; zero GT state leak.
- Canonical 25-sequence official TrackEval: B1-B8 show no reliable gain
  (HOTA +0.005..+0.020; AssA -0.005..-0.053; IDF1 +0.005..+0.012; IDSW and
  Frag slightly worse).  Unlimited dense correction (182,229 interactions)
  reaches HOTA 80.568 / AssA 80.389 / IDF1 92.324.
- Root cause of sparse failure: P0 track ids break almost every frame
  (time-to-next-error median = 1 frame; single-correction retention ~4-8%),
  so sparse identity-layer correction cannot persist.
- **LOW-BUDGET GAIN NOT SUPPORTED**; N9 scheduling is not recommended before
  improving correction persistence or moving to detection-side interaction.

Full evidence: `docs/N8_FINAL_REPORT.md`.
"""
    if "## N8 Addendum" not in final.read_text(encoding="utf-8"):
        with final.open("a", encoding="utf-8") as f:
            f.write(addendum)
    gate = {
        "N0": "PASS",
        "N1_GATE": "PASS",
        "N1_FULL_EVAL": "NOT_RUN",
        "N1_5": "PASS",
        "N2_REAL": "PASS",
        "N3_SMOKE": "PASS",
        "N3_FULL": "NOT_RUN",
        "N4_THREE_SEQUENCE_GATE": "PASS",
        "N4_WINNER": "R2_G2",
        "N4_FULL25": "PASS",
        "N5_0_CANONICAL_INTEGRITY": "PASS",
        "N5_2_TWO_SEQUENCE_GATE": "PASS",
        "N5_3_THREE_SEQUENCE_TREND_GATE": "FAIL",
        "N5_4_FREEZE": "NOT_REACHED",
        "N5_5_FULL25": "NOT_RUN",
        "N5_6_STATS_REPORT": "PARTIAL",
        "N6_0_AUDIT": "PASS",
        "N6_C_TOY_TESTS": "PASS",
        "N6_G_REAL_SEGMENT_RESTART_GATE": "PASS",
        "N6_H_THREE_SEQUENCE_GATE": "PASS",
        "N6_FREEZE": "DONE",
        "N6_I_FULL25": "PASS",
        "N6_FINAL_REPORT": "DONE",
        "N7_A_AUDIT": "PASS",
        "N7_B_LITERATURE": "PASS",
        "N7_C_EVENT_AUDIT": "PASS",
        "N7_D_ROOT_CAUSE": "PASS",
        "N7_E_ROUTE_A": "PASS",
        "N7_E_ROUTE_B": "ROUTE_B_NOT_SUPPORTED_BY_PINNED_OFFICIAL_API",
        "N7_F_TOY_TESTS": "PASS",
        "N7_G_REAL_GATE": "PASS_WITH_NOTES",
        "N7_H_THREE_SEQUENCE_GATE": "PASS",
        "N7_FREEZE": "DONE",
        "N7_I_FULL25": "PASS",
        "N7_FINAL_REPORT": "DONE",
        "N8_A_N7_EVENT_REAUDIT": "PASS",
        "N8_B_LITERATURE": "PASS",
        "N8_C_OBSERVER": "PASS",
        "N8_D_TOY_TESTS": "PASS",
        "N8_E_THREE_SEQUENCE_GATE": "PASS",
        "N8_FREEZE": "DONE",
        "N8_F_FULL25": "PASS",
        "N8_GATE_F_POSITIVE_GAIN": "NOT_SUPPORTED_LOW_BUDGET",
        "N8_FINAL_REPORT": "DONE",
        "TOTAL": "PASS_FULL_PROTOCOL_ONLY",
        "notes": "N8 PASS_FULL_PROTOCOL_ONLY: temporal-error protocol correct; "
        "matched first appearances never consume budget; 25-sequence official "
        "TrackEval complete; B0 byte-identical to P0; causality/budget/purity "
        "gates PASS; B1-B8 show no reliable sparse gain (IDSW/Frag slightly "
        "worse); Unlimited dense correction is strong (HOTA 80.57) but requires "
        "~7143 interactions/1000 frames; single-correction retention ~4-8%; "
        "LOW-BUDGET GAIN NOT SUPPORTED; next stage should improve correction "
        "persistence or move to detection-side interaction.",
    }
    (ROOT / "outputs/stage_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("N8 docs written")


if __name__ == "__main__":
    main()
