Does N72R18 learned identity memory improve future public-ID association on a newly frozen fresh InterMOT candidate lineage?

FINAL GOAL:
Fresh-Lineage Learned Identity Memory Integration

CENTRAL QUESTION:
“在当前可获得的 InterMOT 代码、DanceTrack 数据、冻结 SAM3、N72R18 OSNet encoder 和 N72R18 learned identity memory 上，重新生成并冻结一条新的 candidate/base-score lineage 后，learned identity memory 是否能够改善未来 public-ID association？”

# Decision

`FAIL_LEARNED_MEMORY_DECISION_INACTIVE`

The fresh-lineage identity shadow was strongly positive, but the learned
signal did not cross the frozen exact public-ID assignment boundary on any
future smoke frame. This is not an association gain, so the stage stops here.

# Evidence

The new tape was generated from the current `Sam3Backend` and frozen N72R18
OSNet features, then sealed in
`outputs/N72R20R1/fresh_tape_manifest.json`. It contains only DanceTrack
train development smoke for `dancetrack0001` and `dancetrack0002`:

- 160 frames and 1,193 candidates;
- 160 frames and 1,436 candidates;
- unique candidate UIDs, finite boxes, 512-D feature lineage and verified
  per-file hashes;
- no DanceTrack test data and no runtime GT access.

E0, E1 and E2 consumed the same candidate tape, base-score matrix, public/state
axes, exact solver and explicit `NONE=0.0` policy.

## Identity shadow (post-hoc labels only)

| Horizon | Human-anchor hard-negative win | Learned-memory shadow | Delta |
|---|---:|---:|---:|
| H20 | 0.4324 | 0.8919 | +0.4595 |
| H50 | 0.5213 | 0.9468 | +0.4255 |
| H100 | 0.3913 | 0.9239 | +0.5326 |

At H100, rank-1 improved by `+0.5272`, MRR by `+0.3421`, and mean
positive-minus-hard-negative margin by `+0.1489`. The sequence-cluster 95% CI
for the H100 win-rate delta was `[+0.3864, +0.6667]` over two development
sequences. This establishes a learned identity candidate signal, not public-ID
association success.

## Assignment shadow

Across 318 future frames, E2 changed learned scores on all 318 frames and on
4,549 cells. The exact public assignment changed on `0` frames; changed public
IDs and candidate-vs-NONE changes were also `0`. Row-max preservation had
`0` failures. This directly triggers the frozen inactive-decision stop rule.

The formal primary endpoint, H100 `AssA(E2-E0)`, is therefore **not
estimated**. Formal VAL candidate generation, TrackEval, bootstrap aggregation
and downstream interactive MOT authorization were not started.

# Provenance and safeguards

- The historical N72R20 artifact remains `BLOCKED_NOT_RECOVERED` and was not
  changed or used as an R1 formal input.
- N72R18 GRU and OSNet checkpoints were strict/frozen; no training, LoRA,
  selector, new matcher, SAM3 source change or solver change was made.
- Learned memory followed score → fuse → exact assign → consensus update.
- Human initialization is marked `simulated_from_gt`; future GT was used only
  by post-hoc identity metrics.

`next_full_interactive_mot_stage_authorized=false`.
