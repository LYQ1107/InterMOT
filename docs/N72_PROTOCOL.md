# N72 Mapping and Real-Human Contract Closure Protocol

**Protocol name:** N72_MAPPING_AND_REAL_HUMAN_CONTRACT_CLOSURE  
**Date:** 2026-09-02 (Asia/Shanghai)  
**Project:** InterMOT / SAM3_InterMOT  
**Protocol status:** FROZEN_BEFORE_POSTHOC_DIAGNOSTIC

## Purpose

N72 closes two prerequisites and performs one frozen mechanism diagnosis:

1. define a machine-verifiable candidate/native/local/global/public identity
   mapping contract;
2. define a provenance-complete real-human event ingestion and validation
   contract;
3. diagnose why N71 score changes did not cross the Hungarian assignment
   boundary.

N72 is not a new efficacy experiment, weight scan, production candidate, or
training run.

## Frozen boundaries

- N36-N71 artifacts are read-only inputs and must not be overwritten.
- N72 must not claim real-human efficacy or future identity improvement.
- All existing simulated events remain marked simulated_from_gt.
- Mapping must not infer identity from GT, IoU, appearance similarity, future
  trajectories, or geometric nearest-neighbor heuristics.
- Runtime must not read future GT. GT is allowed only after frozen runtime
  artifacts pass structural validation, for posthoc diagnostics.
- Checkpoints, candidate definitions, Hungarian solver, official metric
  definitions, and sequence-cluster bootstrap rules are unchanged.
- third_party/sam3 is read-only.
- No calibration head, selector, decoder LoRA, or production association
  training is authorized in N72.
- No broad lambda, human_weight, threshold, rank, or LoRA scan is allowed.
- New behavior must use an explicit N72 entry point or switch; old behavior
  remains compatible by default.
- Evidence insufficiency must be represented as BLOCKED or UNMAPPED, never
  filled by an heuristic.

## Mapping contract

The canonical identity chain is:

    official raw SAM native ID
        -> adapter external object ID
        -> segment/chunk-local candidate identity
        -> sequence-global candidate identity
        -> baseline public-ID assignment

Every candidate must retain its provenance. Raw native IDs are not globally
unique across sessions. Unmapped, ambiguous, collided, stale, absent, and
axis-mismatched records remain explicit.

An exact public mapping may come only from an identity namespace/registry
binding, an explicit runtime assignment, a direct user-provided public ID, or
another frozen provenance-complete mapping record.

## Real-human contract

A real-human event must be written by an external UI or annotator and include a
direct public ID, raw BOX/CLICK/CONFIRMED_MASK input, human confirmation,
session/annotator/timestamp provenance, frame and pre-output hashes, candidate
tape reference, mapping reference, and prefix/future ranges.

Fixtures and GT-derived events must never be counted as real_human. The event
frame must not read newly written memory; memory becomes visible from event+1.

## Boundary diagnostic

The assignment-boundary diagnostic uses frozen N71 artifacts only. It reports
score movement, target/competitor margins, forced-assignment regret, required
residual, actual-to-required residual ratio, explicit NONE preference,
candidate/mapping absence, temporal guards, and solver-coupled collateral.

It does not select new parameters, events, checkpoints, or models in N72.

## Authorization flags

    production_authorization = false
    training_authorization = false
    efficacy_claim_authorized = false
    runtime_future_gt_used = false

## Required completion states

N72 may end only as a contract/diagnostic state, not PASS_EFFICACY:

- COMPLETE_CONTRACT_PASS_WAITING_REAL_HUMAN
- PARTIAL_MAPPING_PASS_REAL_HUMAN_RECORDER_READY
- CANDIDATE_PROVENANCE_PASS_PUBLIC_MAPPING_BLOCKED
- BLOCKED_EXACT_MAPPING_SOURCE
- FAILED_NEW_REGRESSION
- COMPLETE_DIAGNOSTIC_ONLY
