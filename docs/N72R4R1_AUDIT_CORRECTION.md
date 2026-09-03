# N72R4R1 Audit Correction

> This is a bookkeeping/provenance correction, not a new scientific experiment.

N72R4 remains `M3_SIGNAL_WAS_SOLVER_ARTIFACT` with research gate `FAIL_FUTURE_EFFECT`; no model, checkpoint, candidate stream, solver definition, or historical output was changed.

## Correct NO versus M0 candidate recall

| Horizon | NO intervention | M0 current-frame correction | M0 − NO |
|---:|---:|---:|---:|
| H20 | 0.692307692 | 0.752136752 | 0.059829060 |
| H50 | 0.683501684 | 0.649831650 | -0.033670034 |
| H100 | 0.710884354 | 0.574829932 | -0.136054422 |

The values above come directly from the canonical Stage10 artifact, not from the stage-status pointer. Runtime future-GT usage remains `false`; GT is used only for posthoc scoring.

## Provenance-hash repair

The effect-assignment wrapper now records separate canonical hashes for the original state×candidate matrix and the solver-facing candidate×state transpose. Existing N72R3R1/N72R4 artifacts are preserved as historical evidence; they are not rewritten in place.

Machine-readable audit: `/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R5/audits/n72r4_stage10_recall_repair.json`
