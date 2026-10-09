# N72R21 code-only delivery

The latest user direction is **MOT primary; SOT deferred**. No further SOT
downloads, fitting, inference or OOM retries are authorized. Historical evidence
remains local and is not removed.

GitHub delivery branch: `codex/n72r21-mot-code-only`. Its new changes contain
source code, tests, the final research report and this delivery explanation only.
The branch is based on the previously published research commit; its existing
history is preserved. New `outputs/` artifacts, experiment logs, checkpoints,
datasets, embeddings, candidate tapes, JUnit XML, images and videos are **not**
included in this delivery. The complete local research branch is retained.

The current frozen system has a **restricted scientific FAIL**, not a MOT
success. The two-sequence TRAIN joint-MOT pilot shows identity/association
regression; target-only frozen VAL diagnostics are not full-MOT evaluation.
Missing lawful cross-recording media remain explicitly not evaluable. No
downstream association stage is authorized.

See [the final report](N72R21_FINAL_REPORT.md) and the previously published
[MOT TRAIN pilot report](N72R21_MOT_TRAIN_PILOT.md) for scope and limitations.
Paths to local evidence in those reports identify reproducibility inputs, not
files newly published by this code-only delivery. Existing asset manifests and
the existing environment must be used; do not recreate or upload datasets.

Actual regression snapshot: N72R21-focused **112 passed**; complete repository
**826 passed, 5 failed**. Four failures are the pinned TrackEval CLI
`SEQMAP_FILE` list-versus-path mismatch; one asserts a historical branch name.
No test or third-party code was patched to conceal those failures.
