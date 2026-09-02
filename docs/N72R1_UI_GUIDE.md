# N72R1 UI guide

The standard-library UI is an ingestion boundary only. It has no simulator, GT
reader, replay runner, evaluator, or public-ID inference. Use a real annotator
account and a frame shown by the external annotation workflow.

The endpoint is `POST /api/events`; `/health` reports readiness. Raw request bytes
are stored under a separate root, and accepted finalized events are hash-chained in
the validated root. The UI does not start full-loop, replay, or training.
