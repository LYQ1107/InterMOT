FINAL GOAL: **Human Correction Driven Persistent Identity Adaptation**

CENTRAL QUESTION: **When a human corrects a tracking error, can the system
learn from this correction and improve future identity tracking?**

These JSONL files are a train-only N72R20 correction-environment artifact.
They were generated from the GT-free SAM3 candidate caches for
`dancetrack0001` and `dancetrack0002`, then labeled with DanceTrack GT only in
offline error discovery. Each record represents a simulated human correction;
it does not claim real user evidence and it does not update identity memory.

The current run contains 1,043 events from 1,497 correction opportunities.
`correction_event_summary.json` is the aggregate source of truth. The
record-only updater is reserved for N72R21; no training, val evaluation,
MOT/TrackEval, or association integration belongs in this directory.
