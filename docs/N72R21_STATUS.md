# N72R21 — active research status

Final Goal: **One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings**.

The application Goal is active; canonical specification is `outputs/N72R21/FINAL_GOAL.json`. This is an intermediate status, not a final scientific decision or next-stage authorization.

M0/M1 history/resource audit and a real CPU development smoke are complete. Historical decisions are preserved; 795 files sealed, 463 model binaries plus frozen OSNet independently verified. Existing DanceTrack is reused, not copied. New focused tests: **20 passed**.

Real smoke: two repeatedly exposed TRAIN sequences, 15 first-eligible `SIMULATED_ONE_CLICK_FROM_GT` targets, 14,154 visible target frames, 376 visible-GT absence frames and 51 reappearances. A separate GT-free runtime process seals results before offline evaluation. Fixed-anchor smoke is weak and makes wrong-person selections; this is not scientific PASS. Missing GT does not identify physical out-of-view versus occlusion; zero writes do not establish useful safe memory.

CHIRLA official metadata acquisition is partial; no media downloaded. Its public HF card requires access/terms acceptance, and direct ScienceDB/server paths remain unresolved. Existing TAO LaSOT/person folders are not complete original LaSOT and cannot supply invented absence labels. Full scientific splits/gates and other baselines/training/evaluation are pending. Current work proceeds on lawful data alternatives and formal one-click methodology, not on global HOTA optimization.

See `outputs/N72R21/EXECUTION_LOG.md`, `historical_audit/`, `datasets/`, `storage/` and `smoke/`. No downstream stage is authorized.
