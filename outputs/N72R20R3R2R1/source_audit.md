# N72R20R3R2R1 Source Audit

- source repository: `LYQ1107/InterMOT`
- source branch: `codex/n72r20r3r2-cross-scene-identity-representation`
- resolved source HEAD: `02dada7664ef0e5c13ec3df74b8b284b41ddaef3`
- working branch: `codex/n72r20r3r2r1-open-set-decision-closure`
- historical source artifact commit recorded by R3R2: `90a141ad85b3b5c7fa2d611f3b3b96e9c182d44a`
- historical R3R2 decision remains immutable: `FAIL_OPEN_SET_CALIBRATION_AFTER_REPRESENTATION`
- historical R3R2 outputs are read-only inputs; this stage writes only under `outputs/N72R20R3R2R1/`

## Frozen source facts

R3R2 representation Rank-1 was `0.6874905231235785` versus the frozen baseline `0.3815011372251706`; oracle-clean Rank-1 was `0.9035633055344958`. The historical direct NONE-logit method had pooled negative FPR `0.23474436503573393` and open-set correct-ID recall `0.5323730098559515`, so the new stage explores the preregistered calibration and state-quality tree rather than rewriting that result.

The candidate tape, OSNet embeddings, N72R18 GRU state, CrossSceneIdentityAdapter candidate tower, exact solver and public-ID authority remain frozen. No SAM3 rerun, DanceTrack VAL/TEST access, association authority, candidate creation or model binary commit is permitted.
