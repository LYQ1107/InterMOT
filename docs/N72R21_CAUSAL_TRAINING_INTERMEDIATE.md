# N72R21 — actual causal training, still active

Final Goal: **One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings**.

This is an intermediate record, not final closure. Current completion counts are derived in `outputs/N72R21/stage_status.json`; neither a training loss decrease nor structural pipeline checks authorize scientific success.

## T1: actual own-state training and state-shift control

The preregistered source is the matching T0_AMP_R1 model, not GT-positive teacher forcing. Full target/frame P0/P1 trajectories are sealed before annotations are opened. Stride-five before-state snapshots contain only current/past candidate references. The post-seal assembler reconstructs every intermediate state from the sole initial click and previous actual selected UID/write, checks exact equality with every saved snapshot, and verifies referenced embedding SHA. It never inserts a GT-positive crop into the bank. FIT uses stride five; INNER is the full frame axis. First-fold preflight reconstructed 50,076 real frames and checked 10,044 snapshots, with all 9,616 INNER target frames included.

T1 trains P0, P1 and deterministic whole-episode MIXED conditions, each with three real seeds and the same initial matching T0 weights. Its 88,391 parameters, losses, optimizer, patience and operating semantics are frozen before fitting. Own-condition INNER loss selects epochs; OUTER outcomes do not select checkpoints. Three complete folds (27 fits) preceded the present continuing folds. Training losses decrease but INNER wrong-accept rates remain high. This is not a reliable identity system.

Five real matched/shifted deployment conditions are preregistered: P0→P0, P1→P1, P1→P0, MIXED→P1, MIXED→P0. First OUTER scene 0001 has all three seeds and five full runtime/evaluation cases. Sequence-macro all-visible recall is respectively 21.11%, 20.91%, 21.35%, 21.41%, 21.14%. These one-scene differences lack cluster uncertainty and are not used to choose a deployed method. F1 scenes have historical research exposure; this is not final independent validation. P1→P0 is explicitly a state-shift control, never described as matched training.

T1 uses frozen behavior states. Refresh using the newly learned policy and future-memory supervision belong to T2; T1 does not silently claim those are complete.

## T2: paired actual future memory trajectories, loss-only labels

The T2 protocol was frozen before collection/fitting. Actual own P1 K8 state now comes from matching T1 MIXED checkpoints. At predetermined stride-50 proposals, two copies of the exact pre-write state process the same current/future candidates. One commits its own selected current observation, the other skips that current write. Both thereafter use their own predictions and writes over the next up-to-ten actual frames. The main trajectory and both counterfactual branches are separate and sealed without annotations. No branch is supplied an oracle positive observation.

Only afterward does a TRAIN/INNER label process identify verified current identity and future hard competitors. Safe supervision requires a verified correct current observation, at least one competitive future frame, no more than 0.02 mean projected hard-negative margin loss, no decrease in correct future selections and no increase in verified wrong selections versus omitting the write. Unknown current matches or missing future competitors stay UNKNOWN and are masked. This is a fixed-behavior paired diagnostic, not a differentiable simulator, a human intervention experiment, or evidence of cross-recording safety.

On the predetermined first FIT scene 0023, first seed: 153 proposals have usable labels, 130 satisfy this offline supervision rule, and 16 remain UNKNOWN. These numbers describe **training labels**, not a deployed memory-safety result. The real-data assembler checks all 13,338 contiguous state frames / 2,673 snapshots; current/risk losses and backward gradients are finite. No completed T2 fitting is claimed at this milestone.

An independent current-input future-safe head is implemented alongside current correctness, bringing the model to 89,672 parameters. Future frames/labels enter loss computation only, not network online inputs. The deployment path combines current correctness, separately predicted future safety, immutable-anchor agreement and two real consecutive confirmations before committing the **current** crop. A no-delay diagnostic and P0/P1/mean controls are available. Synthetic unit tests verify the interface and state transitions; they are not empirical safety or real cross-recording results. K1/4/8 and further module comparisons remain required actual work.

## Evidence, resources and remaining work

Latest complete regression at the 50-focused revision: **764 passed, 5 failed**, four warnings, 51.20s. The same four pinned TrackEval old-CLI list/path failures and historical literal-branch assertion remain. New focused suite including trusted-runtime contracts: **53 passed**; its revision is newer than that full snapshot. Earlier command-name/import mistakes yielded no scientific results and were corrected; no test/third-party code was patched to hide a failure. Raw XML is retained.

T0 24 fits, baseline/encoder/SOT results and original AMP failure remain preserved. All 795 historical sealed files were rechecked unchanged. Free personal storage remains approximately 91.2 GiB, above the separately frozen 60 GiB reserve. Two single-threaded source collectors, a controlled CPU replay and one coupled-state collector are bounded; only one GPU training process is allowed. No extra data copy, deletion or new environment. Current GPU fitter works on T1; T2 scheduling waits for that phase to finish rather than racing it.

CHIRLA public metadata remains complete but lawful media access unresolved. Official HF still requires user acceptance of contact sharing; ScienceDB direct and existing-proxy rechecks timed out. No gated media, token, password or captcha bypass. LaSOT contiguous TRAIN-trim SOT diagnostics are actual but do not supply missing absence/cross-day truth. T2 fits/evaluation, capacities/modules, independent and lawful cross-recording tests, final five tables/report and final Git closure remain active requirements. No downstream stage is authorized.
