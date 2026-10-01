# InterMOT N72R20R3R2R2 Final Report

## Final goal

**Human Identity Representation Probe → Joint Open-Set Identity-Availability Representation Learning**

Central question: “用户点一下这个人以后，我们到底能不能认住他？”

This stage asks whether a frozen cross-scene identity representation can both rank a valid target candidate and reject invalid, absent, or unavailable candidates without changing candidate generation or association authority.

## Lineage and protocol

- Source lineage: N72R20R3R2R1 at `7820d312e2b3c852a7909a4d42dc71bad0c085c6`.
- Eight outer sequence-held-out folds; held-out observations were not used for training or inner selection.
- Episodes: PRESENT `40,662`, P0 `1,809`, P1a `7,840`, P1b `4,151`; counterfactual episodes `121,845`.
- J1/J2/J3/J4/J5/J7 were evaluated. J6 geometry/motion fusion was explicitly skipped because causal target prediction is not available for every runtime frame; see [`J6_geometry_fusion.json`](../outputs/N72R20R3R2R2/joint_representation/J6_geometry_fusion.json).
- Candidate generation, OSNet, N72R18 state, exact solver, public-ID authority, SAM3, VAL/TEST evaluation, and association authority remained frozen.
- GT was used only for labels and post-hoc diagnostics; no future GT feature entered runtime inference.

## Result

Final decision: **`FAIL_P1_LOCALIZATION_QUALITY`**.

The selected open-set pipeline achieved Rank-1 `0.6643`, MRR `0.7735`, candidate-valid AUROC `0.7901`, set-presence AUROC `0.7819`, and recall at pooled FPR 2% `0.1883`. Correct-ID recall was `0.2614`; pooled negative FPR was `0.0665`; macro recall/FPR were `0.2244`/`0.1289`. The P1 localization-quality gap therefore remains the limiting factor.

For historical comparison, the frozen R3R2R1 calibration ceiling was pooled FPR `0.082463`, correct-ID recall `0.227597`, macro FPR/recall `0.067036`/`0.201144`, and shadow wrong-write rate `0.204608`.

The shadow causal replay is diagnostic only: wrong-write rate `0.2530`, retention `0.7470`. The causal gate failed, so `next_association_stage_authorized=false`. No association integration was started.

## Verification

- Focused N72R3R2R2 invariants: **50 passed**.
- Full pytest: **542 passed, 4 failed**.
- The default run’s two N6 failures could not resolve the executable name `python`; the explicit-venv rerun reached the fixed TrackEval dependency and reproduced the same four N6/N7/N8 failures caused by its `SEQMAP_FILE` list-versus-string `os.path.isfile` incompatibility. These are not N72R3R2R2 research-code failures.
- Storage after the run: approximately `110.07 GiB` free; no checkpoint binaries were placed under `outputs/`.

The machine-readable artifacts are under [`outputs/N72R20R3R2R2`](../outputs/N72R20R3R2R2), including the frozen goal, manifests, LOSO results, diagnostics, causal audit, and stage status.

## Authorization and stop boundary

`next_association_stage_authorized=false`. The stage stops at the identity/availability representation result. No automatic SAM3 rerun, solver redesign, LoRA, training continuation, or VAL/TEST evaluation is authorized by this result.
