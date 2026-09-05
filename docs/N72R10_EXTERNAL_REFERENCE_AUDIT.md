# N72R10 external reference audit

Audit date: 2026-09-05 (Asia/Shanghai).  This is a design audit, not evidence that
any external method has been reproduced in InterMOT.  Repository revisions below are
the `HEAD` values returned by `git ls-remote` on the audit date; no external repository
was copied into this worktree.

## Sources and reusable mechanisms

| Method | Paper / official page | Official code | Audited revision and commit date | Relevant mechanism | N72R10 disposition |
|---|---|---|---|---|---|
| SENTRY | [arXiv:2606.24449](https://arxiv.org/abs/2606.24449) | [HamadYA/SENTRY](https://github.com/HamadYA/SENTRY) | `dd4486c7eeadd7e7022854e29e95e3101390ce65`, 2026-07-15 | Candidate-mask admission uses temporal verification and neighbor-aware distractor checks before replacing memory. | Design reference for explicit rescue admission and protected identities; N72R10 keeps SAM3 and its candidate/metric definitions unchanged, so no SENTRY code is imported. |
| InteractTrack / IMAT | [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Interactive_Tracking_A_Human-in-the-Loop_Paradigm_with_Memory-Augmented_Adaptation_CVPR_2026_paper.pdf) | [NorahGreen/InteractTrack](https://github.com/NorahGreen/InteractTrack) | `5f149d4001a84c8b83129192057bf6dd820f71b3`, 2026-06-16 | Interactive tracking benchmark with timestamped instructions and positive/negative memory banks. | Supports separating interaction provenance from automatic re-query and keeping positive/negative memory auditable. N72R10 events remain simulated-from-GT where applicable, not historical human evidence. |
| TCEI | [arXiv:2603.21629](https://arxiv.org/abs/2603.21629) | [1941Zpf/TCEI](https://github.com/1941Zpf/TCEI) | `145d1b8431398156f8d9f854430e306fdee39eaa`, 2026-03-30 | Separates transient intuitive memory from accumulated experiential calibration at test time. | Relevant to source-specific and train/holdout calibration accounting; not used to alter N72R10's frozen association solver or to select events from future labels. |
| SeC | [ICLR 2026 official repository](https://github.com/OpenIXCLab/SeC) | [OpenIXCLab/SeC](https://github.com/OpenIXCLab/SeC) | `0a797af5028623831c016692169df5c621037170`, 2026-03-27 | Concept-driven video segmentation and adaptive semantic/feature reasoning. | A segmentation-memory reference only. It is not an identity association method and is not inserted into the SAM3 decoder or candidate generator. |
| STAR | [OpenReview: Spatial-Temporal Tracklet Matching](https://openreview.net/forum?id=fmCnNQjZrr) | [kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) | `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`, 2025-09-24 | Spatial-temporal tracklet matching for occlusion and viewpoint variation. | Motivates measuring short/long-horizon propagation and sequence-cluster effects. The repository is TrackTrack's code, not an identified STAR implementation; no STAR algorithm is claimed as imported. |
| MOTIP | [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf) | [GISer-WB/MOTIP-2](https://github.com/GISer-WB/MOTIP-2) | `012856c1dc13b324064e79339ae71054518d1b5e`, 2025-03-23 | Treats association as identity prediction using historical trajectory context and ID prompts. | Useful comparison for identity-conditioned association, but N72R10 preserves the exact public-ID solver and tests its future rebind contract rather than replacing it. |
| TrackTrack | [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) | [kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) | `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`, 2025-09-24 | Track-perspective association and track-aware initialization address global matching and occlusion failure modes. | Relevant as an association-boundary diagnostic and hard-negative comparison; N72R10 does not replace Hungarian evaluation. |
| DAM4SAM | [official GitHub / CVPR 2025 description](https://github.com/jovanavidenovic/DAM4SAM) | [jovanavidenovic/DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) | `9c954504b39ebca4c412f207be0787c26bfac85a`, 2026-04-07 | Distractor-aware memory admission for SAM2, with conservative replacement of spatial memory. | Supports preserving trusted memory and explicitly auditing distractor rejection. It is SAM2-specific and is not used as a SAM3 API or checkpoint. |
| SAM2Long | [ICCV 2025 paper / arXiv:2410.16268](https://arxiv.org/abs/2410.16268) | [Mark12Ding/SAM2Long](https://github.com/Mark12Ding/SAM2Long) | `d70b50a7936fec55af201244ecde3d4433aff943`, 2026-08-14 | Maintains diverse memory pathways and prunes them using cumulative quality and uncertainty to reduce long-video error accumulation. | Supports recording trigger-to-H20/H50/H100 drift and keeping future windows independent. Its SAM2 memory tree is not transplanted into SAM3. |

## Scope and exclusions

The search covered GitHub, OpenReview, arXiv, CVF Open Access, and official project
pages, prioritizing 2025--2026 work on human-in-the-loop tracking, interactive
tracking, online appearance/memory, data association, calibration, and parameter-
efficient adaptation.  Older SAM2Long work was retained because it is the official
long-video memory reference explicitly named by the frozen N72R10 protocol.  Results
that were only secondary summaries, unrelated 3-D interaction tracking, or lacked a
verifiable official implementation were not used as implementation authority.

The sources support design hypotheses and audit fields; they do not license changing
the N72R10 checkpoint, SAM3 candidate generation, Hungarian evaluation, future window,
or gate.  In particular, no source above justifies calling a frozen historical
candidate lookup a `FUTURE_FRAME_REQUERY`; that label is reserved for a new current-
frame SAM3 session created at the causal trigger frame.
