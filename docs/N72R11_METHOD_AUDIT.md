# N72R11 public-method audit

This is a design audit, not evidence that any external method was imported or
that N72R11 passed its research gate. The search started with 2025--2026
official papers/project pages and GitHub repositories, covering interactive
MOT, human-in-the-loop tracking, online appearance memory, data association,
score calibration and parameter-efficient adaptation. Only records with a
checkable primary paper/project source and a pinned repository commit were
kept. No external code was copied into the production association path.

| Method | Paper / official page | Repository and audited commit/date | Reusable mechanism | N72R11 disposition |
|---|---|---|---|---|
| MOTIP (CVPR 2025) | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf), [arXiv](https://arxiv.org/abs/2403.16848) | [MCG-NJU/MOTIP](https://github.com/MCG-NJU/MOTIP), `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`, 2026-07-30 | In-context trajectory/ID prediction rather than a hand-written association post-process. | Design comparison only; importing it would change the frozen public-ID solver. |
| TrackTrack (CVPR 2025) | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html) | [TrackTrack](https://github.com/kamkyu94/TrackTrack), `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`, 2025-09-24 | Track-perspective association and track-aware initialization. | External comparator only; N72R11 retains the frozen Hungarian definition. |
| SENTRY (2026) | [arXiv](https://arxiv.org/abs/2606.24449) | [SENTRY](https://github.com/HamadYA/SENTRY), `dd4486c7eeadd7e7022854e29e95e3101390ce65`, 2026-07-15 | Neighbor-aware temporal memory admission, candidate pooling and conservative rescue. | Closest conceptual memory-safety comparator; not copied because the N72R11 candidate/solver protocol is frozen. |
| TCEI (CVPR 2026) | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Guo_Dual-level_Adaptation_for_Multi-Object_Tracking_Building_Test-Time_Calibration_from_Experience_CVPR_2026_paper.pdf), [arXiv](https://arxiv.org/abs/2603.21629) | [TCEI](https://github.com/1941Zpf/TCEI), `145d1b8431398156f8d9f854430e306fdee39eaa`, 2026-03-30 | Separates transient intuition from accumulated experience for test-time calibration. | Reference for role separation only; no code or checkpoint was used. |
| InteractTrack (CVPR 2026) | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Interactive_Tracking_A_Human-in-the-Loop_Paradigm_with_Memory-Augmented_Adaptation_CVPR_2026_paper.pdf) | [InteractTrack](https://github.com/NorahGreen/InteractTrack), `5f149d4001a84c8b83129192057bf6dd820f71b3`, 2026-06-16 | Human-in-the-loop correction and memory-augmented adaptation protocol. | Protocol reference only. N72R11 has no real-human tape; its events remain `simulated_from_gt`. |

The machine-readable source record is `outputs/N72R11/method_audit.json`. The
repository license/source checks and excluded-search rule are recorded there.
These links do not relax N72R11's no-training-before-complete-corpus gate.
