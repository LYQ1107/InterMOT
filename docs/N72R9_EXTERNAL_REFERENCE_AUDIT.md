# N72R9 External Reference Audit

Date of audit: 2026-09-05 (Asia/Shanghai).  The search was performed before the
N72R9 implementation work.  It covered GitHub, CVF Open Access, arXiv and
OpenReview with 2025–2026 queries for human-in-the-loop tracking, interactive
tracking, online ReID/appearance memory, association learning, score
calibration, margin-aware matching and parameter-efficient adaptation.

The references below are design evidence only.  No external repository was
vendored, no external checkpoint was loaded, and no published result is counted
as an InterMOT result.

| Reference | Pinned public revision/date | Audited mechanism | N72R9 use and boundary |
|---|---|---|---|
| [SENTRY paper (ECCV 2026)](https://arxiv.org/abs/2606.24449) / [official GitHub](https://github.com/HamadYA/SENTRY/tree/dd4486c7eeadd7e7022854e29e95e3101390ce65) | `dd4486c7eeadd7e7022854e29e95e3101390ce65`, `2026-07-15T11:19:08Z`, main | Backtrack candidate masks into short tracklets, compare target and neighbor/distractor trajectories, and admit only a conservatively verified rescue. | Reuses the *separation* of trusted target, distractor and neighbor evidence and a shadow/commit gate. It is not copied: SENTRY is training-free and SAM2-host-specific, while N72R9 keeps the frozen SAM3 candidate stream and public-ID solver. |
| [InteractTrack / IMAT paper (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Interactive_Tracking_A_Human-in-the-Loop_Paradigm_with_Memory-Augmented_Adaptation_CVPR_2026_paper.pdf) / [arXiv](https://arxiv.org/abs/2604.01974) / [official GitHub](https://github.com/NorahGreen/InteractTrack/tree/5f149d4001a84c8b83129192057bf6dd820f71b3) | `5f149d4001a84c8b83129192057bf6dd820f71b3`, `2026-06-16T18:37:04Z`, main | Timestamped interaction protocol and memory-augmented adaptation with positive/negative evidence. | Supports N72R9's correction-before-write and event+1 causal boundary. The project has no real human tape; every current interaction remains `simulated_from_gt`, and the single-target/language setting is not treated as a drop-in multi-object public-ID model. |
| [TCEI paper (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Dual-level_Adaptation_for_Multi-Object_Tracking_Building_Test-Time_Calibration_from_Experience_CVPR_2026_paper.html) / [arXiv](https://arxiv.org/abs/2603.21629) / [official GitHub](https://github.com/1941Zpf/TCEI/tree/145d1b8431398156f8d9f854430e306fdee39eaa) | `145d1b8431398156f8d9f854430e306fdee39eaa`, `2026-03-30T02:04:24Z`, main | Separates transient intuition from accumulated experiential calibration for online MOT. | Motivates explicit short-horizon versus durable identity state and a later calibration branch. It does not authorize test-time learning here; calibration is downstream of the frozen N72R9 gate. |
| [Hierarchical Multi-Prototype Appearance Memory (2026)](https://www.mdpi.com/2079-9292/15/11/2357) | No public project repository located in this audit; publication `2026-05-29` | Stable long-term prototypes, a short FIFO transitional queue, reliability-controlled writes and staged association. | Used as a conceptual comparison for trusted-anchor versus recent-observation memory. No unverified implementation or result is attributed to the paper. |
| [UMOT (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12469223/) | No verified official repository pinned in this audit | Track-query memory with temporal enhancement and historical backtracking for short/long association. | Supports separating local temporal evidence from long-range identity state. The frozen SAM3 decoder and Hungarian evaluation are unchanged. |
| [STAR (NeurIPS 2025)](https://openreview.net/forum?id=fmCnNQjZrr) | OpenReview revision visible `2026-04-21`; no verified code revision used | Spatial-temporal tracklet graph matching and higher-order message propagation. | Used only to motivate neighbor/relational diagnostics; no graph module is imported before N72R9's own diagnosis. |
| [FDTA (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_From_Detection_to_Association_Learning_Discriminative_Object_Embeddings_for_Multi-Object_CVPR_2026_paper.html) | Official CVF paper page; no verified code revision used | Refines object embeddings for instance-level association rather than category-only separation. | A diagnostic alternative if source-quality evidence is the bottleneck; no backbone or embedding definition is changed in this round. |
| [TrackTrack (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html) | Official CVF paper page; no verified code revision used | Track-perspective association and track-aware initialization. | Background for track-scoped association; not a replacement for N72R9's explicit public-ID authority. |

## Search limits and exclusions

The 2025–2026 search did not identify a verified public method that simultaneously
provides direct human correction, persistent public-ID memory, a source-aware
target/distractor/neighbor temporal state, true future-frame re-query, and a
global candidate×identity×NONE interface for the frozen SAM3 stream.  LA-MOTR's
[ICCV 2025 paper page](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV2025_paper.html)
advertises `https://github.com/PenK1nG/LA-MOTR`, but that repository was not found
at audit time, so no code or revision was used.  Search results for unrelated
medical-VLM LoRA and generic tracking repositories were excluded.

The implementation decision is therefore deliberately narrow: use the public
mechanisms as protocol inspiration, implement a local source-aware temporal
decoder/requery sidecar, and require the unchanged future-effect and untouched-ID
gates.  This is not a claim that N72R9 reproduces SENTRY, IMAT, TCEI, HMP, UMOT,
STAR, FDTA or TrackTrack.
