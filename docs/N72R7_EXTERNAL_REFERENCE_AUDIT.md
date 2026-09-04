# N72R7 External Reference Audit

This audit records public method repositories inspected before implementing the
candidate-pool/reacquisition branch.  The repositories were cloned read-only at
the pinned commits below on 2026-09-04.  No external code is copied into
InterMOT; the references are design guidance only.  License status is recorded
because an idea reference is not permission to copy implementation code.

| Method | Paper / venue | Repository and inspected commit | Files inspected | License | Reusable mechanism | Code copied |
|---|---|---|---|---|---|---|
| DAM4SAM | [preprint](https://arxiv.org/abs/2411.17576); CVPR 2025 / IJCV 2026 extension | [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) @ `9c954504b39ebca4c412f207be0787c26bfac85a` | `dam4sam_tracker.py`; `sam2/sam2_video_predictor.py` | No repository `LICENSE` file found in inspected commit | reliable recent-memory admission and distractor-resolving memory | No |
| SAM2Long | [paper](https://arxiv.org/abs/2410.16268); ICCV 2025 | [SAM2Long](https://github.com/Mark12Ding/SAM2Long) @ `d70b50a7936fec55af201244ecde3d4433aff943` | `tools/vos_inference.py`; `sam2/sam2_video_predictor.py` | CC BY-NC 4.0 for the majority of the repository | diverse multi-path hypotheses and accumulated path scores | No |
| TrackTrack | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html) | [TrackTrack](https://github.com/kamkyu94/TrackTrack) @ `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da` | `3. Tracker/trackers/tracker.py`; `track.py`; `utils.py` | MIT | track-perspective association and low-confidence track support | No |
| MOTIP | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html) | [MOTIP](https://github.com/MCG-NJU/MOTIP) @ `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` | `models/runtime_tracker.py`; `models/motip/id_decoder.py`; `id_criterion.py`; `trajectory_modeling.py` | Apache-2.0 | trajectory context plus current detections for identity prediction | No |
| SeC | [preprint](https://arxiv.org/abs/2507.15852); ICLR 2026 | [SeC](https://github.com/OpenIXCLab/SeC) @ `0a797af5028623831c016692169df5c621037170` | `inference/modeling_sec.py`; `inference/sam2_video_predictor.py`; `training/sec/models/sec.py`; `sam2_train.py`; `training/TRAIN.md` | Apache-2.0 | progressive concept construction and enhanced pixel-level association | No |

InterMOT remains on its pinned SAM3/SAM3.1 path.  SAM2 private internals are
not imported.  The first N72R7 implementation is restricted to an adapter around
the existing InterMOT candidate rows and exact public-ID solver; a multi-path,
concept, or candidate-generator branch is only allowed if its own diagnostic
gate is triggered.
