# N72R20 Fixed Reference Audit

All repositories below were cloned under
`/data3/liuyeqiang/research_references/N72R20/` with shallow/blobless history
and checked out at the exact required commit. Total checkout size at audit
time is approximately 750MB. These references are design evidence only; none
is imported into the InterMOT runtime.

| Reference (URL, pinned SHA) | Inspected mechanism | What may be borrowed | Explicitly not borrowed |
|---|---|---|---|
| [MOTIP](https://github.com/MCG-NJU/MOTIP), `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` | `models/runtime_tracker.py`, `models/motip/trajectory_modeling.py`, `models/motip/id_decoder.py`; trajectory-local feature/history and learned ID vocabulary | separation of trajectory representation from ID decision; score/decision timing as an audit concept | trained tracker-local vocabulary, newborn vocabulary, decoder architecture, public-ID ownership, assignment replacement |
| [MeMOTR](https://github.com/MCG-NJU/MeMOTR), `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` | `models/query_updater.py`; positive-gated short/long memory and EMA update | confidence-gated memory updates as a comparison point | detector-query memory, query updater, detector thresholds, replacement of human-anchor protocol |
| [TrackTrack](https://github.com/kamkyu94/TrackTrack), `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da` | `3. Tracker/trackers/tracker.py`, `3. Tracker/trackers/track.py`, `update_features()`; feature EMA with score-dependent smoothing | simple normalized feature update and the need to record update provenance | local track IDs, high/low detection policy, its matcher and thresholds |
| [GeneralTrack](https://github.com/qinzheng2000/GeneralTrack), `dbb727bfb63eddd28e97b1f64462cbf7df1413c6` | `core/Point2InstanceRelation.py`, `core/update.py`; learned pair relation | relation features as a conceptual diagnostic | image-pair relation head, ROIAlign/ConvMLP, learned relation matcher |
| [DiffMOT](https://github.com/ShuCvlab/DiffMOT), `eada72e74e54c153b30674d277b97882edc568b8` | `tracker/DiffMOTtracker.py`, `tracker/matching.py`; normalized feature EMA plus motion/appearance cost | separation of motion and appearance evidence | diffusion motion predictor, its cost fusion, Hungarian implementation and local IDs |
| [MASA](https://github.com/siyuanliii/masa), `c5472b9c7615f35abdf1188cb1a0c5408fe50d66` | `masa/models/tracker/masa_tao_tracker.py`; memo lifecycle and embedding momentum | explicit memo lifetime/expiry as a failure-mode audit item | memo-owned local instance IDs, greedy matching, bi-softmax/memo policy |
| [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack), `fb5bfc3a8f067476565e753b3a73df4d757c9d03` | `tracker/boost_track.py`; confidence-adjusted feature update and multi-cue association | confidence/update provenance as a diagnostic field | Kalman local IDs, shape/IoU/Mahalanobis matcher, confidence retuning |
| [BoT-SORT](https://github.com/NirAharon/BoT-SORT), `251985436d6712aaf682aaaf5f71edb4987224bd` | `tracker/bot_sort.py`; normalized smooth feature/history | feature-history representation as a comparison | BoT-SORT association, camera-motion module, local `STrack` IDs |
| [OC_SORT](https://github.com/noahcao/OC_SORT), `8462e7e729a93ccd3bd995c0a79a890336cb3a0b` | `trackers/ocsort_tracker/ocsort.py`; observation-centric geometry and velocity | none beyond documenting geometry as a separate cue | geometry-only tracker, second-pass matching, local track IDs |
| [CKP](https://github.com/zhoujiahuan1991/MM2024-CKP), `183218eb1624027a1991586c931b242dd08d3a35` | `reid/trainer_noisy.py`; noisy-label/pseudo-label training and anti-forgetting | contamination and label-noise terminology for diagnostics | its training pipeline, pseudo-label generation, co-refinement and any new training |

## Reference-level conclusion

The references converge on a narrow reusable abstraction: a state-owned,
normalized identity representation may be updated with explicit provenance and
confidence, while the decision layer remains separate. They do not justify
changing InterMOT's candidate generation, public authority, solver, or human
interaction protocol. N72R20 therefore retains the N72R18 GRU state and frozen
OSNet contract, and tests only whether that state changes the existing public-ID
assignment boundary in a beneficial, auditable way.
