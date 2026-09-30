# N72R20R1 Reference Audit

Final Goal: **Fresh-Lineage Learned Identity Memory Integration**.

This is a targeted mechanism audit at the exact requested commits. Existing
local clones under `/data3/liuyeqiang/research_references/N72R20` were reused;
no duplicate clone or download was made. No reference code or training system
is copied into InterMOT.

| Repository / exact SHA | Files inspected | Mechanism observed | Borrow for R1 | Do not borrow |
|---|---|---|---|---|
| MOTIP `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` | `models/runtime_tracker.py`; `models/motip/trajectory_modeling.py`; `models/motip/id_decoder.py` | Separates historical trajectory from current unknown detections; predicts identity labels before assignment; bounds history by miss tolerance. | Keep the conceptual score-before-commit ordering and explicit history lifetime as an audit check. | No ID decoder, vocabulary IDs, or replacement assignment protocol. |
| MeMOTR `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` | `models/query_updater.py` | Combines short/long memory and gates updates by detection confidence/active-track selection. | Treat confidence and active-state conditions as diagnostics for update eligibility. | No query updater, transformer memory, or trainable update module. |
| TrackTrack `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da` | `3. Tracker/trackers/tracker.py`; `3. Tracker/trackers/track.py` | `update_features()` is called after association/update, with smoothed feature history. | Confirms the causal rule: decision first, feature state update second. | No tracker class, smoothing alpha, or native track ID authority. |
| GeneralTrack `dbb727bfb63eddd28e97b1f64462cbf7df1413c6` | `core/Point2InstanceRelation.py` | Builds relational evidence between current detections and instances from feature/correlation maps. | Use only the abstract idea that candidate-vs-identity evidence can be relational. | No copied network, relation head, or image feature pipeline. |
| DiffMOT `eada72e74e54c153b30674d277b97882edc568b8` | `tracker/DiffMOTtracker.py`; `tracker/matching.py` | Keeps embedding and motion memories, performs staged matching, then optionally updates matched embedding state. | Use as a review reference for staged causal association and matched-observation update. | No diffusion predictor, Kalman/matching replacement, or staged solver. |
| MASA `c5472b9c7615f35abdf1188cb1a0c5408fe50d66` | `masa/models/tracker/masa_tao_tracker.py` | Memo stores embedding/bbox/frame; match score combines bi-softmax and cosine; spatial mask and expiry bound memory. | Audit memory expiry/coverage diagnostics without changing the frozen R1 memory. | No memo momentum, spatial mask, or MMDetection tracker integration. |
| BoostTrack `fb5bfc3a8f067476565e753b3a73df4d757c9d03` | `tracker/boost_track.py` | Combines IoU, Mahalanobis, confidence and embedding evidence; `update_emb()` follows a match. | Confirms that heterogeneous evidence must remain auditable and update after match. | No confidence boosting, Mahalanobis implementation, or alternate association. |
| BoT-SORT `251985436d6712aaf682aaaf5f71edb4987224bd` | `tracker/bot_sort.py` | Smooth feature updates, proximity/appearance gates, and track reactivation. | Use the separation of appearance gate and spatial gate as an audit vocabulary. | No BoT-SORT tracker, thresholds, reactivation policy, or ReID encoder. |
| OC_SORT `8462e7e729a93ccd3bd995c0a79a890336cb3a0b` | `trackers/ocsort_tracker/ocsort.py` | Maintains observation history and predicted state; performs re-update after staged unmatched detection handling. | Keep observed-vs-predicted state distinction visible in diagnostics. | No OC-SORT association or observation-history implementation. |
| CKP `183218eb1624027a1991586c931b242dd08d3a35` | `reid/trainer_noisy.py` | Noisy-label refinement and co-training make assigned/noisy observations different from trusted knowledge. | Supports the R1 rule that assigned observations are not automatically trusted long-term writes; consensus is required. | No noisy ReID training, pseudo-label refinement, or new checkpoint. |

## Fixed reference provenance

All ten local clone HEADs matched the requested SHAs at audit time. The local
clone root is the historical N72R20 reference root, not an R1 result asset;
its 750 MB footprint is already counted in the storage audit. The R1 formal
lineage remains independent of all reference outputs.

## Audit conclusion

The references support the already-frozen R1 design choices—separate state and
observation, score/associate before state update, confidence/active gating,
and bounded memory provenance. They do not authorize importing another tracker,
solver, decoder, or trainable memory. R1 will borrow only these causal design
principles and will evaluate the existing InterMOT seam.
