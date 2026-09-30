# InterMOT N72R18 reference audit

## Scope

This audit was completed before adding N72R18 implementation code. The five
repositories were cloned under `/data3/liuyeqiang/research_refs/` and were
read as design references only. No external training environment, dataset,
image crop, checkpoint, tracker, or association code is imported into
InterMOT.

| method | memory / representation form | update or training signal observed in the reference | fit for N72R18 |
|---|---|---|---|
| STMN | Two learned lookup memories. Spatial memory has key/value slots and refines a frame feature map; temporal memory uses an LSTM query to read attention patterns and aggregate a fixed-length sequence. | Spatial/temporal memory attention is trained with triplet, identity classification, and memory-spread losses. The memory is a learned table, not a causal identity profile updated from one confirmed person. | Useful motivation for separating distractor suppression from temporal aggregation. Not directly reusable: N72R18 has frozen 512-D OSNet vectors, no feature map, no raw crop encoder, and must answer a causal identity-memory question rather than train a video-ReID encoder. |
| KPR | Multiple part embeddings (one vector per learned body part), plus per-part visibility scores; keypoint prompts can identify a target in multi-person ambiguity. | Attention branches pool spatial feature maps into parts; identity and part-triplet losses use visibility. Prompts are optional at inference. | Answers “parts/tokens vs vector”: parts/tokens, not one persistent vector. It motivates reliability as observation quality, but N72R18 has no pose/keypoint or crop re-encoding and therefore keeps one 512-D state plus a scalar gate. |
| M³-ReID | Multi-view spatio-temporal attention over spatial, temporal-width, and temporal-height views; frame-level and video-level representations; visible/infrared modality alignment. | Diverse Attention Constraint separates heads, Orthogonal Frame Regularizer separates frame features, and Multi-Modality Alignment uses retrieval-oriented metric separation. | Useful evidence that temporal aggregation and diversity regularization can matter. Not directly reusable: it expects raw `[B,T,C,H,W]` features, a ResNet/non-local backbone, and cross-modality data; it is not a sparse human-anchor memory updater. |
| CLIP-ReID / OPA067-ReID | One L2-normalized image embedding from a CLIP visual encoder; the online pipeline compares the target vector to candidate crop vectors with cosine similarity. | Image-to-image InfoNCE treats diagonal target/candidate pairs as positives and off-diagonal batch entries as negatives; the repository also contains a standard CLIP encoder path. | The InfoNCE structure is aligned with N72R18 hard-negative training. N72R18 must not fine-tune CLIP or re-encode crops: it uses the inherited frozen 512-D OSNet cache instead. |
| MeMOTR | A long-term memory state and short-term output are carried in tracker queries. The query updater uses a confidence-weight network, short-memory fusion, attention to long memory, and an EMA-like long-memory update. | `is_pos = score > update_threshold`; long memory and last output are updated only for accepted positive tracks. The state is detached before the memory-attention read, and the mechanism is coupled to detection queries and tracking. | Direct motivation for a learned reliability gate and for rejecting unreliable observations. Only the abstraction is transferred; detection confidence, query embeddings, boxes, and MOT association are explicitly out of scope. |

## Repository provenance

| reference | requested / corrected URL | local path | shallow clone commit |
|---|---|---|---|
| STMN | `https://github.com/cvlab-yonsei/STMN` (the requested `chanhopark00/STMN` URL was not a repository) | `/data3/liuyeqiang/research_refs/video_reid/STMN` | `298e2a8eb78f886111e688c8801d727bbf2d7fa5` |
| KPR | `https://github.com/vlsomers/keypoint_promptable_reidentification` | `/data3/liuyeqiang/research_refs/reid/keypoint_promptable_reidentification` | `e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` |
| M³-ReID | `https://github.com/workingcoder/M3-ReID` | `/data3/liuyeqiang/research_refs/reid/M3-ReID` | `9d82aa8c64f0aea3ae7082ffe5940e041e9e6b50` |
| CLIP-ReID / OPA067 | `https://github.com/OPA067/ReID` | `/data3/liuyeqiang/research_refs/reid/ReID` | `8b30ce1c85bb70a6d4e0acb7b6564a4d7e4611a` |
| MeMOTR | `https://github.com/MCG-NJU/MeMOTR` | `/data3/liuyeqiang/research_refs/memory_tracking/MeMOTR` | `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` |

## N72R18 design consequence

The first experiment is deliberately smaller than all five references:

1. Keep the N72R17 OSNet 512-D embeddings and frozen train/validation
   protocol byte-for-byte.
2. Initialize `z_0` with the human anchor, score each future frame before
   consuming that frame's same-identity diagnostic observation, then update
   the state.
3. Compare a GRU update against the fixed EMA(0.90) baseline and add a scalar
   reliability gate to the second learned variant.
4. Train only with hard-negative contrastive loss plus the declared state
   stability penalty. No identity-classification head, new encoder, crop
   generation, SAM3, MOT association, or TrackEval is part of this stage.

The future same-identity observations are ground-truth replay inputs. They are
therefore an offline causal diagnostic, not a claim that a deployable tracker
can identify the correct observation without an association mechanism.
