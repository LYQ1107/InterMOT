# N72R21 source and license audit (in progress)

Final Goal: **One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings**.

CHIRLA's [official dataset card](https://huggingface.co/datasets/bdager/CHIRLA) states CC-BY-4.0 and requires account access plus contact-sharing agreement for repository content. The public card is not acceptance of those terms. No gated media is requested or copied. The [official ScienceDB DOI](https://doi.org/10.57760/sciencedb.20543) resolves to dataset ID `2247f442a9784b5c959e7bead89c0313`; server access remains unresolved. No unverified mirror is a valid source.

Public [CHIRLA benchmark metadata](https://github.com/bdager/CHIRLA/tree/main/benchmark/metadata) is downloaded selectively and verified against the official Git blob SHA. CSV existence does not prove local media, complete annotations, or permission to bypass gated distribution. Official ReID train_0/val/test_0 and gallery/query roles are kept distinct. The project's reported closed-set ReID metrics do not automatically equal N72R21's open-set target-tracking metrics.

Existing DanceTrack train/val data and OSNet weights are reused in place under their previous recorded lineage for research. No data or weights are committed or republished. This phase has not newly completed the upstream license audit needed for distributing demonstration videos; raw imagery remains local. Existing TAO/LaSOT person folders are derived/sampled sequences, not newly licensed full LaSOT. No sparse annotation gap is treated as true target absence.

The [official OSTrack repository](https://github.com/botaoye/OSTrack) declares MIT for code and links pretrained models. That code license alone is not a completed checkpoint-training-lineage or media license audit. B6 is still pending a verified checkpoint and actual inference; no baseline number is invented.

No real names, face identification, or inferred identity mapping is used. Clicks are explicitly `SIMULATED_ONE_CLICK_FROM_GT`, and sequence-local IDs do not imply the same person across videos. No real-human study or surveillance deployment claim is made.
