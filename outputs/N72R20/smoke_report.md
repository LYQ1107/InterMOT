# N72R20 Train Smoke Report

FINAL GOAL: **Human Correction Driven Persistent Identity Adaptation**

CENTRAL QUESTION: **When a human corrects a tracking error, can the system learn from this correction and improve future identity tracking?**

This is a train-only runtime smoke for constructing the N72R20 correction
environment, not a learned correction-benefit result.
The checkpoint and loader audit is recorded in
[`checkpoint_loader_smoke.json`](checkpoint_loader_smoke.json). No completed
val or MOT result is part of this report; a superseded val attempt was
interrupted and quarantined after the Goal correction.

| Sequence | Frames written | Candidates | Candidates/frame | Cache bytes | Wall seconds | Approx. FPS |
|---|---:|---:|---:|---:|---:|---:|
| `dancetrack0001` | 160 | 1,193 | 7.456 | 1,254,784 | 79.57 | 2.01 |
| `dancetrack0002` | 160 | 1,436 | 8.975 | 1,515,192 | 108.05 | 1.48 |
| **total** | **320** | **2,629** | **8.216** | **2,769,976** | **187.62** | **1.71** |

Runtime observations:

- GPU: A100 GPU 1; sampled peak device usage was approximately 28,191 MiB.
- Free space after smoke: 172.04 GiB; the >100 GiB normal reserve remained.
- Storage contains only metadata, float16 embeddings, manifests, and logs; no
  crop images or dense masks were saved.
- `runtime_future_gt_used=false` and `runtime_gt_read=false` for both caches.
- Each sequence used one isolated SAM3 process/session. The generated cache
  is outside Git at `/data3/liuyeqiang/InterMOT_N72R20_assets/candidates/`.

The smoke demonstrates SAM3 candidate generation and frozen OSNet embedding
materialization. The follow-up correction-event report uses GT only offline
for error discovery; it does not update memory or measure future correction
benefit.
