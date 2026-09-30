# N72R20R2 contamination cascade analysis

This is a post-hoc join against the sealed R1 train-smoke tape. Runtime
reconstruction used no GT; GT only labels the recorded writes afterward.
The wrong-write criterion is target-candidate IoU < 0.50.

| Sequence | Wrong writes | Rate | First wrong frame | Exact runs | Near runs (gap≤2) |
|---|---:|---:|---:|---|---|
| dancetrack0001 | 5 | 3.145% | 18 | 18-18, 22-22, 31-31, 34-34, 122-122 | 18-18, 22-22, 31-31, 34-34, 122-122 |
| dancetrack0002 | 28 | 17.610% | 5 | 5-5, 10-10, 61-62, 66-66, 68-69, 77-79, 94-94, 97-97, 99-99, 101-101, 107-108, 110-111, 122-122, 124-125, 127-128, 130-130, 132-132, 142-142, 155-155, 158-158 | 5-5, 10-10, 61-62, 66-69, 77-79, 94-94, 97-101, 107-111, 122-132, 142-142, 155-155, 158-158 |

The detailed JSON records learned rank, learned margin, anchor score,
native continuity, predicted motion IoU, state hashes, state drift,
and target-vs-hard-negative margins for every wrong write.

Interpretation: the R1 assignment shadow did not change public
assignments. These are base-assignment observations admitted by a
non-discriminative consensus rule, not errors caused by a learned
assignment change.
