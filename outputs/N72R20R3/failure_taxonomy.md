# N72R20R3 failure taxonomy

The historical R2 wrong-write definition is preserved: an accepted base assignment is wrong when target-candidate IoU is below 0.50.

Future frames: 8414; P0=154, P1=1665 (P1a=439, P1b=1226), P2=1706, P3=4889.

| Sequence | Future | P0 | P1a | P1b | P2 | P3 | Candidate-set present |
|---|---:|---:|---:|---:|---:|---:|---:|
| dancetrack0001 | 702 | 6 | 25 | 4 | 10 | 657 | 667 |
| dancetrack0002 | 1202 | 6 | 52 | 157 | 716 | 271 | 987 |
| dancetrack0023 | 1482 | 20 | 111 | 243 | 123 | 985 | 1108 |
| dancetrack0024 | 762 | 3 | 47 | 1 | 1 | 710 | 711 |
| dancetrack0039 | 1241 | 41 | 6 | 2 | 11 | 1181 | 1192 |
| dancetrack0057 | 621 | 8 | 19 | 2 | 2 | 590 | 592 |
| dancetrack0062 | 1202 | 0 | 117 | 130 | 677 | 278 | 955 |
| dancetrack0072 | 1202 | 70 | 62 | 687 | 166 | 217 | 383 |

Historical R2 C0 wrong writes: 2000 (the R2 decision remains FAIL_RUNTIME_MEMORY_COMMIT).

| R2 wrong-write category | Count | Fraction |
|---|---:|---:|
| P0_target_absent | 68 | 0.034000 |
| P1_target_candidate_unavailable | 817 | 0.408500 |
| P1a_localization_gray | 439 | 0.219500 |
| P1b_wrong_available_object | 378 | 0.189000 |
| P2_true_candidate_association_error | 1115 | 0.557500 |
| other_insufficient_evidence | 0 | 0.000000 |

P1a is a geometry/localization diagnostic: the base-selected candidate is the best available candidate but remains below the 0.50 target-IoU threshold. P1b records that the base selection is not the best available candidate or is absent. These labels are post-hoc only and never enter runtime decisions.
