FINAL GOAL:
Human Identity Representation Probe → Joint Open-Set Identity-Availability Representation Learning

CENTRAL QUESTION:
“用户点一下这个人以后，我们到底能不能认住他？”

# What did joint open-set training change?

Final decision: `FAIL_P1_LOCALIZATION_QUALITY`.

R3R2 ranking baseline: `{'rows': 8414, 'present_rows': 6595, 'Rank1': 0.6950720242608036, 'Rank2': 0.7840788476118271, 'Rank3': 0.8507960576194087, 'Rank5': 0.9438968915845337, 'MRR': 0.7914611237469463, 'mean_rank': 1.8603487490523123}`.
Best joint selected pipeline: `{'dancetrack0001': 'J4_ENERGY', 'dancetrack0002': 'J5_QUALITY_FUSION', 'dancetrack0023': 'J5_QUALITY_FUSION', 'dancetrack0024': 'J2_JOINT_ABSOLUTE', 'dancetrack0039': 'J2_JOINT_ABSOLUTE', 'dancetrack0057': 'J4_ENERGY', 'dancetrack0062': 'J5_QUALITY_FUSION', 'dancetrack0072': 'J5_QUALITY_FUSION'}`.
Frozen R3R2R1 calibration ceiling: pooled FPR `0.082463`, correct-ID recall `0.227597`, macro FPR/recall `0.067036`/`0.201144`, shadow wrong-write rate `0.204608`.
J6 geometry/motion fusion was skipped: causal target prediction is unavailable for every runtime frame, so partial geometry would violate the no-GT runtime boundary.
Formal selected metrics: `{'rows': 8414, 'present_rows': 6595, 'negative_rows': 1819, 'accepted_rows': 2534, 'correct_id_rows': 1724, 'negative_fpr': 0.06652006597031336, 'correct_id_recall': 0.2614101592115239, 'P0_fpr': 0.14935064935064934, 'P1a_fpr': 0.09415262636273539, 'P1b_fpr': 0.004573170731707317, 'macro_negative_fpr': 0.12889866235902472, 'macro_correct_id_recall': 0.22442792117703725, 'candidate_valid_auroc': 0.7901213783773627, 'candidate_valid_auprc': 0.4443324349813176, 'candidate_recall_at_FPR_2pct': 0.18321942707575978, 'recall_at_FPR_2pct': 0.18832448824867323, 'Rank1': 0.6642911296436694, 'Rank2': 0.7730098559514784, 'Rank3': 0.8447308567096286, 'Rank5': 0.9414708112206217, 'MRR': 0.7735110774155505, 'mean_rank': 1.9131159969673996, 'set_presence_auroc': 0.7818608313143088, 'set_presence_auprc': 0.9200866437321626, 'formal_gate': {'pooled_negative_fpr_le_0.02': False, 'pooled_correct_id_recall_ge_0.60': False, 'macro_fpr_le_0.05': False, 'macro_recall_ge_0.40': False, 'runtime_gt_clean': True, 'no_candidate_creation': True}, 'per_sequence': {'dancetrack0001': {'rows': 702, 'present_rows': 667, 'negative_fpr': 0.05714285714285714, 'correct_id_recall': 0.038980509745127435}, 'dancetrack0002': {'rows': 1202, 'present_rows': 987, 'negative_fpr': 0.0, 'correct_id_recall': 0.0060790273556231}, 'dancetrack0023': {'rows': 1482, 'present_rows': 1108, 'negative_fpr': 0.034759358288770054, 'correct_id_recall': 0.3167870036101083}, 'dancetrack0024': {'rows': 762, 'present_rows': 711, 'negative_fpr': 0.0, 'correct_id_recall': 0.015471167369901548}, 'dancetrack0039': {'rows': 1241, 'present_rows': 1192, 'negative_fpr': 0.46938775510204084, 'correct_id_recall': 0.7416107382550335}, 'dancetrack0057': {'rows': 621, 'present_rows': 592, 'negative_fpr': 0.2413793103448276, 'correct_id_recall': 0.15202702702702703}, 'dancetrack0062': {'rows': 1202, 'present_rows': 955, 'negative_fpr': 0.19433198380566802, 'correct_id_recall': 0.2712041884816754}, 'dancetrack0072': {'rows': 1202, 'present_rows': 383, 'negative_fpr': 0.03418803418803419, 'correct_id_recall': 0.25326370757180156}}}`.
Counterfactual episodes: `121845`; PRESENT/P0/P1a/P1b = `{'PRESENT': 40662, 'P1a': 7840, 'P0': 1809, 'P1b': 4151}`.
Shadow causal wrong-write rate: `0.2530228392297358`; retention: `0.7469771607702642`.

## Lineage and runtime boundary

Source R3R2R1 HEAD: `7820d312e2b3c852a7909a4d42dc71bad0c085c6`. Eight outer sequence folds were run; held-out observations contributed zero training, threshold, or model-selection supervision.
Candidate generation, OSNet, N72R18 state, exact solver, public-ID authority, SAM3, DanceTrack VAL/TEST and association authority remained frozen.
GT was used only for training/post-hoc labels and diagnostics; runtime features contain no GT fields.

## Scientific interpretation

Bottleneck: `BOTTLENECK_P1_LOCALIZATION_QUALITY`. Static gate: `{'pooled_negative_fpr_le_0.02': False, 'pooled_correct_id_recall_ge_0.60': False, 'macro_fpr_le_0.05': False, 'macro_recall_ge_0.40': False, 'runtime_gt_clean': True, 'no_candidate_creation': True}`. Causal gate: `{'static_gate_pass': False, 'wrong_write_rate_le_0.02': False, 'retention_ge_0.60': True, 'pass': False}`.
Oracle ladder and headroom are diagnostic only; they do not authorize runtime association.

## Next-stage authorization

next_association_stage_authorized=False
next_candidate_quality_stage_authorized=True
next_memory_state_learning_stage_authorized=False

No automatic downstream association, LoRA, SAM3 rerun, solver redesign or VAL/TEST evaluation was started.
Test summary: Focused R3R2R2 invariants: 50 passed; full pytest: 542 passed, 4 failed; default run n6 x2 could not resolve executable python, and explicit-venv rerun reached the pinned TrackEval SEQMAP_FILE list-vs-string TypeError in n6/n7/n8.
