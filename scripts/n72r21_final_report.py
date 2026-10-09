"""Render the completed negative/limited-science delivery from verified tables."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from scripts.n72r21_common import ROOT,OUT,read_json,sha256,utcnow


def number(v,percent=False):
    if v is None: return 'NA'
    return f'{100*v:.2f}%' if percent else f'{v:.4f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])+'\n'


def target_rows(records,field):
    rows=[]
    for case,r in records.items():
        p=r[field]
        rows.append([case,number(p['target_recall_all_visible'],True),number(p.get('strict_UID_identity_recall_all_visible'),True),
                     p['verified_wrong_person_takeover_frames'],p['verified_wrong_person_takeover_episodes'],
                     number(p['reacquisition_recall'],True),number(p.get('reacquired_only_median_delay_seconds'))])
    return rows


def run():
    result=read_json(OUT/'FINAL_RESULT.json');tables=read_json(OUT/'evaluation/TABLES_A_TO_E.json')['tables']
    if result['complete_frozen_VAL_scene_cases']!=375 or result['Goal_completion_proven_by_this_result']:
        raise ValueError('completed scientific evidence distinct from Goal closure required')
    tests=read_json(OUT/'tests/REGRESSION_CURRENT.json');history=read_json(OUT/'historical_audit/HISTORY_DELIVERY_CHECK.json')
    focused=list(ET.parse(tests['focused_actual_XML_path']).getroot().iter('testcase'))
    focused_pass=sum(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in focused)
    storage=read_json(OUT/'storage/STORAGE_AFTER.json');videos=read_json(OUT/'visualizations/DELIVERY_VIDEO_RECEIPT.json')
    if not storage['final_snapshot'] or not history['all_history_and_models_unchanged']: raise ValueError('actual final receipts required')
    a,b,c,d,e=(tables[f'Table_{s}'] for s in 'ABCDE')
    full=[case for case in a['independent_sequence_frozen25_VAL_identity_subsystem'] if case.startswith('ACIB_FULL_SEED')]
    gate=result['gates'];families=a['VAL_families_and_sequence_cluster_CI']
    text=[
        '# N72R21 final evidence report — MOT primary, SOT deferred',
        '',f'Generated UTC: {utcnow()}. Scientific decision: **{result["scientific_decision"]}**. Next stage authorized: **false**.',
        '', '## Frozen Goal and latest direction', '',
        'FINAL GOAL: One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings',
        '', 'CENTRAL QUESTION: “用户只需要点击一个人一次，在其离开画面、重新出现、换衣服、跨摄像头或跨独立录制时，系统能否可靠判断他是否出现、在哪里，并持续认住他，而不把其他人误认为他？”',
        '', 'The latest explicit user instruction makes interactive **MOT the primary track** and defers SOT/LaSOT. The original Goal is retained unchanged for provenance; `RESEARCH_DIRECTION_OVERRIDE.json` takes precedence. No new SOT downloads, training, inference or OOM retries were started after the direction change.',
        '', '## Outcome and scientific limits', '',
        'The current frozen learned system has **not established reliable persistent identity or a beneficial MOT intervention**. On two preregistered TRAIN sequences, all 12 learned full-MOT conditions have lower HOTA, AssA and IDF1 and higher IDSW than clicked C0. All 16 conditions have zero registered strict H100 future benefit. These are real joint multi-object trajectories, not independently predicted target episodes merged into a tracker.',
        '', 'The completed 25-sequence frozen VAL is a **single-click identity-subsystem evaluation on shared cached SAM3 candidate tapes**, not full MOT, an end-to-end online SAM3 causal/latency test, a virgin benchmark, or proof of globally unseen persons. No VAL threshold, model, checkpoint, memory or candidate-policy selection was performed. The predetermined TRAIN outer0001 model configuration was used; its FIT/INNER contained no VAL sequences.',
        '', 'No locally authorized CHIRLA video/dense annotation exists. Official metadata and identity/protocol membership were audited, but cross-camera/day/session metrics remain **NA, zero executed episodes**. Within-clip seconds and visible-GT gaps are not days, independent sessions, physical absence or verified occlusion. The failure is scoped to this frozen system and protocol, not a universal impossibility theorem about human identity representations.',
        '', 'The scientific result does not alone mark the app Goal complete; semantic delivery and exact clean Git publication are separately verified.',
        '', '## Quantitative frozen gates', '',
        table(['Gate','Status','Interpretation'],[
            ['0',gate['Gate0']['status'],'GT-free current/past replay checked; upstream cached-pixel causality/latency not proven'],
            ['1',gate['Gate1']['status'],'Frozen independent-sequence effect, safety and usefulness checks'],
            ['2',gate['Gate2']['status'],'Correct identity Recall at primary target-unavailable FPR≤2%, required ≥60%'],
            ['3',gate['Gate3']['status'],'No actual cross-recording media/episodes; no PASS'],
            ['4',gate['Gate4']['status'],'Wrong/unverified writes≤2% AND correct retention≥60%; zero-write is not PASS'],
        ]),
        'Gate thresholds come from the preexisting `F1_WITHIN_VIDEO_DEVELOPMENT.json`, referenced by the frozen VAL protocol. They were not adjusted after results. Detailed boolean checks and paired sequence-cluster intervals are in `FINAL_RESULT.json`.',
        '', '## Table A — One-click target identity (not full MOT)', '',
        'Panel A1: all 15 registered conditions on all 25 VAL sequences, 273 simulated sole clicks and 245,018 future target-frame decisions per condition. Box recall is IoU≥0.5; strict UID recognition is separately reported because an overlapping wrong candidate can satisfy box overlap. Takeover means a verified other identity, not an unmatched UNKNOWN.',
        '',table(['Condition','Box recall','Strict UID recall','Wrong-ID frames','Takeover runs','Reacq recall','Recovered-only median delay (s)'],
                 target_rows(a['independent_sequence_frozen25_VAL_identity_subsystem'],'pooled_fixed_operating_point')),
        'Failed return windows remain in reacquisition recall; conditional median delay includes recovered returns only. It is not an unconditional recovery estimate.',
        '', 'Panel A2: all nine historically exposed eight-TRAIN comparator cohorts. These are development diagnostics with their frozen historical operating points, not independent VAL; do not transplant them into Panel A1. Historical R3R2/Existing Memory comparisons were not registered on this VAL.',
        '',table(['Condition','Box recall','Strict UID recall (if evaluated)','Wrong-ID frames','Takeover runs','Reacq recall','Median delay (if evaluated)'],
                 target_rows(a['F1_historically_exposed_eight_TRAIN_baselines'],'pooled')),
        '', 'Panel A3: seed-average inside each sequence, then sequence macro and clustered 95% CI. All 25 clusters; bootstrap 2,000, fixed seed72104. Seeds and frames are not independent bootstrap units.',
        '',table(['Family','Box recall macro','95% CI','Strict UID macro','Deployed unavailable-target FPR'],[
            [name,number(r['macro_metrics']['target_recall_all_visible']['sequence_macro_mean'],True),
             str(r['macro_metrics']['target_recall_all_visible']['sequence_cluster_95pct_CI']),
             number(r['macro_metrics']['strict_UID_identity_recall_all_visible']['sequence_macro_mean'],True),
             number(r['macro_metrics']['deployed_negative_FPR']['sequence_macro_mean'],True)] for name,r in families.items()]),
        '', '## Table B — Open-set recognition and distinct calibration', '',
        'These are exact pooled tied-score diagnostic curves over the complete cohort, not average per-sequence AP, and never new deployment thresholds. Primary positives mean a strictly matched target candidate exists; primary Recall@FPR2 counts correct ranking among those positives. Primary availability AP/ECE does **not** measure correct-identity claim calibration. Secondary joint-identity calibration excludes only unmatched UNKNOWN rank1 candidates, reports exclusions, and retains structural empty sets as verified negatives.',
        '',table(['VAL condition','Primary correct Recall@FPR2','Availability AP','Availability ECE','Verified identity AP','Verified identity ECE','UNKNOWN excluded'],[
            [case,number(r['primary_candidate_availability']['recall_at_fpr_2pct'],True),
             number(r['primary_candidate_availability']['PR_AUC_average_precision']),number(r['primary_candidate_availability']['ECE']),
             number(r['secondary_verified_identity_claim']['PR_AUC_average_precision']),number(r['secondary_verified_identity_claim']['ECE']),
             r['secondary_UNKNOWN_excluded_frames']] for case,r in b['frozen25_VAL_actual_pooled_curves_all15_cases'].items()]),
        '', 'Exact pooled T2 principal-control curves, all fixed operating-point false-presence counts/rates and UNKNOWN denominators are available in the machine-readable Table B/A and `evaluation/OPEN_SET_METRICS.json`.',
        '', '## Table C — Recovery, density and long-term scope', '',
        table(['Scenario / full seed','Strict UID recall','Return windows','Strict recovered','Failed returns','First accept wrong windows','Takeover seconds'],[
            [case,number(c['within_video_VAL']['cases'][case]['strata']['ALL']['strict_UID_target_recall'],True),
             c['within_video_VAL']['cases'][case]['strict_recovery_and_takeover']['all_return_windows']['valid_return_windows'],
             c['within_video_VAL']['cases'][case]['strict_recovery_and_takeover']['all_return_windows']['strict_identity_recovered_windows'],
             c['within_video_VAL']['cases'][case]['strict_recovery_and_takeover']['all_return_windows']['failed_recovery_windows'],
             c['within_video_VAL']['cases'][case]['strict_recovery_and_takeover']['all_return_windows']['first_accept_verified_wrong_windows'],
             number(c['within_video_VAL']['cases'][case]['strict_recovery_and_takeover']['takeover_total_seconds'])] for case in full]),
        '',table(['Cross-recording scenario','Target recall','ReID','Takeover','Valid episodes'],[
            [r['scenario'],'NA','NA','NA',r['valid_episodes']] for r in c['cross_recording']]),
        '', 'Every compared condition, including failures, has all-visible versus candidate-conditional recall, unavailable-target/NONE errors, actual ≤5 / >5–20 / >20-second within-recording bins, sparse0–4 / medium5–8 / crowded≥9 current-candidate bins, raw-anchor hard-negative strata and GT overlap proxies. Full breakdowns have source seals, complete-frame A/A checks and strict wrong-identity takeover duration/recovery. GT overlap is not physical occlusion truth. `LOW_DENSITY.json`, `HIGH_DENSITY.json`, `REAPPEARANCE.json` and machine-readable Table C retain the complete cohorts; no easy-scene selection.',
        '', '## Table D — Memory safety AND usefulness', '',
        'All three seeds of all 11 T2 controls are shown. These are frozen inference capacity/module state-shift diagnostics, not separately retrained architectures. Correct retention is strictly correct accepted writes divided by available target observations. Wrong means verified other OR unmatched unverified write (upper bound), with the categories separate in frame-count breakdowns.',
        '',table(['T2 control / seed','Writes','Wrong/unverified','Wrong rate','Correct retention','Box recall','Reacq'],[
            [f'{name}/{seed}',p['writes'],p['wrong_writes'],number(p['wrong_write_rate'],True),number(p['correct_observation_retention'],True),
             number(p['target_recall_all_visible'],True),number(p['reacquisition_recall'],True)]
            for name,r in d['T2_all11_controls_three_seeds'].items() for seed,p in r['per_seed_pooled'].items()]),
        '', 'FULL_K8 TRAIN seed1/2 write nothing; seed3 writes185, with101 wrong/unverified (54.59%). A null zero-write wrong rate is not 0%; zero retention fails usefulness. The stricter two-scene full-MOT pilot has zero FULL writes in all three seeds and FULL/P0 byte-identical trajectories. These scoped facts must not be generalized into “all T2/VAL seeds never write.” Actual VAL writes:',
        '',table(['VAL condition','Writes','Wrong/unverified','Wrong rate','Correct retention','Joint gate'],[
            [case,p['writes'],p['wrong_writes'],number(p['wrong_write_rate'],True),number(p['correct_observation_retention'],True),p['memory_joint_gate']]
            for case,p in d['VAL_all15_actual_write_policy_results'].items()]),
        '', 'Frozen, unsafe single-positive, consensus P4/P6, delayed/no-delay, current-safe/no-future and learned reliability policies are all retained in Table D JSON. Removing modules does not by itself establish their retrained causal effect.',
        '', '## Table E — Primary MOT transfer diagnostic', '',
        'Same frozen GT-free candidate input, joint one-to-one assignments, native public identity states and unchanged frozen C0 association path. All32 real complete rollouts, two TRAIN sequences, all16 cases. Official pinned TrackEval HOTA/AssA/IDF1 shown on0–100 scale; IDSW is an actual count. No formal generalization CI is claimed from two scenes.',
        '',table(['MOT condition','HOTA','AssA','IDF1','IDSW'],[
            [case,number(100*r['HOTA']),number(100*r['AssA']),number(100*r['IDF1']),int(r['IDSW'])]
            for case,r in e['official_metrics_all16_cases'].items()]),
        '', 'The model learned target-only candidate states, then was evaluated in actual joint MOT state: this distribution shift is explicit. Same numeric detection multisets are verified per frame, but ID-conditioned HOTA/CLEAR matching can change DetA/LocA/FP/FN; these metrics are not asserted equal. The original erroneous detection-metric equality assertion failed after successful TrackEval. Its exact source, original outputs and failure record remain archived. Versioned evaluator R1 checks exact Decimal detection multisets; no runtime, model, source candidates or trajectories were rerun/altered.',
        '', 'SOT is **DEFERRED_BY_USER**, not a further delivery prerequisite. Prior auxiliary OSTrack/LaSOT trim results and every failed SAM3 SOT resource attempt remain preserved. They are neither complete current identity evaluation nor MOT success; no new SOT continuation is authorized.',
        '', '## Failure branches and lawful resume boundaries', '',
        table(['Branch','Actual evidence / disposition','Safe next branch (not started)'],[
            ['F1 data access','Official CHIRLA metadata obtained; gated HF/contact terms and ScienceDB access did not yield lawful local media. Existing within-video experiments completed; SOT expansion now deferred.','Resume official scoped media access after user handles provider terms; no cookies/tokens requested, no bypass'],
            ['F2 cross-recording labels','Official global identity and membership metadata checked; Tracking/ReID path-prefix overlap and train0 single-camera limits disclosed; zero actual cross-recording video episodes.','Use one official lawful protocol with actual media, dense annotations and frame/PTS metadata; no guessed IDs or lineage mixing'],
            ['F3 candidate missing','All-visible/conditional recall, visible-without-candidate and GT-gap counts reported separately across all25 VAL scenes.','Future TRAIN-only candidate-coverage diagnosis, not identity-controller expansion or heldout tuning'],
            ['F4 encoder confusion','Four frozen representations on actual same competitive candidate crops; second ReID minus OSNet CI includes zero; general features worse; no best-backbone selection.','Research representation failure only with a new preregistered MOT-compatible TRAIN plan'],
            ['F5 open-set','All-future target-unavailable negatives; exact primary and secondary pooled curves/calibration plus UNKNOWN and takeover duration.','No posthoc VAL threshold selection; do not call Rank1 improvement reliable identity'],
            ['F6 contamination','Actual unsafe/control/delayed/anchor conditions and paired own-state training; conservative safe memory fails joint retention/safety.','Immutable-anchor fallback evidence retained; no new policy/integration auto-start'],
            ['F7 cross-day','Not evaluable without independent-recording media; within-video long bins are not appearance-change proof.','Official identity/FPS/recording metadata first, no stitched-day fiction'],
            ['F8 generalization','Frozen25 VAL completed without selection, contrasted with real FIT/INNER and exposed TRAIN; full-MOT TRAIN transfer worsens all12 learned cases.','Inspect coverage/state/scene gap on TRAIN; do not fit VAL or infer all causes from two MOT scenes'],
            ['F9 storage',f'Personal mount retains {storage["free_gib"]:.2f}GiB above60GiB reserve; no material deletion this delivery.','No heavy/duplicate downloads or environment; retain unique historical assets'],
        ]),
        '', '## Reproducibility, repairs and resource receipts', '',
        'Actual124 fits: original archivedT0 four, numerical-repairedT0_AMP_R1 twenty-four, T1_CAUSAL_V1 seventy-two, T2_COUPLED_V1 twenty-four. The120 repaired/causal fits are eligible; original four are retained, not mixed into repaired results. Actual248 best/latest files have SHA, schema, finite tensors, strict architecture loads and CPU probability-contract tests. Best is selected by TRAIN INNER, latest is resume state. Loader checks are engineering, never identity performance. All weights stay local, not Git.',
        '', 'Original T0 FP16 scale/gradient failure is retained. The repaired trainer is versioned; the exact original trainer is proven by Git095c5b8 and its frozen SHA. The first catalog preflight incorrectly required that archived trainer equal the repaired current source; that catalog-only error and its repair are recorded. No original weight, trainer record, runtime or sealed schema was silently rewritten.',
        '', f'Latest actual complete repository regression: **{tests["passed"]} passed, {tests["failed"]} failed**, {tests["skipped"]} skipped. Four failures are the pinned old TrackEval CLI SEQMAP_FILE list-versus-path interface; one asserts a historical literal branch name. Focused N72R21:{focused_pass} passed. The first delivery run omitted per-process venv PATH; three bare-python legacy subprocesses failed to launch. Its XML/diagnostic are preserved; the corrected R2 process PATH restores the actual four-CLI/one-branch classification. No test/third-party source was patched to hide failures.',
        '', f'Final history receipt: {history["metadata_and_historical_code_checked"]} historical metadata/code files and {history["historical_model_count"]} models unchanged. All{videos["actual_video_count"]} local licensed real examples decode with correct frame counts/FPS. Four MOT clips were visually inspected; both recovery and failure retained. GT is posthoc overlay only, sole-click inset is not a second click, cross-recording examples are unavailable. No raw pixels, faces, features, candidate caches, datasets or checkpoints are published to Git.',
        '', 'No other-user process was terminated. Existing environment/data reused; no new training, candidate inference, Hungarian redesign, LoRA or downstream association stage was launched during final evidence assembly.',
        '', '## Commands and source evidence', '',
        'Use the existing environment, repository PYTHONPATH and single-thread CPU limits. These commands reproduce posthoc summaries/audits only; they do not authorize additional experiments:',
        '', '```bash', 'cd /data3/liuyeqiang/InterMOT','source .venv/bin/activate','export PYTHONPATH="$PWD"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'python scripts/n72r21_validation_summary.py',
        'python scripts/n72r21_validation_summary.py --pooled-curves',
        'python scripts/n72r21_failure_breakdowns.py --scope VAL',
        'python scripts/n72r21_pooled_t2_curves.py',
        'python scripts/n72r21_final_evidence.py',
        'python scripts/n72r21_delivery_receipts.py --final',
        'python scripts/n72r21_final_report.py',
        'python -m pytest -q tests/test_n72r21*.py', '```',
        '', '`FINAL_RESULT.json`, `evaluation/TABLES_A_TO_E.json`, frozen protocols, runtime/evaluation seals, checkpoint manifest, failure analyses, JUnit XML and history/storage/video receipts carry the complete machine-readable evidence and source SHA. `docs/N72R21_MOT_TRAIN_PILOT.md` retains the focused MOT report.',
        '', 'Publication state is verified separately by the delivery audit and Git receipt; no report text can make an unverified push true. **NEXT_ASSOCIATION_STAGE_AUTHORIZED=false.** Stop at this scoped result; no autonomous next stage.',
    ]
    report='\n'.join(text)+'\n'
    for path in (OUT/'FINAL_REPORT.md',ROOT/'docs/N72R21_FINAL_REPORT.md'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(report)
    print(json.dumps({'report_SHA256':sha256(OUT/'FINAL_REPORT.md'),'mirrored_docs_SHA256':sha256(ROOT/'docs/N72R21_FINAL_REPORT.md'),
                      'scientific_decision':result['scientific_decision'],'full_VAL':375,'new_training':False}))


if __name__=='__main__':run()
