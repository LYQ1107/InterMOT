"""Read-only comparison: identity-score changes are not trajectory authority."""
from pathlib import Path
import json
from scripts.n72r21_common import OUT,read_json,write_json,sha256
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl


def run():
    cases=('B2_R4_CAUSAL_P1','B3_R4R1_NATIVE','B4_P0','B4_P1','B4_P4','B4_P6')
    expected=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'];result={}
    for sequence in expected:
        reference=OUT/'baselines/runtime_seals/B5_ONLINE_NO_LTM'/f'{sequence}.json'
        paths=[OUT/'baselines/runtime_seals'/c/f'{sequence}.json' for c in cases]
        if not reference.exists() or not all(p.exists() for p in paths):continue
        ref=read_json(reference)
        baseline={r['episode_uid']:r for r in ref['artifacts']};parts={}
        for case,path in zip(cases,paths):
            count=changed=score_frames=score_changed=0
            for record in read_json(path)['artifacts']:
                old=baseline[record['episode_uid']]
                if sha256(record['path'])!=record['sha256'] or sha256(old['path'])!=old['sha256']:raise ValueError('runtime seal')
                actual=read_zstd_jsonl(Path(record['path']));original=read_zstd_jsonl(Path(old['path']))
                if [r['frame'] for r in actual]!=[r['frame'] for r in original]:raise ValueError('comparison axis')
                count+=len(actual);changed+=sum(a['selected_candidate_uid']!=b['selected_candidate_uid'] for a,b in zip(actual,original))
                score_frames+=sum(a['identity_score'] is not None for a in actual)
            parts[case]={'frames':count,'target_UID_decisions_changed_vs_existing_tracker':changed,
                         'identity_evidence_frames':score_frames,'complete_target_decision_equivalence':changed==0,
                         'identity_scores_are_not_a_substitute_for_actual_decision_changes':True}
        result[sequence]=parts
    write_json('baselines/AUTHORITY_VS_SCORE_AUDIT.json',{'status':'REAL_SEALED_DECISION_COMPARISON','sequences':result,
        'complete_registered_cohort':len(result)==len(expected),'reference':'B5_ONLINE_NO_LTM',
        'no_GT_needed_for_equivalence_test':True,'no_baseline_or_historical_result_modified':True,
        'not_a_claim_that_memory_improves_tracking':True})
    print(json.dumps({'compared_sequences':len(result)}))


if __name__=='__main__':run()
