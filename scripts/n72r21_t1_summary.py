"""Paired state-condition diagnostic, complete cohort/seed status explicit."""
import json
import numpy as np
from scripts.n72r21_common import OUT,read_json,write_json,sha256
from scripts.n72r21_baseline_summary import summarize
from scripts.n72r21_t1_replay import COMPARISONS


def run():
    protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json');expected=protocol['sequences'];seeds=[72101,72102,72103];results={}
    for comparison in COMPARISONS:
        per_seed={}
        for seed in seeds:
            case=f'T1_{comparison}_K8_SEED{seed}';records={};sources={}
            for sequence in expected:
                path=OUT/'experiments/T1/evaluations'/case/f'{sequence}.json'
                if not path.exists():continue
                # Legacy summary needs an initialization_failure field. Every
                # represented episode has a sealed successful real click;
                # no failed initialization is dropped from these run records.
                records[sequence]=[{**e,'initialization_failure':False} for e in read_json(path)['episodes']];sources[sequence]=sha256(path)
            if records:
                per_seed[str(seed)]={'per_sequence':{s:summarize(e) for s,e in records.items()},
                    'pooled':summarize([e for episodes in records.values() for e in episodes]),'source_evaluation_sha256':sources,
                    'cohort_complete':set(records)==set(expected)}
        if not per_seed:continue
        complete_sequences=[s for s in expected if all(str(seed) in per_seed and s in per_seed[str(seed)]['per_sequence'] for seed in seeds)]
        values=np.array([np.mean([per_seed[str(seed)]['per_sequence'][s]['target_recall_all_visible'] for seed in seeds]) for s in complete_sequences])
        ci=None
        if len(values)>1:
            rng=np.random.default_rng(protocol['bootstrap']['seed']);draw=rng.integers(0,len(values),(protocol['bootstrap']['replicates'],len(values)))
            ci=np.quantile(values[draw].mean(1),[.025,.975]).tolist()
        results[comparison]={'per_seed':per_seed,'sequences_with_all_three_seeds':complete_sequences,
            'seed_mean_sequence_macro_recall':float(values.mean()) if len(values) else None,'sequence_cluster_95pct_CI':ci,
            'complete_registered_cohort':len(complete_sequences)==len(expected),'not_frame_IID':True,
            'mismatched_control_not_selected_as_deployed_method':comparison=='P1_TO_P0'}
    write_json('experiments/T1/DEVELOPMENT_SUMMARY.json',{'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','state_comparisons':results,
        'scope':'Historically exposed F1 TRAIN scenes, condition-by-seed completion explicit','independent_final_validation':False,
        'T2_T3_complete':False,'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps({k:{'three_seed_sequences':len(r['sequences_with_all_three_seeds']),'macro_recall':r['seed_mean_sequence_macro_recall']} for k,r in results.items()}))


if __name__=='__main__':run()
