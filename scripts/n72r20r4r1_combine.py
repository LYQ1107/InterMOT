"""Official combined DEV metrics and paired sequence-cluster uncertainty."""
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_evaluate import EvaluationBatch


def paired_bootstrap(metrics):
    indices=np.random.default_rng(720401).integers(0,len(SEQUENCES),size=(2000,len(SEQUENCES)))
    result={}
    for name,m in metrics.items():
        result[name]={}
        for key in ('HOTA','AssA','DetA','IDF1','IDSW'):
            differences=np.asarray([m['per_sequence'][s][key]-metrics['BASELINE']['per_sequence'][s][key] for s in SEQUENCES])
            samples=differences[indices].mean(axis=1)
            result[name][key]={'macro_delta':float(differences.mean()),'CI95':np.quantile(samples,[.025,.975]).tolist(),
                'official_combined_delta':m[key]-metrics['BASELINE'][key],'per_sequence_delta':dict(zip(SEQUENCES,differences.tolist()))}
    return {'seed':720401,'draws':2000,'cluster':'sequence','paired':True,'estimator':'sequence-macro delta, not a CI of the official combined nonlinear HOTA','variants':result}


def run():
    done=OUT/'dev_trackeval/ALL_VARIANTS.json'
    if done.exists():return read_json(done)
    sources={s:read_json(OUT/'evaluations/outer'/f'{s}.json') for s in SEQUENCES}
    names=list(next(iter(sources.values()))['metrics'])
    if any(set(r['metrics'])!=set(names) for r in sources.values()):raise ValueError('formal variant axes differ across outer folds')
    batch=EvaluationBatch('combined_DEV',SEQUENCES)
    try:
        for name in names:
            for s in SEQUENCES:batch.add_existing(name,s,sources[s]['manifests'][name][s])
        result=batch.evaluate()
    finally:batch.close()
    metrics=result['metrics'];source=read_json(OUT/'audit/M0_TRACK_EVAL.json')['metrics']
    if any(metrics['BASELINE'][k]!=source[k] for k in ('HOTA','AssA','DetA','LocA','IDF1','MOTA','IDSW','FP','FN')):raise RuntimeError('combined baseline does not reproduce M0')
    bootstrap=paired_bootstrap(metrics);write_json(OUT/'dev_trackeval/PAIRED_BOOTSTRAP.json',bootstrap)
    write_json(done,{'metrics':metrics,'source_outer_evaluations':{s:sha256(OUT/'evaluations/outer'/f'{s}.json') for s in SEQUENCES},'official_combined_not_mean_of_fold_HOTA':True})
    write_json(OUT/'dev_trackeval/BASELINE.json',metrics['BASELINE'])
    write_json(OUT/'dev_trackeval/PER_SEQUENCE.json',{n:r['per_sequence'] for n,r in metrics.items()})
    final=metrics['FINAL_SELECTED'];base=metrics['BASELINE'];ci=bootstrap['variants']['FINAL_SELECTED']['HOTA']['CI95']
    gate={'delta_HOTA':final['HOTA']-base['HOTA'],'delta_AssA':final['AssA']-base['AssA'],'delta_DetA':final['DetA']-base['DetA'],'macro_HOTA_CI95':ci}
    gate['numerical_DEVELOPMENT_gate_pass']=gate['delta_HOTA']>=.005 and gate['delta_AssA']>0 and gate['delta_DetA']>=-.005
    gate['CI_lower_gt_zero']=ci[0]>0
    write_json(OUT/'dev_trackeval/FINAL_SELECTED.json',{'metrics':final,'baseline':base,'gate':gate,'frozen_final_policies':{s:sha256(OUT/'authority/frozen_final'/f'{s}.json') for s in SEQUENCES},
        'scientific_PASS_requires_opportunity_causal_and_memory_audits_also':True})
    return metrics


if __name__=='__main__':torch.set_num_threads(1);run()
