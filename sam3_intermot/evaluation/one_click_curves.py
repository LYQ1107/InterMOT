"""Equivalent tied-score open-set summaries in O(n log n), not O(n²)."""
import numpy as np


def open_set_metrics_fast(probability,available,ranking_correct,*,bins=10):
    p=np.asarray(probability,float);y=np.asarray(available,bool);correct=np.asarray(ranking_correct,bool)
    if p.ndim!=1 or p.shape!=y.shape or y.shape!=correct.shape:raise ValueError('metric array shape mismatch')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid availability probability')
    positive=int(y.sum());negative=int((~y).sum());recall2=None;ap=None
    if len(p):
        order=np.argsort(-p,kind='stable');scores=p[order];labels=y[order]
        ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(p)-1]
        tp=np.cumsum(labels)[ends];fp=(ends+1)-tp
        identity=np.cumsum(labels&correct[order])[ends]
        if positive:
            recall=tp/positive;ap=float(np.sum(np.diff(np.r_[0.,recall])*(tp/(ends+1))))
            if negative:
                legal=fp/negative<=.02;recall2=max(0.,float((identity[legal]/positive).max())) if legal.any() else 0.
    ece=0.;records=[]
    for i in range(bins):
        mask=(p>=i/bins)&((p<(i+1)/bins) if i<bins-1 else p<=1)
        if mask.any():
            confidence=float(p[mask].mean());rate=float(y[mask].mean());ece+=mask.mean()*abs(confidence-rate)
            records.append({'bin':i,'n':int(mask.sum()),'confidence':confidence,'positive_rate':rate})
    return {'positive_frames':positive,'negative_frames':negative,'recall_at_fpr_2pct':recall2,
        'PR_AUC_average_precision':ap,'ECE':float(ece) if len(p) else None,'calibration_bins':records,
        'diagnostic_curve_not_deployment_threshold_tuning':True}
