import numpy as np
import pytest
from sam3_intermot.evaluation.one_click_curves import open_set_metrics_fast
from sam3_intermot.evaluation.one_click_protocol import open_set_metrics


@pytest.mark.parametrize('mode',['mixed','positive','negative','empty','ties'])
def test_fast_tied_score_curve_agrees_with_original_endpoint(mode):
    rng=np.random.default_rng(72104);p=rng.random(117);y=rng.integers(0,2,117).astype(bool);correct=rng.integers(0,2,117).astype(bool)
    if mode=='positive':y[:]=True
    if mode=='negative':y[:]=False
    if mode=='empty':p,y,correct=p[:0],y[:0],correct[:0]
    if mode=='ties':p=np.round(p,1)
    expected=open_set_metrics(p,y,correct);actual=open_set_metrics_fast(p,y,correct)
    for key in ('positive_frames','negative_frames','recall_at_fpr_2pct','PR_AUC_average_precision','ECE'):
        if expected[key] is None:assert actual[key] is None
        else:assert actual[key]==pytest.approx(expected[key],abs=1e-14)
    assert actual['calibration_bins']==expected['calibration_bins']
