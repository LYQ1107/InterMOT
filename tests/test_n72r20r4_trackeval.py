from copy import deepcopy
import pytest
from scripts.n72r20r4_trackeval import paired_metrics, METRICS
from scripts.n72r20r4_run_loso import choose_inner, configurations
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from sam3_intermot.association.identity_authority import AuthorityConfig


def metric(hota=0.5,deta=0.6):
    result={k:0.5 for k in METRICS}
    result.update(HOTA=hota,DetA=deta,IDSW=3,FP=20,FN=30)
    result["per_sequence"]={"a":{k:v for k,v in result.items()},"b":{k:v for k,v in result.items()}}
    return result


def test_paired_cluster_bootstrap_reports_own_estimator():
    b=metric();t=metric(0.52)
    r=paired_metrics(b,t,samples=200)
    assert r["combined_delta"]["HOTA"] == pytest.approx(0.02)
    assert r["bootstrap"]["HOTA"]["interval_95"] == pytest.approx([0.02,0.02])
    assert r["HOTA_CI_lower_gt_zero"]
    t["per_sequence"].pop("a")
    with pytest.raises(ValueError):paired_metrics(b,t)


def test_inner_selection_deta_constraint_and_g0_fallback():
    configs=configurations(AuthorityConfig().to_dict())
    m={"G0":metric(0.5),"G1_FIXED_1":metric(0.8,0.59),"G1_FIXED_0.1":metric(0.499)}
    assert choose_inner(m,configs)["name"]=="G0"
    m["G1_FIXED_0.1"]=metric(0.51)
    assert choose_inner(m,configs)["name"]=="G1_FIXED_0.1"


def test_trajectory_export_is_prevalidated_and_deterministic():
    row={"frame":0,"outputs":[{"public_id":100001,"candidate_uid":"a","box_xyxy":[1,2,11,22],"confidence":0.9}]}
    result=trajectory_text([row])
    assert result=="1,100001,1.0000,2.0000,10.0000,20.0000,0.900000,-1,-1,-1\n"
    corrupt=deepcopy(row);corrupt["outputs"].append(corrupt["outputs"][0])
    with pytest.raises(RuntimeError,match="duplicate"):trajectory_text([corrupt])
