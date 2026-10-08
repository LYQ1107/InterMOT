"""Trajectory persistence and evaluator selection integrity."""
from copy import deepcopy
from pathlib import Path
import hashlib
import pytest
from scripts import n72r20r4r1_evaluate as evaluation
from scripts.n72r20r4r1_common import stream_zstd
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r4r1_inner import metric_key


def trace():
    return [{"frame":0,"state_after":"fixture-state","outputs":[{"public_id":1,"candidate_uid":"a","box_xyxy":[0,0,10,10],"confidence":1.},
        {"public_id":2,"candidate_uid":"b","box_xyxy":[20,0,30,10],"confidence":.8}]}]


@pytest.mark.parametrize("changed",[False,True])
def test_ownership_delta_reconstructs_exact_full_mot(tmp_path,monkeypatch,changed):
    base=trace();source=tmp_path/"source.jsonl.zst";artifact=stream_zstd(source,base)
    metadata={"trajectory_sha256":hashlib.sha256(trajectory_text(base).encode()).hexdigest(),"trace_path":str(source),"trace_sha256":artifact["sha256"]}
    monkeypatch.setattr(evaluation,"ASSETS",tmp_path/"assets")
    monkeypatch.setattr(evaluation,"baseline_trace",lambda sequence:(deepcopy(base),metadata))
    treatment=deepcopy(base)
    if changed:
        treatment[0]["outputs"][0]["public_id"]=2;treatment[0]["outputs"][1]["public_id"]=1
    batch=evaluation.EvaluationBatch("fixture",["s"])
    try:manifest=batch.add("v","s",treatment,{"committed_state_stream_sha256":hashlib.sha256(b"fixture-state").hexdigest()})
    finally:batch.close()
    assert evaluation.reconstruct_mot(manifest)==trajectory_text(treatment)


def test_missing_non_target_output_is_not_exported_as_full_tracker(tmp_path,monkeypatch):
    base=trace();metadata={"trajectory_sha256":"x","trace_sha256":"y","trace_path":"not-read"}
    monkeypatch.setattr(evaluation,"ASSETS",tmp_path/"assets")
    monkeypatch.setattr(evaluation,"baseline_trace",lambda sequence:(base,metadata))
    damaged=deepcopy(base);damaged[0]["outputs"].pop()
    batch=evaluation.EvaluationBatch("fixture",["s"])
    try:
        with pytest.raises(RuntimeError,match="coverage changed"):batch.add("v","s",damaged,{})
    finally:batch.close()


def test_inner_deta_guard_has_priority_over_hota_and_zero_write_safety():
    baseline={"HOTA":.5,"AssA":.5,"DetA":.6,"IDSW":10}
    metrics={"base":baseline,"bad":{**baseline,"HOTA":.9,"DetA":.5}}
    empty={"memory":{"safety_pass":False,"wrong_write_rate":None,"correct_write_retention":0},"funnel":{}}
    assert metric_key("base",metrics,baseline,{"base":empty,"bad":empty})>metric_key("bad",metrics,baseline,{"base":empty,"bad":empty})
