"""Prove the explicit fit/deployment switch does not alter frozen inference."""
import subprocess
import torch
import pytest
from sam3_intermot.association.opportunity_models import ActionValueModel
from scripts.n72r20r4r1_common import ROOT


@pytest.mark.parametrize("family",["C2","C3","C4","C5","C6"])
def test_default_inference_exactly_matches_pre_inner_frozen_commit(family):
    source=subprocess.check_output(["git","show","8fb0fdbe16c0e5512f36434a2f4b126a356fce4a:sam3_intermot/association/opportunity_models.py"],cwd=ROOT,text=True)
    namespace={"__name__":"frozen_stage_reference","__package__":"sam3_intermot.association"}
    exec(compile(source,"frozen_8fb0fdb_model","exec"),namespace)
    old=namespace["ActionValueModel"](family);current=ActionValueModel(family)
    current.load_state_dict(old.state_dict(),strict=True)
    current.mean[3]=1.;old.mean[3]=1.;current.scale.fill_(.05);old.scale.fill_(.05)
    x=torch.randn(8,24)
    a=old(x);b=current(x)
    assert a.keys()==b.keys()
    for key in a:assert torch.equal(a[key],b[key])
