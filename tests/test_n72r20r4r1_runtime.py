"""Engineering-only fixtures: no synthetic row is a scientific training sample."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.opportunity_solver import AssociationAction
from sam3_intermot.association.opportunity_scores import decompose_scores
from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from scripts.n72r20r4r1_supervision import match_frame,measure_branch,supervision_gate
from scripts.n72r20r4r1_common import assert_fit_sequence,events


def vec(i):
    x=np.zeros(512,dtype=np.float32);x[i]=1.;return x


def rows(frame=0):
    return [{"candidate_uid":f"{frame}:{i}","feature":vec(i),"box_xyxy":[i*20.,0.,i*20.+10.,10.],"native_tid":i+10,"native_scope":None,"conf":1.,"confidence":1.} for i in range(2)]


class TestUpdater(torch.nn.Module):
    __test__=False
    def forward(self,state,observation):return observation,torch.ones(len(state)),observation


def bank():return LearnedIdentityMemoryBank.from_updater(TestUpdater())


def tracker(memory="P0",event_frame=0):
    event={"event_frame":event_frame,"human_anchor":vec(0),"target_candidate_uid":f"{event_frame}:0","target_box_xyxy":[0,0,10,10]}
    return OpportunityTracker(config=AuthorityConfig(source="raw",memory=memory if memory in ("P0","P1","P2") else "P0"),event=event,bank=bank(),intervention_policy=InterventionPolicy(source="raw"),memory_policy=MemoryPolicy(memory))


@pytest.mark.parametrize("memory",["P0","P1","P2"])
def test_exact_keep_scores_lifecycle_and_anchor(memory):
    new=tracker(memory);old=CausalIdentityTracker(config=new.config,event=new.event,bank=bank())
    for f in range(4):
        a=new.step(rows(f),f);b=old.step(rows(f),f)
        assert a["assignments"]==b["assignments"] and a["state_after"]==b["state_after"]
        assert a["memory"]==b["memory"]
        assert np.array_equal(a["base_matrix"],b["base_matrix"])


def test_clone_owns_prototypes_memory_and_pending():
    t=tracker("P4");t.step(rows(0),0);t.step(rows(1),1);c=t.clone()
    c.states[t.target_public].prototype[0]=.2
    c.bank.records[t.target_public]._current_state[0]=.5
    c.pending[0]["frame"]=-99
    assert t.states[t.target_public].prototype[0]==1.
    assert t.bank.records[t.target_public].current_state[0]==1.
    assert t.pending[0]["frame"]==1
    assert c.bank.updater is t.bank.updater


@pytest.mark.parametrize("family",["GLOBAL_SWAP","REJECT_TARGET"])
def test_force_full_state_feedback_not_target_output_merge(family):
    t=tracker();t.step(rows(0),0)
    act=AssociationAction(family,t.target_public,"1:1" if family=="GLOBAL_SWAP" else None)
    b=t.step(rows(1),1,forced_action=act)
    assert len(b["outputs"])==2
    assert len({r["candidate_uid"] for r in b["outputs"]})==2
    if family=="GLOBAL_SWAP":
        assert t.states[t.target_public].last_native_tid==11
        # Its only alternative edge is negative: the exact remainder chooses
        # NONE and births the unowned candidate, not a fabricated pair swap.
        assert t.states[100002].state=="LOST"
        assert len(b["births"])==1
        assert t.states[b["births"][0]].last_native_tid==10
    else:
        assert t.states[t.target_public].state=="LOST" and len(b["births"])==1
    t.step(rows(2),2)


def test_hard_negative_is_immutable_after_bonus_discount():
    t=tracker();t.step(rows(0),0);s=t.states[t.target_public]
    s.add_negative(11)
    t.intervention_policy=InterventionPolicy(source="raw",native_discount=0.,positive_discount=0.)
    with pytest.raises(ValueError,match="HARD_NEGATIVE"):
        t.step(rows(1),1,forced_action=AssociationAction("GLOBAL_SWAP",t.target_public,"1:1"))


@pytest.mark.parametrize("key",["gt","gt_id","target_gt_id","oracle","label","label_index","runtime_gt_read","runtime_future_gt_used"])
def test_runtime_rejects_truth_keys(key):
    t=tracker();r=rows(0);r[0][key]=True
    with pytest.raises(ValueError):t.step(r,0)


@pytest.mark.parametrize("memory",["P3","P4","P6"])
def test_confirmation_occurs_only_on_later_frame(memory):
    t=tracker(memory);t.step(rows(0),0)
    anchor=t.bank.records[t.target_public].anchor_hash()
    first=t.step(rows(1),1);second=t.step(rows(2),2)
    assert not first["memory"]["accepted"]
    assert second["memory"]["accepted"]
    assert second["memory"]["confirmation_frame"]==2
    assert t.bank.update_audit[-1]["frame"]==2
    assert t.bank.records[t.target_public].anchor_hash()==anchor


def test_rollback_has_no_gt_and_never_backdates():
    t=tracker("P6")
    for f in range(3):t.step(rows(f),f)
    anchor=t.bank.records[t.target_public].anchor_hash()
    for f in range(3,6):
        r=rows(f);r[0]["feature"]=vec(1)
        result=t.step(r,f)
    assert result["memory"]["rollback"]
    assert result["memory"]["rollback_observed_frame"]==5
    assert t.bank.records[t.target_public].last_update_frame==5
    assert t.bank.records[t.target_public].anchor_hash()==anchor


def test_preclick_cannot_intervene():
    t=tracker(event_frame=2)
    with pytest.raises(ValueError,match="before human"):
        t.step(rows(0),0,forced_action=AssociationAction("KEEP",100001))


@pytest.mark.parametrize("f",[0,1,20])
def test_exact_decomposition_and_positive_provenance(f):
    t=tracker();t.step(rows(0),0);s=t.states[100001];s.add_positive(10)
    part=decompose_scores([s],rows(f),f)
    assert part["positive_bonus"][0,0]==5.
    assert part["native_bonus"][0,0]==3.
    assert np.allclose(sum(v for k,v in part.items() if k not in ("scores","hard_mask")),part["scores"])


def test_offline_matching_is_one_to_one_and_retains_unmatched_competitor():
    r=rows();r[1]["box_xyxy"]=r[0]["box_xyxy"]
    matched=match_frame(r,[(7,[0,0,10,10])])
    assert list(matched.values()).count(7)==1 and list(matched.values()).count(None)==1


def test_counterfactual_value_keep_zero_and_damage_unhidden():
    b={"frame":0,"assignments":{"1":"a","2":"b"},"target_uid":"a","memory":{"accepted":False}}
    matches=[{"a":7,"b":8}]
    keep=measure_branch([b],[deepcopy(b)],matches,{1:7,2:8},1,7)
    assert keep["value"]==0 and not keep["beneficial"]
    t={**b,"assignments":{"1":"b","2":"a"},"target_uid":"b"}
    harmed=measure_branch([b],[t],matches,{1:7,2:8},1,7)
    assert harmed["current_other_damage"]==1 and harmed["harmful"]


def test_cannot_train_on_all_negative_supervision():
    x=[{"H5":{"beneficial":False,"harmful":True},"identity_key":["s",1],"sequence":"s"} for _ in range(40)]
    assert not supervision_gate(x)["pass"]


@pytest.mark.parametrize("sequence",["dancetrack0001","dancetrack0002"])
def test_fold_reader_excludes_outer_and_inner(sequence):
    with pytest.raises(ValueError):assert_fit_sequence(sequence,"dancetrack0001")


@pytest.mark.parametrize("split",["val","test","MOT17"])
def test_val_is_gated_and_test_never_authorized(split):
    with pytest.raises(ValueError):events(split)
