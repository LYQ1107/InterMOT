import itertools
import numpy as np
import pytest

from sam3_intermot.association.opportunity_solver import AssociationAction,solve_counterfactual_global_assignment,objective
from sam3_intermot.association.public_assignment import solve_exact_public_assignment,validate_exact_public_assignment
from sam3_intermot.association.global_assignment_adapter import public_map


def solve(matrix,action,mask=None):
    n,m=np.asarray(matrix).shape
    return solve_counterfactual_global_assignment([{"candidate_uid":f"u{i}","candidate_index":i} for i in range(n)],np.asarray(matrix,dtype=float),list(range(11,11+m)),list(range(101,101+m)),action,hard_negative_mask=mask)


@pytest.mark.parametrize("n,m",[(0,0),(0,2),(1,0),(1,1),(2,2),(3,3),(2,3),(3,2)])
def test_keep_matches_complete_original_solver(n,m):
    matrix=np.arange(n*m).reshape(n,m)-2
    if m==0:
        result=solve(matrix,AssociationAction("KEEP",101));assert not result["feasible"];return
    result=solve(matrix,AssociationAction("KEEP",101))
    expected=solve_exact_public_assignment([{"candidate_uid":f"u{i}","candidate_index":i} for i in range(n)],matrix,list(range(11,11+m)),list(range(101,101+m)),source_run_id="N72R20R4:0")
    assert result["solver"]==expected and result["global_cost"]==0


@pytest.mark.parametrize("matrix",[np.array([[9,2],[1,8]]),np.array([[9,2,1],[1,8,2],[3,1,7]])])
def test_forced_edge_changes_all_ownership_and_is_exact_constrained_optimum(matrix):
    result=solve(matrix,AssociationAction("GLOBAL_SWAP",101,"u1"))
    assert result["feasible"] and result["assignment_changed"]
    mapping=public_map(result["solver"])
    assert mapping[101]=="u1" and mapping[102]!="u1"
    n,m=matrix.shape
    feasible=[]
    # Brute-force all per-candidate public/NONE decisions, enforce the edge.
    for choice in itertools.product(range(-1,m),repeat=n):
        real=[v for v in choice if v>=0]
        if choice[1]!=0 or len(real)!=len(set(real)):continue
        feasible.append(sum(0 if j<0 else matrix[i,j] for i,j in enumerate(choice)))
    assert result["assignment_objective"]==max(feasible)
    assert result["global_cost"]>=0
    assert not validate_exact_public_assignment(result["solver"])


@pytest.mark.parametrize("scores,mask",[([[2,1],[-1e9,2]],None),([[2,1],[100,2]],[[False,False],[True,False]])])
def test_hard_negative_is_infeasible_even_with_huge_identity_score(scores,mask):
    result=solve(scores,AssociationAction("SWITCH_TO_CANDIDATE",101,"u1"),mask)
    assert not result["feasible"] and result["hard_constraint_violation"]
    assert result["solver"] is None


def test_reject_is_public_none_not_omitted_other_assignments():
    result=solve([[9,1],[1,8]],AssociationAction("REJECT_TARGET",101))
    assert result["feasible"] and public_map(result["solver"])[101] is None
    assert public_map(result["solver"])[102]=="u1"
    assert len(result["solver"]["assignment_rows"])==2
    assert result["unassigned_candidate_uids"]==["u0"]


def test_recover_from_none_and_duplicate_uid_or_axes_are_rejected():
    result=solve([[-2,4]],AssociationAction("RECOVER_FROM_NONE",101,"u0"))
    assert result["feasible"] and public_map(result["solver"])[101]=="u0"
    assert result["displaced_public_ids"]==[102]
    with pytest.raises(ValueError,match="collision"):
        solve_counterfactual_global_assignment([{"candidate_uid":"u"},{"candidate_uid":"u"}],np.ones((2,1)),[11],[101],AssociationAction("KEEP",101))


@pytest.mark.parametrize("family,uid",[("SWITCH_TO_CANDIDATE",None),("KEEP","u"),("REJECT_TARGET","u"),("UNKNOWN",None)])
def test_invalid_action_schema_does_not_reach_solver(family,uid):
    with pytest.raises(ValueError):AssociationAction(family,101,uid)


def test_mask_shape_stale_candidate_and_recovery_preconditions():
    with pytest.raises(ValueError,match="axis mismatch"):
        solve([[3]],AssociationAction("KEEP",101),np.zeros((2,1)))
    assert not solve([[3]],AssociationAction("SWITCH_TO_CANDIDATE",101,"future_uid"))["feasible"]
    assert not solve([[3]],AssociationAction("RECOVER_FROM_NONE",101,"u0"))["feasible"]
