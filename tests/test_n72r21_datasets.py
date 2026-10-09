import pytest
from sam3_intermot.one_click.datasets import strict_candidate_matching,validate_cross_session_pair


def test_matching_is_one_to_one_and_unmatched_not_other_person_proof():
    candidates=[{'candidate_uid':uid,'box_xyxy':(0,0,10,10)} for uid in ('a','b')]
    result=strict_candidate_matching(candidates,[{'identity':0,'box':(0,0,10,10)}])
    assert list(result.values()).count(0)==1 and list(result.values()).count(None)==1


@pytest.mark.parametrize('global_ids',[False,True])
def test_local_numeric_id_cannot_create_cross_recording_identity(global_ids):
    anchor={'recording_id':'a','global_identity':0,'timestamp':'2023-01-01'}
    query={'recording_id':'b','global_identity':0,'timestamp':'2023-01-02'}
    if global_ids:assert validate_cross_session_pair(anchor,query,authoritative_global_ids=True)
    else:
        with pytest.raises(ValueError):validate_cross_session_pair(anchor,query)
