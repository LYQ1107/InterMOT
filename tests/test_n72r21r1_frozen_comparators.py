import numpy as np
import pytest
from tests.test_n72r21_mot_bridge import inputs
from sam3_intermot.one_click.frozen_comparator_controls import FrozenAdapterActor,ShuffledEvidenceMOTBridge,shuffled_evidence
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge


class FakeAdapter:
    def scores(self,anchor,features):return features@anchor


def test_donor_is_lexical_current_UID_not_truth_or_future_best():
    rows,event=inputs();anchor,uid=shuffled_evidence(rows,'p')
    assert uid=='q' and np.array_equal(anchor,rows[1]['feature'])
    with pytest.raises(ValueError):shuffled_evidence([rows[0]],'p')


def test_adapter_preserves_UID_axis_none_threshold_and_no_bank():
    rows,event=inputs();e={'frame':0,'box_xyxy':event['target_box_xyxy'],'sequence':'unit'}
    actor=FrozenAdapterActor(FakeAdapter(),event['human_anchor'],e,score_min=1.1,margin_min=.05)
    result=actor.step(1,rows)
    assert result['rank1_candidate_uid']=='p' and result['selected_candidate_uid'] is None
    assert not result['memory_write'] and not actor.bank
    with pytest.raises(ValueError):actor.step(1,rows)
    with pytest.raises(ValueError):actor.step(2,[dict(rows[0],gt_id=7),rows[1]])


def test_shuffled_shadow_preserves_complete_C0_and_disrupts_gate_human_anchor():
    rows,event=inputs();donor,uid=shuffled_evidence(rows,'p')
    actor=FrozenAdapterActor(FakeAdapter(),donor,{'frame':0,'box_xyxy':event['target_box_xyxy'],'sequence':'unit'},score_min=.5,margin_min=.05)
    bridge=ShuffledEvidenceMOTBridge(event,actor,policy=GatePolicy(family='shadow'));bridge.configure_fps(20)
    baseline=MOTIdentityBridge(event)
    for frame in range(5):
        a,b=bridge.step(frame,rows),baseline.step(frame,rows)
        assert a['outputs']==b['outputs'] and a['state_after']==b['state_after']
        assert np.array_equal(bridge.tracker.event['human_anchor'],event['human_anchor'])
        if frame:
            assert a['identity_decision']['proposed_candidate_uid']=='q'
            assert a['authority']['features']['anchor_cosine']==pytest.approx(1.)
            assert a['authority']['features']['KEEP_anchor_cosine']==pytest.approx(0.)
            assert not a['joint_identity_memory_write']
