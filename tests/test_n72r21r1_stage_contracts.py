from copy import deepcopy
import numpy as np
import pytest
from scripts.n72r21r1_common import HISTORY


def test_new_fit_reader_ignores_VAL_TEST_confirmation_and_exposed_outer(monkeypatch):
    from scripts import n72r21r1_train_authority as fit
    def row(role,*,harm=False):
        metric={'complete':True,'N01':int(not harm),'N10':int(harm),'non_target_damage':0,
            'wrong_or_UNKNOWN_writes':0,'non_target_benefit':0,'same_target_fragment_displacements':0}
        return {'initial_clicked_UID_verified_target':True,'role':role,'state_source':'JOINT_BASELINE_SHADOW_P0',
            'raw_trajectory_labels':{'future':{'H100':metric}},'causal_previous_feature_vectors':[[0.]*32]*3,
            'feature_vector':[0.]*32,'sequence':'FIT_UNIT' if role=='FIT' else role,'episode_uid':'unit',
            'frame':1,'action':{'family':'KEEP','candidate_uid':None}}
    records=[row('FIT'),row('FIT',harm=True),row('INNER')]+[row(k) for k in
        ['VAL','TEST','CONFIRMATION','HISTORICAL_OUTER_NOT_VIRGIN']]
    manifest={'path':'unit','sha256':'unit','label_components_SHA':'unit','origin_definition_SHA':'unit'}
    monkeypatch.setattr(fit,'read_json',lambda p:manifest)
    monkeypatch.setattr(fit,'sha256',lambda p:'unit')
    monkeypatch.setattr(fit,'read_zstd_jsonl',lambda p:records)
    FIT,INNER,_,_=fit.load_records('BASELINE_ONLY','H100')
    assert len(FIT)==2 and len(INNER)==1
    assert {r['role'] for r in FIT+INNER}=={'FIT','INNER'}


def test_pinned_CLEAR_FP_FN_can_change_with_ID_continuity_without_new_detections(monkeypatch):
    # This is a pinned metric diagnostic, never a patch to the third party.
    monkeypatch.syspath_prepend(str(HISTORY/'third_party/MOTIP/TrackEval'))
    from trackeval.metrics import CLEAR
    data={'num_tracker_dets':4,'num_gt_dets':4,'num_gt_ids':2,'num_tracker_ids':2,'num_timesteps':2,
        'gt_ids':[np.array([0,1]),np.array([0,1])],'tracker_ids':[np.array([0,1]),np.array([0,1])],
        'similarity_scores':[np.eye(2),np.array([[.6,.7],[.55,.4]])]}
    treatment=deepcopy(data);treatment['tracker_ids'][1]=np.array([1,0])
    metric=CLEAR({'PRINT_CONFIG':False});a,b=metric.eval_sequence(data),metric.eval_sequence(treatment)
    assert a['CLR_FP']==a['CLR_FN']==1 and b['CLR_FP']==b['CLR_FN']==0
    assert all(np.array_equal(x,y) for x,y in zip(data['similarity_scores'],treatment['similarity_scores']))


def test_posthoc_oracle_state_never_becomes_online_runtime_truth():
    from tests.test_n72r21_mot_bridge import inputs,actor
    from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
    rows,event=inputs();runtime=SafeMOTIdentityBridge(event,actor(event));runtime.configure_fps(20)
    runtime.step(0,rows)
    for key in ['target_gt_identity','future_target_uid','oracle']:
        with pytest.raises(ValueError):runtime.step(1,[dict(rows[0],**{key:'p'}),rows[1]])
        assert runtime.tracker.frame==0
