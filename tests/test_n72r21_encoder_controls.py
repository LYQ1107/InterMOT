"""Input/rank/denominator contracts, not scientific performance fixtures."""
import numpy as np
import torch
from PIL import Image
from scripts.n72r21_encoder_controls import roi_tensor, matched_tensor
from scripts.n72r21_encoder_evaluate import observation, unit
from scripts.n72r21_sot_summary import summarize
from scripts.n72r20r1_candidate_stream_smoke import crop_batch


def test_exact_existing_crop_tensor_preprocessing(tmp_path):
    pixels=np.arange(3*25*30,dtype=np.uint8).reshape(25,30,3)
    path=tmp_path/'frame.png'; Image.fromarray(pixels).save(path)
    image=torch.from_numpy(pixels.copy()).permute(2,0,1)
    box=[-1.4,2.6,20.2,23.1]
    crop,empty=roi_tensor(image,box)
    assert not empty
    assert torch.equal(matched_tensor(crop),crop_batch(path,[box])[0])


def test_offimage_uid_preserved_with_explicit_empty_flag():
    crop,empty=roi_tensor(torch.ones(3,20,20,dtype=torch.uint8),[30,1,40,10])
    assert empty and crop.shape==(3,8,8) and not crop.any()


def test_ties_are_not_identity_wins_and_unresolved_full_rank_separate():
    margin,rank,mrr,fullrank=observation(np.array([.8,.8,.9]),[0],[1])
    assert margin==0 and rank==2 and mrr==.5 and fullrank==3


def test_l2_normalization_and_zero_rejection():
    import pytest
    assert np.allclose(unit(np.array([[3.,4.]])),[[.6,.8]])
    with pytest.raises(ValueError): unit(np.zeros((2,3)))


def test_sot_pooled_denominator_not_average_of_target_rates():
    def episode(n,correct):
        return {'visible_frames':n,'correct_frames_IoU_0_5':correct,'frames':n+2,
                'SOT_success_AUC_21_thresholds':.2,'SOT_normalized_precision_at_0_2':.3,
                'verified_wrong_person_takeover_frames':0,'visible_GT_gap_false_presence_frames':2,
                'reappearance_episodes':[]}
    result=summarize([episode(10,10),episode(100,0)])
    assert result['all_visible_target_recall']==10/110
    assert result['no_NONE_head'] and not result['memory_safety_PASS']
