from scripts.n72r21r1_density_trackeval import mask_original_frames


def test_density_mask_retains_original_frame_numbers_public_ids_and_empty_gaps():
    text='1,41,1,2,3,4,1,-1,-1,-1\n2,42,1,2,3,4,1,-1,-1,-1\n8,41,1,2,3,4,1,-1,-1,-1\n'
    assert mask_original_frames(text,{0,7})=='1,41,1,2,3,4,1,-1,-1,-1\n8,41,1,2,3,4,1,-1,-1,-1\n'
    assert mask_original_frames(text,set())==''
