from scripts.n72r21_lasot_candidates import valid_crop


def test_invalid_or_clipped_empty_crop_not_fabricated_as_machine_observation():
    assert valid_crop([0,0,10,20],100,100)
    assert not valid_crop([105,0,110,20],100,100)
    assert not valid_crop([0,0,0,20],100,100)
    assert not valid_crop([0,0,float('nan'),20],100,100)
