import pytest
from scripts.n72r21r1_learned_shadow_repair import corrected_protocol


def test_explanatory_suffix_repair_is_unique_fixed_checkpoint_not_selection():
    uid='LOGISTIC__MIXED__H100_GLOBAL_RISK__seed72111'
    original={'shadow_head':uid+' fixed before effects.','cases':[{'case':uid}]}
    corrected=corrected_protocol(original)
    assert corrected['shadow_head']==uid and original['shadow_head']!=uid
    assert corrected['cases']==original['cases']
    with pytest.raises(ValueError):corrected_protocol({'shadow_head':'OTHER','cases':[{'case':'OTHER'}]})
