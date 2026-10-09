import pytest
from scripts.n72r21_verify_risk_labels import json_equal


def test_JSON_integer_key_roundtrip_is_equivalent_not_changed_supervision():
    assert json_equal({'margins':{12:.1,20:-.2}},{'margins':{'12':.1,'20':-.2}})
    assert json_equal({'margins':{9:.1,10:-.2}},{'margins':{'9':.1,'10':-.2}})
    assert not json_equal({'margins':{12:.1}},{'margins':{'12':.2}})


def test_nonfinite_labels_rejected_and_extra_metadata_not_ignored():
    with pytest.raises(ValueError):json_equal({'margin':float('nan')},{'margin':float('nan')})
    assert not json_equal({'label':True},{'label':True,'extra':1})
