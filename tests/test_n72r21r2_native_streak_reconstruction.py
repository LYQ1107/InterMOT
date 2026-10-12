from copy import deepcopy
import pytest
from sam3_intermot.evaluation.native_streak_reconstruction import observed_native_before_after, controlled_prefix_probe_values


def fixtures():
    trace=[{"frame":i,"target_uid":uid,"runtime_GT_read":False,"runtime_future_GT_used":False,"identity_memory_write":False}
           for i,uid in enumerate(["a","a","b","c",None,"d","e"])]
    axes={0:{"a":("s",1)},1:{"a":("s",1)},2:{"b":("s",1)},3:{"c":("other_scope",1)},
          4:{},5:{"d":("other_scope",1)},6:{"e":("other_scope",1)}}
    return trace,axes


def test_event_frame_not_incremented_scope_switch_none_and_consecutive_keys():
    trace,axes=fixtures();before,after=observed_native_before_after(trace,axes,0)
    assert list(before.values())==[0,0,1,2,1,0,1]
    assert list(after.values())==[0,1,2,1,0,1,2]
    assert after[2]==2  # UID changes, but exact native key continues.


@pytest.mark.parametrize("bad",["skip_frame","missing_axis","unknown_UID","GT","future_GT","write","bad_key","bool_frame"])
def test_invalid_axis_truth_or_non_P0_cannot_supply_derived_measurement(bad):
    trace,axes=fixtures()
    if bad=="skip_frame":trace[2]["frame"]=3
    elif bad=="missing_axis":axes.pop(2)
    elif bad=="unknown_UID":trace[2]["target_uid"]="oracle"
    elif bad=="GT":trace[2]["runtime_GT_read"]=True
    elif bad=="future_GT":trace[2]["runtime_future_GT_used"]=True
    elif bad=="write":trace[2]["identity_memory_write"]=True
    elif bad=="bool_frame":trace[0]["frame"]=False
    else:axes[2]["b"]=("s",True)
    with pytest.raises(ValueError):observed_native_before_after(trace,axes,0)


def test_actual_controlled_prefix_not_baseline_future_supplies_probe_counter():
    trace,axes=fixtures();prefix=deepcopy(trace[1:]);prefix[0]["target_uid"]=None
    parent={"source_frame":1,"own_actual_source_prefix":prefix}
    probes=[{"frame":2,"causal_state_distribution_observables":{"observed_native_streak":0}}]
    assert controlled_prefix_probe_values(trace,parent,axes,0,probes)=={2:0}
    probes[0]["causal_state_distribution_observables"]["observed_native_streak"]=1
    with pytest.raises(ValueError,match="recorded source counter"):
        controlled_prefix_probe_values(trace,parent,axes,0,probes)


def test_missing_source_probe_cannot_use_nearest_frame():
    trace,axes=fixtures();parent={"source_frame":1,"own_actual_source_prefix":trace[1:3]}
    probe={"frame":4,"causal_state_distribution_observables":{"observed_native_streak":1}}
    with pytest.raises(ValueError):controlled_prefix_probe_values(trace,parent,axes,0,[probe])
