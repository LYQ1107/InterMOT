import copy
from tests.test_n72r21_causal_data import fixture
from sam3_intermot.one_click.acib_runtime import ACIBRecognizer
from scripts.n72r21_collect_coupled_states import Capture,clone_state,paired_rollout


def initialized():
    model,anchor,event=fixture();runtime=ACIBRecognizer(Capture(model),anchor,'token',policy='P1')
    runtime.start_recording('record_a',fps=20,width=100,height=100,initial_frame=1,initial_box=event['box_xyxy'])
    row={'candidate_uid':'2','feature':anchor,'box_xyxy':event['box_xyxy'],'conf':.8}
    return runtime,row


def test_branches_are_actual_own_rollouts_and_do_not_mutate_main_state():
    runtime,row=initialized();before=copy.deepcopy(runtime.snapshot())
    future=[({'frame':frame},[{**row,'candidate_uid':str(frame)}]) for frame in range(3,6)]
    result=paired_rollout(runtime,2,[row],future)
    assert runtime.snapshot()==before
    assert result['branches']['with_current_write'][0]['machine_bank_size']==2
    assert result['branches']['without_current_write'][0]['machine_bank_size']==1
    assert result['branches']['with_current_write'][0]['current_candidate_logits'].keys()=={'3'}
    assert result['GT_read'] is False


def test_clone_mutable_state_does_not_alias_main():
    runtime,row=initialized();runtime.step(2,[row]);clone=clone_state(runtime)
    clone.bank.clear();clone.last_box[0]=999
    assert len(runtime.bank)==1 and runtime.last_box[0]!=999
