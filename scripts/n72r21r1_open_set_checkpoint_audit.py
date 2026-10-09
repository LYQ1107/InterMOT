"""Extension receipt: preserve114-head audit, strictly verify9 new E6 heads."""
import json
import torch
from scripts.n72r21r1_common import ROOT,OUT,read_json,write_json,sha256,update_status
from scripts.n72r21r1_train_open_set import PROTOCOL
from sam3_intermot.one_click.open_set_verifier import OpenSetVerifierPredictor


def run():
    torch.set_num_threads(1);p=read_json(PROTOCOL);records=[];weights={};steps=0
    old=OUT/'checkpoints/SEALED_MANIFEST_V1.json';previous=read_json(old);assert previous['actual_new_fits']==114
    for family in p['families']:
        for seed in p['seeds']:
            path=OUT/'training/current_axis_verifier'/(family+'__seed'+str(seed)+'.json');r=read_json(path)
            assert r['FIT_sequences']==p['FIT_sequences'] and r['INNER_sequences']==p['INNER_sequences']
            assert sha256(r['checkpoint_path'])==r['checkpoint_SHA'] and sha256(r['epoch_log_path'])==r['epoch_log_SHA']
            assert r['optimizer_steps']==r['nonzero_gradient_steps']>0 and r['changed_state_tensors']>0
            assert all(sha256(ROOT/k)==v for k,v in r['source_code_SHA'].items())
            loader=OpenSetVerifierPredictor(r['checkpoint_path']);assert loader.selection==r['selection']
            saved=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=True)
            assert saved['epoch']==r['selected_epoch'] and saved['seed']==seed and saved['protocol_SHA']==sha256(PROTOCOL)
            for e in r['all_epoch_checkpoints']:
                assert sha256(e['checkpoint_path'])==e['checkpoint_SHA'];weights[e['checkpoint_path']]=e['checkpoint_SHA']
            weights[r['checkpoint_path']]=r['checkpoint_SHA'];steps+=r['optimizer_steps']
            records.append({'fit_record_path':str(path),'fit_record_SHA':sha256(path),'checkpoint_SHA':r['checkpoint_SHA'],
                'family':family,'seed':seed,'strict_load_pass':True,'epochs':r['epochs_completed'],'steps':r['optimizer_steps'],
                'selection_status':r['selection']['status'],'protocol_SHA':r['protocol_SHA']})
    assert len(records)==9 and steps==4121
    write_json('checkpoints/CURRENT_AXIS_EXTENSION_V1.json',{'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'status':'COMPLETE_ACTUAL_NINE_CURRENT_AXIS_STRICT_LOADS','prior114_manifest_SHA':sha256(old),'additional_strict_loads':9,
        'total_actual_new_heads_including_prior':123,'additional_epoch_selected_SHAs_verified':len(weights),'additional_optimizer_nonzero_steps':steps,
        'total_optimizer_nonzero_steps_including_prior':previous['optimizer_steps']+steps,'additional_weights_SHA':weights,'records':records,
        'source_auditor_SHA':sha256(__file__),'not_scientific_success':True,'next_stage_authorized':False})
    update_status(actual_new_head_strict_load_count=123,actual_new_current_axis_strict_loader_audit_complete=True)
    print(json.dumps({'strict_loader_total_new_heads':123,'additional_weight_SHAs':len(weights),'additional_nonzero_steps':steps}),flush=True)


if __name__=='__main__':run()
