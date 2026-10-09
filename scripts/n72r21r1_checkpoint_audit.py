"""Versioned selected/epoch weight SHA, strict loader and exact FIT lineage audit."""
from collections import Counter
import json
from pathlib import Path
import torch
from scripts.n72r21r1_common import ROOT,OUT,read_json,write_json,sha256,update_status
from sam3_intermot.one_click.learned_authority import TrajectoryAuthorityPredictor
from sam3_intermot.one_click.on_policy_authority import OnPolicyAuthorityPredictor
from sam3_intermot.one_click.joint_write_authority import JointWritePredictor


def run():
    torch.set_num_threads(1);records=[];counts=Counter();weights={};steps=0;gradient_steps=0
    expected_fit={'dancetrack0023','dancetrack0024','dancetrack0039','dancetrack0057','dancetrack0062','dancetrack0072'}
    for folder,loader in [('authority',TrajectoryAuthorityPredictor),('on_policy_correction',OnPolicyAuthorityPredictor),('joint_write',JointWritePredictor)]:
        for path in sorted((OUT/'training'/folder).glob('*.json')):
            r=read_json(path)
            if 'checkpoint_path' not in r:continue  # namespace manifests are not fitted weights
            assert set(r['FIT_sequences'])==expected_fit and r['INNER_sequences']==['dancetrack0002']
            assert sha256(r['checkpoint_path'])==r['checkpoint_SHA'] and sha256(r['epoch_log_path'])==r['epoch_log_SHA']
            assert r['optimizer_steps']>0 and r['nonzero_gradient_steps']>0 and r['changed_state_tensors']>0
            assert all(sha256(ROOT/p)==s for p,s in r['source_code_SHA'].items())
            selected=loader(r['checkpoint_path']);assert selected.selection==r['selection']
            saved=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=True)
            assert saved['seed']==r['seed'] and saved['epoch']==r['selected_epoch'] and saved['protocol_SHA']==r['protocol_SHA']
            assert len(r['all_epoch_checkpoints'])==r['epochs_completed']
            for epoch in r['all_epoch_checkpoints']:
                assert sha256(epoch['checkpoint_path'])==epoch['checkpoint_SHA'];weights[epoch['checkpoint_path']]=epoch['checkpoint_SHA']
            weights[r['checkpoint_path']]=r['checkpoint_SHA'];counts[folder]+=1;steps+=r['optimizer_steps'];gradient_steps+=r['nonzero_gradient_steps']
            records.append({'fit_uid':r['fit_uid'],'fit_record_path':str(path),'fit_record_SHA':sha256(path),
                'selected_checkpoint_path':r['checkpoint_path'],'selected_checkpoint_SHA':r['checkpoint_SHA'],
                'FIT_sequences':r['FIT_sequences'],'INNER_sequences':r['INNER_sequences'],'seed':r['seed'],
                'source_code_SHA':r['source_code_SHA'],'protocol_SHA':r['protocol_SHA'],'epochs':r['epochs_completed'],
                'optimizer_steps':r['optimizer_steps'],'nonzero_gradient_steps':r['nonzero_gradient_steps'],
                'actual_new_strict_state_dict_loader_pass':True,'selection_status':r['selection']['status']})
    assert counts=={'authority':93,'on_policy_correction':18,'joint_write':3}
    assert steps==gradient_steps==24879
    write_json('checkpoints/SEALED_MANIFEST_V1.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_NEW_HEAD_CHECKPOINT_AUDIT',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','actual_new_fits':len(records),'fits_by_namespace':dict(counts),
        'actual_new_selected_checkpoint_strict_loads':len(records),'all_epoch_and_selected_weight_files_SHA_verified':len(weights),
        'weight_files_SHA':weights,'optimizer_steps':steps,'nonzero_gradient_steps':gradient_steps,'records':records,
        'no_VAL_TEST_confirmation_parameter_fit_or_selection':True,'older248_weights_are_separate_inherited_SHA_catalog_not_new_training':True,
        'source_auditor_SHA':sha256(Path(__file__)),'not_scientific_success':True,'next_stage_authorized':False})
    update_status(actual_new_head_strict_loader_audit_complete=True,actual_new_head_strict_load_count=len(records))
    print(json.dumps({'actual_new_strict_loads':len(records),'weight_SHAs':len(weights),'optimizer_steps':steps}),flush=True)


if __name__=='__main__':run()
