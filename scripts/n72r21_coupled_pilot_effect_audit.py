"""Actual paired behavior effect, not inferred from different bank sizes."""
from pathlib import Path
import numpy as np
from scripts.n72r21_common import OUT,read_json,write_json,sha256
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl


def run():
    case='dancetrack0001__seed72101__T1MIXED_P1_K8';sequence='dancetrack0023'
    sealpath=OUT/'training/coupled_states/seals'/case/f'{sequence}.json';seal=read_json(sealpath)
    labelpath=OUT/'training/coupled_states/labels'/case/f'{sequence}.json';labels=read_json(labelpath)
    if labels['source_seal_sha256']!=sha256(sealpath):raise ValueError('source label seal')
    pairs=changed_decisions=changed_logits=0;proposals=0;maximum_logit_difference=0.;nonzero_margins=[]
    for artifact in seal['artifacts']:
        if sha256(artifact['branches_path'])!=artifact['branches_sha256']:raise ValueError('branches changed')
        for proposal in read_zstd_jsonl(Path(artifact['branches_path'])):
            proposals+=1;branches=proposal['branches']
            for without,with_write in zip(branches['without_current_write'],branches['with_current_write'],strict=True):
                if without['frame']!=with_write['frame']:raise ValueError('paired causal frame axis')
                first=without['current_candidate_logits'];second=with_write['current_candidate_logits']
                if set(first)!=set(second):raise ValueError('same future candidate UID axis')
                difference=max((abs(first[u]-second[u]) for u in first),default=0.)
                pairs+=1;changed_decisions+=without['selected_candidate_uid']!=with_write['selected_candidate_uid']
                changed_logits+=difference>1e-4;maximum_logit_difference=max(maximum_logit_difference,difference)
    margins=[r['mean_projected_margin_delta'] for e in labels['episodes'] for r in e['labels'] if r['mean_projected_margin_delta'] is not None]
    report={'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','scope':'Predetermined FIT pilot, one sequence/seed, IN_SAMPLE, no heldout generalization',
        'source_seal_sha256':sha256(sealpath),'source_label_sha256':sha256(labelpath),'actual_paired_proposals':proposals,
        'actual_future_frame_pairs':pairs,'different_selected_UID_future_pairs':int(changed_decisions),
        'different_candidate_logits_above_1e_minus4_pairs':int(changed_logits),'max_candidate_logit_absolute_difference':maximum_logit_difference,
        'competitive_mean_projected_margin_delta_mean':float(np.mean(margins)) if margins else None,
        'competitive_mean_projected_margin_delta_median':float(np.median(margins)) if margins else None,
        'bank_difference_does_not_itself_prove_identity_gain':True,'future_labels_are_training_supervision_not_safety_PASS':True,
        'scientific_success':False,'next_stage_authorized':False}
    write_json('training/coupled_states/PILOT_COUNTERFACTUAL_EFFECT_AUDIT.json',report);print(report)


if __name__=='__main__':run()
