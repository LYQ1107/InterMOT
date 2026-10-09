"""Current correctness and separately learned future-safe delayed commits.

No annotation/future trajectory is consulted by this deployment runtime.
The future-safe head is trained from offline paired branches, not an oracle
at deployment. A high predicted probability is not itself a safety PASS.
"""
import numpy as np
from sam3_intermot.one_click.acib_runtime import ACIBRecognizer,Evidence,unit


class TrustedACIBRecognizer(ACIBRecognizer):
    def __init__(self,model,anchor,token,*,capacity=8,policy='FULL',safe_delay=2):
        if policy not in ('FULL','NO_DELAY','P0','P1','MEAN','SAFE_FIXED'):raise ValueError('registered learned-write policy')
        self.deploy_policy=policy
        super().__init__(model,anchor,token,capacity=capacity,policy='P0' if policy in ('FULL','NO_DELAY') else policy,safe_delay=safe_delay)

    def step(self,frame,rows):
        result=super().step(frame,rows);selected=result['selected_candidate_uid'];probability=None
        if selected is not None:
            slot=next(i for i,r in enumerate(rows) if str(r['candidate_uid'])==selected)
            probability=float(self.model.last_future_safe_probability[0,slot])
        learned=self.deploy_policy in ('FULL','NO_DELAY')
        if learned:
            eligible=bool(result['eligible_safe_write'] and probability is not None and probability>=.95)
            confirmations=self.pending['count'] if self.pending is not None else 0
            written=eligible and (self.deploy_policy=='NO_DELAY' or confirmations>=self.safe_delay)
            if not eligible:self.pending=None
            if written:
                row=rows[slot];vector=unit(row['feature']);vector.setflags(write=False)
                self.bank.append(Evidence(vector,self.recording,int(frame),self.camera,frame/self.fps,result['candidate_available_probability'],
                    self.deploy_policy+'_LEARNED_CURRENT_AND_FUTURE_SAFE_CURRENT_CROP',float(np.clip(row.get('conf',0.),0,1)),
                    float(vector@self.anchor),selected))
                self.bank=self.bank[-self.capacity:]
            result.update(memory_write=bool(written),memory_write_candidate_uid=selected if written else None,
                          machine_bank_size=len(self.bank),bank_source_recordings=[e.recording_id for e in self.bank],
                          eligible_learned_future_safe_write=eligible)
        result.update(future_safe_probability=probability,future_safe_gate_used=learned,
                      write_future_harm_head_trained=True,
                      future_safe_semantics='CURRENT_INPUT_PREDICTION_OF_TRAIN_PAIRED_BRANCH_SUPERVISION_NOT_ONLINE_ORACLE',
                      deploy_policy=self.deploy_policy)
        return result
