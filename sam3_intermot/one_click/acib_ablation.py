"""Explicit inference deletion controls; not claimed retrained architectures.

FULL reproduces the frozen ACIB equations and parameter names. All controls
use the same trained weights. Their induced distribution shift is disclosed;
no independent module-learning benefit follows merely from these probes.
"""
import torch
from torch.nn import functional as F
from sam3_intermot.one_click.acib import ACIBNetwork


class ACIBVariantNetwork(ACIBNetwork):
    def __init__(self,*,anchor_mode='full',selection='learned',availability=True):
        super().__init__()
        if anchor_mode not in ('full','bank_only') or selection not in ('learned','uniform'):raise ValueError('registered inference variant')
        self.anchor_mode=anchor_mode;self.selection=selection;self.availability=bool(availability)

    def forward(self,anchor,candidates,valid,quality,*,bank=None,bank_valid=None,bank_metadata=None,context=None):
        batch,count,dimension=candidates.shape
        if dimension!=self.input_dim or anchor.shape!=(batch,dimension) or quality.shape!=(batch,count,5) or valid.shape!=(batch,count) or count<1:raise ValueError('ACIB variant schema')
        if bank is None:
            bank=anchor[:,None,:];bank_valid=torch.zeros(batch,1,dtype=torch.bool,device=anchor.device);bank_metadata=torch.zeros(batch,1,2,device=anchor.device,dtype=anchor.dtype)
        if bank.shape[0]!=batch or bank.shape[2]!=dimension or bank_valid.shape!=bank.shape[:2] or bank_metadata.shape!=(*bank.shape[:2],2):raise ValueError('bank schema')
        if context is None:context=torch.zeros(batch,3,device=anchor.device,dtype=anchor.dtype)
        has_bank=bank_valid.any(1);effective=bank_valid.clone();effective[~has_bank,0]=True
        bank=torch.where(has_bank[:,None,None],bank,anchor[:,None,:].expand_as(bank))
        raw_anchor=F.normalize(anchor.float(),dim=-1);raw_candidates=F.normalize(candidates.float(),dim=-1);raw_bank=F.normalize(bank.float(),dim=-1)
        learned_anchor=F.normalize(self.identity(anchor).float(),dim=-1);learned_candidates=F.normalize(self.identity(candidates).float(),dim=-1);learned_bank=F.normalize(self.identity(bank).float(),dim=-1)
        raw_anchor_cos=(raw_candidates*raw_anchor[:,None,:]).sum(-1);anchor_cos=(learned_candidates*learned_anchor[:,None,:]).sum(-1)
        raw_sim=torch.einsum('bnd,bkd->bnk',raw_candidates,raw_bank);learned_sim=torch.einsum('bnd,bkd->bnk',learned_candidates,learned_bank)
        mask=effective[:,None,:];denominator=effective.sum(-1).clamp_min(1)[:,None].float()
        raw_mean=(raw_sim*mask).sum(-1)/denominator;raw_std=(((raw_sim-raw_mean[:,:,None])**2*mask).sum(-1)/denominator).sqrt()
        selector=self.bank_selector(bank_metadata).float().squeeze(-1)
        weights=torch.softmax((8*learned_sim+selector[:,None,:]).masked_fill(~mask,-1e4),dim=-1)
        if self.selection=='uniform':weights=mask.float().expand_as(weights)/denominator[:,:,None]
        weighted=(learned_sim*weights).sum(-1)
        consistency=((raw_bank*raw_anchor[:,None,:]).sum(-1)*effective).sum(-1)/effective.sum(-1).clamp_min(1)
        recency=(bank_metadata[:,:,0]*effective).sum(-1)/effective.sum(-1).clamp_min(1);camera=(bank_metadata[:,:,1]*effective).sum(-1)/effective.sum(-1).clamp_min(1)
        repeat=lambda value:value[:,None].expand(batch,count)
        evidence=torch.stack([raw_anchor_cos,anchor_cos,raw_sim.masked_fill(~mask,-1e4).max(-1).values,raw_mean,
            learned_sim.masked_fill(~mask,-1e4).max(-1).values,weighted,repeat(consistency),*quality.float().unbind(-1),repeat(recency),repeat(camera),repeat(has_bank.float()),raw_std],dim=-1)
        ranking_identity=anchor_cos
        if self.anchor_mode=='bank_only':
            # Sole human observation is the initial fallback bank, not a
            # new click. Once a machine bank exists, fixed-anchor scoring and
            # consistency evidence are removed. Use with P1 for an unsafe
            # identity-scoring ablation, not a deployable success claim.
            evidence=evidence.clone();evidence[:,:,0]=0.;evidence[:,:,1]=0.;evidence[:,:,6]=0.;ranking_identity=weighted
        logits=(self.log_scale.float().exp().clamp(1,30)*ranking_identity+self.candidate_head(evidence).float().squeeze(-1)).masked_fill(~valid,-1e4)
        ranking=torch.softmax(logits,dim=-1)*valid;top=logits.max(1).values;second=logits.topk(min(2,count),dim=1).values[:,-1];second=torch.where(valid.sum(-1)>1,second,top-2.)
        pooled_mean=(evidence*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None];pooled_max=evidence.masked_fill(~valid[:,:,None],-1e4).max(1).values
        pooled_max=torch.where(valid.any(1)[:,None],pooled_max,torch.zeros_like(pooled_max))
        pooled=torch.cat([pooled_mean,pooled_max,top.clamp(-30,30)[:,None],(top-second).clamp(0,30)[:,None],torch.log1p(valid.sum(-1).float())[:,None],context.float()],dim=1)
        availability_logits=self.availability_head(pooled).float();availability_logits=torch.stack([availability_logits[:,0].masked_fill(~valid.any(-1),-1e4),availability_logits[:,1],availability_logits[:,2]],dim=1)
        availability=torch.softmax(availability_logits,-1);probability=availability[:,0]
        if not self.availability:probability=valid.any(-1).float()
        joint=torch.cat([ranking*probability[:,None],(1-probability)[:,None]],dim=1)
        write_input=torch.cat([evidence,ranking[:,:,None],probability[:,None,None].expand(batch,count,1)],dim=-1)
        write_logits=self.write_head(write_input).float().squeeze(-1).masked_fill(~valid,-1e4)
        return {'candidate_logits':logits,'availability_logits':availability_logits,'availability_probabilities':availability,
            'joint_probabilities':joint,'write_logits':write_logits,'evidence':evidence,'candidate_valid_probability':probability,'physical_presence_probability':None,
            'probability_semantics':'LEARNED_F1_CANDIDATE_AVAILABILITY_VISIBLE_GT_GAP_NOT_PHYSICAL_PRESENCE' if self.availability else 'ABLATION_ALWAYS_CANDIDATE_IF_NONEMPTY_NO_LEARNED_AVAILABILITY',
            'inference_variant':{'anchor_mode':self.anchor_mode,'bank_attention':self.selection,'availability_used':self.availability,'retrained_architecture':False}}
