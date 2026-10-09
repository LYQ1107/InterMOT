"""GT-free candidate ACIB network; no scientific success is assumed.

The F1 three-class head predicts candidate availability / visible-GT-gap
semantics, NOT experimentally validated physical presence. True cross-session
training requires authoritative data, not a guessed same numeric local ID.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

AVAILABILITY_CLASSES = ('TARGET_VALID_CANDIDATE','VISIBLE_TARGET_NO_VALID_CANDIDATE',
                        'VISIBLE_GT_GAP_PHYSICAL_ABSENCE_UNRESOLVED')
EVIDENCE_SCHEMA = ('raw_anchor_cos','learned_anchor_cos','raw_bank_max','raw_bank_mean',
                   'learned_bank_max','learned_bank_selected','anchor_bank_consistency',
                   'confidence','normalized_box_area','aspect_scaled','motion_iou',
                   'last_accept_recency','bank_recency','known_camera_difference',
                   'has_machine_bank','raw_bank_similarity_std')


class ACIBNetwork(nn.Module):
    """One anchor, finite previous bank, current candidates → target or NONE."""
    def __init__(self,input_dim=512,hidden_dim=128):
        super().__init__()
        self.input_dim=input_dim;self.hidden_dim=hidden_dim
        self.identity=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
        self.bank_selector=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,1))
        self.candidate_head=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,1))
        self.availability_head=nn.Sequential(nn.Linear(38,64),nn.GELU(),nn.Linear(64,3))
        self.write_head=nn.Sequential(nn.Linear(18,64),nn.GELU(),nn.Linear(64,1))
        self.log_scale=nn.Parameter(torch.tensor(2.0794415))

    def forward(self,anchor,candidates,valid,quality,*,bank=None,bank_valid=None,bank_metadata=None,context=None):
        # Shapes: BxD anchor, BxNxD current candidate, BxN mask, BxNx5
        # quality [confidence, normalized_area, aspect_scaled, motion_iou,
        # last_accept_recency]. Bank metadata: [within-clip recency, known
        # camera difference]; absence of metadata is not an inferred camera.
        batch,count,dimension=candidates.shape
        if dimension!=self.input_dim or anchor.shape!=(batch,dimension) or quality.shape!=(batch,count,5):
            raise ValueError('ACIB input/schema shape')
        if valid.shape!=(batch,count) or count<1: raise ValueError('padded slot required even for empty current set')
        if bank is None:
            bank=anchor[:,None,:];bank_valid=torch.zeros((batch,1),dtype=torch.bool,device=anchor.device)
            bank_metadata=torch.zeros((batch,1,2),device=anchor.device,dtype=anchor.dtype)
        if bank.shape[0]!=batch or bank.shape[2]!=dimension or bank_valid.shape!=bank.shape[:2] or bank_metadata.shape!=(*bank.shape[:2],2):
            raise ValueError('bounded-bank schema')
        if context is None:context=torch.zeros((batch,3),device=anchor.device,dtype=anchor.dtype)
        has_bank=bank_valid.any(dim=1)
        effective=bank_valid.clone();effective[~has_bank,0]=True
        bank=torch.where(has_bank[:,None,None],bank,anchor[:,None,:].expand_as(bank))
        raw_anchor=F.normalize(anchor.float(),dim=-1);raw_candidates=F.normalize(candidates.float(),dim=-1)
        raw_bank=F.normalize(bank.float(),dim=-1)
        learned_anchor=F.normalize(self.identity(anchor).float(),dim=-1)
        learned_candidates=F.normalize(self.identity(candidates).float(),dim=-1)
        learned_bank=F.normalize(self.identity(bank).float(),dim=-1)
        raw_anchor_cos=(raw_candidates*raw_anchor[:,None,:]).sum(-1)
        anchor_cos=(learned_candidates*learned_anchor[:,None,:]).sum(-1)
        raw_sim=torch.einsum('bnd,bkd->bnk',raw_candidates,raw_bank)
        learned_sim=torch.einsum('bnd,bkd->bnk',learned_candidates,learned_bank)
        mask=effective[:,None,:];denominator=effective.sum(-1).clamp_min(1)[:,None].float()
        raw_mean=(raw_sim*mask).sum(-1)/denominator
        raw_std=(((raw_sim-raw_mean[:,:,None])**2*mask).sum(-1)/denominator).sqrt()
        selector=self.bank_selector(bank_metadata).float().squeeze(-1)
        weights=torch.softmax((8*learned_sim+selector[:,None,:]).masked_fill(~mask,-1e4),dim=-1)
        weighted=(learned_sim*weights).sum(-1)
        bank_consistency=((raw_bank*raw_anchor[:,None,:]).sum(-1)*effective).sum(-1)/effective.sum(-1).clamp_min(1)
        recency=(bank_metadata[:,:,0]*effective).sum(-1)/effective.sum(-1).clamp_min(1)
        camera=(bank_metadata[:,:,1]*effective).sum(-1)/effective.sum(-1).clamp_min(1)
        repeat=lambda value:value[:,None].expand(batch,count)
        evidence=torch.stack([raw_anchor_cos,anchor_cos,raw_sim.masked_fill(~mask,-1e4).max(-1).values,raw_mean,
            learned_sim.masked_fill(~mask,-1e4).max(-1).values,weighted,repeat(bank_consistency),
            *quality.float().unbind(-1),repeat(recency),repeat(camera),repeat(has_bank.float()),raw_std],dim=-1)
        logits=(self.log_scale.float().exp().clamp(1,30)*anchor_cos+self.candidate_head(evidence).float().squeeze(-1)).masked_fill(~valid,-1e4)
        ranking=torch.softmax(logits,dim=-1)*valid
        top=logits.max(dim=1).values
        second=logits.topk(min(2,count),dim=1).values[:,-1]
        second=torch.where(valid.sum(-1)>1,second,top-2.)
        pooled_mean=(evidence*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        pooled_max=evidence.masked_fill(~valid[:,:,None],-1e4).max(1).values
        pooled_max=torch.where(valid.any(1)[:,None],pooled_max,torch.zeros_like(pooled_max))
        pooled=torch.cat([pooled_mean,pooled_max,top.clamp(-30,30)[:,None],(top-second).clamp(0,30)[:,None],
                          torch.log1p(valid.sum(-1).float())[:,None],context.float()],dim=1)
        availability_logits=self.availability_head(pooled).float()
        availability_logits=torch.stack([availability_logits[:,0].masked_fill(~valid.any(-1),-1e4),
                                         availability_logits[:,1],availability_logits[:,2]],dim=1)
        availability=torch.softmax(availability_logits,dim=-1)
        valid_probability=availability[:,0]
        joint=torch.cat([ranking*valid_probability[:,None],(1-valid_probability)[:,None]],dim=1)
        write_input=torch.cat([evidence,ranking[:,:,None],valid_probability[:,None,None].expand(batch,count,1)],dim=-1)
        write_logits=self.write_head(write_input).float().squeeze(-1).masked_fill(~valid,-1e4)
        return {'candidate_logits':logits,'availability_logits':availability_logits,'availability_probabilities':availability,
                'joint_probabilities':joint,'write_logits':write_logits,'evidence':evidence,
                'candidate_valid_probability':valid_probability,'physical_presence_probability':None,
                'probability_semantics':'LEARNED_F1_CANDIDATE_AVAILABILITY_VISIBLE_GT_GAP_NOT_PHYSICAL_PRESENCE'}


def training_losses(output,target_slot,availability_label,write_labels,write_verified):
    joint=output['joint_probabilities'].float().clamp_min(1e-8)
    decision=F.nll_loss(joint.log(),target_slot)
    availability=F.cross_entropy(output['availability_logits'],availability_label)
    if write_verified.any():
        write=F.binary_cross_entropy_with_logits(output['write_logits'][write_verified],write_labels[write_verified].float())
    else:write=output['write_logits'].sum()*0.
    total=decision+.5*availability+.2*write
    return {'total':total,'decision':decision,'availability':availability,'write_correctness_proxy':write}
