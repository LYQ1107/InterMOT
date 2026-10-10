"""Small distinguishable event authority families, never direct MOT splices."""
import torch
from torch import nn
from .event_authority_learning import CURRENT_DIM
from .intervention_features import FEATURE_NAMES

FAMILIES = ("SCALAR", "LOGISTIC_RISK", "SMALL_MLP", "PAIRWISE_RANKER", "HARM_FIRST_DUAL_HEAD", "CAUSAL_TEMPORAL",
            "COUNTERFACTUAL_ACTION_VALUE", "DUAL_IDENTITY_STATE")


class EventAuthorityHead(nn.Module):
    """Output independent benefit/risk logits and same-prestate paired value.

    Frozen models can propose authority; actual commit must still pass the
    unchanged full global hard-constraint solver. No model receives labels,
    future vectors, GT IDs or future-selected candidates.
    """
    def __init__(self, family, *, hidden=32):
        super().__init__()
        if family not in FAMILIES or hidden not in (32, 64):
            raise ValueError("Unregistered family; relational model requires a separate measured trigger")
        self.family = family
        if family == "SCALAR":
            self.scalar_weights = nn.Parameter(torch.zeros(3))
            self.bias = nn.Parameter(torch.zeros(3))
        elif family == "LOGISTIC_RISK":
            self.net = nn.Linear(CURRENT_DIM, 3)
        elif family == "HARM_FIRST_DUAL_HEAD":
            self.risk = nn.Sequential(nn.Linear(CURRENT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.benefit_value = nn.Sequential(nn.Linear(CURRENT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        elif family == "CAUSAL_TEMPORAL":
            self.gru = nn.GRU(len(FEATURE_NAMES), hidden, batch_first=True)
            self.net = nn.Sequential(nn.Linear(hidden + CURRENT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        elif family == "DUAL_IDENTITY_STATE":
            self.anchor_indices = [FEATURE_NAMES.index(k) for k in ("anchor_cosine", "anchor_margin", "anchor_advantage_vs_KEEP", "KEEP_anchor_cosine")]
            self.state_indices = [FEATURE_NAMES.index(k) for k in ("prototype_anchor_agreement", "proposed_prototype_cosine", "actor_bank_size", "trusted_gap")]
            self.anchor = nn.Sequential(nn.Linear(4, hidden // 2), nn.ReLU())
            self.state = nn.Sequential(nn.Linear(4, hidden // 2), nn.ReLU())
            self.trust = nn.Sequential(nn.Linear(CURRENT_DIM, 1), nn.Sigmoid())
            self.net = nn.Sequential(nn.Linear(CURRENT_DIM + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        else:
            self.net = nn.Sequential(nn.Linear(CURRENT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, current, past, *, keep_current=None):
        if current.ndim != 2 or current.shape[-1] != CURRENT_DIM or past.ndim != 3 or past.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("Wrong causal input axes")
        if self.family == "SCALAR":
            ids = [FEATURE_NAMES.index(k) for k in ("anchor_advantage_vs_KEEP", "proposed_probability", "global_regret")]
            score = (current[:, ids] * self.scalar_weights).sum(-1)
            return torch.stack((score, -score, score), -1) + self.bias
        if self.family == "HARM_FIRST_DUAL_HEAD":
            bv = self.benefit_value(current)
            return torch.cat((bv[:, :1], self.risk(current), bv[:, 1:]), -1)
        if self.family == "CAUSAL_TEMPORAL":
            _, hidden = self.gru(past)
            return self.net(torch.cat((hidden[-1], current), -1))
        if self.family == "DUAL_IDENTITY_STATE":
            trust = self.trust(current)
            dual = torch.cat((self.anchor(current[:, self.anchor_indices]), trust * self.state(current[:, self.state_indices])), -1)
            return self.net(torch.cat((current, dual), -1))
        output = self.net(current)
        if self.family in ("PAIRWISE_RANKER", "COUNTERFACTUAL_ACTION_VALUE"):
            if keep_current is None or keep_current.shape != current.shape:
                raise ValueError("Actual same-prestate KEEP features required, not random training pairs")
            keep = self.net(keep_current)
            output = torch.cat((output[:, :2], output[:, 2:3] - keep[:, 2:3]), -1)
        return output


def independent_head_loss(output, target, *, harm_weight=5.):
    benefit = nn.functional.binary_cross_entropy_with_logits(output[:, 0], target[:, 0], reduction="none")
    risk = nn.functional.binary_cross_entropy_with_logits(output[:, 1], target[:, 1], reduction="none")
    risk = risk * torch.where(target[:, 1] > .5, harm_weight, 1.)
    value = nn.functional.smooth_l1_loss(output[:, 2], target[:, 2], reduction="none")
    return benefit + risk + value


def runtime_predictions(output):
    probabilities = output[:, :2].sigmoid().detach().cpu().tolist()
    values = output[:, 2].detach().cpu().tolist()
    return [{"beneficial": p[0], "harmful": p[1], "value": v} for p, v in zip(probabilities, values, strict=True)]
