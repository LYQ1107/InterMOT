"""Development-only, explicitly versioned learned-authority hand-filter ablation.

The original controller, scores, history, full solver and feedback are reused.
No GT labels select actions, and no ablation is qualified for confirmation.
"""
from contextlib import contextmanager
import math
from . import event_authority_runtime as original
from .intervention_features import FEATURE_NAMES

POLICIES = ("LEGACY_RAW_ANCHOR", "NO_GLOBAL_COST_CEILING", "NO_COST_OR_DISPLACEMENT_CEILING",
            "NO_COST_DISPLACEMENT_OR_ANCHOR_ADVANTAGE_CEILING")
LEGACY_PREDICATE = original.qualified_current_candidate


def ablated_qualification(features, prediction, point, *, feasible, changed, confirmations, policy):
    if policy not in POLICIES or set(features) != set(FEATURE_NAMES) or not all(math.isfinite(v) for v in features.values()):
        raise ValueError("Registered policy and exact causal features required; no GT/future fields")
    if policy == POLICIES[0]:
        return LEGACY_PREDICATE(features, prediction, point, feasible=feasible, changed=changed, confirmations=confirmations)
    if not feasible or not changed or confirmations < point["confirmation_delay"]:
        return False
    if prediction["beneficial"] < point["claim_min"] or prediction["harmful"] > point["risk_max"] or prediction["value"] <= 0:
        return False
    # Only named hard filters are removed. Real features and model inputs are
    # NOT zeroed/clipped or rewritten to impersonate support in the old gate.
    if policy == POLICIES[1] and features["displaced_count"] > 0:
        return False
    if features["proposal_NONE"]:
        return features["NONE_probability"] >= .95 and features["NONE_advantage"] >= .05 and features["KEEP_anchor_cosine"] < .6
    if policy != POLICIES[3] and features["anchor_advantage_vs_KEEP"] < point["anchor_advantage_min"]:
        return False
    return features["quality"] >= .5 and features["anchor_cosine"] >= .6


@contextmanager
def scoped_qualification(policy):
    previous = original.qualified_current_candidate
    def qualify(features, prediction, point, **kwargs):
        return ablated_qualification(features, prediction, point, policy=policy, **kwargs)
    original.qualified_current_candidate = qualify
    try:
        yield
    finally:
        original.qualified_current_candidate = previous


class SupportAblationBridge(original.LearnedEventBridge):
    def __init__(self, *args, support_policy, **kwargs):
        if support_policy not in POLICIES:
            raise ValueError("Unregistered support-policy variant")
        self.support_policy = support_policy
        super().__init__(*args, **kwargs)

    def step(self, frame, rows):
        with scoped_qualification(self.support_policy):
            result = super().step(frame, rows)
        result["authority"].update(support_ablation_policy=self.support_policy,
                                   support_ablation_is_UNQUALIFIED_TRAIN_diagnostic=True,
                                   model_scores_or_current_features_modified=False)
        return result
