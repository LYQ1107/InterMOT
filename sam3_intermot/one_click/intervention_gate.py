"""Baseline-preserving authority, independent from identity ranking quality."""
from dataclasses import dataclass, asdict
import numpy as np
from .intervention_features import feature_vector


@dataclass(frozen=True)
class GatePolicy:
    family: str = 'shadow'
    probability_min: float = .9
    identity_margin_min: float = .05
    base_margin_max: float = 1.
    quality_min: float = .5
    anchor_cosine_min: float = .6
    regret_max: float = 1.
    displaced_max: int = 0
    delay_frames: int = 2
    benefit_min: float = .9
    harm_max: float = .02
    value_min: float = 0.
    allow_reject: bool = False

    def __post_init__(self):
        if self.family not in ('off', 'shadow', 'original', 'confidence', 'recovery', 'global_regret', 'protected', 'delayed', 'two_branch', 'risk', 'abstain'):
            raise ValueError('unregistered authority family')
        if not 0 <= self.probability_min <= 1 or self.delay_frames < 1 or self.displaced_max < 0:
            raise ValueError('invalid frozen gate configuration')


def decide_authority(policy, features, *, feasible, proposed_change, predictor=None):
    prediction = None
    if not feasible: return False, ['HARD_INFEASIBLE'], prediction
    if not proposed_change: return False, ['SAME_AS_KEEP'], prediction
    if policy.family in ('off', 'shadow'): return False, ['BASELINE_PRESERVING_' + policy.family.upper()], prediction
    if policy.family == 'original': return True, ['ORIGINAL_FEASIBILITY_ONLY_CONTROL'], prediction
    reasons = []
    reject = bool(features['proposal_NONE'])
    if reject and not policy.allow_reject: reasons.append('REJECT_NOT_AUTHORIZED')
    if features['proposed_probability'] < policy.probability_min: reasons.append('LOW_IDENTITY_CONFIDENCE')
    if not reject:
        if features['candidate_margin'] < policy.identity_margin_min: reasons.append('SMALL_IDENTITY_MARGIN')
        if features['quality'] < policy.quality_min: reasons.append('LOW_CANDIDATE_QUALITY')
        if features['anchor_cosine'] < policy.anchor_cosine_min: reasons.append('LOW_ANCHOR_AGREEMENT')
    if not features['KEEP_NONE'] and features['base_KEEP_margin'] > policy.base_margin_max:
        reasons.append('KEEP_HAS_STRONG_CONTINUITY')
    if policy.family == 'recovery' and not (features['KEEP_NONE'] or features['target_LOST'] or features['base_KEEP_margin'] <= 0.):
        reasons.append('NOT_RECOVERY_OPPORTUNITY')
    if policy.family in ('global_regret', 'protected', 'delayed', 'two_branch', 'risk', 'abstain'):
        if features['global_regret'] > policy.regret_max: reasons.append('EXCESS_GLOBAL_REGRET')
    if policy.family in ('protected', 'delayed', 'two_branch', 'risk', 'abstain'):
        if features['displaced_count'] > policy.displaced_max: reasons.append('COMPETITOR_PROTECTED')
    if policy.family in ('delayed', 'two_branch', 'abstain'):
        if features['pending_confirmation_count'] < policy.delay_frames: reasons.append('AWAIT_CAUSAL_CONFIRMATION')
    if policy.family in ('two_branch', 'risk'):
        if predictor is None: raise ValueError('TRAIN-fitted current-input risk/value predictor required')
        prediction = predictor.predict(feature_vector(features))
        if any(k not in prediction for k in ('beneficial', 'harmful', 'abstain', 'value')):
            raise ValueError('risk predictor must expose benefit/harm/abstain/value')
        if not np.isfinite(list(prediction.values())).all(): raise ValueError('non-finite risk output')
        if prediction['beneficial'] < policy.benefit_min: reasons.append('INSUFFICIENT_PREDICTED_BENEFIT')
        if prediction['harmful'] > policy.harm_max: reasons.append('PREDICTED_HARM')
        if prediction['value'] <= policy.value_min: reasons.append('NONPOSITIVE_PREDICTED_VALUE')
        if prediction['abstain'] >= prediction['beneficial']: reasons.append('PREDICTED_ABSTENTION')
    return not reasons, reasons or ['APPROVED_CURRENT_PAST_EVIDENCE'], prediction
