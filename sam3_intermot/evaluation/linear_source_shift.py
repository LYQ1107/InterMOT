"""Read-only linear logit attribution for FIT-constant deployment dimensions."""
import numpy as np


def constant_feature_decomposition(fit, own_normalized, weight, bias, *, causal_dim=32):
    fit = np.asarray(fit, np.float32)
    own = np.asarray(own_normalized, np.float32)
    weight, bias = np.asarray(weight, np.float32), np.asarray(bias, np.float32)
    if (fit.ndim != 2 or own.ndim != 2 or not len(fit) or not len(own)
            or fit.shape[1] != own.shape[1] or causal_dim > own.shape[1]
            or weight.shape != (3, own.shape[1]) or bias.shape != (3,)
            or not all(np.isfinite(a).all() for a in (fit, own, weight, bias))):
        raise ValueError("Exact finite linear causal input/output axes required")
    columns = np.flatnonzero(np.ptp(fit[:, :causal_dim], axis=0) == 0)
    contribution = own[:, columns] @ weight[:, columns].T
    full = own @ weight.T + bias
    residual = full - contribution
    return columns, full, contribution, residual


def score_only_pass(logits, point):
    logits = np.asarray(logits, np.float32)
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError("Finite benefit/risk/value logits required")
    probabilities = 1. / (1. + np.exp(-np.clip(logits[:, :2], -80., 80.)))
    # This is NOT feasibility, confirmation, action selection or future safety.
    return ((probabilities[:, 0] >= point["claim_min"]) & (probabilities[:, 1] <= point["risk_max"])
            & (logits[:, 2] > 0))
