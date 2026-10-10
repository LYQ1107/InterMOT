"""Read-only support measurements; neither OOD counts nor rows are G1 roots."""
import numpy as np


def _matrix(value, width, name, *, empty=False):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != width or not empty and not len(result):
        raise ValueError("Wrong nonempty feature axis: " + name)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite feature observations: " + name)
    return result


def support_comparison(source, observed, names, mean, scale):
    """Exact float32 ranges, no new policy threshold or feature clipping.

    Values are the existing pre-normalizer scaled/clipped encoder values,
    NOT original-frame units. Zero history padding must be removed by caller.
    Marginal range membership does not establish joint distribution support.
    """
    names = tuple(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Unique named feature axis required")
    src = _matrix(source, len(names), "FIT")
    own = _matrix(observed, len(names), "actual", empty=True)
    mean, scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
    if (mean.shape != (len(names),) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Exact finite frozen normalizer required")
    low, high = src.min(0), src.max(0)
    outside = (own < low) | (own > high)
    summaries = {}
    for i, name in enumerate(names):
        quantiles = [0., .1, .25, .5, .75, .9, 1.]
        summaries[name] = {
            "FIT_min": float(low[i]), "FIT_max": float(high[i]),
            "FIT_exactly_constant": bool(low[i] == high[i]),
            "FIT_quantiles_0_10_25_50_75_90_100": np.quantile(src[:, i], quantiles).tolist(),
            "actual_quantiles_0_10_25_50_75_90_100": np.quantile(own[:, i], quantiles).tolist() if len(own) else None,
            "outside_FIT_marginal_range_rows": int(outside[:, i].sum()),
            "outside_FIT_marginal_range_fraction": float(outside[:, i].mean()) if len(own) else None,
            "max_abs_actual_frozen_normalized_value": float(np.abs((own[:, i] - mean[i]) / scale[i]).max()) if len(own) else None,
            "frozen_mean": float(mean[i]), "frozen_scale": float(scale[i]),
        }
    return {"FIT_rows_correlated": len(src), "actual_rows_correlated": len(own),
            "any_marginal_outside_rows": int(outside.any(1).sum()),
            "features": summaries, "marginal_membership_is_not_joint_support": True,
            "rows_not_independent_scientific_units": True}


def unpadded_history(past, width):
    """Use the deployed encoder's EXACT all-zero padding convention."""
    past = np.asarray(past, np.float32)
    if past.ndim != 3 or past.shape[1:] != (3, width) or not np.isfinite(past).all():
        raise ValueError("Exact past3 causal feature axis required")
    flat = past.reshape(-1, width)
    mask = np.all(flat == 0, axis=1)
    return flat[~mask], int(mask.sum())


def prediction_error(actual, saved):
    names = ("beneficial", "harmful", "value")
    if set(actual) != set(names) or set(saved) != set(names):
        raise ValueError("Three original predictor outputs required")
    a, b = np.array([actual[k] for k in names]), np.array([saved[k] for k in names])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite frozen predictor outputs")
    return float(np.abs(a - b).max())
