"""Direct, IPW, SNIPS, and doubly robust policy-value estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .data import BinaryPolicy, LoggedBanditData
from .nuisance import NuisanceModel


def estimate_from_predictions(
    action: ArrayLike,
    reward: ArrayLike,
    target_action: ArrayLike,
    propensity1: ArrayLike,
    mu0: ArrayLike,
    mu1: ArrayLike,
    *,
    propensity_clip: float = 0.02,
) -> dict[str, float]:
    """Compute contextual-bandit OPE estimators from nuisance predictions."""

    a = np.asarray(action, dtype=np.int64)
    y = np.asarray(reward, dtype=float)
    pi = np.asarray(target_action, dtype=np.int64)
    p1 = np.asarray(propensity1, dtype=float)
    m0 = np.asarray(mu0, dtype=float)
    m1 = np.asarray(mu1, dtype=float)

    arrays = [a, y, pi, p1, m0, m1]
    if any(arr.ndim != 1 for arr in arrays):
        raise ValueError("all OPE inputs must be one-dimensional")
    if len({arr.size for arr in arrays}) != 1 or a.size == 0:
        raise ValueError("all OPE inputs must have the same positive length")
    if not 0.0 < propensity_clip < 0.5:
        raise ValueError("propensity_clip must lie in (0, 0.5)")
    if np.any((a < 0) | (a > 1)) or np.any((pi < 0) | (pi > 1)):
        raise ValueError("actions must be binary")

    p1 = np.clip(p1, propensity_clip, 1.0 - propensity_clip)
    observed_propensity = np.where(a == 1, p1, 1.0 - p1)
    matched = a == pi
    weights = matched.astype(float) / observed_propensity

    mu_pi = np.where(pi == 1, m1, m0)
    mu_observed = np.where(a == 1, m1, m0)

    dm = float(np.mean(mu_pi))
    ipw = float(np.mean(weights * y))
    weight_sum = float(np.sum(weights))
    snips = float(np.sum(weights * y) / weight_sum) if weight_sum > 0.0 else float("nan")
    dr_scores = mu_pi + weights * (y - mu_observed)
    dr = float(np.mean(dr_scores))
    dr_standard_error = (
        float(np.std(dr_scores, ddof=1) / np.sqrt(dr_scores.size)) if dr_scores.size > 1 else 0.0
    )

    weight_square_sum = float(np.sum(weights**2))
    ess = weight_sum**2 / weight_square_sum if weight_square_sum > 0.0 else 0.0

    return {
        "dm": dm,
        "ipw": ipw,
        "snips": snips,
        "dr": dr,
        "dr_standard_error": dr_standard_error,
        "dr_ci95_lower": dr - 1.96 * dr_standard_error,
        "dr_ci95_upper": dr + 1.96 * dr_standard_error,
        "matched_rate": float(np.mean(matched)),
        "effective_sample_size": float(ess),
        "min_observed_propensity": float(np.min(observed_propensity)),
        "max_importance_weight": float(np.max(weights)),
    }


def estimate_policy_value(
    data: LoggedBanditData,
    policy: BinaryPolicy,
    nuisance: NuisanceModel,
    *,
    propensity_clip: float = 0.02,
) -> dict[str, float]:
    """Evaluate a fixed target policy on held-out logged observational data."""

    p1 = nuisance.propensity1(data.x)
    mu0, mu1 = nuisance.outcomes(data.x)
    target_action = policy.actions(data.x)
    return estimate_from_predictions(
        data.action,
        data.reward,
        target_action,
        p1,
        mu0,
        mu1,
        propensity_clip=propensity_clip,
    )
