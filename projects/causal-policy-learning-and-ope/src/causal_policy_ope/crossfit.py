"""Cross-fitted off-policy evaluation for a fixed target policy."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from .data import BinaryPolicy, LoggedBanditData
from .nuisance import fit_nuisance
from .ope import estimate_from_predictions


def cross_fitted_policy_value(
    data: LoggedBanditData,
    policy: BinaryPolicy,
    *,
    folds: int = 5,
    seed: int = 0,
    propensity_clip: float = 0.02,
) -> dict[str, float]:
    """Estimate policy value with out-of-fold nuisance predictions.

    The target policy must be fixed before this function is called. Every observation
    receives propensity and outcome predictions from nuisance models trained without
    that observation.
    """

    if folds < 2 or folds > data.n:
        raise ValueError("folds must lie between 2 and the number of observations")

    p1 = np.empty(data.n, dtype=float)
    mu0 = np.empty(data.n, dtype=float)
    mu1 = np.empty(data.n, dtype=float)

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(data.x):
        fold = LoggedBanditData(
            x=data.x[train_idx],
            action=data.action[train_idx],
            reward=data.reward[train_idx],
            true_propensity=data.true_propensity[train_idx],
        )
        nuisance = fit_nuisance(fold)
        p1[test_idx] = nuisance.propensity1(data.x[test_idx])
        mu0[test_idx], mu1[test_idx] = nuisance.outcomes(data.x[test_idx])

    estimates = estimate_from_predictions(
        data.action,
        data.reward,
        policy.actions(data.x),
        p1,
        mu0,
        mu1,
        propensity_clip=propensity_clip,
    )
    estimates["folds"] = float(folds)
    return estimates
