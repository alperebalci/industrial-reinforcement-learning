"""Propensity and outcome nuisance models for observational OPE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge

from .data import LoggedBanditData


@dataclass(frozen=True)
class NuisanceModel:
    propensity: LogisticRegression
    outcome0: Ridge
    outcome1: Ridge

    def propensity1(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.propensity.predict_proba(x)[:, 1]

    def outcomes(
        self,
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.outcome0.predict(x), self.outcome1.predict(x)


def fit_nuisance(
    data: LoggedBanditData,
    *,
    ridge_alpha: float = 1.0,
) -> NuisanceModel:
    """Fit behavior propensity and action-conditional reward regressions."""

    if np.unique(data.action).size < 2:
        raise ValueError("both actions must be represented in nuisance-training data")

    propensity = LogisticRegression(max_iter=2000).fit(data.x, data.action)
    outcome0 = Ridge(alpha=ridge_alpha).fit(
        data.x[data.action == 0],
        data.reward[data.action == 0],
    )
    outcome1 = Ridge(alpha=ridge_alpha).fit(
        data.x[data.action == 1],
        data.reward[data.action == 1],
    )
    return NuisanceModel(propensity=propensity, outcome0=outcome0, outcome1=outcome1)
