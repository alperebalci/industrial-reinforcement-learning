"""Budget-constrained policy learning from estimated heterogeneous treatment effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from .data import LoggedBanditData


@dataclass(frozen=True)
class BudgetThresholdPolicy:
    """Treat contexts whose estimated treatment effect exceeds a learned threshold."""

    outcome0: Ridge
    outcome1: Ridge
    threshold: float
    budget: float

    def treatment_score(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.outcome1.predict(x) - self.outcome0.predict(x)

    def actions(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        return (self.treatment_score(x) >= self.threshold).astype(np.int64)


def _fit_outcome_models(data: LoggedBanditData, alpha: float) -> tuple[Ridge, Ridge]:
    if np.unique(data.action).size < 2:
        raise ValueError("both actions must be represented in policy-training data")
    outcome0 = Ridge(alpha=alpha).fit(data.x[data.action == 0], data.reward[data.action == 0])
    outcome1 = Ridge(alpha=alpha).fit(data.x[data.action == 1], data.reward[data.action == 1])
    return outcome0, outcome1


def fit_budget_policy(
    data: LoggedBanditData,
    budget: float = 0.40,
    *,
    ridge_alpha: float = 1.0,
) -> BudgetThresholdPolicy:
    """Learn a simple constrained treatment policy using outcome-model treatment scores."""

    if not 0.0 < budget < 1.0:
        raise ValueError("budget must lie strictly between zero and one")

    outcome0, outcome1 = _fit_outcome_models(data, ridge_alpha)
    scores = outcome1.predict(data.x) - outcome0.predict(data.x)
    threshold = float(np.quantile(scores, 1.0 - budget))
    return BudgetThresholdPolicy(
        outcome0=outcome0,
        outcome1=outcome1,
        threshold=threshold,
        budget=float(budget),
    )
