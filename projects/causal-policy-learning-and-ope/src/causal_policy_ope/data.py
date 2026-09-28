"""Synthetic observational contextual-bandit data with known potential-outcome structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class BinaryPolicy(Protocol):
    def actions(self, x: NDArray[np.float64]) -> NDArray[np.int64]: ...


@dataclass(frozen=True)
class LoggedBanditData:
    x: NDArray[np.float64]
    action: NDArray[np.int64]
    reward: NDArray[np.float64]
    true_propensity: NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def baseline_reward(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2.0 + 0.8 * x[:, 0] - 0.45 * x[:, 1] + 0.20 * x[:, 2] ** 2


def treatment_effect(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.15 * x[:, 0] - 0.75 * x[:, 1] + 0.55 * np.sin(x[:, 2]) - 0.20


def behavior_propensity(x: NDArray[np.float64]) -> NDArray[np.float64]:
    logits = -0.35 + 0.90 * x[:, 0] - 0.65 * x[:, 1] + 0.40 * x[:, 2]
    propensity = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(propensity, 0.05, 0.95)


def generate_logged_data(seed: int, n: int) -> LoggedBanditData:
    """Generate logged observational decisions under selection on observed covariates."""

    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    propensity = behavior_propensity(x)
    action = rng.binomial(1, propensity).astype(np.int64)
    noise = rng.normal(0.0, 0.6, size=n)
    reward = baseline_reward(x) + action * treatment_effect(x) + noise
    return LoggedBanditData(x=x, action=action, reward=reward, true_propensity=propensity)


def true_policy_value(policy: BinaryPolicy, seed: int = 9001, n: int = 200_000) -> float:
    """Monte Carlo policy value using the known synthetic potential-outcome model.

    This oracle is only for benchmark evaluation and is unavailable to policy fitting/OPE.
    """

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    action = policy.actions(x)
    return float(np.mean(baseline_reward(x) + action * treatment_effect(x)))
