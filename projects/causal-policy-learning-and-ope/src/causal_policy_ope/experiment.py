"""Held-out observational policy-learning and OPE benchmark."""

from __future__ import annotations

import json

import numpy as np

from .data import generate_logged_data, true_policy_value
from .nuisance import fit_nuisance
from .ope import estimate_policy_value
from .policy import fit_budget_policy


def run_experiment(seed: int = 17) -> dict[str, object]:
    policy_train = generate_logged_data(seed, 4000)
    ope_train = generate_logged_data(seed + 1, 4000)
    evaluation = generate_logged_data(seed + 2, 5000)

    policy = fit_budget_policy(policy_train, budget=0.40)
    nuisance = fit_nuisance(ope_train)
    estimates = estimate_policy_value(evaluation, policy, nuisance)
    oracle_value = true_policy_value(policy, seed=seed + 10_000)

    evaluation_actions = policy.actions(evaluation.x)
    return {
        "policy_budget": policy.budget,
        "evaluation_treatment_rate": float(np.mean(evaluation_actions)),
        "ope": estimates,
        "synthetic_oracle_value": oracle_value,
        "absolute_error": {
            name: float(abs(estimates[name] - oracle_value))
            for name in ("dm", "ipw", "snips", "dr")
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
