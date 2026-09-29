import numpy as np

from causal_policy_ope import (
    cross_fitted_policy_value,
    estimate_from_predictions,
    estimate_policy_value,
    fit_budget_policy,
    fit_nuisance,
    generate_logged_data,
    true_policy_value,
)


def test_budget_policy_respects_training_treatment_fraction() -> None:
    data = generate_logged_data(seed=1, n=2500)
    policy = fit_budget_policy(data, budget=0.30)
    rate = np.mean(policy.actions(data.x))
    assert abs(rate - 0.30) < 0.02


def test_doubly_robust_reduces_to_direct_method_with_exact_outcomes() -> None:
    action = np.array([0, 1, 1, 0, 1, 0])
    target = np.array([1, 1, 0, 0, 1, 0])
    propensity1 = np.array([0.3, 0.7, 0.6, 0.4, 0.8, 0.2])
    mu0 = np.array([1.0, 1.2, 0.8, 1.4, 1.1, 0.9])
    mu1 = np.array([1.5, 1.7, 1.0, 1.9, 1.8, 1.3])
    reward = np.where(action == 1, mu1, mu0)

    estimates = estimate_from_predictions(
        action,
        reward,
        target,
        propensity1,
        mu0,
        mu1,
    )

    assert np.isclose(estimates["dr"], estimates["dm"], atol=1e-12)
    assert estimates["effective_sample_size"] > 0.0


def test_end_to_end_ope_is_finite() -> None:
    policy_train = generate_logged_data(seed=2, n=1800)
    nuisance_train = generate_logged_data(seed=3, n=1800)
    evaluation = generate_logged_data(seed=4, n=2000)

    policy = fit_budget_policy(policy_train, budget=0.40)
    nuisance = fit_nuisance(nuisance_train)
    estimates = estimate_policy_value(evaluation, policy, nuisance)

    for name in ("dm", "ipw", "snips", "dr"):
        assert np.isfinite(estimates[name])
    assert estimates["effective_sample_size"] > 0.0
    assert estimates["max_importance_weight"] >= 1.0


def test_synthetic_oracle_is_evaluation_only_and_finite() -> None:
    data = generate_logged_data(seed=5, n=2000)
    policy = fit_budget_policy(data, budget=0.35)
    value = true_policy_value(policy, seed=99, n=20_000)
    assert np.isfinite(value)


def test_logged_propensities_have_overlap() -> None:
    data = generate_logged_data(seed=6, n=1000)
    assert np.min(data.true_propensity) >= 0.05
    assert np.max(data.true_propensity) <= 0.95



def test_cross_fitted_dr_is_finite_and_reports_interval() -> None:
    policy_train = generate_logged_data(seed=21, n=1800)
    evaluation = generate_logged_data(seed=22, n=2400)
    policy = fit_budget_policy(policy_train, budget=0.35)

    estimates = cross_fitted_policy_value(evaluation, policy, folds=4, seed=9)

    assert np.isfinite(estimates["dr"])
    assert estimates["dr_standard_error"] > 0.0
    assert estimates["dr_ci95_lower"] < estimates["dr"] < estimates["dr_ci95_upper"]
    assert estimates["folds"] == 4.0
