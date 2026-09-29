# Causal Policy Learning and Off-Policy Evaluation

A compact observational-data benchmark for **policy learning under a treatment budget** and **off-policy evaluation (OPE)** with direct, inverse-propensity, self-normalized, and doubly robust estimators.

This project fills a different gap from the umbrella repository's offline-RL work. It studies a one-step contextual treatment-allocation problem rather than a sequential MDP.

## Research question

> Given historical decisions made by a non-random behavior policy, can we learn a constrained target policy and estimate its value without deploying it first?

The benchmark separates three datasets:

1. **policy-training data** — learn a treatment rule;
2. **nuisance-training data** — estimate behavior propensity and outcome regressions;
3. **held-out logged evaluation data** — estimate the fixed target policy's value.

A fourth synthetic Monte Carlo oracle is used only to measure estimator error in the benchmark. It is never available to policy learning or OPE.

## Observational data model

Contexts `X` are observed before treatment. The behavior policy assigns a binary action with a context-dependent probability

```text
e(X) = P(A=1 | X),
```

so the logged treatment groups are not marginally randomized.

Potential outcomes are generated from a heterogeneous response surface. All variables that drive treatment assignment in the synthetic benchmark are included in `X`; therefore the experiment is deliberately a **selection-on-observables** setting.

## Policy learning

Two action-conditional ridge regressions estimate

```text
mu_0(x) = E[Y | X=x, A=0]
mu_1(x) = E[Y | X=x, A=1].
```

The estimated treatment score is

```text
tau_hat(x) = mu_1(x) - mu_0(x).
```

For a treatment budget `q`, the policy threshold is chosen on policy-training data so that approximately the top `q` fraction of estimated treatment scores receive action 1.

This is a transparent budget-constrained policy learner, not a reproduction of every algorithm in Athey and Wager (2021).

## Off-policy evaluation

A logistic regression estimates the behavior propensity and separate outcome regressions estimate `mu_0` and `mu_1`.

For a deterministic target policy `pi(x)`, the project reports:

- **DM** — direct method / outcome regression;
- **IPW** — inverse propensity weighting;
- **SNIPS** — self-normalized importance weighting;
- **DR** — doubly robust estimation.

The DR estimate uses

```text
mu_hat(X, pi(X))
+ I[A = pi(X)] / e_hat(A|X) * (Y - mu_hat(X,A)).
```

Diagnostics include target/behavior action match rate, effective sample size, minimum observed-action propensity, and maximum importance weight.

## Identification assumptions

OPE from observational data is not automatically causal. The benchmark relies on:

- consistency / well-defined treatments;
- conditional exchangeability (no unobserved confounding after conditioning on `X`);
- positivity / overlap for actions required by the target policy;
- stable data-generating conditions between logged data and the target deployment population.

Propensity clipping is an engineering variance control and does not repair a true overlap violation.

## Run

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m causal_policy_ope.experiment
```

The experiment reports OPE estimates alongside a synthetic oracle value computed from the known data generator. In real observational applications that oracle is unavailable.

## Scope boundary

Implemented in v0.1:

- binary treatment;
- contextual one-step policy;
- treatment-budget threshold policy;
- estimated propensity model;
- action-specific outcome regression;
- DM, IPW, SNIPS, and DR estimators;
- support/weight diagnostics;
- strict sample separation between policy learning, nuisance fitting, and OPE evaluation;
- K-fold cross-fitted nuisance prediction for fixed-policy OPE;
- doubly robust influence-score standard error and normal-approximation 95% interval.

Natural extensions:

- doubly robust policy scores for direct empirical-welfare maximization;
- policy trees and other interpretable constrained classes;
- sensitivity analysis for unobserved confounding;
- multi-action policies;
- confidence intervals / influence-function inference;
- sequential OPE for MDPs.

## References

- Susan Athey and Stefan Wager (2021), "Policy Learning With Observational Data," *Econometrica* 89(1):133–161. DOI: 10.3982/ECTA15732.
- Miroslav Dudik et al., doubly robust policy evaluation for contextual bandits.
- Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudik (2017), "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits," ICML.

## License

This project inherits the umbrella repository's license.
