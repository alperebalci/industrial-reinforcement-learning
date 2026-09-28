# Research Notes

## Relationship to offline RL

The umbrella repository already contains offline sequential decision-making, including FQE. This project intentionally isolates the contextual/causal case where each observation has one context, one action, and one outcome.

That simpler setting makes propensity weighting, overlap, direct outcome modeling, and doubly robust correction directly inspectable.

## What "causal" means here

The synthetic design satisfies conditional exchangeability because the context contains every variable used by the behavior policy and potential-outcome functions. The code does not infer that property from data. Applying the estimators to a real observational dataset requires a domain-specific identification argument.

## Policy-learning boundary

The budget-threshold learner uses heterogeneous-effect scores from action-specific outcome regressions. Athey and Wager (2021) develop a more general policy-learning framework based on doubly robust scores and constrained policy classes. Their paper motivates the research direction; this v0.1 should not be described as a reproduction of their full method.
