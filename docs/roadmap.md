# Roadmap

The first release emphasizes transparent small-state benchmarks. Future additions should increase modeling complexity only when the evaluation protocol remains auditable.

## Phase 1 — transparent foundations (implemented)

- exact finite-horizon dynamic programming;
- base-stock inventory baseline;
- tabular Q-learning;
- constrained capacity allocation with a hard overtime budget;
- shadow-price / Lagrangian-shaped Q-learning;
- offline behavior cloning;
- pessimistic tabular fitted-Q learning.

## Phase 2 — function approximation (implemented)

Implemented:

- DQN for regime-switching inventory with a neural Q-network, replay buffer, target network, and feasibility masking;
- PPO for dynamic flex-workforce allocation across three work centers;
- SAC for continuous energy-aware production-rate decisions with twin critics and short-horizon MPC-style comparison.

Future neural expansion:

- larger masked-action scheduling/resource-allocation benchmarks;
- multi-output or recurrent policies where partial history is operationally meaningful.

Every neural benchmark should retain an exact or optimization reference on a smaller validation regime.

## Phase 3 — constrained and safe RL (in progress)

Implemented:

- CMDP formulation for stochastic flex-workforce allocation;
- explicit reward critic and constraint critic;
- primal-dual / Lagrangian PPO with projected dual updates;
- expected service-violation budget reported separately from economic reward;
- safety shield / action repair based on expected next-period backlog;
- shield intervention rate reported separately from raw policy quality.

Implemented risk-aware extension:

- empirical VaR/CVaR reporting for rare-demand inventory;
- mean-optimal versus CVaR-optimal base-stock policy search;
- tail-weighted PPO as a transparent worst-episode training heuristic.

Remaining:

- formal chance-constrained policy optimization;
- continuous-action constrained RL for production/energy systems;
- stronger statistical validation of constraint satisfaction across training seeds.

## Partial observability / maintenance benchmark (implemented)

- hidden four-state equipment degradation model;
- noisy categorical condition sensor;
- exact Bayesian belief filtering;
- reactive sensor and posterior-threshold maintenance rules;
- discretized finite-horizon belief-state dynamic programming;
- belief-state DQN that never receives hidden health labels;
- held-out cost, failure exposure, intervention-rate, and belief-entropy evaluation.

This extends the separate fully observed maintenance MDP project rather than duplicating it.

## Phase 4 — offline RL (in progress)

Implemented:

- fixed logged inventory trajectories with no simulator access during fitting;
- behavior-policy and state/action coverage diagnostics;
- tabular behavior cloning;
- pessimistic tabular fitted-Q iteration;
- neural discrete CQL-style conservative offline Q-learning;
- target-policy support reporting;
- Fitted Q Evaluation (FQE) from logged transitions before simulator testing.

Remaining:

- IQL-style offline policy learning;
- richer off-policy evaluation diagnostics;
- offline RL on scheduling or maintenance logs;
- dataset-shift and behavior-policy sensitivity studies.

## Phase 5 — model-based and hybrid decision systems

- learned dynamics + model predictive control;
- RL selecting among optimization/heuristic operators;
- RL warm-starting or parameterizing mathematical optimization;
- digital-twin policy evaluation under distribution shift.

## Planned industrial benchmark families

- multi-echelon inventory;
- predictive maintenance with partial observability — implemented;
- energy-aware production;
- dynamic workforce/capacity allocation;
- rolling-horizon scheduling with stochastic arrivals;
- hybrid RL + OR decision systems.

## Candidate manufacturing benchmark families

The following additions are deliberately narrower than a generic list of RL-in-manufacturing applications. Each candidate should become a benchmark only if it supports an explicit sequential decision model, a credible non-RL baseline, reproducible uncertainty, and operational evaluation beyond training reward.

### Adaptive quality inspection and sampling

A policy chooses whether, when, and how intensively to inspect while defect risk evolves over time.

Useful formulations include:

- inspection / skip / escalate actions with explicit inspection cost;
- partial observability where latent process health is inferred from sparse quality signals;
- adaptive sampling rates under changing defect prevalence;
- active sensing where the policy selects which measurement or inspection station to use.

References should include Bayesian risk rules, fixed sampling plans, dynamic programming or optimal-stopping formulations where tractable, and transparent threshold policies.

### Multi-agent AMR material handling and intralogistics

A fleet-level benchmark should study dispatching, task allocation, congestion, charging, and just-in-time line-side delivery rather than only shortest-path navigation.

Evaluation should report:

- tardiness or line-starvation exposure;
- travel and waiting time;
- charger utilization;
- congestion / conflict events;
- fleet utilization and task completion rate.

Baselines should include rolling-horizon assignment, vehicle-routing or task-allocation heuristics, and optimization-based dispatch where scale permits.

### Dynamic assembly-line balancing and reconfiguration

The decision system reallocates tasks, workers, robots, or stations as product mix, processing times, disturbances, and resource availability change.

This benchmark family should distinguish:

- static line balancing solved once;
- dynamic re-balancing after disruptions or demand changes;
- modular line reconfiguration with setup / changeover penalties;
- human-robot task allocation when human constraints can be modeled explicitly.

Strong references include MILP / CP-SAT, large-neighborhood search, and rule-based rebalancing policies.

### Hierarchical multi-timescale production control

A hierarchical benchmark should connect decisions made at different time scales instead of treating planning, scheduling, and dispatching as unrelated tasks.

A possible hierarchy is:

```text
capacity / campaign targets
        ↓
rolling-horizon scheduling
        ↓
dispatching / production-rate control
```

The research question is whether hierarchical RL adds value over decomposed optimization, MPC, or rolling-horizon policies when uncertainty and execution feedback couple the layers.

### Circular manufacturing and remanufacturing decisions

Returned products or components can be inspected, repaired, remanufactured, harvested for parts, recycled, or discarded under uncertain condition and recovery yield.

A benchmark should expose the sequential value of information and recourse through:

- uncertain return quality;
- inspection decisions;
- repair / remanufacture / recycle routing;
- recovered-material inventory;
- capacity and service constraints.

References should include stochastic programming, approximate dynamic programming, threshold rules, or network-flow formulations on smaller validation instances.

### Digital-twin process-control track

Low-level process-control applications such as CNC machining, additive manufacturing, welding, injection molding, furnaces, and batch processes should not be added as isolated RL demonstrations.

They become suitable benchmark candidates only when the repository can provide:

- a validated simulator, surrogate, or digital twin with declared fidelity limits;
- physically meaningful state and action bounds;
- safety / quality constraints that remain explicit;
- classical-control, Bayesian-optimization, MPC, or response-surface baselines;
- evaluation under sensor noise, drift, disturbances, and model mismatch.

This gate keeps actuator-level RL from being presented as useful merely because it can optimize a synthetic reward.
