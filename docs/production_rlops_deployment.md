# Production Deployment of Industrial RL

This note describes how a reinforcement-learning policy can be connected to a manufacturing system without treating the live plant as an unrestricted training environment.

The key distinction is:

```text
real-time inference != real-time policy learning
```

A policy may make decisions frequently while policy updates happen much more slowly, under explicit validation and rollback rules. In many factories, that separation is the difference between a useful deployment architecture and unsafe online experimentation.

## What "RLOps" means here

"RLOps" is used here as an informal engineering label for the operational discipline around deployed RL systems. It is not treated as a standardized replacement for MLOps.

A production RL stack combines concerns from:

- MLOps: model/version management, reproducibility, observability, deployment and rollback;
- control engineering: stability, latency, operating envelopes and fallback control;
- Operations Research: constraints, optimization baselines, feasibility and multi-objective trade-offs;
- OT/industrial automation: PLC/SCADA/MES integration, historian data, deterministic execution and cyber-physical safety;
- RL: state/action/reward semantics, exploration, off-policy data, policy evaluation and non-stationarity.

The operational object being versioned is therefore larger than a neural-network file. A deployable policy release should identify at least:

```text
policy weights
+ observation/state schema
+ action schema and units
+ reward/constraint definition
+ feature transforms
+ safety-filter configuration
+ training dataset / simulator version
+ evaluation suite
+ fallback policy
```

## Recommended control architecture

A generic plant-side architecture is:

```text
Sensors / PLC / SCADA / MES
          |
          v
state estimation + feature validation
          |
          v
RL policy inference
          |
          v
safety shield / action projection / rule checks
          |
          v
supervisory setpoint or operational action
          |
          v
PLC / machine / robot / line
          |
          +----------------------+
          |                      |
          v                      v
historian / event log       KPI + constraint log
          |                      |
          +----------+-----------+
                     v
          offline / scheduled learning
                     |
                     v
          candidate policy validation
                     |
                     v
          shadow -> assisted -> controlled rollout
```

For most manufacturing use cases, the RL policy should initially sit above the fastest actuator loop. The PLC, drive controller, robot controller or MPC layer can retain hard-real-time interlocks and local regulation, while RL changes supervisory variables such as:

- production-rate targets;
- machine-mode selection;
- dispatching decisions;
- buffer thresholds;
- energy setpoints;
- maintenance actions;
- inspection intensity;
- routing or resource-allocation decisions.

Low-level actuator control is possible in research and specialized applications, but it requires stronger evidence for timing, stability, fault handling and safe exploration than a planning-level benchmark.

## Sampling, decision and learning clocks

There should not be one universal "RL frequency." Separate at least three clocks.

### 1. Observation clock

Sensor data may arrive at milliseconds, seconds or minutes depending on the process.

Examples:

- vibration or servo signals: high-frequency raw acquisition;
- temperatures, pressures, WIP counts: slower process sampling;
- quality, order or maintenance state: event-driven or cycle-level updates.

The RL state should normally consume engineered or aggregated signals at the resolution justified by the decision problem, rather than every raw sensor sample.

### 2. Decision clock

The action frequency should match the plant's controllable dynamics.

Examples:

- production-rate or energy supervisory control: seconds to minutes;
- dispatching: on job arrival / machine release;
- scheduling: minutes to hours;
- maintenance: hours to days or condition-triggered.

A policy should not be called at 100 ms merely because the infrastructure can do so.

### 3. Learning clock

Policy updates should usually be slower than inference.

Common patterns include:

- offline retraining from historian data;
- scheduled updates after a shift/day/week;
- event-triggered retraining after validated drift;
- constrained online adaptation with a frozen safety layer;
- fleet learning across similar assets followed by site-specific validation.

Every candidate update should be treated as a new controller release, not as an invisible weight mutation.

## Deployment stages

### Stage 0 — simulator / digital twin

Train and stress-test before plant actuation.

Required tests should include:

- nominal scenarios;
- demand/process disturbances;
- sensor noise and missing values;
- parameter drift;
- actuator saturation;
- delayed observations/actions;
- model mismatch;
- rare but plausible safety events.

The simulator should expose its fidelity limits. A digital twin is not evidence of real-plant safety by itself.

### Stage 1 — shadow mode

The policy receives live states and produces actions, but actions are not executed.

Log:

- proposed action;
- incumbent-controller action;
- estimated advantage;
- constraint margin;
- policy confidence / support diagnostics;
- disagreement rate;
- downstream KPI outcome.

This stage tests integration, state semantics and timing without changing production.

### Stage 2 — assisted mode

The policy may recommend actions to an operator or automation layer.

Use this stage to measure:

- operator acceptance/override rate;
- reasons for override;
- action feasibility;
- latency;
- repeated disagreement patterns;
- whether the policy exploits simulator artifacts absent in the plant.

### Stage 3 — bounded autonomy

Allow automatic actions only inside a declared operating envelope.

Typical controls:

- hard action bounds;
- rate-of-change limits;
- invariant or rule-based interlocks;
- MPC or optimization-based safety filter;
- fallback to incumbent policy;
- watchdog timeout;
- automatic rollback on KPI or constraint degradation.

### Stage 4 — broader autonomy

Expand the envelope only after evidence accumulates. New operating regions, products, recipes, machines or seasons should be treated as distribution-shift events requiring revalidation.

## Safe online adaptation

Online learning is not categorically impossible, but unrestricted exploration on a production asset is usually the wrong starting point.

Safer alternatives include:

- offline RL from historian trajectories;
- model-based RL with a validated simulator;
- conservative policy improvement;
- constrained/safe RL;
- residual RL around a trusted controller;
- contextual bandits for bounded one-step decisions;
- policy adaptation only inside an action envelope;
- exploration in simulation and exploitation in the plant.

If live exploration is allowed, specify:

```text
who permits it
where it is permitted
which actions are forbidden
maximum deviation from baseline
abort conditions
rollback policy
data-retention requirements
how exploration cost is charged
```

"Exploration" should be an engineered operating mode, not an epsilon-greedy default.

## Reward and constraints in production

A production objective should not collapse every concern into one opaque scalar.

Separate:

### Economic objective

Examples:

- throughput;
- tardiness;
- WIP;
- energy;
- scrap;
- overtime;
- changeover;
- maintenance cost.

### Hard or regulated constraints

Examples:

- temperature/pressure limits;
- buffer capacity;
- robot/cell exclusion zones;
- maximum ramp rate;
- quality limits;
- equipment health limits;
- contractual or service-level constraints.

### Risk metrics

Examples:

- probability of violation;
- worst-case or tail cost;
- CVaR;
- maximum consecutive overload periods;
- recovery time after disturbance.

Reward changes are policy changes. Changing weights because management priorities changed should trigger a new validation cycle.

## Monitoring after deployment

A production RL monitor should cover four categories.

### Data health

- missing/stale sensors;
- schema mismatch;
- unit changes;
- timestamp skew;
- distribution drift;
- out-of-range values.

### Policy health

- action distribution;
- action saturation;
- disagreement versus baseline;
- safety-filter intervention rate;
- out-of-support actions;
- inference latency.

### Operational performance

- throughput;
- service/tardiness;
- WIP;
- energy;
- quality/scrap;
- downtime;
- maintenance burden.

### Safety and governance

- constraint violations;
- near-limit operation;
- fallback activations;
- operator overrides;
- policy version active at each decision;
- audit trail linking state -> proposed action -> filtered action -> executed action -> outcome.

## Non-stationarity

Factories are non-stationary because of:

- tool wear;
- maintenance;
- new product mix;
- operator behavior;
- seasonal ambient conditions;
- raw-material changes;
- supplier changes;
- calibration drift;
- upstream/downstream bottleneck changes.

The first response to drift should not automatically be "learn online." Diagnose whether the issue is:

1. bad data,
2. changed constraints,
3. changed economics,
4. changed dynamics,
5. changed action availability,
6. insufficient support in the training data.

The remediation may be recalibration, feature repair, re-optimization, simulator update or policy retraining rather than online gradient updates.

## Integration patterns

### Supervisory RL over PLC / SCADA

The RL service writes bounded setpoints or mode commands through an industrial integration layer. PLC logic retains deterministic interlocks.

### RL + MPC

RL can choose references, weights, horizons, modes or operating regions while MPC enforces dynamic feasibility.

### RL + mathematical optimization

RL can warm-start, parameterize or choose among optimization policies. Optimization remains the feasibility authority.

### Offline-RL recommendation service

Historical data train a policy that is deployed first as a recommender. This is appropriate when action coverage is narrow and online exploration is unacceptable.

### Multi-agent plant systems

Do not start with independent agents on every machine. First establish:

- ownership of shared constraints;
- conflict resolution;
- communication failure behavior;
- global-versus-local reward decomposition;
- coordinator/fallback logic.

Multi-agent RL is justified only when distributed control structure itself is part of the research question.

## Minimal production-readiness checklist

Before autonomous execution, answer yes to all relevant items:

- Is the decision problem genuinely sequential?
- Is a strong rule/MPC/optimization baseline documented?
- Are state and action units unambiguous?
- Are hard constraints enforced outside the learned reward?
- Is there a deterministic fallback?
- Can the policy be rolled back immediately?
- Can every executed action be traced to a policy version?
- Has shadow-mode disagreement been reviewed?
- Has distribution shift been stress-tested?
- Are operator override and abort paths tested?
- Is policy-update frequency separated from inference frequency?
- Is online exploration either disabled or explicitly bounded?
- Are evaluation seeds/scenarios separated from training?
- Are KPI gains reported together with risk and constraint metrics?

## Research implication for this repository

A useful next benchmark is not "an RL agent connected to MQTT." It is a reproducible deployment experiment that compares:

```text
incumbent controller
vs
RL in shadow mode
vs
RL behind a safety filter
vs
updated RL after detected drift
```

and reports:

- operational KPI change;
- constraint violations;
- intervention/override rate;
- action-support diagnostics;
- latency;
- rollback events;
- sensitivity to sensor/process drift.

That would make deployment engineering itself part of the benchmark rather than treating integration as an afterthought.

## Further reading

- Dogru et al., "Reinforcement Learning in Process Industries: Review and Perspective," IEEE/CAA Journal of Automatica Sinica, 2024. https://doi.org/10.1109/JAS.2024.124227
- Schäfer et al., "An Architecture for Deploying Reinforcement Learning in Industrial Environments," 2023. https://arxiv.org/abs/2306.01420
- "Reinforcement Learning for Autonomous Process Control in Industry 4.0: Advantages and Challenges," Applied Artificial Intelligence, 2024. https://doi.org/10.1080/08839514.2024.2383101
- "Reinforcement learning for autonomous production planning and control: A systematic literature review," Journal of Manufacturing Systems, 2026. The review reports a persistent gap between simulation-heavy academic validation and sustained live closed-loop deployment.
