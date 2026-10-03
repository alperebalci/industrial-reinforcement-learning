# Tabular RL for Joint Manufacturing Control

A compact reinforcement-learning benchmark for joint production-rate, preventive-maintenance, and quality-inspection decisions in a stochastic manufacturing line.

The project started from a simple five-state Q-learning demonstrator and turns it into an auditable industrial-decision benchmark. The central question is not whether Q-learning can maximize a synthetic reward. It is whether a learned discrete policy can improve joint operating decisions relative to transparent engineering rules when production, degradation, inspection, and downtime interact.

## Decision problem

The observable state contains five normalized variables:

    machine temperature
    machine wear
    workload
    intrinsic process quality
    production rate

The controller chooses one of five actions:

    0  decrease production speed
    1  hold production speed
    2  increase production speed
    3  perform preventive maintenance
    4  run intensive quality inspection

The simulator deliberately separates intrinsic process quality from inspection effectiveness. Inspection does not magically improve the underlying process; it increases defect detection, creates rework/inspection cost, and reduces effective throughput. Preventive maintenance consumes a full period of downtime and therefore has an explicit opportunity cost.

## Compared policies

Three policies are evaluated on matched random seeds:

- fixed_speed: hold nominal speed, with maintenance only at an emergency threshold;
- threshold_rule: transparent engineering thresholds for speed, maintenance, and inspection;
- q_learning: tabular Q-learning over a discretized five-dimensional state.

A negative RL result is valid. The benchmark is designed to reveal when a simple operating rule remains stronger than the learned policy.

## Q-learning implementation

Each normalized state dimension is discretized into 10 bins. The Q-table therefore has 10 x 10 x 10 x 10 x 10 x 5 entries.

State indices are clipped on both lower and upper bounds, avoiding accidental negative NumPy indexing. Exploration uses epsilon-greedy action selection with episode-level epsilon decay rather than decaying after every plant transition.

## Manufacturing economics

Per-period profit includes production revenue, operating cost, escaped-defect cost, detected-defect rework cost, intensive-inspection cost, and preventive-maintenance cost.

The physical state evolves with stochastic shocks plus production-load effects on temperature, wear, and process quality. Failure can terminate the episode if temperature or wear exceeds its declared limit.

## Evaluation protocol

Policies are evaluated with common episode seeds. Every step draws the same fixed-size exogenous shock vector, so different policies face matched stochastic scenarios even when they take different actions.

Reported metrics include:

- mean and lower-tail profit;
- throughput;
- escaped and detected defects;
- inspection and rework cost;
- maintenance cost and maintenance count;
- intensive-inspection count;
- maximum wear;
- minimum process quality;
- failure rate;
- episode length.

Reward is therefore not treated as the only performance measure.

## Installation

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e '.[test]'

## Run tests

    pytest

## Run benchmark

    python -m manufacturing_rl.benchmark --training-episodes 600 --episodes 100 --seed 2026

The command prints a JSON KPI report for all three policies.

## Repository layout

    src/manufacturing_rl/
      environment.py   stochastic manufacturing simulator and action definitions
      q_learning.py    tabular Q-learning and bounded discretization
      baselines.py     fixed-speed and threshold engineering controls
      evaluation.py    matched-seed multi-KPI evaluation
      benchmark.py     train/evaluate command-line benchmark

    tests/
      test_environment.py
      test_q_learning.py
      test_evaluation.py

## Claims boundary

This project is an educational/research benchmark, not a calibrated factory digital twin. The degradation, quality, and economic parameters are synthetic. It does not claim production deployment readiness, a general optimal policy, or that tabular Q-learning scales to high-dimensional industrial control.

Useful extensions include calibrated degradation/failure models, backlog and due dates, partially observed machine health, offline RL from plant logs, constrained RL, small-state dynamic-programming references, and multi-machine production systems.
