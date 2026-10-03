"""Train and benchmark tabular Q-learning against transparent controls."""

from __future__ import annotations

import argparse
import json

from .baselines import fixed_speed_policy, threshold_policy
from .environment import ManufacturingEnv
from .evaluation import evaluate_policy
from .q_learning import QLearningAgent


def run_benchmark(
    training_episodes: int,
    evaluation_episodes: int,
    seed: int,
) -> dict:
    env = ManufacturingEnv()
    agent = QLearningAgent(n_actions=env.n_actions, seed=seed)
    agent.train(env, episodes=training_episodes, seed=seed * 100)

    def learned_policy(state):
        return agent.choose_action(state, training=False)

    policies = {
        "fixed_speed": fixed_speed_policy,
        "threshold_rule": threshold_policy,
        "q_learning": learned_policy,
    }
    evaluation_seed = seed * 10_000
    return {
        name: evaluate_policy(
            policy,
            episodes=evaluation_episodes,
            seed=evaluation_seed,
        )
        for name, policy in policies.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-episodes", type=int, default=600)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_benchmark(
        args.training_episodes,
        args.episodes,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
