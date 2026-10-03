"""Tabular Q-learning for the manufacturing control environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import ManufacturingEnv


@dataclass(frozen=True)
class QLearningConfig:
    bins_per_dimension: int = 10
    learning_rate: float = 0.12
    discount_factor: float = 0.97
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_per_episode: float = 0.992


class QLearningAgent:
    def __init__(
        self,
        n_actions: int,
        config: QLearningConfig | None = None,
        seed: int = 0,
    ):
        self.config = config or QLearningConfig()
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        bins = self.config.bins_per_dimension
        self.q_table = np.zeros(
            (bins, bins, bins, bins, bins, n_actions), dtype=np.float32
        )
        self.epsilon = self.config.epsilon_start
        self.training_rewards: list[float] = []

    def discretize_state(self, state: np.ndarray) -> tuple[int, ...]:
        bins = self.config.bins_per_dimension
        clipped = np.clip(np.asarray(state, dtype=float), 0.0, 1.0)
        indices = np.floor(clipped * bins).astype(int)
        indices = np.clip(indices, 0, bins - 1)
        return tuple(int(i) for i in indices)

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        discrete = self.discretize_state(state)
        values = self.q_table[discrete]
        best = np.flatnonzero(values == values.max())
        return int(best[0] if not training else self.rng.choice(best))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        current_idx = self.discretize_state(state)
        next_idx = self.discretize_state(next_state)
        current = float(self.q_table[current_idx][action])
        continuation = 0.0 if done else float(np.max(self.q_table[next_idx]))
        target = reward + self.config.discount_factor * continuation
        updated = current + self.config.learning_rate * (target - current)
        self.q_table[current_idx][action] = updated

    def train(
        self,
        env: ManufacturingEnv,
        episodes: int = 600,
        seed: int = 2026,
    ) -> list[float]:
        self.training_rewards = []
        self.epsilon = self.config.epsilon_start
        for episode in range(episodes):
            state = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            while not done:
                action = self.choose_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            self.training_rewards.append(float(total_reward))
            self.epsilon = max(
                self.config.epsilon_min,
                self.epsilon * self.config.epsilon_decay_per_episode,
            )
        return self.training_rewards
