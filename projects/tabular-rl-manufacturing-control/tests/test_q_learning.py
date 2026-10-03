import numpy as np

from manufacturing_rl.environment import ManufacturingEnv
from manufacturing_rl.q_learning import QLearningAgent


def test_discretization_clips_both_bounds():
    agent = QLearningAgent(n_actions=5)
    state = np.array([-0.5, 0.0, 0.51, 1.0, 1.7])
    idx = agent.discretize_state(state)
    assert idx[0] == 0
    assert idx[1] == 0
    assert idx[2] == 5
    assert idx[3] == 9
    assert idx[4] == 9


def test_short_training_is_reproducible():
    env1 = ManufacturingEnv()
    env2 = ManufacturingEnv()
    agent1 = QLearningAgent(n_actions=env1.n_actions, seed=9)
    agent2 = QLearningAgent(n_actions=env2.n_actions, seed=9)

    rewards1 = agent1.train(env1, episodes=8, seed=100)
    rewards2 = agent2.train(env2, episodes=8, seed=100)

    assert np.allclose(rewards1, rewards2)
    assert np.array_equal(agent1.q_table, agent2.q_table)
