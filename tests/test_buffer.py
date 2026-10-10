"""Tests for the rollout buffer."""

import numpy as np
import pytest
import torch

from cce.algo.buffer import RolloutBuffer


def test_buffer_insert():
    obs_spec = {"obs1": (3, 3), "obs2": (2,)}
    buf = RolloutBuffer(num_envs=4, num_agents=2, rollout_length=10, obs_spec=obs_spec)

    obs = {"obs1": np.random.randn(4, 2, 3, 3).astype(np.float32),
           "obs2": np.random.randn(4, 2, 2).astype(np.float32)}
    actions_move = np.zeros((4, 2), dtype=np.int64)
    actions_send = np.zeros((4, 2), dtype=np.int64)
    log_probs = np.zeros((4, 2), dtype=np.float32)
    rewards = np.zeros((4, 2), dtype=np.float32)
    values = np.zeros((4, 2), dtype=np.float32)
    dones = np.zeros(4, dtype=bool)
    comm_cost = np.zeros(4, dtype=np.float32)

    buf.insert(obs, actions_move, actions_send, log_probs, rewards, values, dones, comm_cost)

    assert buf.step == 1


def test_buffer_compute_returns():
    obs_spec = {"obs1": (3, 3)}
    buf = RolloutBuffer(num_envs=2, num_agents=2, rollout_length=5, obs_spec=obs_spec,
                        gamma=0.99, gae_lambda=0.95)

    # Fill with dummy data
    for t in range(5):
        obs = {"obs1": np.random.randn(2, 2, 3, 3).astype(np.float32)}
        buf.insert(obs, np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64),
                   np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32),
                   np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=bool),
                   np.zeros(2, dtype=np.float32))

    next_value = np.zeros((2, 2), dtype=np.float32)
    next_done = np.zeros(2, dtype=bool)

    buf.compute_returns(next_value, next_done)

    assert buf.returns.shape == (5, 2, 2)
    assert buf.advantages.shape == (5, 2, 2)


def test_buffer_get_batches():
    obs_spec = {"obs1": (3, 3)}
    buf = RolloutBuffer(num_envs=4, num_agents=2, rollout_length=10, obs_spec=obs_spec)

    # Fill
    for t in range(10):
        obs = {"obs1": np.random.randn(4, 2, 3, 3).astype(np.float32)}
        buf.insert(obs, np.zeros((4, 2), dtype=np.int64), np.zeros((4, 2), dtype=np.int64),
                   np.zeros((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32),
                   np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=bool),
                   np.zeros(4, dtype=np.float32))

    buf.compute_returns(np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=bool))

    batches = list(buf.get_batches(batch_size=20, num_epochs=1, shuffle=False))
    assert len(batches) > 0
    assert "obs_obs1" in batches[0]
    assert "returns" in batches[0]
