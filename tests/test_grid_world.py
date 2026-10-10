"""Tests for the vectorised grid world."""

import numpy as np
import pytest

from cce.envs.grid_world import VectorGridWorld, GridWorldConfig


def test_grid_world_reset():
    cfg = GridWorldConfig(grid_size=10, num_agents=2, fov_size=3, max_steps=10)
    env = VectorGridWorld(cfg, num_envs=4)
    obs = env.reset()

    assert "local_map" in obs
    assert obs["local_map"].shape == (4, 2, 3, 3)
    assert "visited_self" in obs
    assert obs["visited_self"].shape == (4, 2, 3, 3)
    assert "self_pos" in obs
    assert obs["self_pos"].shape == (4, 2, 2)


def test_grid_world_step():
    cfg = GridWorldConfig(grid_size=10, num_agents=2, fov_size=3, max_steps=10)
    env = VectorGridWorld(cfg, num_envs=4)
    obs = env.reset()

    actions_move = np.zeros((4, 2), dtype=np.int64)  # all move up
    send_flags = np.zeros((4, 2), dtype=bool)

    obs_new, rewards, dones, infos = env.step(actions_move, send_flags)

    assert obs_new["local_map"].shape == (4, 2, 3, 3)
    assert rewards.shape == (4, 2)
    assert dones.shape == (4,)
    assert "coverage" in infos


def test_grid_world_movement():
    cfg = GridWorldConfig(grid_size=10, num_agents=1, fov_size=1, max_steps=10)
    env = VectorGridWorld(cfg, num_envs=1)
    obs = env.reset()

    # Move right
    actions_move = np.array([[3]], dtype=np.int64)
    send_flags = np.zeros((1, 1), dtype=bool)

    initial_pos = env.agent_pos[0, 0].copy()
    obs_new, rewards, dones, infos = env.step(actions_move, send_flags)
    new_pos = env.agent_pos[0, 0]

    # Should have moved right by 1 (or stayed if blocked)
    assert np.all(np.abs(new_pos - initial_pos) <= 1)


def test_grid_world_coverage_increases():
    cfg = GridWorldConfig(grid_size=10, num_agents=2, fov_size=3, max_steps=20)
    env = VectorGridWorld(cfg, num_envs=1)
    env.reset()

    # Move around randomly
    rng = np.random.default_rng(42)
    coverages = []
    for t in range(20):
        actions_move = rng.integers(0, 4, size=(1, 2))
        send_flags = np.zeros((1, 2), dtype=bool)
        obs, rewards, dones, infos = env.step(actions_move, send_flags)
        coverages.append(infos["coverage"][0])

    # Coverage should generally increase
    assert coverages[-1] >= coverages[0]


def test_grid_world_communication():
    cfg = GridWorldConfig(grid_size=10, num_agents=2, fov_size=3, max_steps=10, comm_range=3)
    env = VectorGridWorld(cfg, num_envs=1)
    env.reset()

    # Comm mask should have correct shape
    in_range = env.get_comm_mask()
    assert in_range.shape == (1, 2, 2)
    # Diagonal should be False (no self-communication)
    assert not in_range[0, 0, 0]
    assert not in_range[0, 1, 1]
