"""Tests for MAPPO (integration test)."""

import numpy as np
import pytest
import torch

from cce.envs.grid_world import VectorGridWorld, GridWorldConfig
from cce.algo.mappo import MAPPO


def test_mappo_init():
    cfg = GridWorldConfig(grid_size=10, num_agents=2, fov_size=3, max_steps=10,
                          message_embed_dim=4)
    env = VectorGridWorld(cfg, num_envs=4, message_embed_dim=4)
    agent = MAPPO(env, actor_hidden=32, critic_hidden=64, n_codes=8, code_dim=4, n_slots=2)

    assert agent.actor is not None
    assert agent.critic is not None
    assert agent.buffer is not None


def test_mappo_select_actions():
    # Keep the env's message width consistent with the agent's code_dim.
    cfg = GridWorldConfig(grid_size=10, num_agents=2, fov_size=3, max_steps=10,
                          message_embed_dim=4)
    env = VectorGridWorld(cfg, num_envs=4)
    agent = MAPPO(env, actor_hidden=32, critic_hidden=64, n_codes=8, code_dim=4, n_slots=2)

    obs = env.reset()
    actions_move, actions_send, log_probs, values, info = agent.select_actions(obs)

    assert actions_move.shape == (4, 2)
    assert actions_send.shape == (4, 2)
    assert log_probs.shape == (4, 2)
    assert values.shape == (4, 2)
    assert "message_indices" in info
    assert "vq_loss" in info
    assert "perplexity" in info


def test_mappo_train_step():
    # message_embed_dim must equal the agent's code_dim: the environment derives
    # the message_embed observation width from it, and the rollout buffer sizes
    # itself from env.obs_spec. train.py/evaluate.py wire these identically.
    cfg = GridWorldConfig(grid_size=10, num_agents=2, fov_size=3, max_steps=10,
                          message_embed_dim=4)
    env = VectorGridWorld(cfg, num_envs=4)
    agent = MAPPO(
        env, actor_hidden=32, critic_hidden=64, n_codes=8, code_dim=4, n_slots=2,
        rollout_length=10, batch_size=160, num_epochs=2
    )

    stats = agent.train_step()

    assert "coverage" in stats
    assert "actor_loss" in stats
    assert "critic_loss" in stats
    assert "entropy" in stats
    assert "vq_loss" in stats
