"""Tests for the budget controller."""

import pytest
import torch

from cce.comm.budget import BudgetController


def test_budget_controller_policy_mode():
    bc = BudgetController(mode="policy", target_beta=0.25, pid_kp=0.5)
    assert bc.mode == "policy"
    assert bc.target_beta == 0.25

    bias = bc.get_send_logit_bias()
    assert bias.item() == 0.0


def test_budget_controller_utility_mode():
    bc = BudgetController(mode="utility", target_beta=0.5)
    assert bc.mode == "utility"

    threshold = bc.get_threshold()
    assert threshold.item() == 0.0


def test_budget_controller_update():
    bc = BudgetController(mode="policy", target_beta=0.25, pid_kp=0.5, pid_ki=0.0, pid_kd=0.0)

    # Simulate an episode with average cost 0.5 (over budget)
    bc.reset_episode()
    for _ in range(10):
        bc.on_step(0.5)

    stats = bc.on_episode_end()
    assert stats["actual_beta"] == 0.5
    assert stats["target_beta"] == 0.25
    assert stats["error"] > 0
    assert stats["lambda"] > 0  # Should increase lambda to reduce sending


def test_budget_controller_converges():
    bc = BudgetController(mode="policy", target_beta=0.25, pid_kp=0.5, pid_ki=0.1, pid_kd=0.0)

    # Simulate multiple episodes
    for _ in range(20):
        bc.reset_episode()
        for _ in range(10):
            bc.on_step(0.5)
        bc.on_episode_end()

    # Lambda should have increased significantly
    assert bc.lambda.item() > 1.0


def test_budget_controller_decide_send_utility():
    bc = BudgetController(mode="utility", target_beta=0.5)
    bc.lambda.fill_(0.3)  # Set threshold to 0.3

    utility = torch.tensor([0.1, 0.5, 0.2, 0.8])
    hard, soft = bc.decide_send_utility(utility)

    assert hard.shape == utility.shape
    assert soft.shape == utility.shape
    # Hard decisions: send if u > 0.3
    expected_hard = torch.tensor([False, True, False, True])
    assert torch.equal(hard, expected_hard)
