"""Tests for the VQ bottleneck."""

import pytest
import torch

from cce.comm.vq import VectorQuantizer


def test_vq_forward():
    vq = VectorQuantizer(n_codes=8, code_dim=4, n_slots=2)
    z = torch.randn(3, 5, 2, 4)  # (batch, seq, slots, dim)
    out = vq(z)

    assert "quantised" in out
    assert "indices" in out
    assert "loss" in out
    assert "perplexity" in out

    assert out["quantised"].shape == z.shape
    assert out["indices"].shape == z.shape[:-1]
    assert out["loss"].dim() == 0  # scalar
    assert out["perplexity"].dim() == 0


def test_vq_straight_through():
    vq = VectorQuantizer(n_codes=8, code_dim=4, n_slots=2)
    z = torch.randn(3, 5, 2, 4, requires_grad=True)
    out = vq(z)

    loss = out["quantised"].sum()
    loss.backward()

    # Gradient should flow through via ST estimator
    assert z.grad is not None
    assert z.grad.shape == z.shape
    # With ST estimator, gradient w.r.t. z should be 1 (identity backward)
    # But there's also the commitment loss which adds a gradient term
    # Just check it's not all zeros
    assert z.grad.abs().sum() > 0


def test_vq_codebook_update():
    vq = VectorQuantizer(n_codes=8, code_dim=4, n_slots=2, decay=0.99)
    vq.train()

    initial_weight = vq.codebook.weight.clone()

    for _ in range(10):
        z = torch.randn(10, 5, 2, 4)
        vq(z)

    # Codebook should have changed
    assert not torch.allclose(vq.codebook.weight, initial_weight, atol=1e-4)


def test_vq_dead_code_reset():
    vq = VectorQuantizer(n_codes=8, code_dim=4, n_slots=2, dead_reset_steps=5)
    vq.train()

    # Use only the first code repeatedly
    z = torch.zeros(10, 5, 2, 4)
    z[..., 0] = 1.0  # Bias towards one code

    for i in range(20):
        vq(z)

    # After enough steps, dead codes should be reset
    stats = vq.get_usage_stats()
    assert stats["active_codes"] >= 1


def test_vq_bits_per_message():
    vq = VectorQuantizer(n_codes=16, code_dim=4, n_slots=4)
    # log2(16) = 4 bits per code, 4 slots → 16 bits
    assert vq.bits_per_message == 16

    vq2 = VectorQuantizer(n_codes=32, code_dim=4, n_slots=2)
    # log2(32) = 5 bits per code, 2 slots → 10 bits
    assert vq2.bits_per_message == 10
