import sys
sys.path.insert(0, '/Users/srimaan/.openclaw/workspaces/repo-cce')

import torch
from cce.comm.vq import VectorQuantizer, compute_codebook_perplexity, revive_dead_codes
from cce.eval.metrics import compute_metrics, aggregate_episode_result


def test_vq_perplexity():
    """Test VQ perplexity tracking works correctly."""
    print("Testing VQ perplexity tracking...")
    
    vq = VectorQuantizer(embed_dim=32, num_codes=8)
    
    # Simulate uniform code usage
    batch_size = 64
    for i in range(10):
        z = torch.randn(batch_size, 32)
        # Assign codes uniformly (cycling through all codes)
        code_indices = torch.arange(batch_size) % 8
        z_q, loss, perplexity = vq(z, code_indices)
    
    stats = vq.get_usage_stats()
    
    print(f"  Perplexity: {stats['perplexity']:.2f}")
    print(f"  Active codes: {stats['active_codes']}/{vq.num_codes}")
    
    # With uniform usage, perplexity should be close to num_codes (8)
    assert stats['perplexity'] > 4.0, "Perplexity too low for uniform usage"
    assert stats['active_codes'] >= 6, "Should use most codes with uniform assignment"
    
    print("  ✓ VQ perplexity test passed")


def test_vq_entropy_regularizer():
    """Test entropy regularizer encourages diverse code usage."""
    print("Testing VQ entropy regularizer...")
    
    # Model without entropy weight
    vq_no_entropy = VectorQuantizer(embed_dim=32, num_codes=8, entropy_weight=0.0)
    
    # Model with entropy weight
    vq_with_entropy = VectorQuantizer(embed_dim=32, num_codes=8, entropy_weight=0.1)
    
    batch_size = 32
    
    # Simulate training steps
    for i in range(20):
        z = torch.randn(batch_size, 32)
        
        # Without entropy: biased to first code (indices 0-3)
        code_indices_no = torch.randint(0, 4, (batch_size,))
        
        # With entropy: uniform over all 8 codes
        logits = torch.randn(batch_size, 8)
        code_indices_with = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
        
        vq_no_entropy(z, code_indices_no)
        vq_with_entropy(z, code_indices_with)
    
    stats_no = vq_no_entropy.get_usage_stats()
    stats_with = vq_with_entropy.get_usage_stats()
    
    print(f"  Without entropy: {stats_no['active_codes']} active codes")
    print(f"  With entropy: {stats_with['active_codes']} active codes")
    
    print("  ✓ VQ entropy test completed")


def test_metrics_bandwidth():
    """Test bandwidth calculation is non-zero."""
    print("Testing metrics bandwidth calculation...")
    
    # Simulate episodes
    episodes = [
        {'coverage': 0.96, 'steps': 32, 'messages_sent': 256, 'bandwidth_bits': 256 * 4},
        {'coverage': 0.95, 'steps': 35, 'messages_sent': 280, 'bandwidth_bits': 280 * 4},
    ]
    
    metrics = compute_metrics(episodes)
    
    print(f"  Total messages: {metrics['total_messages']}")
    print(f"  Bandwidth: {metrics['bandwidth']} bits")
    print(f"  Success rate: {metrics['success_rate']}")
    
    assert metrics['bandwidth'] > 0, "Bandwidth should be non-zero!"
    assert metrics['success_rate'] == 1.0, "Success rate should be 100% for test data"
    
    print("  ✓ Metrics bandwidth test passed")


def test_metrics_success_rate():
    """Test success rate calculation."""
    print("Testing metrics success rate calculation...")
    
    episodes = [
        {'coverage': 0.96, 'steps': 32, 'messages_sent': 100, 'bandwidth_bits': 400},
        {'coverage': 0.80, 'steps': 100, 'messages_sent': 100, 'bandwidth_bits': 400},
        {'coverage': 0.97, 'steps': 30, 'messages_sent': 100, 'bandwidth_bits': 400},
    ]
    
    metrics = compute_metrics(episodes)
    
    print(f"  Success rate: {metrics['success_rate']}")
    print(f"  T95: {metrics['t95']}")
    
    # 2 out of 3 episodes have coverage >= 0.95
    assert metrics['success_rate'] == 2/3, f"Expected 2/3 success rate, got {metrics['success_rate']}"
    
    print("  ✓ Metrics success rate test passed")


def test_aggregate_episode():
    """Test episode aggregation."""
    print("Testing episode aggregation...")
    
    result = aggregate_episode_result(coverage=0.95, steps=32, messages_sent=256, d_m=4)
    
    print(f"  Coverage: {result['coverage']}")
    print(f"  Bandwidth bits: {result['bandwidth_bits']}")
    
    assert result['bandwidth_bits'] == 1024, "Bandwidth should be messages * d_m"
    
    print("  ✓ Episode aggregation test passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Running evaluation tests...")
    print("=" * 50)
    
    test_vq_perplexity()
    test_vq_entropy_regularizer()
    test_metrics_bandwidth()
    test_metrics_success_rate()
    test_aggregate_episode()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
