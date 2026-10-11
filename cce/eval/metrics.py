import numpy as np
from typing import Dict, List, Any


def compute_metrics(episodes: List[Dict[str, Any]], max_steps: int = 100) -> Dict[str, float]:
    """
    Compute evaluation metrics from a list of episode results.
    
    Args:
        episodes: List of episode dictionaries with keys:
            - coverage: final coverage achieved (0 to 1)
            - steps: number of steps taken
            - messages_sent: number of messages sent in this episode
            - bandwidth_bits: bits transmitted (messages * d_m)
        max_steps: Maximum allowed steps per episode
        
    Returns:
        Dictionary with aggregated metrics
    """
    n = len(episodes)
    if n == 0:
        return {
            'success_rate': 0.0,
            't95': float('inf'),
            'coverage_mean': 0.0,
            'coverage_std': 0.0,
            'total_messages': 0,
            'bandwidth': 0.0,
            'comm_efficiency': 0.0
        }
    
    # Coverage metrics
    coverages = [ep['coverage'] for ep in episodes]
    coverage_mean = np.mean(coverages)
    coverage_std = np.std(coverages)
    
    # Success rate: fraction of episodes with coverage >= 0.95
    successes = [ep['coverage'] >= 0.95 for ep in episodes]
    success_rate = sum(successes) / n
    
    # T95: median steps to reach 95% coverage (among successful episodes only)
    t95_values = [ep['steps'] for ep, success in zip(episodes, successes) if success]
    if t95_values:
        t95 = float(np.median(t95_values))
    else:
        t95 = float('inf')
    
    # Communication metrics
    total_messages = sum(ep.get('messages_sent', 0) for ep in episodes)
    bandwidth = sum(ep.get('bandwidth_bits', 0) for ep in episodes)
    
    # Communication efficiency: bandwidth per successful episode
    # Avoid division by zero
    n_success = sum(successes)
    comm_efficiency = bandwidth / n_success if n_success > 0 else 0.0
    
    return {
        'success_rate': success_rate,
        't95': t95,
        'coverage_mean': coverage_mean,
        'coverage_std': coverage_std,
        'total_messages': total_messages,
        'bandwidth': bandwidth,
        'comm_efficiency': comm_efficiency
    }


def aggregate_episode_result(coverage: float, steps: int, 
                              messages_sent: int, d_m: int = 4) -> Dict[str, Any]:
    """
    Process a single episode result into standardized metrics.
    
    Args:
        coverage: Final coverage achieved (0 to 1)
        steps: Number of steps taken
        messages_sent: Number of messages sent
        d_m: Message dimension (bits per message)
        
    Returns:
        Episode metrics dictionary
    """
    return {
        'coverage': coverage,
        'steps': steps,
        'messages_sent': messages_sent,
        'bandwidth_bits': messages_sent * d_m  # Fixed: messages * message dimension
    }


if __name__ == "__main__":
    # Self-test with sample data
    sample_episodes = [
        {'coverage': 0.96, 'steps': 32, 'messages_sent': 256, 'bandwidth_bits': 256 * 4},
        {'coverage': 0.95, 'steps': 35, 'messages_sent': 280, 'bandwidth_bits': 280 * 4},
        {'coverage': 0.97, 'steps': 30, 'messages_sent': 240, 'bandwidth_bits': 240 * 4},
    ]
    
    metrics = compute_metrics(sample_episodes)
    
    print("Metrics test:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Verify bandwidth is non-zero
    assert metrics['bandwidth'] > 0, "Bandwidth should be non-zero!"
    assert metrics['success_rate'] > 0, "Success rate should be > 0 for test data!"
    
    print("\nMetrics computation test passed!")
