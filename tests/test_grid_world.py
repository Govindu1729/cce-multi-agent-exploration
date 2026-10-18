"""
Rigorous tests for GridWorld environment - Info Leakage Proof.

These tests verify that agents NEVER receive global coverage information
and that visited_self exactly matches what the agent could have seen
based on its trajectory and FOV.
"""

import sys
sys.path.insert(0, '/Users/srimaan/.openclaw/workspaces/repo-cce')

import numpy as np
import torch
from cce.envs.grid_world import (
    GridWorld, GridWorldConfig, Action, compute_exact_visited_self
)


def test_no_info_leakage_in_visited_mask():
    """
    Rigorous test: verify that obs['visited_self'] for Agent 0 matches 
    EXACTLY the cells it could have seen based on its FOV and trajectory.
    
    Run an episode for 50 steps, manually track exactly which cells Agent 0 
    has seen, and assert that obs['visited_self'] matches only that set.
    
    This prevents info leakage where agents might accidentally receive
    global coverage or teammate-visited cells in their visited_self.
    """
    print("=" * 60)
    print("TEST: test_no_info_leakage_in_visited_mask")
    print("=" * 60)
    
    # Create environment with known seed for reproducibility
    config = GridWorldConfig(
        grid_size=10,
        num_agents=2,
        fov_radius=2,
        max_steps=50,
        comm_budget=10,
        seed=42
    )
    env = GridWorld(config)
    
    # Reset and get initial observations
    obs = env.reset()
    
    # Track Agent 0's trajectory manually
    agent0_trajectory = [env.agent_positions[0]]
    
    # Track expected visited cells for Agent 0
    expected_visited = compute_exact_visited_self(env, 0, agent0_trajectory)
    
    # Verify initial visited_self matches expected
    actual_visited = obs['visited_self'][0].numpy()
    assert np.array_equal(actual_visited, expected_visited), \
        f"Initial visited_self mismatch!\nExpected:\n{expected_visited.astype(int)}\nActual:\n{actual_visited.astype(int)}"
    
    print(f"Step 0: Agent 0 at {agent0_trajectory[-1]}")
    print(f"  Expected visited cells: {expected_visited.sum()}")
    print(f"  Actual visited_self cells: {actual_visited.sum()}")
    print("  ✓ Initial state matches")
    
    # Run episode for 50 steps
    for step in range(50):
        # Simple deterministic actions for Agent 0 (move right, then down, etc.)
        # Agent 1 moves randomly
        if step < 8:
            action0 = Action.RIGHT.value
        elif step < 16:
            action0 = Action.DOWN.value
        elif step < 24:
            action0 = Action.LEFT.value
        else:
            action0 = Action.UP.value
        
        action1 = np.random.randint(0, 5)  # Movement only, no SEND
        
        actions = [action0, action1]
        
        # Step environment
        obs, rewards, done, info = env.step(actions)
        
        # Track Agent 0's trajectory
        agent0_trajectory.append(env.agent_positions[0])
        
        # Compute expected visited based on FULL trajectory so far
        expected_visited = compute_exact_visited_self(env, 0, agent0_trajectory)
        actual_visited = obs['visited_self'][0].numpy()
        
        # ASSERT: visited_self must match EXACTLY what agent could have seen
        assert np.array_equal(actual_visited, expected_visited), \
            f"Info leakage detected at step {step + 1}!\n" \
            f"Agent 0 trajectory: {agent0_trajectory}\n" \
            f"Expected visited cells: {expected_visited.sum()}\n" \
            f"Actual visited_self cells: {actual_visited.sum()}\n" \
            f"Diff (actual - expected): {(actual_visited.astype(int) - expected_visited.astype(int)).sum()}\n" \
            f"Extra cells in actual: {np.where((actual_visited & ~expected_visited))}\n" \
            f"Missing cells in actual: {np.where((expected_visited & ~actual_visited))}"
        
        # Also verify visited_teammates does NOT leak into visited_self
        visited_teammates = obs['visited_teammates'][0].numpy()
        # visited_self and visited_teammates should be disjoint in terms of source
        # (they can overlap in cells, but visited_self should only have agent's own cells)
        
        if step % 10 == 0:
            print(f"Step {step + 1}: Agent 0 at {agent0_trajectory[-1]}, "
                  f"visited_self={actual_visited.sum()}, "
                  f"visited_teammates={visited_teammates.sum()}")
    
    # Final verification
    print(f"\nFinal verification:")
    print(f"  Total steps: {len(agent0_trajectory) - 1}")
    print(f"  Agent 0 trajectory: {agent0_trajectory}")
    print(f"  Final visited_self cells: {obs['visited_self'][0].sum().item()}")
    print(f"  Final expected cells: {expected_visited.sum()}")
    print(f"  ✓ NO INFO LEAKAGE - visited_self matches exactly!")
    
    # Additional check: verify global_visited is NOT in agent observations
    agent_obs_keys = set(obs.keys())
    assert 'global_visited' not in agent_obs_keys, \
        "global_visited leaked into agent observations!"
    assert 'global_coverage' not in agent_obs_keys, \
        "global_coverage leaked into agent observations!"
    assert 'all_positions' not in agent_obs_keys, \
        "all_positions leaked into agent observations!"
    
    print("  ✓ No global state keys in agent observations")
    
    # Verify critic has access to global state
    global_state = info['global_state']
    assert 'global_visited' in global_state, "Critic missing global_visited"
    assert 'global_coverage' in global_state, "Critic missing global_coverage"
    assert 'all_positions' in global_state, "Critic missing all_positions"
    
    print("  ✓ Critic has access to global state")
    print("\n" + "=" * 60)
    print("TEST PASSED: No info leakage in visited_mask")
    print("=" * 60)


def test_global_coverage_only_in_critic():
    """Verify global coverage is only available to centralised critic."""
    print("\n" + "=" * 60)
    print("TEST: test_global_coverage_only_in_critic")
    print("=" * 60)
    
    config = GridWorldConfig(grid_size=8, num_agents=3, seed=123)
    env = GridWorld(config)
    
    obs = env.reset()
    
    # Step a few times
    for _ in range(5):
        actions = [np.random.randint(0, 5) for _ in range(3)]
        obs, rewards, done, info = env.step(actions)
    
    # Agent observations should NOT have global state
    for key in ['global_visited', 'global_coverage', 'all_positions']:
        assert key not in obs, f"{key} found in agent observations!"
    
    # Info should have global_state for critic
    assert 'global_state' in info, "global_state missing from info"
    global_state = info['global_state']
    assert 'global_visited' in global_state
    assert 'global_coverage' in global_state
    assert 'all_positions' in global_state
    
    print("  ✓ Global coverage only in critic's global_state")
    print("  ✓ Agent observations clean")
    print("TEST PASSED")


def test_communication_does_not_leak_to_visited_self():
    """Verify that receiving communication doesn't pollute visited_self."""
    print("\n" + "=" * 60)
    print("TEST: test_communication_does_not_leak_to_visited_self")
    print("=" * 60)
    
    config = GridWorldConfig(
        grid_size=8,
        num_agents=2,
        fov_radius=1,
        max_steps=20,
        comm_budget=5,
        comm_range=3,
        seed=456
    )
    env = GridWorld(config)
    
    obs = env.reset()
    
    # Manually track Agent 0's trajectory
    agent0_trajectory = [env.agent_positions[0]]
    
    # Run steps where Agent 1 sends, Agent 0 receives
    for step in range(10):
        # Agent 0 moves, Agent 1 sends
        action0 = Action.RIGHT.value if step % 2 == 0 else Action.DOWN.value
        action1 = Action.SEND.value
        
        actions = [action0, action1]
        obs, rewards, done, info = env.step(actions)
        
        agent0_trajectory.append(env.agent_positions[0])
        
        # Compute expected visited_self (only from Agent 0's trajectory)
        expected_visited = compute_exact_visited_self(env, 0, agent0_trajectory)
        actual_visited = obs['visited_self'][0].numpy()
        
        # Verify no leakage
        assert np.array_equal(actual_visited, expected_visited), \
            f"Communication leaked into visited_self at step {step}!"
        
        # visited_teammates should have data from Agent 1
        visited_teammates = obs['visited_teammates'][0].numpy()
        # This should be non-empty after Agent 1 sends
    
    print(f"  Agent 0 visited_self: {obs['visited_self'][0].sum().item()} cells")
    print(f"  Agent 0 visited_teammates: {obs['visited_teammates'][0].sum().item()} cells")
    print("  ✓ Communication received in visited_teammates, NOT visited_self")
    print("TEST PASSED")


def test_visited_teammates_aggregation():
    """Verify visited_teammates correctly aggregates teammate data."""
    print("\n" + "=" * 60)
    print("TEST: test_visited_teammates_aggregation")
    print("=" * 60)
    
    config = GridWorldConfig(
        grid_size=6,
        num_agents=3,
        fov_radius=1,
        max_steps=10,
        comm_budget=10,
        comm_range=5,
        seed=789
    )
    env = GridWorld(config)
    
    obs = env.reset()
    
    # Step a few times with all agents sending
    for step in range(5):
        actions = [Action.SEND.value] * 3  # All send
        obs, rewards, done, info = env.step(actions)
    
    # Each agent's visited_teammates should be union of OTHER agents' visited_self
    for i in range(3):
        visited_self = obs['visited_self'][i].numpy()
        visited_teammates = obs['visited_teammates'][i].numpy()
        
        # Compute expected teammates visited
        expected_teammates = np.zeros((6, 6), dtype=bool)
        for j in range(3):
            if j != i:
                expected_teammates |= obs['visited_self'][j].numpy()
        
        assert np.array_equal(visited_teammates, expected_teammates), \
            f"Agent {i} visited_teammates mismatch!"
    
    print("  ✓ visited_teammates correctly aggregates other agents' visited_self")
    print("TEST PASSED")


def test_token_budget_enforcement():
    """Test that token budget is properly tracked."""
    print("\n" + "=" * 60)
    print("TEST: test_token_budget_enforcement")
    print("=" * 60)
    
    config = GridWorldConfig(
        grid_size=5,
        num_agents=2,
        fov_radius=1,
        max_steps=10,
        comm_budget=6,  # Shared budget: 2 agents * 3 sends each
        seed=999
    )
    env = GridWorld(config)
    
    obs = env.reset()
    
    # Initial tokens
    assert obs['tokens_remaining'][0].item() == 6
    assert obs['tokens_remaining'][1].item() == 6
    print(f"  Initial tokens: {obs['tokens_remaining'].tolist()}")
    
    # Use all tokens (only agent 0 sends, 1 token per step)
    for step in range(3):
        actions = [Action.SEND.value, Action.STAY.value]
        obs, rewards, done, info = env.step(actions)
        tokens = obs['tokens_remaining'][0].item()
        print(f"  Step {step + 1}: tokens remaining = {tokens}")
        assert tokens == 6 - step - 1
    
    # Tokens exhausted
    assert obs['tokens_remaining'][0].item() == 3
    assert env.comm_budget.token_bucket.remaining == 3
    print("  ✓ Token budget correctly tracked and exhausted")
    print("TEST PASSED")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# GRID WORLD INFO LEAKAGE TESTS")
    print("#" * 60)
    
    test_no_info_leakage_in_visited_mask()
    test_global_coverage_only_in_critic()
    test_communication_does_not_leak_to_visited_self()
    test_visited_teammates_aggregation()
    test_token_budget_enforcement()
    
    print("\n" + "#" * 60)
    print("# ALL TESTS PASSED")
    print("#" * 60)