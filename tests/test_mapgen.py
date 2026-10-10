"""Tests for map generation."""

import numpy as np
import pytest

from cce.envs.mapgen import generate_map, keep_largest_component, sample_free_positions


def test_generate_map_shape():
    rng = np.random.default_rng(42)
    m = generate_map(rng, grid_size=20, obstacle_density=0.2)
    assert m.shape == (20, 20)
    assert m.dtype == np.int8


def test_generate_map_has_free_cells():
    rng = np.random.default_rng(42)
    m = generate_map(rng, grid_size=20, obstacle_density=0.2)
    n_free = (m == 0).sum()
    assert n_free > 50  # At least 50 free cells


def test_keep_largest_component():
    # Create a map with two disconnected free regions
    m = np.ones((10, 10), dtype=np.int8)
    m[0:3, 0:3] = 0  # Region 1
    m[7:10, 7:10] = 0  # Region 2 (smaller)
    new_m = keep_largest_component(m)
    # Only one region should remain
    n_free = (new_m == 0).sum()
    assert n_free == 9  # 3x3


def test_sample_free_positions():
    rng = np.random.default_rng(42)
    m = np.zeros((10, 10), dtype=np.int8)
    m[0, :] = 1  # Top row is obstacle
    positions = sample_free_positions(rng, m, 5)
    assert positions.shape == (5, 2)
    # All positions should be free
    for y, x in positions:
        assert m[y, x] == 0
    # All positions should be distinct
    assert len(set(map(tuple, positions))) == 5


def test_generate_map_connectivity():
    rng = np.random.default_rng(42)
    m = generate_map(rng, grid_size=20, obstacle_density=0.3)
    # Check that there's a large connected component
    from cce.envs.mapgen import _connected_components
    free = m == 0
    labels, sizes = _connected_components(free)
    assert sizes.max() > 100  # At least 100 cells in largest component
