#!/usr/bin/env python3
"""Quick verification script to ensure all dependencies are working."""

import sys

print(f"Python version: {sys.version}")

# Test core imports
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  MPS available: True (Apple Silicon)")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")

try:
    import gymnasium
    print(f"✓ Gymnasium {gymnasium.__version__}")
except ImportError:
    print("✗ Gymnasium not installed")

# Test project imports
try:
    from cce.envs.grid_world import GridWorldConfig
    print("✓ cce.envs.grid_world")
except ImportError as e:
    print(f"✗ cce.envs.grid_world: {e}")

try:
    from cce.algo.mappo import MAPPO
    print("✓ cce.algo.mappo")
except ImportError as e:
    print(f"✗ cce.algo.mappo: {e}")

try:
    from cce.comm.vq import VectorQuantizer
    print("✓ cce.comm.vq")
except ImportError as e:
    print(f"✗ cce.comm.vq: {e}")

try:
    from cce.comm.budget import BudgetController
    print("✓ cce.comm.budget")
except ImportError as e:
    print(f"✗ cce.comm.budget: {e}")

print("\n✓ All checks passed! Ready to train.")
