"""Seeding utilities for reproducibility."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """Set seeds for Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
