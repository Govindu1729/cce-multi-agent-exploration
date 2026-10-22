"""Classical baselines for the multi-agent exploration task."""

from cce.baselines.frontier import (
    FrontierBaseline,
    RandomBaseline,
    make_baseline,
)

__all__ = ["FrontierBaseline", "RandomBaseline", "make_baseline"]