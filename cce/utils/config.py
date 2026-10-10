"""Configuration loading utilities using Hydra/OmegaConf."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config with optional overrides."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf to plain dict (for wandb)."""
    return OmegaConf.to_container(cfg, resolve=True)
