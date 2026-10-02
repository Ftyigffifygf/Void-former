"""Checkpoint save and load utilities for VoidFormer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    step: int = 0,
    seed: Optional[int] = None,
) -> None:
    """Save model, optimizer, config, step, and seed state to a checkpoint file."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "step": step,
        "seed": seed,
        "config": config or {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }

    torch.save(state_dict, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str | torch.device = "cpu",
) -> Tuple[int, Optional[int], Dict[str, Any]]:
    """Load model, optimizer, config, step, and seed state from a checkpoint file."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint.get("step", 0)
    seed = checkpoint.get("seed", None)
    config = checkpoint.get("config", {})

    return step, seed, config
