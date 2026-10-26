"""
Checkpoint saving and loading utilities with integrity verification.
"""

import os
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None
) -> None:
    """
    Saves training state dictionary to disk.
    If is_best is True, also creates a copy at best_filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        os.makedirs(os.path.dirname(best_filepath), exist_ok=True)
        torch.save(state, best_filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Loads model state_dict and optional optimizer state_dict from a checkpoint file.
    Returns the metadata dictionary containing epoch, metrics, and config.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    try:
        checkpoint = torch.load(filepath, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Corrupted or invalid checkpoint file at '{filepath}': {e}")

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint at '{filepath}' to be a dictionary, got {type(checkpoint).__name__}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint at '{filepath}' is missing 'model_state_dict'")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
