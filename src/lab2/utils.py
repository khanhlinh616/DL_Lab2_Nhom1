"""
Utility functions for seed setting, device management, and visualization.
"""

import os
import random
import time
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "cpu") -> torch.device:
    """
    Returns torch.device based on preference.
    Defaults to CPU as per project requirements, but can support CUDA if requested and available.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_time(seconds: float) -> str:
    """Formats duration in seconds into human-readable mm:ss string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plots and saves a normalized confusion matrix using matplotlib.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize rows to display recall per class
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_history_json(results: dict, cfg, path: str) -> None:
    """
    Saves training history, configuration, environment metadata, and summary metrics to a JSON file.
    """
    import json
    import torchvision

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    config_dict = vars(cfg) if hasattr(cfg, "__dict__") else (cfg if isinstance(cfg, dict) else {})
    history_data = results.get("history", {})

    payload = {
        "experiment_name": getattr(cfg, "name", results.get("experiment_name", "unknown")),
        "config": config_dict,
        "history": {
            "train_loss": history_data.get("train_loss", []),
            "train_acc": history_data.get("train_acc", []),
            "val_loss": history_data.get("val_loss", []),
            "val_acc": history_data.get("val_acc", []),
            "epoch_times": history_data.get("epoch_times", [])
        },
        "best_epoch": results.get("best_epoch"),
        "best_metric": results.get("best_metric"),
        "total_time_seconds": results.get("total_time_seconds"),
        "environment": {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(getattr(cfg, "device", "cpu"))
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

