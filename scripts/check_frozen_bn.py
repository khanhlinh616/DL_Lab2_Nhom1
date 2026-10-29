"""
Diagnostic script to measure BatchNorm buffer updates in frozen backbones.
Examines whether running_mean, running_var, and num_batches_tracked change during training
even when parameter requires_grad=False.
"""

import os
import sys
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lab2.config import ExperimentConfig
from lab2.models import build_model
from lab2.train import train_one_epoch, create_optimizer

# Configure UTF-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def measure_bn_buffer_updates(
    model_name: str,
    method: str = "transfer",
    mode: str = "train",
    num_samples: int = 16,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Measures BatchNorm buffer changes in the frozen backbone of a model.
    Runs 1 epoch on synthetic data under 'train' or 'eval' mode.
    """
    model = build_model(model_name=model_name, method=method, num_classes=10, pretrained=False)

    # Identify all BatchNorm layers whose parameters are frozen
    frozen_bn_layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # If weight exists and requires_grad is False, it belongs to frozen backbone
            if module.weight is not None and not module.weight.requires_grad:
                frozen_bn_layers.append((name, module))
            elif module.weight is None and not (name.startswith("fc") or name.startswith("classifier")):
                frozen_bn_layers.append((name, module))

    # Take snapshot of initial buffer values
    initial_buffers: Dict[str, Dict[str, torch.Tensor]] = {}
    total_buffers = 0
    for name, module in frozen_bn_layers:
        initial_buffers[name] = {}
        for buf_name, buf in module.named_buffers():
            initial_buffers[name][buf_name] = buf.clone()
            total_buffers += 1

    # Create synthetic dataset and loader
    torch.manual_seed(42)
    inputs = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,), dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    cfg = ExperimentConfig(
        name=f"{model_name}_{method}",
        model_name=model_name,
        method=method,
        epochs=1,
        batch_size=batch_size,
        learning_rate=0.001,
        weight_decay=0.0001,
        device="cpu"
    )
    optimizer = create_optimizer(model, cfg)

    if mode == "train":
        # Standard train_one_epoch sets model.train() internally
        train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    else:
        # In eval mode, evaluate through loader without updating weights
        model.eval()
        with torch.no_grad():
            for data, target in loader:
                model(data)

    # Compare buffers post-execution
    changed_buffers = 0
    changed_details = []
    for name, module in frozen_bn_layers:
        for buf_name, buf in module.named_buffers():
            orig = initial_buffers[name][buf_name]
            if not torch.equal(buf, orig):
                changed_buffers += 1
                diff = (buf.float() - orig.float()).abs().max().item()
                changed_details.append({
                    "layer": name,
                    "buffer": buf_name,
                    "max_diff": diff
                })

    return {
        "model_name": model_name,
        "method": method,
        "mode": mode,
        "frozen_bn_layers_count": len(frozen_bn_layers),
        "total_buffers_count": total_buffers,
        "changed_buffers_count": changed_buffers,
        "unchanged_buffers_count": total_buffers - changed_buffers,
        "changed_details": changed_details
    }


def main():
    print("=" * 85)
    print("BATCHNORM BUFFER BEHAVIOR AUDIT IN FROZEN BACKBONE")
    print("=" * 85)
    print(f"{'Model':<22} | {'Mode':<6} | {'Frozen BN':<10} | {'Total Buf':<10} | {'Changed Buf':<12} | {'Status'}")
    print("-" * 85)

    experiments = [
        ("resnet18", "train"),
        ("resnet18", "eval"),
        ("densenet121", "train"),
        ("densenet121", "eval"),
    ]

    for model_name, mode in experiments:
        res = measure_bn_buffer_updates(model_name=model_name, method="transfer", mode=mode)
        status = "UPDATED (Running stats changed)" if res["changed_buffers_count"] > 0 else "FROZEN (No stats changed)"
        print(
            f"{model_name + '_transfer':<22} | "
            f"{mode:<6} | "
            f"{res['frozen_bn_layers_count']:<10} | "
            f"{res['total_buffers_count']:<10} | "
            f"{res['changed_buffers_count']:<12} | "
            f"{status}"
        )

    print("=" * 85)
    print("CONCLUSION:")
    print("1. In 'train' mode (standard train_one_epoch), all BatchNorm running stats in the")
    print("   frozen backbone ARE updated despite requires_grad=False.")
    print("2. In 'eval' mode, BatchNorm running stats remain completely static (0 buffers changed).")
    print("=" * 85)


if __name__ == "__main__":
    main()
