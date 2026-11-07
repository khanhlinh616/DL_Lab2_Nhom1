"""
Model architecture and parameter inspection script for Lab 2 (Step 3: Explore the Model Architecture).
Displays total, trainable, and frozen parameters for all experiment configurations,
with options to print full architecture string, explore named layers, and inspect parameter tensors.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

# Ensure src directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lab2.models import build_model, count_parameters, get_trainable_param_names


def print_architecture(model_name: str, method: str = "transfer", num_classes: int = 10):
    """Step 3.1: Print the complete model structure using print(model)."""
    model = build_model(model_name=model_name, method=method, num_classes=num_classes, pretrained=False)
    print("\n" + "=" * 95)
    print(f"STEP 3.1: COMPLETE PYTORCH ARCHITECTURE PRINTOUT - {model_name.upper()} ({method.upper()})")
    print("=" * 95)
    print(model)
    print("=" * 95 + "\n")


def inspect_layers(model_name: str, method: str = "transfer", num_classes: int = 10):
    """Step 3.2: Identify named children layers/stages and their parameter distribution."""
    model = build_model(model_name=model_name, method=method, num_classes=num_classes, pretrained=False)
    print("\n" + "=" * 95)
    print(f"STEP 3.2: LAYER-BY-LAYER STRUCTURAL EXPLORATION - {model_name.upper()} ({method.upper()})")
    print("=" * 95)
    print(f"{'Child Layer Name':<28} | {'Layer Type':<32} | {'Params':<12} | {'Status'}")
    print("-" * 95)

    # If the model has a container like 'features' (DenseNet), expand its sub-modules for granular exploration
    items_to_inspect = []
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        for sub_name, sub_child in model.features.named_children():
            items_to_inspect.append((f"features.{sub_name}", sub_child))
        if hasattr(model, "classifier"):
            items_to_inspect.append(("classifier", model.classifier))
    else:
        for name, child in model.named_children():
            items_to_inspect.append((name, child))

    for name, child in items_to_inspect:
        layer_type = child.__class__.__name__
        child_params = sum(p.numel() for p in child.parameters())
        trainable_params = sum(p.numel() for p in child.parameters() if p.requires_grad)

        if child_params == 0:
            status = "No trainable weights (Pooling/Activation)"
        elif trainable_params == child_params:
            status = "100% Trainable (Unfrozen)"
        elif trainable_params == 0:
            status = "100% Frozen (Pretrained)"
        else:
            status = f"Partially Unfrozen ({trainable_params:,}/{child_params:,})"

        print(f"{name:<28} | {layer_type:<32} | {child_params:<12,d} | {status}")

    print("=" * 95 + "\n")


def inspect_parameters_detailed(model_name: str, method: str = "transfer", num_classes: int = 10):
    """Step 3.3: Identify all parameter tensors, their shape, numel, and requires_grad status."""
    model = build_model(model_name=model_name, method=method, num_classes=num_classes, pretrained=False)
    print("\n" + "=" * 95)
    print(f"STEP 3.3: PARAMETER TENSOR REGISTRY - {model_name.upper()} ({method.upper()})")
    print("=" * 95)
    print(f"{'Tensor Name':<45} | {'Shape':<22} | {'Numel':<10} | {'Requires Grad'}")
    print("-" * 95)

    for name, param in model.named_parameters():
        shape_str = str(list(param.shape))
        grad_str = "YES (Trainable)" if param.requires_grad else "NO (Frozen)"
        print(f"{name:<45} | {shape_str:<22} | {param.numel():<10,d} | {grad_str}")

    counts = count_parameters(model)
    print("-" * 95)
    print(f"SUMMARY: Total={counts['total']:,d} | Trainable={counts['trainable']:,d} | Frozen={counts['frozen']:,d}")
    print("=" * 95 + "\n")


def print_summary_table():
    """Default summary view for all 4 core experiment configurations."""
    experiments = [
        ("resnet18_transfer", "resnet18", "transfer"),
        ("resnet18_finetune", "resnet18", "finetune"),
        ("densenet121_transfer", "densenet121", "transfer"),
        ("densenet121_finetune", "densenet121", "finetune"),
    ]

    print("=" * 95)
    print("LAB 2: MODEL ARCHITECTURE AND PARAMETER INSPECTION")
    print("=" * 95)
    print(f"{'Experiment ID':<24} | {'Model':<12} | {'Method':<10} | {'Total':<11} | {'Trainable':<11} | {'Frozen':<11} | {'Trainable %'}")
    print("-" * 95)

    detailed_info = {}

    for exp_id, model_name, method in experiments:
        model = build_model(model_name=model_name, method=method, num_classes=10, pretrained=False)
        counts = count_parameters(model)
        pct = (counts["trainable"] / counts["total"]) * 100

        print(
            f"{exp_id:<24} | {model_name:<12} | {method:<10} | "
            f"{counts['total']:<11,d} | {counts['trainable']:<11,d} | {counts['frozen']:<11,d} | {pct:>6.2f}%"
        )
        detailed_info[exp_id] = get_trainable_param_names(model)

    print("-" * 95)
    print("\nDETAILED TRAINABLE PARAMETER GROUPS:")
    print("=" * 95)
    for exp_id, param_names in detailed_info.items():
        print(f"\n[{exp_id}] ({len(param_names)} trainable parameter tensors):")
        if len(param_names) <= 8:
            for p in param_names:
                print(f"  - {p}")
        else:
            for p in param_names[:5]:
                print(f"  - {p}")
            print(f"  ... [and {len(param_names) - 7} more tensors] ...")
            for p in param_names[-2:]:
                print(f"  - {p}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Lab 2 Model Architecture & Parameter Inspection Tool.")
    parser.add_argument("--print-model", choices=["resnet18", "densenet121", "all"], help="Print complete PyTorch model architecture.")
    parser.add_argument("--layers", choices=["resnet18", "densenet121", "all"], help="Inspect named layers and structural hierarchy.")
    parser.add_argument("--params", choices=["resnet18", "densenet121", "all"], help="Inspect detailed parameter tensor shapes and gradients.")
    parser.add_argument("--method", choices=["transfer", "finetune"], default="transfer", help="Adaptation method (default: transfer).")

    args = parser.parse_args()

    if args.print_model:
        targets = ["resnet18", "densenet121"] if args.print_model == "all" else [args.print_model]
        for m in targets:
            print_architecture(model_name=m, method=args.method)
    elif args.layers:
        targets = ["resnet18", "densenet121"] if args.layers == "all" else [args.layers]
        for m in targets:
            inspect_layers(model_name=m, method=args.method)
    elif args.params:
        targets = ["resnet18", "densenet121"] if args.params == "all" else [args.params]
        for m in targets:
            inspect_parameters_detailed(model_name=m, method=args.method)
    else:
        print_summary_table()


if __name__ == "__main__":
    main()
