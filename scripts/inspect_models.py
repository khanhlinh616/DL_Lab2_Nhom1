"""
Model architecture and parameter inspection script.
Displays total, trainable, and frozen parameters for all 4 experiment configurations.
"""

import os
import sys

# Ensure src directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lab2.models import build_model, count_parameters, get_trainable_param_names


def main():
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
        # Note: pretrained=False can be used for structural parameter counting without downloading weights,
        # or pretrained=True. Since parameter count is identical regardless of weight initialization,
        # we can build the architecture structure safely.
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
        # Show first 5 and last 2 if many
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


if __name__ == "__main__":
    main()
