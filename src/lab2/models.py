"""
Model builders for ResNet18 and DenseNet121 with Transfer Learning and Fine-tuning.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(
    method: str = "transfer",
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """
    Builds ResNet18 configured for CIFAR-10.
    - transfer: All backbone layers frozen; only fc is trainable.
    - finetune: Earlier layers (conv1..layer3) frozen; layer4 and fc are trainable with differential learning rates.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Step 1: Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: In fine-tuning, unfreeze the final backbone stage (layer4)
    if method == "finetune":
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif method != "transfer":
        raise ValueError(f"Unknown method '{method}'. Expected 'transfer' or 'finetune'.")

    # Step 3: Replace final classification layer (newly created layer has requires_grad=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def build_densenet121(
    method: str = "transfer",
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """
    Builds DenseNet121 configured for CIFAR-10.
    - transfer: All feature extraction layers frozen; only classifier is trainable.
    - finetune: Earlier layers (conv0..denseblock3) frozen; denseblock4, norm5, and classifier are trainable.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    # Step 1: Freeze all feature extraction parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: In fine-tuning, unfreeze the final denseblock (denseblock4) and final norm layer (norm5)
    if method == "finetune":
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
    elif method != "transfer":
        raise ValueError(f"Unknown method '{method}'. Expected 'transfer' or 'finetune'.")

    # Step 3: Replace final classification layer (newly created layer has requires_grad=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model


def build_model(
    model_name: str,
    method: str,
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """Factory function to build model based on model_name and method."""
    if model_name == "resnet18":
        return build_resnet18(method=method, num_classes=num_classes, pretrained=pretrained)
    elif model_name == "densenet121":
        return build_densenet121(method=method, num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'. Choose 'resnet18' or 'densenet121'.")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total, trainable, and frozen parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen
    }


def get_trainable_param_names(model: nn.Module) -> List[str]:
    """Returns list of names of all parameters that have requires_grad=True."""
    return [name for name, param in model.named_parameters() if param.requires_grad]
