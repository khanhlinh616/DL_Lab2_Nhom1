"""
Unit tests for model builders, freezing/unfreezing mechanism, and parameter counts.
"""

import pytest
import torch

from lab2.models import (
    build_model,
    count_parameters,
    get_trainable_param_names
)


def test_resnet18_transfer_freezing():
    """Verify ResNet18 transfer learning only trains fc layer."""
    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    counts = count_parameters(model)

    assert counts["trainable"] == (512 * 10 + 10)  # weights + bias of fc
    assert counts["frozen"] > 0
    assert counts["total"] == counts["trainable"] + counts["frozen"]

    trainable_names = get_trainable_param_names(model)
    assert set(trainable_names) == {"fc.weight", "fc.bias"}


def test_resnet18_finetune_freezing():
    """Verify ResNet18 fine-tuning trains layer4 and fc layer, freezing earlier layers."""
    model = build_model("resnet18", method="finetune", num_classes=10, pretrained=False)
    trainable_names = get_trainable_param_names(model)

    assert any(name.startswith("fc.") for name in trainable_names)
    assert any(name.startswith("layer4.") for name in trainable_names)
    assert not any(name.startswith("layer1.") for name in trainable_names)
    assert not any(name.startswith("layer2.") for name in trainable_names)
    assert not any(name.startswith("layer3.") for name in trainable_names)
    assert not any(name.startswith("conv1.") for name in trainable_names)


def test_densenet121_transfer_freezing():
    """Verify DenseNet121 transfer learning only trains classifier layer."""
    model = build_model("densenet121", method="transfer", num_classes=10, pretrained=False)
    counts = count_parameters(model)

    assert counts["trainable"] == (1024 * 10 + 10)  # weights + bias of classifier
    trainable_names = get_trainable_param_names(model)
    assert set(trainable_names) == {"classifier.weight", "classifier.bias"}


def test_densenet121_finetune_freezing():
    """Verify DenseNet121 fine-tuning trains denseblock4, norm5, and classifier."""
    model = build_model("densenet121", method="finetune", num_classes=10, pretrained=False)
    trainable_names = get_trainable_param_names(model)

    assert any(name.startswith("classifier.") for name in trainable_names)
    assert any("denseblock4" in name for name in trainable_names)
    assert any("norm5" in name for name in trainable_names)
    assert not any("denseblock1" in name for name in trainable_names)
    assert not any("denseblock2" in name for name in trainable_names)
    assert not any("denseblock3" in name for name in trainable_names)


def test_model_forward_shape():
    """Verify forward pass returns logits of shape (batch, 10)."""
    dummy_input = torch.randn(2, 3, 64, 64)

    res_model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    dense_model = build_model("densenet121", method="transfer", num_classes=10, pretrained=False)

    res_out = res_model(dummy_input)
    dense_out = dense_model(dummy_input)

    assert res_out.shape == (2, 10)
    assert dense_out.shape == (2, 10)
