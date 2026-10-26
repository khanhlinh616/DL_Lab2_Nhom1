"""
Unit tests for training step, gradient flow, and config validation.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from lab2.config import ExperimentConfig
from lab2.models import build_model
from lab2.train import train_one_epoch, validate, create_optimizer


def test_training_step_gradient_flow_transfer():
    """Verify that in transfer learning, frozen layers receive no gradients while fc layer updates."""
    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)

    # Synthetic batch of 4 samples
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=4)

    initial_fc_weight = model.fc.weight.clone()
    initial_layer1_weight = model.layer1[0].conv1.weight.clone()

    loss, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert loss > 0
    assert 0 <= acc <= 100

    # Layer 1 is frozen: should have no gradient and weights must NOT change
    assert model.layer1[0].conv1.weight.grad is None
    assert torch.equal(model.layer1[0].conv1.weight, initial_layer1_weight)

    # FC is trainable: should have gradient and weights MUST change
    assert model.fc.weight.grad is not None
    assert not torch.equal(model.fc.weight, initial_fc_weight)


def test_validate_no_grad():
    """Verify that validate runs without modifying parameters or computing gradients."""
    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=4)

    val_loss, val_acc = validate(model, loader, criterion, torch.device("cpu"))

    assert val_loss > 0
    assert 0 <= val_acc <= 100
    for param in model.parameters():
        assert param.grad is None


def test_config_validation():
    """Verify config validator catches invalid fields."""
    with pytest.raises(ValueError, match="Invalid model_name"):
        cfg = ExperimentConfig(
            name="test",
            model_name="vgg16",  # unsupported in this prompt
            method="transfer",
            epochs=5,
            batch_size=16,
            learning_rate=0.001,
            weight_decay=0.0001
        )
        cfg.validate()

    with pytest.raises(ValueError, match="epochs must be > 0"):
        cfg = ExperimentConfig(
            name="test",
            model_name="resnet18",
            method="transfer",
            epochs=0,
            batch_size=16,
            learning_rate=0.001,
            weight_decay=0.0001
        )
        cfg.validate()


@pytest.mark.parametrize("model_name,expected_head_params", [
    ("resnet18", 5130),
    ("densenet121", 10250),
])
def test_create_optimizer_finetune(model_name, expected_head_params):
    """Verify finetune creates 2 param groups with differential LR and correct head param count."""
    cfg = ExperimentConfig(
        name=f"{model_name}_finetune",
        model_name=model_name,
        method="finetune",
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        backbone_lr=0.0001,
        weight_decay=0.0001,
        optimizer="adam"
    )
    model = build_model(model_name, method="finetune", num_classes=10, pretrained=False)
    opt = create_optimizer(model, cfg)

    assert len(opt.param_groups) == 2
    # Group 0 is backbone
    assert opt.param_groups[0]["lr"] == 0.0001
    assert len(opt.param_groups[0]["params"]) > 0
    # Group 1 is head
    assert opt.param_groups[1]["lr"] == 0.001
    head_numel = sum(p.numel() for p in opt.param_groups[1]["params"])
    assert head_numel == expected_head_params


@pytest.mark.parametrize("model_name", ["resnet18", "densenet121"])
def test_create_optimizer_transfer(model_name):
    """Verify transfer learning creates exactly 1 param group with learning_rate."""
    cfg = ExperimentConfig(
        name=f"{model_name}_transfer",
        model_name=model_name,
        method="transfer",
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.0001,
        optimizer="adam"
    )
    model = build_model(model_name, method="transfer", num_classes=10, pretrained=False)
    opt = create_optimizer(model, cfg)

    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 0.001


def test_create_optimizer_unsupported():
    """Verify create_optimizer raises ValueError for unsupported optimizer."""
    cfg = ExperimentConfig(
        name="test_unsupported_opt",
        model_name="resnet18",
        method="transfer",
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.0001,
        optimizer="rmsprop"
    )
    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        create_optimizer(model, cfg)

