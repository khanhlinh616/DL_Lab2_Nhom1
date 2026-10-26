"""
Unit tests for model evaluation and metrics computation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lab2.evaluate import evaluate_model


class TinySyntheticModel(nn.Module):
    """Small linear model for offline evaluation testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        return self.fc(x)


def test_evaluate_model_synthetic():
    """Verify evaluate_model produces correct metrics on synthetic DataLoader."""
    num_samples = 40
    num_classes = 10
    class_names = [f"cls_{i}" for i in range(num_classes)]

    # Deterministic inputs and labels
    torch.manual_seed(42)
    inputs = torch.randn(num_samples, 16)
    labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = TinySyntheticModel(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    results = evaluate_model(model, dataloader, criterion, device, class_names)

    # 1. Total samples check
    assert results["total_test_samples"] == num_samples

    # 2. Confusion matrix sum equals total samples
    cm = np.array(results["confusion_matrix"])
    assert cm.shape == (num_classes, num_classes)
    assert cm.sum() == num_samples

    # 3. per_class_accuracy has all class keys
    assert len(results["per_class_accuracy"]) == num_classes
    for name in class_names:
        assert name in results["per_class_accuracy"]
        assert 0.0 <= results["per_class_accuracy"][name] <= 100.0

    # 4. test_accuracy within [0, 100]
    assert 0.0 <= results["test_accuracy"] <= 100.0
    assert results["test_loss"] >= 0.0


def test_evaluate_model_perfect_accuracy():
    """Verify evaluate_model correctly computes 100% accuracy when outputs match labels."""
    num_classes = 4
    class_names = ["c0", "c1", "c2", "c3"]
    # 4 samples, one for each class
    inputs = torch.eye(num_classes)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=2)

    # Identity model: output equals input logits
    class IdentityModel(nn.Module):
        def forward(self, x):
            return x * 10.0  # High confidence

    model = IdentityModel()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    results = evaluate_model(model, dataloader, criterion, device, class_names)

    assert results["test_accuracy"] == 100.0
    assert results["total_test_samples"] == 4
    for name in class_names:
        assert results["per_class_accuracy"][name] == 100.0
    cm = np.array(results["confusion_matrix"])
    assert np.all(cm == np.eye(num_classes))
