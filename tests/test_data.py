"""
Unit tests for data transformations, dataset wrapping, and splitting.
"""

import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset, random_split

from lab2.data import (
    CIFAR10_CLASSES,
    DatasetWithTransform,
    get_transforms
)


def test_transforms_output_shape():
    """Verify transforms produce (3, H, W) float tensors."""
    img = Image.new("RGB", (32, 32), color="red")
    train_transform = get_transforms(image_size=64, is_train=True)
    eval_transform = get_transforms(image_size=64, is_train=False)

    tensor_train = train_transform(img)
    tensor_eval = eval_transform(img)

    assert tensor_train.shape == (3, 64, 64)
    assert tensor_eval.shape == (3, 64, 64)
    assert tensor_train.dtype == torch.float32
    assert tensor_eval.dtype == torch.float32


def test_classes_list():
    """Verify CIFAR10 class list is correct and contains 10 classes."""
    assert len(CIFAR10_CLASSES) == 10
    assert "airplane" in CIFAR10_CLASSES
    assert "truck" in CIFAR10_CLASSES


def test_dataset_split_no_overlap():
    """Verify random_split with fixed generator produces distinct subsets without leakage."""
    synthetic_data = TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
    gen = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(synthetic_data, [80, 20], generator=gen)

    assert len(train_set) == 80
    assert len(val_set) == 20

    train_indices = set(train_set.indices)
    val_indices = set(val_set.indices)
    assert len(train_indices.intersection(val_indices)) == 0


def test_dataset_with_transform():
    """Verify DatasetWithTransform applies transformation on getitem."""
    dummy_imgs = [Image.new("RGB", (32, 32)) for _ in range(5)]
    dummy_labels = list(range(5))
    raw_dataset = list(zip(dummy_imgs, dummy_labels))

    transform = get_transforms(image_size=48, is_train=False)
    wrapped = DatasetWithTransform(raw_dataset, transform=transform)

    assert len(wrapped) == 5
    img, label = wrapped[0]
    assert img.shape == (3, 48, 48)
    assert label == 0
