"""
Unit tests for checkpoint saving and loading functionality.
"""

import os
import pytest
import torch

from lab2.checkpoint import load_checkpoint, save_checkpoint
from lab2.models import build_model


def test_checkpoint_save_and_load(tmp_path):
    """Verify that saving and loading checkpoint faithfully recovers model weights and metadata."""
    ckpt_path = os.path.join(tmp_path, "test_ckpt.pth")
    best_path = os.path.join(tmp_path, "test_best.pth")

    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    # Modify fc weight with known values
    with torch.no_grad():
        model.fc.weight.fill_(3.1415)

    state = {
        "epoch": 3,
        "model_name": "resnet18",
        "method": "transfer",
        "model_state_dict": model.state_dict(),
        "val_acc": 85.5,
        "val_loss": 0.42
    }

    save_checkpoint(state, filepath=ckpt_path, is_best=True, best_filepath=best_path)

    assert os.path.exists(ckpt_path)
    assert os.path.exists(best_path)

    new_model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    loaded_state = load_checkpoint(ckpt_path, model=new_model)

    assert loaded_state["epoch"] == 3
    assert loaded_state["val_acc"] == 85.5
    assert torch.equal(new_model.fc.weight, model.fc.weight)


def test_load_nonexistent_checkpoint():
    """Verify FileNotFoundError is raised for missing checkpoint."""
    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_file.pth", model=model)
