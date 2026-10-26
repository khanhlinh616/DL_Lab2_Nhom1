"""
Unit tests for configuration loading and validation.
"""

import pytest
import yaml
from lab2.config import load_config, load_yaml, get_all_experiment_names, ExperimentConfig


def test_load_config_merging(tmp_path):
    """Verify load_config merges global and experiment settings correctly."""
    sample_yaml = {
        "global": {
            "num_classes": 10,
            "image_size": 128,
            "val_split": 0.15,
            "seed": 42,
            "device": "cpu",
            "checkpoint_metric": "val_acc",
            "data_dir": "./custom_data",
        },
        "experiments": {
            "exp_transfer": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 7,
                "batch_size": 32,
                "learning_rate": 0.002,
                "weight_decay": 0.0005,
                "description": "Test transfer",
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    cfg = load_config("exp_transfer", config_path=str(config_file))

    assert cfg.name == "exp_transfer"
    assert cfg.model_name == "resnet18"
    assert cfg.method == "transfer"
    assert cfg.epochs == 7
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 0.002
    assert cfg.weight_decay == 0.0005
    # Merged from global
    assert cfg.val_split == 0.15
    assert cfg.num_classes == 10
    assert cfg.image_size == 128
    assert cfg.data_dir == "./custom_data"
    assert cfg.checkpoint_metric == "val_acc"


def test_load_config_overrides(tmp_path):
    """Verify overrides dict overrides both global and experiment settings."""
    sample_yaml = {
        "global": {
            "val_split": 0.1,
            "device": "cpu",
        },
        "experiments": {
            "exp_test": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    cfg = load_config(
        "exp_test",
        config_path=str(config_file),
        overrides={"epochs": 1, "batch_size": 64, "learning_rate": 0.0005}
    )

    assert cfg.epochs == 1
    assert cfg.batch_size == 64
    assert cfg.learning_rate == 0.0005


def test_load_config_nonexistent_experiment(tmp_path):
    """Verify loading non-existent experiment raises KeyError."""
    sample_yaml = {
        "global": {},
        "experiments": {
            "existing_exp": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    with pytest.raises(KeyError, match="not found"):
        load_config("nonexistent_exp", config_path=str(config_file))


@pytest.mark.parametrize("invalid_field,invalid_val,expected_match", [
    ("val_split", 0.7, "val_split must be between 0.0 and 0.5"),
    ("val_split", 0.0, "val_split must be between 0.0 and 0.5"),
    ("checkpoint_metric", "invalid_metric", "checkpoint_metric must be 'val_acc' or 'val_loss'"),
    ("epochs", -1, "epochs must be > 0"),
    ("batch_size", 0, "batch_size must be > 0"),
    ("learning_rate", 0.0, "learning_rate must be > 0"),
    ("model_name", "alexnet", "Invalid model_name"),
    ("method", "pretrain", "Invalid method"),
])
def test_load_config_invalid_values(tmp_path, invalid_field, invalid_val, expected_match):
    """Verify invalid configuration values raise ValueError."""
    sample_yaml = {
        "global": {
            "val_split": 0.1,
            "checkpoint_metric": "val_acc",
        },
        "experiments": {
            "exp_test": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    with pytest.raises(ValueError, match=expected_match):
        load_config("exp_test", config_path=str(config_file), overrides={invalid_field: invalid_val})
