"""
Unit tests for history logging and results table generation.
"""

import csv
import json
import os
import pytest
from lab2.config import ExperimentConfig
from lab2.utils import save_history_json

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from make_results_table import generate_tables, collect_experiment_row


def test_save_history_json(tmp_path):
    """Verify save_history_json saves complete history and environment metadata."""
    cfg = ExperimentConfig(
        name="test_exp",
        model_name="resnet18",
        method="transfer",
        epochs=3,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.0001,
        device="cpu"
    )
    results = {
        "experiment_name": "test_exp",
        "total_time_seconds": 12.5,
        "best_epoch": 2,
        "best_metric": 85.0,
        "history": {
            "train_loss": [2.1, 1.5, 1.2],
            "train_acc": [40.0, 60.0, 75.0],
            "val_loss": [2.0, 1.4, 1.3],
            "val_acc": [45.0, 85.0, 80.0],
            "epoch_times": [4.0, 4.2, 4.3]
        }
    }
    output_path = tmp_path / "metrics" / "test_exp_history.json"
    save_history_json(results, cfg, str(output_path))

    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "test_exp"
    assert data["best_epoch"] == 2
    assert data["best_metric"] == 85.0
    assert data["total_time_seconds"] == 12.5
    assert data["config"]["model_name"] == "resnet18"
    assert data["config"]["method"] == "transfer"
    assert data["history"]["train_loss"] == [2.1, 1.5, 1.2]
    assert data["history"]["val_acc"] == [45.0, 85.0, 80.0]
    assert "environment" in data
    assert "torch_version" in data["environment"]
    assert "torchvision_version" in data["environment"]
    assert data["environment"]["device"] == "cpu"


def test_generate_tables_with_synthetic_data(tmp_path):
    """Verify generate_tables creates CSV and Markdown with matching values and handles missing files."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)

    # 1. Create complete synthetic history + test_metrics for exp1
    exp1_history = {
        "experiment_name": "exp1",
        "config": {
            "name": "exp1",
            "model_name": "resnet18",
            "method": "transfer",
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 0.001
        },
        "best_epoch": 3,
        "best_metric": 88.5,
        "total_time_seconds": 65.0,
        "history": {
            "train_loss": [2.0, 1.5, 1.2, 1.0, 0.8],
            "train_acc": [50.0, 60.0, 70.0, 80.0, 85.0],
            "epoch_times": [13.0, 13.0, 13.0, 13.0, 13.0]
        }
    }
    with open(metrics_dir / "exp1_history.json", "w", encoding="utf-8") as f:
        json.dump(exp1_history, f)

    exp1_test = {
        "test_loss": 0.85,
        "test_accuracy": 84.5
    }
    with open(metrics_dir / "exp1_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(exp1_test, f)

    # 2. Create only history (missing test_metrics) for exp2
    exp2_history = {
        "experiment_name": "exp2",
        "config": {
            "name": "exp2",
            "model_name": "densenet121",
            "method": "finetune",
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 0.001
        },
        "best_epoch": 4,
        "best_metric": 91.0,
        "total_time_seconds": 120.0,
        "history": {
            "train_loss": [1.5, 1.0, 0.8, 0.6, 0.5],
            "train_acc": [60.0, 75.0, 82.0, 88.0, 92.0],
            "epoch_times": [24.0, 24.0, 24.0, 24.0, 24.0]
        }
    }
    with open(metrics_dir / "exp2_history.json", "w", encoding="utf-8") as f:
        json.dump(exp2_history, f)

    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "results_tables.md"

    rows = generate_tables(
        metrics_dir=str(metrics_dir),
        config_path=str(tmp_path / "nonexistent_config.yaml"),
        output_csv=str(output_csv),
        write_md=str(output_md)
    )

    assert len(rows) == 2
    row_map = {r["Experiment ID"]: r for r in rows}

    # Verify exp1 values
    r1 = row_map["exp1"]
    assert r1["Epochs"] == "5"
    assert r1["Batch"] == "16"
    assert r1["LR"] == "0.001"
    assert r1["Trainable params"] == "5,130"
    assert r1["Train Loss/Acc (epoch cuối)"] == "0.8000 / 85.00%"
    assert r1["Best Val Acc (epoch)"] == "88.50% (ep 3)"
    assert r1["Test Loss"] == "0.8500"
    assert r1["Test Acc"] == "84.50%"

    # Verify exp2 missing test file produces N/A
    r2 = row_map["exp2"]
    assert r2["Test Loss"] == "N/A"
    assert r2["Test Acc"] == "N/A"
    assert r2["Trainable params"] == "2,170,378"

    # Verify CSV file output
    assert os.path.exists(output_csv)
    with open(output_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["Experiment ID"] in ["exp1", "exp2"]

    # Verify Markdown file output
    assert os.path.exists(output_md)
    with open(output_md, "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "| exp1 |" in md_text
        assert "| exp2 |" in md_text
        assert "| N/A |" in md_text
