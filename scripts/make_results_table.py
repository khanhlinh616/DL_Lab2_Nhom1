"""
Generate summary table from experiment result JSONs.
Produces outputs/metrics/summary.csv and prints Markdown table to stdout.
Supports optional --write argument to save Markdown to a file.
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lab2.config import load_config, load_yaml, get_all_experiment_names
from lab2.models import build_model, count_parameters
from lab2.utils import format_time

# Configure UTF-8 for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_trainable_params(model_name: Optional[str], method: Optional[str], num_classes: int = 10) -> str:
    """Computes trainable parameter count for a model configuration (pretrained=False)."""
    if not model_name or not method:
        return "N/A"
    try:
        model = build_model(model_name=model_name, method=method, num_classes=num_classes, pretrained=False)
        params = count_parameters(model)
        return f"{params['trainable']:,}"
    except Exception:
        return "N/A"


def collect_experiment_row(
    exp_id: str,
    metrics_dir: str,
    config_path: str
) -> Dict[str, str]:
    """Collects metrics and config values for a single experiment ID."""
    history_file = os.path.join(metrics_dir, f"{exp_id}_history.json")
    test_file = os.path.join(metrics_dir, f"{exp_id}_test_metrics.json")

    history_data: Optional[Dict[str, Any]] = None
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except Exception:
            history_data = None

    test_data: Optional[Dict[str, Any]] = None
    if os.path.exists(test_file):
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
        except Exception:
            test_data = None

    # Load configuration from history, config.yaml, or test_data
    cfg_dict = {}
    if history_data and isinstance(history_data.get("config"), dict):
        cfg_dict = history_data["config"]
    elif os.path.exists(config_path):
        try:
            cfg = load_config(exp_id, config_path=config_path)
            cfg_dict = vars(cfg)
        except Exception:
            cfg_dict = {}

    epochs = str(cfg_dict.get("epochs")) if cfg_dict.get("epochs") is not None else "N/A"
    batch = str(cfg_dict.get("batch_size")) if cfg_dict.get("batch_size") is not None else "N/A"
    lr = str(cfg_dict.get("learning_rate")) if cfg_dict.get("learning_rate") is not None else "N/A"

    model_name = cfg_dict.get("model_name")
    method = cfg_dict.get("method")
    trainable_params = get_trainable_params(model_name, method)

    # Train Loss / Acc (epoch cuối)
    train_loss_acc = "N/A"
    avg_epoch_time = "N/A"
    best_val_acc = "N/A"
    total_time_str = "N/A"

    if history_data:
        h = history_data.get("history", {})
        t_losses = h.get("train_loss", [])
        t_accs = h.get("train_acc", [])
        if t_losses and t_accs:
            train_loss_acc = f"{t_losses[-1]:.4f} / {t_accs[-1]:.2f}%"

        best_metric = history_data.get("best_metric")
        best_epoch = history_data.get("best_epoch")
        if best_metric is not None and best_epoch is not None:
            best_val_acc = f"{best_metric:.2f}% (ep {best_epoch})"

        epoch_times = h.get("epoch_times", [])
        if epoch_times:
            avg_sec = sum(epoch_times) / len(epoch_times)
            avg_epoch_time = format_time(avg_sec)

        tot_sec = history_data.get("total_time_seconds")
        if tot_sec is not None:
            total_time_str = format_time(tot_sec)

    # Test metrics
    test_loss = "N/A"
    test_acc = "N/A"
    if test_data:
        tl = test_data.get("test_loss")
        ta = test_data.get("test_accuracy")
        if tl is not None:
            test_loss = f"{tl:.4f}"
        if ta is not None:
            test_acc = f"{ta:.2f}%"

    return {
        "Experiment ID": exp_id,
        "Epochs": epochs,
        "Batch": batch,
        "LR": lr,
        "Trainable params": trainable_params,
        "Train Loss/Acc (epoch cuối)": train_loss_acc,
        "Best Val Acc (epoch)": best_val_acc,
        "Test Loss": test_loss,
        "Test Acc": test_acc,
        "Thời gian TB mỗi epoch": avg_epoch_time,
        "Tổng thời gian": total_time_str
    }


def generate_tables(
    metrics_dir: str = "./outputs/metrics",
    config_path: str = "./configs/config.yaml",
    output_csv: str = "./outputs/metrics/summary.csv",
    write_md: Optional[str] = None
) -> List[Dict[str, str]]:
    """Generates summary CSV and Markdown table."""
    # Resolve relative paths against PROJECT_ROOT if not found in current working directory
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        candidate = os.path.join(PROJECT_ROOT, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    if not os.path.isabs(metrics_dir) and not os.path.exists(metrics_dir):
        candidate = os.path.join(PROJECT_ROOT, metrics_dir)
        if os.path.exists(candidate):
            metrics_dir = candidate

    # Discover all experiment IDs from config.yaml and available JSON files
    exp_ids = []
    if os.path.exists(config_path):
        try:
            exp_ids.extend(get_all_experiment_names(config_path))
        except Exception:
            pass

    # Also discover any metrics files in metrics_dir
    if os.path.exists(metrics_dir):
        for pattern in ["*_history.json", "*_test_metrics.json"]:
            for path in glob.glob(os.path.join(metrics_dir, pattern)):
                fname = os.path.basename(path)
                if fname.endswith("_history.json"):
                    eid = fname[:-len("_history.json")]
                else:
                    eid = fname[:-len("_test_metrics.json")]
                if eid not in exp_ids:
                    exp_ids.append(eid)

    columns = [
        "Experiment ID",
        "Epochs",
        "Batch",
        "LR",
        "Trainable params",
        "Train Loss/Acc (epoch cuối)",
        "Best Val Acc (epoch)",
        "Test Loss",
        "Test Acc",
        "Thời gian TB mỗi epoch",
        "Tổng thời gian"
    ]

    rows = []
    for eid in exp_ids:
        row = collect_experiment_row(eid, metrics_dir=metrics_dir, config_path=config_path)
        rows.append(row)

    # Save summary CSV
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    # Build Markdown table string
    header_line = "| " + " | ".join(columns) + " |"
    separator_line = "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(columns))]) + " |"
    table_lines = [header_line, separator_line]
    for r in rows:
        line = "| " + " | ".join([str(r.get(c, "N/A")) for c in columns]) + " |"
        table_lines.append(line)
    md_content = "\n".join(table_lines)

    # Print to stdout
    print(md_content)

    # Write Markdown to file if requested
    if write_md:
        os.makedirs(os.path.dirname(os.path.abspath(write_md)), exist_ok=True)
        with open(write_md, "w", encoding="utf-8") as f:
            f.write(md_content + "\n")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables from experiment results.")
    parser.add_argument("--metrics-dir", type=str, default="./outputs/metrics", help="Directory containing metrics JSONs.")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--output-csv", type=str, default="./outputs/metrics/summary.csv", help="Path to save summary.csv.")
    parser.add_argument("--write", type=str, default=None, help="Optional path to write markdown table (e.g. docs/results_tables.md).")

    args = parser.parse_args()
    generate_tables(
        metrics_dir=args.metrics_dir,
        config_path=args.config,
        output_csv=args.output_csv,
        write_md=args.write
    )


if __name__ == "__main__":
    main()
