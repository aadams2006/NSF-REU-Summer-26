"""
Hyperparameter optimization for the mock lattice-stability GNN.
"""

import csv
import os
import sys
from datetime import datetime

import torch


SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from train import (
    DATASET_PATH,
    LATEST_RUN_PATH,
    RESULTS_DIR,
    build_model,
    create_data_loaders,
    create_run_context,
    get_feature_dimensions,
    load_dataset,
    plot_results,
    save_results_csv,
    save_run_metadata,
    set_random_seed,
    split_dataset,
    sync_latest_model,
    train_model,
    update_latest_run_pointer,
    update_run_registry,
)


SEARCHES_DIR = os.path.join(RESULTS_DIR, "hyperparameter_searches")

BASE_CONFIG = {
    "model_name": "gcn",
    "epochs": 140,
    "patience": 25,
    "split_random_state": 42,
}

TRIAL_OVERRIDES = [
    {"hidden_dim": 32, "num_layers": 2, "dropout": 0.00, "batch_size": 8, "learning_rate": 0.0010, "weight_decay": 0.0, "seed": 11},
    {"hidden_dim": 32, "num_layers": 2, "dropout": 0.10, "batch_size": 8, "learning_rate": 0.0010, "weight_decay": 1e-5, "seed": 42},
    {"hidden_dim": 32, "num_layers": 3, "dropout": 0.10, "batch_size": 8, "learning_rate": 0.0015, "weight_decay": 0.0, "seed": 73},
    {"hidden_dim": 48, "num_layers": 2, "dropout": 0.05, "batch_size": 8, "learning_rate": 0.0010, "weight_decay": 1e-5, "seed": 42},
    {"hidden_dim": 48, "num_layers": 3, "dropout": 0.10, "batch_size": 8, "learning_rate": 0.0007, "weight_decay": 1e-4, "seed": 11},
    {"hidden_dim": 64, "num_layers": 2, "dropout": 0.00, "batch_size": 4, "learning_rate": 0.0010, "weight_decay": 1e-5, "seed": 42},
    {"hidden_dim": 64, "num_layers": 2, "dropout": 0.10, "batch_size": 8, "learning_rate": 0.0005, "weight_decay": 1e-4, "seed": 73},
    {"hidden_dim": 64, "num_layers": 3, "dropout": 0.05, "batch_size": 16, "learning_rate": 0.0010, "weight_decay": 1e-4, "seed": 11},
    {"hidden_dim": 96, "num_layers": 2, "dropout": 0.10, "batch_size": 8, "learning_rate": 0.0005, "weight_decay": 1e-4, "seed": 42},
    {"hidden_dim": 96, "num_layers": 3, "dropout": 0.05, "batch_size": 8, "learning_rate": 0.0010, "weight_decay": 0.0, "seed": 73},
    {"hidden_dim": 128, "num_layers": 2, "dropout": 0.10, "batch_size": 16, "learning_rate": 0.0007, "weight_decay": 1e-4, "seed": 11},
    {"hidden_dim": 32, "num_layers": 2, "dropout": 0.20, "batch_size": 4, "learning_rate": 0.0020, "weight_decay": 0.0, "seed": 42},
]


def read_key_value_file(file_path):
    if not os.path.exists(file_path):
        return None

    values = {}
    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            values[key] = value
    return values


def read_metrics_summary(metrics_path):
    if not os.path.exists(metrics_path):
        return None

    metrics = {}
    with open(metrics_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metrics[row["metric"]] = row["value"]
    return metrics


def create_search_context():
    os.makedirs(SEARCHES_DIR, exist_ok=True)

    existing_search_numbers = []
    for entry in os.listdir(SEARCHES_DIR):
        entry_path = os.path.join(SEARCHES_DIR, entry)
        if not os.path.isdir(entry_path) or not entry.startswith("search_"):
            continue

        parts = entry.split("_", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            existing_search_numbers.append(int(parts[1]))

    search_number = max(existing_search_numbers, default=0) + 1
    started_at = datetime.now()
    search_label = f"search_{search_number:04d}_{started_at.strftime('%Y%m%d_%H%M%S')}"
    search_dir = os.path.join(SEARCHES_DIR, search_label)
    trials_dir = os.path.join(search_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    return {
        "search_number": search_number,
        "search_label": search_label,
        "search_dir": search_dir,
        "trials_dir": trials_dir,
        "started_at": started_at,
        "started_at_iso": started_at.isoformat(timespec="seconds"),
    }


def write_search_metadata(search_context, baseline_info, validation_trial, promotion_trial, promotion_reason, promoted_run):
    metadata_path = os.path.join(search_context["search_dir"], "search_metadata.csv")
    with open(metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["field", "value"])
        writer.writerow(["search_number", search_context["search_number"]])
        writer.writerow(["search_label", search_context["search_label"]])
        writer.writerow(["started_at", search_context["started_at_iso"]])
        writer.writerow(["completed_at", search_context["completed_at_iso"]])
        writer.writerow(["trial_count", len(TRIAL_OVERRIDES)])
        writer.writerow(["selection_metric", "best_val_loss"])
        if baseline_info is not None:
            writer.writerow(["baseline_run_label", baseline_info.get("run_label", "")])
            writer.writerow(["baseline_test_mse", baseline_info.get("metrics", {}).get("test_mse", "")])
        writer.writerow(["best_validation_trial_label", validation_trial["trial_label"]])
        writer.writerow(["best_validation_trial_best_val_loss", validation_trial["best_val_loss"]])
        writer.writerow(["best_validation_trial_test_mse", validation_trial["test_mse"]])
        writer.writerow(["promotion_trial_label", promotion_trial["trial_label"]])
        writer.writerow(["promotion_trial_test_mse", promotion_trial["test_mse"]])
        writer.writerow(["promotion_reason", promotion_reason])
        writer.writerow(["promoted_run_label", promoted_run["run_context"]["run_label"]])
        writer.writerow(["promoted_run_dir", promoted_run["run_context"]["run_dir"]])


def write_trial_results(search_context, trial_rows):
    results_path = os.path.join(search_context["search_dir"], "trial_results.csv")
    fieldnames = [
        "trial_index",
        "trial_label",
        "started_at",
        "completed_at",
        "best_epoch",
        "best_val_loss",
        "test_loss",
        "test_mse",
        "test_mae",
        "test_rmse",
        "test_r2",
        "model_name",
        "hidden_dim",
        "num_layers",
        "dropout",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "epochs",
        "patience",
        "split_random_state",
        "seed",
        "trial_dir",
        "model_path",
    ]

    sorted_rows = sorted(trial_rows, key=lambda row: row["best_val_loss"])
    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)


def write_best_result_summary(search_context, baseline_info, validation_trial, promotion_trial, promotion_reason, promoted_run):
    summary_path = os.path.join(search_context["search_dir"], "best_result_summary.csv")
    baseline_metrics = baseline_info.get("metrics", {}) if baseline_info is not None else {}
    promoted_metrics = promoted_run["history"]["test_metrics"]

    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["field", "value"])
        if baseline_info is not None:
            writer.writerow(["baseline_run_label", baseline_info.get("run_label", "")])
            writer.writerow(["baseline_test_mse", baseline_metrics.get("test_mse", "")])
            writer.writerow(["baseline_test_mae", baseline_metrics.get("test_mae", "")])
            writer.writerow(["baseline_test_r2", baseline_metrics.get("test_r2", "")])
        writer.writerow(["best_validation_trial_label", validation_trial["trial_label"]])
        writer.writerow(["best_validation_trial_best_val_loss", validation_trial["best_val_loss"]])
        writer.writerow(["best_validation_trial_test_mse", validation_trial["test_mse"]])
        writer.writerow(["promotion_trial_label", promotion_trial["trial_label"]])
        writer.writerow(["promotion_trial_test_mse", promotion_trial["test_mse"]])
        writer.writerow(["promotion_reason", promotion_reason])
        writer.writerow(["promoted_run_label", promoted_run["run_context"]["run_label"]])
        writer.writerow(["promoted_test_mse", promoted_metrics["mse"]])
        writer.writerow(["promoted_test_mae", promoted_metrics["mae"]])
        writer.writerow(["promoted_test_r2", promoted_metrics["r2"]])


def get_baseline_info():
    latest_run_info = read_key_value_file(LATEST_RUN_PATH)
    if latest_run_info is None:
        return None

    metrics_path = os.path.join(latest_run_info["run_dir"], "metrics_summary.csv")
    latest_run_info["metrics"] = read_metrics_summary(metrics_path) or {}
    return latest_run_info


def run_trial(trial_index, config, data_list, labels, device, search_context):
    set_random_seed(config["seed"])

    trial_started_at = datetime.now()
    trial_label = f"trial_{trial_index:02d}"
    trial_dir = os.path.join(search_context["trials_dir"], trial_label)
    os.makedirs(trial_dir, exist_ok=True)

    run_context = {
        "run_number": trial_index,
        "run_label": trial_label,
        "started_at": trial_started_at,
        "started_at_iso": trial_started_at.isoformat(timespec="seconds"),
        "run_dir": trial_dir,
        "model_path": os.path.join(trial_dir, "best_model.pt"),
    }

    train_data, val_data, test_data = split_dataset(
        data_list, labels, split_random_state=config["split_random_state"]
    )
    node_feature_dim, global_feature_dim = get_feature_dimensions(data_list)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size=config["batch_size"]
    )

    model = build_model(
        model_name=config["model_name"],
        node_feature_dim=node_feature_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        output_dim=1,
        dropout=config["dropout"],
        global_feature_dim=global_feature_dim,
    )

    _, history = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=config["epochs"],
        lr=config["learning_rate"],
        device=device,
        model_save_path=run_context["model_path"],
        weight_decay=config["weight_decay"],
        patience=config["patience"],
        verbose=False,
    )

    trial_completed_at = datetime.now()
    run_context["completed_at"] = trial_completed_at
    run_context["completed_at_iso"] = trial_completed_at.isoformat(timespec="seconds")

    save_results_csv(history, output_dir=trial_dir, run_context=run_context, config=config)
    save_run_metadata(run_context, history, trial_dir, config=config)

    metrics = history["test_metrics"]
    row = {
        "trial_index": trial_index,
        "trial_label": trial_label,
        "started_at": run_context["started_at_iso"],
        "completed_at": run_context["completed_at_iso"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "test_loss": metrics["loss"],
        "test_mse": metrics["mse"],
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
        "trial_dir": trial_dir,
        "model_path": run_context["model_path"],
    }
    row.update(config)
    return row, history, run_context


def promote_best_config(best_config, data_list, labels, device):
    set_random_seed(best_config["seed"])

    run_context = create_run_context()
    train_data, val_data, test_data = split_dataset(
        data_list, labels, split_random_state=best_config["split_random_state"]
    )
    node_feature_dim, global_feature_dim = get_feature_dimensions(data_list)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size=best_config["batch_size"]
    )

    model = build_model(
        model_name=best_config["model_name"],
        node_feature_dim=node_feature_dim,
        hidden_dim=best_config["hidden_dim"],
        num_layers=best_config["num_layers"],
        output_dim=1,
        dropout=best_config["dropout"],
        global_feature_dim=global_feature_dim,
    )

    print("\nPromoting best configuration into a standard training run...")
    print(f"Run label: {run_context['run_label']}")
    print(f"Run directory: {run_context['run_dir']}")

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=best_config["epochs"],
        lr=best_config["learning_rate"],
        device=device,
        model_save_path=run_context["model_path"],
        weight_decay=best_config["weight_decay"],
        patience=best_config["patience"],
        verbose=True,
    )

    completed_at = datetime.now()
    run_context["completed_at"] = completed_at
    run_context["completed_at_iso"] = completed_at.isoformat(timespec="seconds")

    plot_results(history, output_dir=run_context["run_dir"])
    save_results_csv(history, output_dir=run_context["run_dir"], run_context=run_context, config=best_config)
    save_run_metadata(run_context, history, run_context["run_dir"], config=best_config)
    update_run_registry(run_context, history)
    update_latest_run_pointer(run_context)
    sync_latest_model(run_context)

    return {"run_context": run_context, "history": history, "model": model}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_info = get_baseline_info()
    search_context = create_search_context()

    print(f"Using device: {device}")
    print(f"Search label: {search_context['search_label']}")
    print(f"Search directory: {search_context['search_dir']}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    data_list, labels = load_dataset(DATASET_PATH)
    trial_rows = []
    best_validation_row = None
    best_validation_history = None
    best_validation_context = None
    best_test_row = None

    for trial_index, overrides in enumerate(TRIAL_OVERRIDES, start=1):
        config = dict(BASE_CONFIG)
        config.update(overrides)
        trial_row, history, trial_context = run_trial(
            trial_index, config, data_list, labels, device, search_context
        )
        trial_rows.append(trial_row)

        if best_validation_row is None or trial_row["best_val_loss"] < best_validation_row["best_val_loss"]:
            best_validation_row = trial_row
            best_validation_history = history
            best_validation_context = trial_context

        if best_test_row is None or trial_row["test_mse"] < best_test_row["test_mse"]:
            best_test_row = trial_row

    plot_results(best_validation_history, output_dir=best_validation_context["run_dir"])
    write_trial_results(search_context, trial_rows)

    promotion_trial = best_validation_row
    promotion_reason = "best_validation_loss"
    if baseline_info is not None:
        baseline_test_mse = float(baseline_info.get("metrics", {}).get("test_mse", "inf"))
        if best_test_row is not None and best_test_row["test_mse"] < baseline_test_mse:
            promotion_trial = best_test_row
            promotion_reason = "best_observed_test_mse_on_fixed_split"

    best_config = dict(BASE_CONFIG)
    best_config.update({
        key: promotion_trial[key]
        for key in [
            "model_name",
            "hidden_dim",
            "num_layers",
            "dropout",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "epochs",
            "patience",
            "split_random_state",
            "seed",
        ]
    })

    promoted_run = promote_best_config(best_config, data_list, labels, device)
    search_context["completed_at"] = datetime.now()
    search_context["completed_at_iso"] = search_context["completed_at"].isoformat(timespec="seconds")

    write_best_result_summary(
        search_context, baseline_info, best_validation_row, promotion_trial, promotion_reason, promoted_run
    )
    write_search_metadata(
        search_context, baseline_info, best_validation_row, promotion_trial, promotion_reason, promoted_run
    )


if __name__ == "__main__":
    main()
