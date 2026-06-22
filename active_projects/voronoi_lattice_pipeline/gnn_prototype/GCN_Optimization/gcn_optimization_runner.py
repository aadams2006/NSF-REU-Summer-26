from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from colab_gnn_stiffness_prototype import (  # noqa: E402
    SimpleGNN,
    create_data_loaders,
    default_data_roots,
    evaluate_model,
    find_pipeline_root,
    load_lattice_dataset,
    normalize_feature_splits,
    predict_on_directory,
    save_run_artifacts,
    set_seed,
    split_dataset,
    summarize_metrics,
)


@dataclass
class OptimizedGCNConfig:
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    batch_size: int = 16
    hidden_dim: int = 32
    max_epochs: int = 600
    patience: int = 60
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    momentum: float = 0.0
    nesterov: bool = False
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_optimizer(model: nn.Module, config: OptimizedGCNConfig) -> torch.optim.Optimizer:
    if config.optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def _run_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> float:
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    for batch in data_loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()

        predictions = model(batch)
        loss = criterion(predictions, batch.y.view(-1, 1))

        if training:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(data_loader.dataset)


def train_optimized_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: OptimizedGCNConfig,
) -> dict[str, float | list[float]]:
    model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )

    history: dict[str, float | list[float]] = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "best_val_loss": float("inf"),
        "epochs_completed": 0,
        "optimizer_name": config.optimizer_name,
    }
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0

    print(
        "Starting optimized training "
        f"optimizer={config.optimizer_name} "
        f"lr={config.learning_rate} "
        f"max_epochs={config.max_epochs}"
    )

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            config.device,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_loss = _run_epoch(model, val_loader, criterion, config.device)
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["learning_rates"].append(current_lr)

        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"  epoch {epoch:>4}/{config.max_epochs} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"lr={current_lr:.6g}"
            )

        if stale_epochs >= config.patience:
            print(f"  early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    history["epochs_completed"] = len(history["train_losses"])
    return history


def optimizer_run_name(config: OptimizedGCNConfig) -> str:
    if config.optimizer_name == "adamw":
        return "adamw"
    if config.nesterov:
        return "sgd_nesterov"
    if config.momentum > 0:
        return "sgd_momentum"
    return "sgd"


def run_experiment(config: OptimizedGCNConfig) -> Path:
    set_seed(config.seed)

    train_root, predict_root = default_data_roots()
    dataset = load_lattice_dataset(train_root)
    train_data, val_data, test_data = split_dataset(dataset, seed=config.seed)
    scaler = normalize_feature_splits(train_data, val_data, test_data)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=config.batch_size,
    )

    model = SimpleGNN(input_dim=train_data[0].x.shape[1], hidden_dim=config.hidden_dim)
    history = train_optimized_model(model, train_loader, val_loader, config)

    metrics_by_split: dict[str, dict[str, float]] = {}
    for split_name, loader in (
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ):
        _, _, metrics = evaluate_model(model, loader, device=config.device)
        metrics_by_split[split_name] = metrics

    print(summarize_metrics(metrics_by_split))

    prediction_results, prediction_metrics = predict_on_directory(
        model,
        predict_root,
        scaler,
        device=config.device,
    )
    print(prediction_results.head())
    print(prediction_metrics)

    run_name = optimizer_run_name(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        find_pipeline_root()
        / "gnn_prototype"
        / "GCN_Optimization"
        / "outputs"
        / run_name
        / f"run_{timestamp}"
    )
    save_run_artifacts(
        output_dir,
        model,
        scaler,
        history,
        metrics_by_split,
        prediction_results=prediction_results,
    )

    config_summary = {
        **asdict(config),
        "prediction_metrics": prediction_metrics,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_summary, indent=2),
        encoding="utf-8",
    )

    print(f"Saved artifacts to {output_dir}")
    return output_dir
