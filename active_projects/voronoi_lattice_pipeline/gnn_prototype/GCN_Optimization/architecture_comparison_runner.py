from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, SAGEConv, global_mean_pool

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from active_projects.voronoi_lattice_pipeline.gnn_prototype.GCN_Optimization.colab_gnn_stiffness_prototype import (  # noqa: E402
    SimpleGNN,
    create_data_loaders,
    compute_regression_metrics,
    default_data_roots,
    find_pipeline_root,
    load_lattice_dataset,
    normalize_feature_splits,
    save_run_artifacts,
    set_seed,
    split_dataset,
    summarize_metrics,
)


@dataclass
class ArchitectureConfig:
    architecture_name: str
    architecture_label: str | None = None
    batch_size: int = 16
    hidden_dim: int = 24
    dropout: float = 0.1
    lr_phase1: float = 0.003
    lr_phase2: float = 0.0005
    epochs_phase1: int = 200
    epochs_phase2: int = 700
    patience: int = 999
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_group: str = "architecture_comparison"

    @property
    def total_epochs(self) -> int:
        return self.epochs_phase1 + self.epochs_phase2

    @property
    def run_name(self) -> str:
        return self.architecture_name

    @property
    def display_name(self) -> str:
        return self.architecture_label or ARCHITECTURE_LABELS.get(self.architecture_name, self.architecture_name)


ARCHITECTURE_LABELS = {
    "gcn2_control": "GCN-2 Control",
    "gcn3": "GCN-3",
    "gin": "GIN",
    "graphsage": "GraphSAGE",
}


def _concat_graph_features(x: torch.Tensor, data: Data, graph_feature_dim: int) -> torch.Tensor:
    pooled = global_mean_pool(x, data.batch)
    if graph_feature_dim > 0 and hasattr(data, "graph_attr"):
        graph_attr = data.graph_attr
        if graph_attr.dim() == 1:
            graph_attr = graph_attr.unsqueeze(0)
        pooled = torch.cat((pooled, graph_attr), dim=1)
    return pooled


class ThreeLayerGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, graph_feature_dim: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.graph_feature_dim = graph_feature_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim + graph_feature_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)

        x = self.conv1(data.x, data.edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, data.edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return self.output(_concat_graph_features(x, data, self.graph_feature_dim))


class GINRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, graph_feature_dim: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.graph_feature_dim = graph_feature_dim
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=1,
            train_eps=True,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=1,
            train_eps=True,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim + graph_feature_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        edge_attr = getattr(data, "edge_weight", None)
        if edge_attr is None:
            edge_attr = torch.ones((data.edge_index.shape[1], 1), dtype=data.x.dtype, device=data.x.device)
        else:
            edge_attr = edge_attr.view(-1, 1)

        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return self.output(_concat_graph_features(x, data, self.graph_feature_dim))


class GraphSAGERegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, graph_feature_dim: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.graph_feature_dim = graph_feature_dim
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim + graph_feature_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        # SAGEConv does not consume scalar edge weights directly, so weighted connectivity
        # only enters through the derived node and graph summary features.
        x = self.conv1(data.x, data.edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return self.output(_concat_graph_features(x, data, self.graph_feature_dim))


def build_model(
    config: ArchitectureConfig,
    input_dim: int,
    graph_feature_dim: int,
) -> nn.Module:
    if config.architecture_name == "gcn2_control":
        return SimpleGNN(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            graph_feature_dim=graph_feature_dim,
        )
    if config.architecture_name == "gcn3":
        return ThreeLayerGCN(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            graph_feature_dim=graph_feature_dim,
            dropout=config.dropout,
        )
    if config.architecture_name == "gin":
        return GINRegressor(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            graph_feature_dim=graph_feature_dim,
            dropout=config.dropout,
        )
    if config.architecture_name == "graphsage":
        return GraphSAGERegressor(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            graph_feature_dim=graph_feature_dim,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported architecture: {config.architecture_name}")


def normalize_target_splits(
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data],
) -> StandardScaler:
    train_targets = np.asarray([float(sample.y.item()) for sample in train_data], dtype=np.float32).reshape(-1, 1)
    target_scaler = StandardScaler().fit(train_targets)

    for split in (train_data, val_data, test_data):
        for sample in split:
            scaled_target = target_scaler.transform(np.asarray([[float(sample.y.item())]], dtype=np.float32)).astype(
                np.float32
            )
            sample.y = torch.from_numpy(scaled_target.reshape(-1))

    return target_scaler


def apply_feature_scaler(dataset: list[Data], scaler: StandardScaler) -> None:
    for sample in dataset:
        transformed = scaler.transform(sample.x.numpy()).astype(np.float32)
        sample.x = torch.from_numpy(transformed)
        if hasattr(scaler, "graph_mean_") and hasattr(scaler, "graph_scale_"):
            graph_mean = np.asarray(scaler.graph_mean_, dtype=np.float32)
            graph_scale = np.asarray(scaler.graph_scale_, dtype=np.float32)
            graph_scale = np.where(graph_scale == 0.0, 1.0, graph_scale)
            graph_transformed = ((sample.graph_attr.numpy() - graph_mean) / graph_scale).astype(np.float32)
            sample.graph_attr = torch.from_numpy(graph_transformed)


def apply_target_scaler(dataset: list[Data], target_scaler: StandardScaler) -> None:
    for sample in dataset:
        scaled_target = target_scaler.transform(np.asarray([[float(sample.y.item())]], dtype=np.float32)).astype(
            np.float32
        )
        sample.y = torch.from_numpy(scaled_target.reshape(-1))


def inverse_transform_vector(values: np.ndarray, target_scaler: StandardScaler) -> np.ndarray:
    return target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1).astype(np.float32)


def run_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
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
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(data_loader.dataset)


def train_target_normalized_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: ArchitectureConfig,
) -> dict[str, float | list[float]]:
    model.to(config.device)
    criterion = nn.MSELoss()

    history: dict[str, float | list[float]] = {
        "train_losses": [],
        "val_losses": [],
        "best_val_loss": float("inf"),
        "epochs_completed": 0,
    }
    best_state = deepcopy(model.state_dict())

    phases = (
        ("phase_1", config.lr_phase1, config.epochs_phase1),
        ("phase_2", config.lr_phase2, config.epochs_phase2),
    )

    epochs_completed = 0
    for phase_name, learning_rate, phase_epochs in phases:
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
        stale_epochs = 0

        print(
            f"[{config.display_name}] Starting {phase_name}: "
            f"lr={learning_rate}, epochs={phase_epochs}"
        )
        for _ in range(phase_epochs):
            train_loss = run_epoch(model, train_loader, criterion, config.device, optimizer=optimizer)
            val_loss = run_epoch(model, val_loader, criterion, config.device)

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            epochs_completed += 1

            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epochs_completed == 1 or epochs_completed % 20 == 0:
                print(
                    f"[{config.display_name}] "
                    f"epoch {epochs_completed:>4}/{config.total_epochs} "
                    f"train={train_loss:.6f} val={val_loss:.6f}"
                )

            if stale_epochs >= config.patience:
                print(f"[{config.display_name}] early stopping triggered during {phase_name}")
                break

    model.load_state_dict(best_state)
    history["epochs_completed"] = epochs_completed
    return history


def evaluate_target_normalized_model(
    model: nn.Module,
    data_loader,
    target_scaler: StandardScaler,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    model.to(device)

    predictions: list[float] = []
    ground_truth: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            batch_predictions = model(batch).cpu().numpy().ravel()
            batch_targets = batch.y.cpu().numpy().ravel()
            predictions.extend(batch_predictions.tolist())
            ground_truth.extend(batch_targets.tolist())

    scaled_predictions = np.asarray(predictions, dtype=np.float32)
    scaled_ground_truth = np.asarray(ground_truth, dtype=np.float32)
    prediction_array = inverse_transform_vector(scaled_predictions, target_scaler)
    ground_truth_array = inverse_transform_vector(scaled_ground_truth, target_scaler)
    metrics = compute_regression_metrics(prediction_array, ground_truth_array)
    return prediction_array, ground_truth_array, metrics


def predict_on_directory_target_normalized(
    model: nn.Module,
    data_root: str | Path,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size: int = 1,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, float]]:
    prediction_data = load_lattice_dataset(data_root)
    apply_feature_scaler(prediction_data, feature_scaler)
    apply_target_scaler(prediction_data, target_scaler)

    loader = DataLoader(prediction_data, batch_size=batch_size, shuffle=False)
    predictions, ground_truth, metrics = evaluate_target_normalized_model(
        model,
        loader,
        target_scaler,
        device=device,
    )

    results = pd.DataFrame(
        {
            "Lattice_Index": np.arange(len(predictions)),
            "Predicted_Stiffness": predictions,
            "Actual_Stiffness": ground_truth,
            "Absolute_Error": np.abs(predictions - ground_truth),
            "Percent_Difference": np.abs(predictions - ground_truth)
            / np.clip(np.abs(ground_truth), a_min=np.finfo(np.float32).eps, a_max=None)
            * 100.0,
        }
    )
    return results, metrics


def build_training_history_figure(history: dict[str, float | list[float]], phase_1_epochs: int) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(history["train_losses"], label="Training loss", linewidth=2)
    axis.plot(history["val_losses"], label="Validation loss", linewidth=2)
    axis.axvline(phase_1_epochs, color="gray", linestyle="--", linewidth=1.5, label="Phase switch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("Training history")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    return fig


def build_prediction_splits_figure(split_results: list[tuple[str, np.ndarray, np.ndarray]]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(split_results), figsize=(6 * len(split_results), 5))
    if len(split_results) == 1:
        axes = [axes]

    for axis, (title, predictions, ground_truth) in zip(axes, split_results):
        axis.scatter(ground_truth, predictions, alpha=0.7, s=40)
        lower = min(ground_truth.min(), predictions.min())
        upper = max(ground_truth.max(), predictions.max())
        axis.plot([lower, upper], [lower, upper], "r--", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Actual stiffness")
        axis.set_ylabel("Predicted stiffness")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def build_result_summary(
    config: ArchitectureConfig,
    metrics_by_split: dict[str, dict[str, float]],
    prediction_metrics: dict[str, float],
    history: dict[str, float | list[float]],
    input_dim: int,
    graph_feature_dim: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Architecture": config.display_name,
                "Architecture_Key": config.architecture_name,
                "Seed": config.seed,
                "Input_Dim": input_dim,
                "Graph_Feature_Dim": graph_feature_dim,
                "Epochs_Completed": history["epochs_completed"],
                "Best_Val_Loss": history["best_val_loss"],
                "Validation_RMSE": metrics_by_split["Validation"]["RMSE"],
                "Validation_MAE": metrics_by_split["Validation"]["MAE"],
                "Validation_R2": metrics_by_split["Validation"]["R2"],
                "Test_RMSE": metrics_by_split["Test"]["RMSE"],
                "Test_MAE": metrics_by_split["Test"]["MAE"],
                "Test_R2": metrics_by_split["Test"]["R2"],
                "Prediction_RMSE": prediction_metrics["RMSE"],
                "Prediction_MAE": prediction_metrics["MAE"],
                "Prediction_R2": prediction_metrics["R2"],
            }
        ]
    )


def run_architecture_experiment(
    config: ArchitectureConfig,
    train_root: str | Path | None = None,
    predict_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    set_seed(config.seed)

    if train_root is None or predict_root is None:
        default_train_root, default_predict_root = default_data_roots()
        train_root = Path(train_root) if train_root is not None else default_train_root
        predict_root = Path(predict_root) if predict_root is not None else default_predict_root
    else:
        train_root = Path(train_root)
        predict_root = Path(predict_root)

    dataset = load_lattice_dataset(train_root)
    train_data, val_data, test_data = split_dataset(dataset, seed=config.seed)
    feature_scaler = normalize_feature_splits(train_data, val_data, test_data)
    target_scaler = normalize_target_splits(train_data, val_data, test_data)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=config.batch_size,
    )

    input_dim = train_data[0].x.shape[1]
    graph_feature_dim = train_data[0].graph_attr.shape[1]
    model = build_model(config, input_dim=input_dim, graph_feature_dim=graph_feature_dim)
    history = train_target_normalized_model(model, train_loader, val_loader, config)

    metrics_by_split: dict[str, dict[str, float]] = {}
    split_results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for split_name, loader in (
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ):
        predictions, ground_truth, metrics = evaluate_target_normalized_model(
            model,
            loader,
            target_scaler,
            device=config.device,
        )
        metrics_by_split[split_name] = metrics
        split_results.append((split_name, predictions, ground_truth))

    prediction_results, prediction_metrics = predict_on_directory_target_normalized(
        model,
        predict_root,
        feature_scaler,
        target_scaler,
        batch_size=1,
        device=config.device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_root is None:
        output_dir = (
            find_pipeline_root()
            / "gnn_prototype"
            / "outputs"
            / config.output_group
            / config.run_name
            / f"run_{timestamp}"
        )
    else:
        output_dir = Path(output_root) / config.run_name / f"run_{timestamp}"

    save_run_artifacts(
        output_dir,
        model,
        feature_scaler,
        history,
        metrics_by_split,
        prediction_results=prediction_results,
    )

    metrics_frame = summarize_metrics(metrics_by_split)
    metrics_frame.to_csv(output_dir / "metrics_summary.csv", index=True)
    summary_frame = build_result_summary(
        config,
        metrics_by_split,
        prediction_metrics,
        history,
        input_dim=input_dim,
        graph_feature_dim=graph_feature_dim,
    )
    summary_frame.to_csv(output_dir / "run_summary.csv", index=False)

    target_scaler_payload = {
        "mode": "standardize_y",
        "mean": float(target_scaler.mean_[0]),
        "scale": float(target_scaler.scale_[0]),
    }
    (output_dir / "target_scaler.json").write_text(json.dumps(target_scaler_payload, indent=2), encoding="utf-8")

    config_payload = {
        **asdict(config),
        "display_name": config.display_name,
        "input_dim": input_dim,
        "graph_feature_dim": graph_feature_dim,
        "prediction_metrics": prediction_metrics,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    history_figure = build_training_history_figure(history, config.epochs_phase1)
    history_figure.savefig(output_dir / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(history_figure)

    split_figure = build_prediction_splits_figure(split_results)
    split_figure.savefig(output_dir / "prediction_splits.png", dpi=200, bbox_inches="tight")
    plt.close(split_figure)

    print(f"[{config.display_name}] Metrics")
    print(metrics_frame)
    print(f"[{config.display_name}] Prediction metrics")
    print(pd.Series(prediction_metrics))
    print(f"[{config.display_name}] Saved artifacts to {output_dir}")

    return {
        "config": config,
        "output_dir": output_dir,
        "history": history,
        "metrics_by_split": metrics_by_split,
        "metrics_frame": metrics_frame,
        "prediction_metrics": prediction_metrics,
        "prediction_results": prediction_results,
        "summary_frame": summary_frame,
    }
