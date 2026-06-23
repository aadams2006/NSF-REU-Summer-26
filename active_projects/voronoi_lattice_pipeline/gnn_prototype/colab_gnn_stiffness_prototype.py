from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_pipeline_root(start_dir: str | Path | None = None) -> Path:
    start = Path(start_dir or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "gnn_prototype").is_dir() and (candidate / "datasets").is_dir():
            return candidate
        nested = candidate / "active_projects" / "voronoi_lattice_pipeline"
        if nested.is_dir():
            return nested.resolve()
    raise FileNotFoundError("Could not locate the voronoi_lattice_pipeline directory.")


def default_data_roots(pipeline_root: str | Path | None = None) -> tuple[Path, Path]:
    root = find_pipeline_root(pipeline_root)
    train_root = root / "source_archives" / "lattice_data" / "Randomness_Sweep"
    predict_root = root / "datasets" / "Lattice_Guess_Prediction_Input_Data"
    return train_root, predict_root


@dataclass
class TrainingConfig:
    batch_size: int = 16
    hidden_dim: int = 24
    lr_phase1: float = 0.003
    lr_phase2: float = 0.0005
    epochs_phase1: int = 200
    epochs_phase2: int = 700
    patience: int = 999
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def total_epochs(self) -> int:
        return self.epochs_phase1 + self.epochs_phase2


class SimpleGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)

        x = self.conv1(data.x, data.edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, data.batch)
        return self.output(x)


def _build_graph_edges(adjacency_matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    undirected_edges = np.argwhere(np.triu(adjacency_matrix, k=1) > 0)
    if undirected_edges.size == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    bidirectional_edges = np.vstack((undirected_edges, undirected_edges[:, ::-1]))
    edge_index = torch.from_numpy(bidirectional_edges.T.astype(np.int64)).contiguous()
    edge_weights = adjacency_matrix[undirected_edges[:, 0], undirected_edges[:, 1]].astype(np.float32)
    bidirectional_weights = np.concatenate((edge_weights, edge_weights))
    edge_weight = torch.from_numpy(bidirectional_weights).contiguous()
    return edge_index, edge_weight


def load_lattice_sample(folder_path: str | Path) -> Data:
    folder = Path(folder_path)
    node_features = pd.read_csv(folder / "node_features.csv", usecols=["x", "y"]).to_numpy(dtype=np.float32)
    adjacency_matrix = pd.read_csv(folder / "adjacency_area.csv", index_col=0).to_numpy()
    stiffness = float(pd.read_csv(folder / "lattice_stiffness.csv").iloc[0, 0])
    edge_index, edge_weight = _build_graph_edges(adjacency_matrix)

    return Data(
        x=torch.from_numpy(node_features),
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([stiffness], dtype=torch.float32),
    )


def load_lattice_dataset(data_root: str | Path) -> list[Data]:
    root = Path(data_root)
    folders = sorted(path for path in root.glob("randomness_*") if path.is_dir())
    if not folders:
        raise FileNotFoundError(f"No randomness folders found in {root}")

    dataset: list[Data] = []
    failed_folders: list[tuple[Path, Exception]] = []

    for folder in folders:
        try:
            dataset.append(load_lattice_sample(folder))
        except Exception as exc:  # pragma: no cover - keeps notebook runs from failing on one bad sample
            failed_folders.append((folder, exc))

    if not dataset:
        raise ValueError(f"Unable to load any lattice samples from {root}")

    print(f"Loaded {len(dataset)} lattice samples from {root}")
    if failed_folders:
        print(f"Skipped {len(failed_folders)} folders due to read errors.")
        for folder, exc in failed_folders[:5]:
            print(f"  {folder.name}: {exc}")

    return dataset


def split_dataset(
    dataset: list[Data],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    train_data, remainder = train_test_split(
        dataset,
        train_size=train_ratio,
        shuffle=True,
        random_state=seed,
    )
    relative_val_ratio = val_ratio / (1.0 - train_ratio)
    val_data, test_data = train_test_split(
        remainder,
        train_size=relative_val_ratio,
        shuffle=True,
        random_state=seed,
    )
    return train_data, val_data, test_data


def normalize_feature_splits(
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data],
) -> StandardScaler:
    train_features = torch.cat([sample.x for sample in train_data], dim=0).numpy()
    scaler = StandardScaler().fit(train_features)

    for split in (train_data, val_data, test_data):
        for sample in split:
            transformed = scaler.transform(sample.x.numpy()).astype(np.float32)
            sample.x = torch.from_numpy(transformed)

    return scaler


def create_data_loaders(
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
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

        print(f"Starting {phase_name}: lr={learning_rate}, epochs={phase_epochs}")
        for _ in range(phase_epochs):
            train_loss = _run_epoch(model, train_loader, criterion, config.device, optimizer=optimizer)
            val_loss = _run_epoch(model, val_loader, criterion, config.device)

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
                    f"  epoch {epochs_completed:>4}/{config.total_epochs} "
                    f"train={train_loss:.6f} val={val_loss:.6f}"
                )

            if stale_epochs >= config.patience:
                print(f"  early stopping triggered during {phase_name}")
                break

    model.load_state_dict(best_state)
    history["epochs_completed"] = epochs_completed
    return history


def _safe_r2_score(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    baseline = np.sum((ground_truth - ground_truth.mean()) ** 2)
    if baseline == 0:
        return float("nan")
    residual = np.sum((ground_truth - predictions) ** 2)
    return float(1 - residual / baseline)


def compute_regression_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    residuals = predictions - ground_truth
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    denominator = np.clip(np.abs(ground_truth), a_min=np.finfo(np.float32).eps, a_max=None)
    relative_errors = np.abs(residuals) / denominator * 100.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Mean Relative Error (%)": float(np.mean(relative_errors)),
        "Max Relative Error (%)": float(np.max(relative_errors)),
        "R2": _safe_r2_score(ground_truth, predictions),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
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
            predictions.extend(batch_predictions.tolist())
            ground_truth.extend(batch.y.cpu().numpy().ravel().tolist())

    prediction_array = np.asarray(predictions, dtype=np.float32)
    ground_truth_array = np.asarray(ground_truth, dtype=np.float32)
    metrics = compute_regression_metrics(prediction_array, ground_truth_array)
    return prediction_array, ground_truth_array, metrics


def summarize_metrics(metrics_by_split: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(metrics_by_split).T


def plot_training_history(history: dict[str, float | list[float]], phase_1_epochs: int) -> None:
    train_losses = history["train_losses"]
    val_losses = history["val_losses"]

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training loss", linewidth=2)
    plt.plot(val_losses, label="Validation loss", linewidth=2)
    plt.axvline(phase_1_epochs, color="gray", linestyle="--", linewidth=1.5, label="Phase switch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_splits(split_results: Iterable[tuple[str, np.ndarray, np.ndarray]]) -> None:
    results = list(split_results)
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for axis, (title, predictions, ground_truth) in zip(axes, results):
        axis.scatter(ground_truth, predictions, alpha=0.7, s=40)
        lower = min(ground_truth.min(), predictions.min())
        upper = max(ground_truth.max(), predictions.max())
        axis.plot([lower, upper], [lower, upper], "r--", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Actual stiffness")
        axis.set_ylabel("Predicted stiffness")
        axis.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def predict_on_directory(
    model: nn.Module,
    data_root: str | Path,
    scaler: StandardScaler,
    batch_size: int = 1,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, float]]:
    prediction_data = load_lattice_dataset(data_root)
    for sample in prediction_data:
        transformed = scaler.transform(sample.x.numpy()).astype(np.float32)
        sample.x = torch.from_numpy(transformed)

    loader = DataLoader(prediction_data, batch_size=batch_size, shuffle=False)
    predictions, ground_truth, metrics = evaluate_model(model, loader, device=device)

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


def save_run_artifacts(
    output_dir: str | Path,
    model: nn.Module,
    scaler: StandardScaler,
    history: dict[str, float | list[float]],
    metrics_by_split: dict[str, dict[str, float]],
    prediction_results: pd.DataFrame | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "history": history,
            "metrics": metrics_by_split,
        },
        output_path / "lattice_gnn_model.pt",
    )

    summarize_metrics(metrics_by_split).to_csv(output_path / "metrics_summary.csv", index=True)
    if prediction_results is not None:
        prediction_results.to_csv(output_path / "prediction_results.csv", index=False)

    return output_path


def main() -> None:
    config = TrainingConfig()
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
    history = train_model(model, train_loader, val_loader, config)

    metrics_by_split: dict[str, dict[str, float]] = {}
    split_results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for split_name, loader in (
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ):
        predictions, ground_truth, metrics = evaluate_model(model, loader, device=config.device)
        metrics_by_split[split_name] = metrics
        split_results.append((split_name, predictions, ground_truth))

    print(summarize_metrics(metrics_by_split))
    plot_training_history(history, config.epochs_phase1)
    plot_prediction_splits(split_results)

    prediction_results, prediction_metrics = predict_on_directory(
        model,
        predict_root,
        scaler,
        device=config.device,
    )
    print(prediction_results.head())
    print(pd.Series(prediction_metrics, name="Prediction Set"))

    output_dir = find_pipeline_root() / "gnn_prototype" / "outputs"
    save_run_artifacts(
        output_dir,
        model,
        scaler,
        history,
        metrics_by_split,
        prediction_results=prediction_results,
    )
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
