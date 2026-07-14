from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from architecture_comparison_runner import (  # noqa: E402
    ArchitectureConfig,
    apply_feature_scaler,
    apply_target_scaler,
    build_model,
    evaluate_target_normalized_model,
    normalize_target_splits,
    predict_on_prepared_data_target_normalized,
)
from colab_gnn_stiffness_prototype import (  # noqa: E402
    compute_regression_metrics,
    create_data_loaders,
    load_lattice_dataset,
    normalize_feature_splits,
    save_run_artifacts,
    set_seed,
)
from residual_error_analysis import load_lattice_feature_frame  # noqa: E402


DOMAIN_FEATURES = [
    "Randomness_Value",
    "Node_Count",
    "Graph_Density",
    "Std_Degree",
    "Mean_Center_Distance",
    "Std_Center_Distance",
]
SUPPORTED_VARIANTS = ("control", "domain_weighted")


@dataclass
class OODExperimentConfig:
    variants: tuple[str, ...] = SUPPORTED_VARIANTS
    seed: int = 42
    ood_fraction: float = 0.10
    validation_fraction: float = 0.10
    weight_strength: float = 2.0
    batch_size: int = 16
    hidden_dim: int = 24
    dropout: float = 0.10
    lr_phase1: float = 0.003
    lr_phase2: float = 0.0005
    epochs_phase1: int = 200
    epochs_phase2: int = 700
    patience: int = 150
    weight_decay: float = 1e-5
    checkpoint_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = True
    output_group: str = "gcn3_ood_domain_weighting"

    @property
    def total_epochs(self) -> int:
        return self.epochs_phase1 + self.epochs_phase2


def validate_config(config: OODExperimentConfig) -> None:
    unsupported = sorted(set(config.variants) - set(SUPPORTED_VARIANTS))
    if unsupported:
        raise ValueError(f"Unsupported OOD experiment variants: {unsupported}")
    if not config.variants:
        raise ValueError("At least one variant is required")
    if not 0.0 < config.ood_fraction < 0.5:
        raise ValueError("ood_fraction must be between 0 and 0.5")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if config.ood_fraction + config.validation_fraction >= 0.8:
        raise ValueError("OOD and validation fractions leave too little training data")
    if config.weight_strength < 0.0:
        raise ValueError("weight_strength must be non-negative")


def build_ood_partition(
    source_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    config: OODExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    source_matrix = source_features[DOMAIN_FEATURES].to_numpy(dtype=np.float64)
    prediction_matrix = prediction_features[DOMAIN_FEATURES].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(source_matrix)
    source_scaled = scaler.transform(source_matrix)
    prediction_scaled = scaler.transform(prediction_matrix)

    distances = np.sqrt(
        np.min(np.sum((source_scaled[:, None, :] - prediction_scaled[None, :, :]) ** 2, axis=2), axis=1)
    )
    closeness = pd.Series(distances).rank(method="average", ascending=False, pct=True).to_numpy(dtype=np.float64)
    sample_count = len(source_features)
    ood_count = max(1, int(round(sample_count * config.ood_fraction)))
    validation_count = max(1, int(round(sample_count * config.validation_fraction)))

    ordered_indices = np.argsort(distances, kind="stable")
    ood_indices = np.sort(ordered_indices[:ood_count])
    remaining_indices = np.setdiff1d(np.arange(sample_count), ood_indices, assume_unique=True)
    train_indices, validation_indices = train_test_split(
        remaining_indices,
        test_size=validation_count,
        shuffle=True,
        random_state=config.seed,
    )

    raw_weights = 1.0 + config.weight_strength * closeness[train_indices]
    normalized_weights = raw_weights / raw_weights.mean()
    assignments = source_features.copy()
    assignments["Domain_Distance"] = distances
    assignments["Domain_Closeness_Percentile"] = closeness
    assignments["Partition"] = "Train"
    assignments.loc[validation_indices, "Partition"] = "Validation"
    assignments.loc[ood_indices, "Partition"] = "OOD_Holdout"
    assignments["Domain_Weight"] = 0.0
    assignments.loc[train_indices, "Domain_Weight"] = normalized_weights

    partitions = {
        "train": np.asarray(train_indices, dtype=np.int64),
        "validation": np.asarray(validation_indices, dtype=np.int64),
        "ood": np.asarray(ood_indices, dtype=np.int64),
    }
    return assignments, partitions


def _prepare_variant_data(
    raw_source_data: list[Data],
    raw_prediction_data: list[Data],
    assignments: pd.DataFrame,
    partitions: dict[str, np.ndarray],
    use_domain_weights: bool,
) -> tuple[list[Data], list[Data], list[Data], list[Data], StandardScaler, StandardScaler]:
    train_data = [deepcopy(raw_source_data[index]) for index in partitions["train"]]
    validation_data = [deepcopy(raw_source_data[index]) for index in partitions["validation"]]
    ood_data = [deepcopy(raw_source_data[index]) for index in partitions["ood"]]
    prediction_data = deepcopy(raw_prediction_data)

    train_weights = assignments.loc[partitions["train"], "Domain_Weight"].to_numpy(dtype=np.float32)
    for sample, weight in zip(train_data, train_weights, strict=True):
        sample.domain_weight = torch.tensor([weight if use_domain_weights else 1.0], dtype=torch.float32)

    feature_scaler = normalize_feature_splits(train_data, validation_data, ood_data)
    target_scaler = normalize_target_splits(train_data, validation_data, ood_data)
    apply_feature_scaler(prediction_data, feature_scaler)
    apply_target_scaler(prediction_data, target_scaler)
    return train_data, validation_data, ood_data, prediction_data, feature_scaler, target_scaler


def _run_weighted_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    use_domain_weights: bool,
) -> float:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predictions = model(batch).view(-1)
        targets = batch.y.view(-1)
        per_sample_loss = (predictions - targets) ** 2
        if training and use_domain_weights:
            weights = batch.domain_weight.view(-1)
            loss = torch.sum(per_sample_loss * weights) / torch.sum(weights)
        else:
            loss = per_sample_loss.mean()
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / total_graphs


def train_ood_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: OODExperimentConfig,
    variant: str,
    checkpoint_path: Path,
) -> dict[str, float | int | list[float]]:
    use_domain_weights = variant == "domain_weighted"
    model.to(config.device)
    history: dict[str, float | int | list[float]] = {
        "train_losses": [],
        "val_losses": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "epochs_completed": 0,
    }
    best_state = deepcopy(model.state_dict())
    phases = (
        ("phase_1", config.lr_phase1, config.epochs_phase1),
        ("phase_2", config.lr_phase2, config.epochs_phase2),
    )
    checkpoint_config = {
        **asdict(config),
        "variant": variant,
    }
    start_phase = 0
    start_epoch = 0
    stale_epochs = 0
    optimizer_state = None

    if config.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        if checkpoint.get("config") != checkpoint_config:
            raise ValueError(f"Checkpoint configuration mismatch at {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        best_state = checkpoint["best_model_state_dict"]
        history = checkpoint["history"]
        if checkpoint.get("completed", False):
            model.load_state_dict(best_state)
            print(f"[{variant}] Loaded completed checkpoint")
            return history
        start_phase = int(checkpoint["phase_index"])
        start_epoch = int(checkpoint["epoch_in_phase"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        optimizer_state = checkpoint.get("optimizer_state_dict")
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(f"[{variant}] Resuming at epoch {int(history['epochs_completed']) + 1}")

    def save_checkpoint(
        phase_index: int,
        epoch_in_phase: int,
        optimizer: torch.optim.Optimizer | None,
        current_stale_epochs: int,
        completed: bool = False,
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": checkpoint_config,
            "completed": completed,
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "history": history,
            "phase_index": phase_index,
            "epoch_in_phase": epoch_in_phase,
            "stale_epochs": current_stale_epochs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)

    for phase_index, (phase_name, learning_rate, phase_epochs) in enumerate(phases):
        if phase_index < start_phase:
            continue
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
        epoch_start = start_epoch if phase_index == start_phase else 0
        current_stale_epochs = stale_epochs if phase_index == start_phase else 0
        if phase_index == start_phase and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        print(f"[{variant}] Starting {phase_name}: lr={learning_rate}, epochs={phase_epochs}")

        for epoch_in_phase in range(epoch_start, phase_epochs):
            train_loss = _run_weighted_epoch(
                model,
                train_loader,
                config.device,
                optimizer,
                use_domain_weights,
            )
            validation_loss = _run_weighted_epoch(
                model,
                validation_loader,
                config.device,
                optimizer=None,
                use_domain_weights=False,
            )
            history["train_losses"].append(train_loss)
            history["val_losses"].append(validation_loss)
            history["epochs_completed"] = int(history["epochs_completed"]) + 1
            if validation_loss < float(history["best_val_loss"]):
                history["best_val_loss"] = validation_loss
                history["best_epoch"] = int(history["epochs_completed"])
                best_state = deepcopy(model.state_dict())
                current_stale_epochs = 0
            else:
                current_stale_epochs += 1

            epochs_completed = int(history["epochs_completed"])
            if epochs_completed == 1 or epochs_completed % 20 == 0:
                print(
                    f"[{variant}] epoch {epochs_completed:>4}/{config.total_epochs} "
                    f"train={train_loss:.6f} val={validation_loss:.6f}"
                )
            if config.checkpoint_interval > 0 and epochs_completed % config.checkpoint_interval == 0:
                save_checkpoint(phase_index, epoch_in_phase + 1, optimizer, current_stale_epochs)
            if current_stale_epochs >= config.patience:
                print(f"[{variant}] early stopping during {phase_name}")
                break
        save_checkpoint(phase_index + 1, 0, None, 0)

    model.load_state_dict(best_state)
    save_checkpoint(len(phases), 0, None, 0, completed=True)
    return history


def _predictions_frame(
    dataset_name: str,
    sample_ids: list[str],
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> pd.DataFrame:
    residual = predictions - ground_truth
    return pd.DataFrame(
        {
            "Dataset": dataset_name,
            "Sample_ID": sample_ids,
            "Predicted_Stiffness": predictions,
            "Actual_Stiffness": ground_truth,
            "Residual": residual,
            "Absolute_Error": np.abs(residual),
        }
    )


def _variant_summary(
    variant: str,
    config: OODExperimentConfig,
    history: dict[str, float | int | list[float]],
    metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    row: dict[str, float | int | str | bool] = {
        "Variant": variant,
        "Uses_Domain_Weights": variant == "domain_weighted",
        "Seed": config.seed,
        "OOD_Fraction": config.ood_fraction,
        "Validation_Fraction": config.validation_fraction,
        "Weight_Strength": config.weight_strength if variant == "domain_weighted" else 0.0,
        "Epochs_Completed": int(history["epochs_completed"]),
        "Best_Epoch": int(history["best_epoch"]),
        "Best_Validation_Loss": float(history["best_val_loss"]),
    }
    for dataset_name, dataset_metrics in metrics.items():
        for metric_name in ("RMSE", "MAE", "R2"):
            row[f"{dataset_name}_{metric_name}"] = dataset_metrics[metric_name]
    return pd.DataFrame([row])


def _save_training_history(history: dict[str, float | int | list[float]], phase_1_epochs: int, save_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(history["train_losses"], label="Training loss", linewidth=2)
    axis.plot(history["val_losses"], label="Validation loss", linewidth=2)
    axis.axvline(phase_1_epochs, color="gray", linestyle="--", label="Phase switch")
    axis.set(title="Training history", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_ood_variant(
    variant: str,
    config: OODExperimentConfig,
    raw_source_data: list[Data],
    raw_prediction_data: list[Data],
    source_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    assignments: pd.DataFrame,
    partitions: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, object]:
    variant_dir = output_dir / variant
    summary_path = variant_dir / "variant_summary.csv"
    predictions_path = variant_dir / "all_predictions.csv"
    completion_path = variant_dir / "variant_complete.json"
    if config.resume and all(path.is_file() for path in (summary_path, predictions_path, completion_path)):
        print(f"[{variant}] Reusing completed variant")
        return {
            "summary": pd.read_csv(summary_path),
            "predictions": pd.read_csv(predictions_path),
            "output_dir": variant_dir,
        }

    set_seed(config.seed)
    train_data, validation_data, ood_data, prediction_data, feature_scaler, target_scaler = _prepare_variant_data(
        raw_source_data,
        raw_prediction_data,
        assignments,
        partitions,
        use_domain_weights=variant == "domain_weighted",
    )
    train_loader, validation_loader, ood_loader = create_data_loaders(
        train_data,
        validation_data,
        ood_data,
        batch_size=config.batch_size,
    )
    train_evaluation_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False)
    architecture_config = ArchitectureConfig(
        architecture_name="gcn3",
        architecture_label=f"GCN-3 OOD {variant}",
        batch_size=config.batch_size,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        lr_phase1=config.lr_phase1,
        lr_phase2=config.lr_phase2,
        epochs_phase1=config.epochs_phase1,
        epochs_phase2=config.epochs_phase2,
        patience=config.patience,
        weight_decay=config.weight_decay,
        seed=config.seed,
        split_seed=config.seed,
        device=config.device,
        output_group=config.output_group,
    )
    model = build_model(
        architecture_config,
        input_dim=train_data[0].x.shape[1],
        graph_feature_dim=train_data[0].graph_attr.shape[1],
    )
    history = train_ood_model(
        model,
        train_loader,
        validation_loader,
        config,
        variant,
        output_dir / "checkpoints" / f"{variant}_training.pt",
    )

    predictions_frames: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, float]] = {}
    split_specs = (
        ("Train", train_evaluation_loader, partitions["train"]),
        ("Validation", validation_loader, partitions["validation"]),
        ("OOD_Holdout", ood_loader, partitions["ood"]),
    )
    for dataset_name, loader, indices in split_specs:
        predictions, ground_truth, dataset_metrics = evaluate_target_normalized_model(
            model,
            loader,
            target_scaler,
            device=config.device,
        )
        metrics[dataset_name] = dataset_metrics
        sample_ids = source_features.iloc[indices]["Sample_ID"].tolist()
        predictions_frames.append(_predictions_frame(dataset_name, sample_ids, predictions, ground_truth))

    prediction_results, prediction_metrics = predict_on_prepared_data_target_normalized(
        model,
        prediction_data,
        target_scaler,
        batch_size=config.batch_size,
        device=config.device,
    )
    metrics["Prediction"] = prediction_metrics
    predictions_frames.append(
        _predictions_frame(
            "Prediction",
            prediction_features["Sample_ID"].tolist(),
            prediction_results["Predicted_Stiffness"].to_numpy(dtype=np.float32),
            prediction_results["Actual_Stiffness"].to_numpy(dtype=np.float32),
        )
    )
    all_predictions = pd.concat(predictions_frames, ignore_index=True)
    for dataset_name in metrics:
        dataset_predictions = all_predictions[all_predictions["Dataset"] == dataset_name]
        metrics[dataset_name]["Bias"] = float(dataset_predictions["Residual"].mean())

    summary = _variant_summary(variant, config, history, metrics)
    for dataset_name in metrics:
        summary[f"{dataset_name}_Bias"] = metrics[dataset_name]["Bias"]

    variant_dir.mkdir(parents=True, exist_ok=True)
    save_run_artifacts(
        variant_dir,
        model,
        feature_scaler,
        history,
        metrics,
        prediction_results=prediction_results,
    )
    summary.to_csv(summary_path, index=False)
    all_predictions.to_csv(predictions_path, index=False)
    (variant_dir / "target_scaler.json").write_text(
        json.dumps({"mean": float(target_scaler.mean_[0]), "scale": float(target_scaler.scale_[0])}, indent=2),
        encoding="utf-8",
    )
    _save_training_history(history, config.epochs_phase1, variant_dir / "training_history.png")
    completion_path.write_text(json.dumps({"completed": True, "variant": variant}, indent=2), encoding="utf-8")
    return {"summary": summary, "predictions": all_predictions, "output_dir": variant_dir}


def _save_domain_partition_figure(assignments: pd.DataFrame, save_path: Path) -> None:
    colors = {"Train": "#355070", "Validation": "#6d597a", "OOD_Holdout": "#b56576"}
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for partition_name, frame in assignments.groupby("Partition", sort=True):
        axes[0].scatter(
            frame["Randomness_Value"],
            frame["Graph_Density"],
            c=colors[partition_name],
            alpha=0.65,
            s=28,
            label=partition_name,
        )
    axes[0].set(title="OOD partition in structural feature space", xlabel="Randomness", ylabel="Graph density")
    axes[0].legend()
    ordered = [assignments.loc[assignments["Partition"] == name, "Domain_Distance"] for name in colors]
    axes[1].boxplot(ordered, tick_labels=list(colors))
    axes[1].set(title="Distance to prediction domain", ylabel="Standardized nearest-neighbor distance")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_variant_comparison(summary: pd.DataFrame, save_path: Path) -> None:
    datasets = ["Validation", "OOD_Holdout", "Prediction"]
    variants = summary["Variant"].tolist()
    x = np.arange(len(datasets))
    width = 0.8 / len(variants)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for variant_index, (_, row) in enumerate(summary.iterrows()):
        offset = (variant_index - (len(variants) - 1) / 2) * width
        axes[0].bar(x + offset, [row[f"{name}_R2"] for name in datasets], width, label=row["Variant"])
        axes[1].bar(x + offset, [row[f"{name}_RMSE"] for name in datasets], width, label=row["Variant"])
        axes[2].bar(x + offset, [row[f"{name}_Bias"] for name in datasets], width, label=row["Variant"])
    for axis, title, ylabel in (
        (axes[0], "R2 comparison", "R2"),
        (axes[1], "RMSE comparison", "RMSE"),
        (axes[2], "Signed bias comparison", "Prediction - actual"),
    ):
        axis.set_xticks(x, datasets, rotation=15)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_ood_experiment(
    config: OODExperimentConfig,
    train_root: str | Path,
    predict_root: str | Path,
    output_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_config(config)
    train_root = Path(train_root)
    predict_root = Path(predict_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir is not None:
        output_dir = Path(run_dir)
    elif output_root is not None:
        output_dir = Path(output_root) / f"run_{timestamp}"
    else:
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / config.output_group / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "experiment_config.json"
    manifest = {
        **asdict(config),
        "domain_features": DOMAIN_FEATURES,
        "train_root": str(train_root),
        "predict_root": str(predict_root),
    }
    comparable_manifest = {key: value for key, value in manifest.items() if key not in {"device", "resume", "variants"}}
    if config.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_comparable = {key: value for key, value in existing.items() if key not in {"device", "resume", "variants"}}
        if existing_comparable != json.loads(json.dumps(comparable_manifest)):
            raise ValueError(f"Existing run configuration differs in {output_dir}; use a new run directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    source_features = load_lattice_feature_frame(train_root)
    prediction_features = load_lattice_feature_frame(predict_root)
    raw_source_data = load_lattice_dataset(train_root)
    raw_prediction_data = load_lattice_dataset(predict_root)
    if len(raw_source_data) != len(source_features) or len(raw_prediction_data) != len(prediction_features):
        raise ValueError("Feature records and graph datasets do not align")

    assignments, partitions = build_ood_partition(source_features, prediction_features, config)
    assignments.to_csv(output_dir / "ood_partition_assignments.csv", index=False)
    pd.DataFrame(
        [
            {
                "Partition": name,
                "Count": len(indices),
                "Mean_Domain_Distance": float(assignments.loc[indices, "Domain_Distance"].mean()),
                "Mean_Stiffness": float(assignments.loc[indices, "Actual_Stiffness"].mean()),
            }
            for name, indices in partitions.items()
        ]
    ).to_csv(output_dir / "ood_partition_summary.csv", index=False)
    domain_summary_rows: list[dict[str, float | int | str]] = []
    partition_labels = {"train": "Train", "validation": "Validation", "ood": "OOD_Holdout"}
    for partition_name, indices in partitions.items():
        row: dict[str, float | int | str] = {
            "Dataset": partition_labels[partition_name],
            "Count": len(indices),
        }
        row.update(source_features.iloc[indices][DOMAIN_FEATURES].mean().to_dict())
        domain_summary_rows.append(row)
    prediction_row: dict[str, float | int | str] = {"Dataset": "Prediction", "Count": len(prediction_features)}
    prediction_row.update(prediction_features[DOMAIN_FEATURES].mean().to_dict())
    domain_summary_rows.append(prediction_row)
    pd.DataFrame(domain_summary_rows).to_csv(output_dir / "ood_domain_feature_summary.csv", index=False)
    _save_domain_partition_figure(assignments, output_dir / "ood_partition_diagnostics.png")

    variant_results: dict[str, dict[str, object]] = {}
    for variant in config.variants:
        variant_results[variant] = run_ood_variant(
            variant,
            config,
            raw_source_data,
            raw_prediction_data,
            source_features,
            prediction_features,
            assignments,
            partitions,
            output_dir,
        )

    comparison = pd.concat([result["summary"] for result in variant_results.values()], ignore_index=True)
    comparison.to_csv(output_dir / "ood_variant_comparison.csv", index=False)
    if {"control", "domain_weighted"}.issubset(comparison["Variant"]):
        control = comparison.set_index("Variant").loc["control"]
        weighted = comparison.set_index("Variant").loc["domain_weighted"]
        numeric_columns = comparison.select_dtypes(include=[np.number]).columns
        delta = {"Comparison": "domain_weighted_minus_control"}
        delta.update({column: float(weighted[column] - control[column]) for column in numeric_columns})
        pd.DataFrame([delta]).to_csv(output_dir / "ood_weighting_delta.csv", index=False)
    _save_variant_comparison(comparison, output_dir / "ood_variant_comparison.png")
    (output_dir / "experiment_complete.json").write_text(
        json.dumps({"completed": True, "variants": list(config.variants)}, indent=2),
        encoding="utf-8",
    )
    print(f"OOD experiment saved to {output_dir}")
    print(comparison)
    return {
        "output_dir": output_dir,
        "assignments": assignments,
        "partitions": partitions,
        "variant_results": variant_results,
        "comparison": comparison,
    }
