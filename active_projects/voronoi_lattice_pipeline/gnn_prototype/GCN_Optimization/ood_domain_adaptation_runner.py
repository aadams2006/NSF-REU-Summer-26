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
    load_lattice_dataset,
    normalize_feature_splits,
    save_run_artifacts,
    set_seed,
)
from ood_validation_runner import (  # noqa: E402
    DOMAIN_FEATURES,
    OODExperimentConfig,
    _predictions_frame,
    _run_weighted_epoch,
    _save_training_history,
    build_ood_partition,
    train_ood_model,
)
from residual_error_analysis import load_lattice_feature_frame  # noqa: E402


VARIANTS = ("control", "domain_adapted")


@dataclass
class OODDomainAdaptationConfig:
    seed: int = 42
    partition_seed: int = 42
    adaptation_split_seed: int = 42
    ood_fraction: float = 0.10
    validation_fraction: float = 0.10
    adaptation_train_fraction: float = 0.40
    adaptation_validation_fraction: float = 0.10
    adaptation_repeats: int = 4
    preservation_weight: float = 0.50
    batch_size: int = 16
    hidden_dim: int = 24
    dropout: float = 0.10
    lr_phase1: float = 0.003
    lr_phase2: float = 0.0005
    epochs_phase1: int = 200
    epochs_phase2: int = 700
    patience: int = 150
    weight_decay: float = 1e-5
    finetune_lr: float = 1e-4
    finetune_epochs: int = 150
    finetune_patience: int = 30
    checkpoint_interval: int = 25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = True
    output_group: str = "gcn3_ood_domain_adaptation"

    def base_config(self) -> OODExperimentConfig:
        return OODExperimentConfig(
            variants=("control",),
            seed=self.seed,
            ood_fraction=self.ood_fraction,
            validation_fraction=self.validation_fraction,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            lr_phase1=self.lr_phase1,
            lr_phase2=self.lr_phase2,
            epochs_phase1=self.epochs_phase1,
            epochs_phase2=self.epochs_phase2,
            patience=self.patience,
            weight_decay=self.weight_decay,
            checkpoint_interval=self.checkpoint_interval,
            device=self.device,
            resume=self.resume,
            output_group=self.output_group,
        )


def validate_config(config: OODDomainAdaptationConfig) -> None:
    if not 0.0 < config.adaptation_train_fraction < 1.0:
        raise ValueError("adaptation_train_fraction must be between 0 and 1")
    if not 0.0 < config.adaptation_validation_fraction < 1.0:
        raise ValueError("adaptation_validation_fraction must be between 0 and 1")
    if config.adaptation_train_fraction + config.adaptation_validation_fraction >= 1.0:
        raise ValueError("Adaptation splits must leave an untouched OOD evaluation subset")
    if config.adaptation_repeats < 1:
        raise ValueError("adaptation_repeats must be at least 1")
    if not 0.0 <= config.preservation_weight <= 1.0:
        raise ValueError("preservation_weight must be between 0 and 1")
    if config.finetune_lr <= 0.0 or config.finetune_epochs < 1 or config.finetune_patience < 1:
        raise ValueError("Fine-tuning parameters must be positive")


def build_adaptation_partitions(
    source_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    config: OODDomainAdaptationConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    partition_config = config.base_config()
    partition_config.seed = config.partition_seed
    assignments, base_partitions = build_ood_partition(source_features, prediction_features, partition_config)
    ood_indices = base_partitions["ood"].copy()
    rng = np.random.default_rng(config.adaptation_split_seed)
    rng.shuffle(ood_indices)

    adaptation_train_count = max(1, int(round(len(ood_indices) * config.adaptation_train_fraction)))
    adaptation_validation_count = max(
        1,
        int(round(len(ood_indices) * config.adaptation_validation_fraction)),
    )
    if adaptation_train_count + adaptation_validation_count >= len(ood_indices):
        raise ValueError("Adaptation split leaves no OOD evaluation samples")

    adaptation_train = np.sort(ood_indices[:adaptation_train_count])
    adaptation_validation = np.sort(
        ood_indices[adaptation_train_count : adaptation_train_count + adaptation_validation_count]
    )
    ood_evaluation = np.sort(ood_indices[adaptation_train_count + adaptation_validation_count :])
    assignments["Partition"] = "Train"
    assignments.loc[base_partitions["validation"], "Partition"] = "Validation"
    assignments.loc[adaptation_train, "Partition"] = "Adaptation_Train"
    assignments.loc[adaptation_validation, "Partition"] = "Adaptation_Validation"
    assignments.loc[ood_evaluation, "Partition"] = "OOD_Evaluation"
    partitions = {
        "train": base_partitions["train"],
        "validation": base_partitions["validation"],
        "adaptation_train": adaptation_train,
        "adaptation_validation": adaptation_validation,
        "ood_evaluation": ood_evaluation,
    }
    return assignments, partitions


def prepare_data(
    raw_source_data: list[Data],
    raw_prediction_data: list[Data],
    partitions: dict[str, np.ndarray],
) -> tuple[dict[str, list[Data]], StandardScaler, StandardScaler]:
    datasets = {
        name: [deepcopy(raw_source_data[index]) for index in indices]
        for name, indices in partitions.items()
    }
    domain_data = (
        datasets["adaptation_train"]
        + datasets["adaptation_validation"]
        + datasets["ood_evaluation"]
    )
    feature_scaler = normalize_feature_splits(
        datasets["train"],
        datasets["validation"],
        domain_data,
    )
    target_scaler = normalize_target_splits(
        datasets["train"],
        datasets["validation"],
        domain_data,
    )
    prediction_data = deepcopy(raw_prediction_data)
    apply_feature_scaler(prediction_data, feature_scaler)
    apply_target_scaler(prediction_data, target_scaler)
    datasets["prediction"] = prediction_data
    return datasets, feature_scaler, target_scaler


def fine_tune_domain_model(
    model: torch.nn.Module,
    replay_loader: DataLoader,
    validation_loader: DataLoader,
    adaptation_validation_loader: DataLoader,
    config: OODDomainAdaptationConfig,
    checkpoint_path: Path,
) -> dict[str, float | int | list[float]]:
    model.to(config.device)
    optimizer = Adam(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
    history: dict[str, float | int | list[float]] = {
        "train_losses": [],
        "validation_losses": [],
        "adaptation_validation_losses": [],
        "selection_losses": [],
        "best_selection_loss": float("inf"),
        "best_epoch": 0,
        "epochs_completed": 0,
    }
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    start_epoch = 0
    checkpoint_config = asdict(config)

    if config.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        if checkpoint.get("config") != checkpoint_config:
            raise ValueError(f"Checkpoint configuration mismatch at {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_state = checkpoint["best_model_state_dict"]
        history = checkpoint["history"]
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["next_epoch"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if checkpoint.get("completed", False):
            model.load_state_dict(best_state)
            print("[domain_adapted] Loaded completed fine-tuning checkpoint")
            return history
        print(f"[domain_adapted] Resuming fine-tuning at epoch {start_epoch + 1}")

    def save_checkpoint(next_epoch: int, completed: bool = False) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": checkpoint_config,
            "completed": completed,
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "stale_epochs": stale_epochs,
            "next_epoch": next_epoch,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)

    for epoch in range(start_epoch, config.finetune_epochs):
        train_loss = _run_weighted_epoch(model, replay_loader, config.device, optimizer, False)
        validation_loss = _run_weighted_epoch(model, validation_loader, config.device, None, False)
        adaptation_validation_loss = _run_weighted_epoch(
            model,
            adaptation_validation_loader,
            config.device,
            None,
            False,
        )
        selection_loss = (
            config.preservation_weight * validation_loss
            + (1.0 - config.preservation_weight) * adaptation_validation_loss
        )
        history["train_losses"].append(train_loss)
        history["validation_losses"].append(validation_loss)
        history["adaptation_validation_losses"].append(adaptation_validation_loss)
        history["selection_losses"].append(selection_loss)
        history["epochs_completed"] = epoch + 1
        if selection_loss < float(history["best_selection_loss"]):
            history["best_selection_loss"] = selection_loss
            history["best_epoch"] = epoch + 1
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[domain_adapted] epoch {epoch + 1:>3}/{config.finetune_epochs} "
                f"replay={train_loss:.6f} val={validation_loss:.6f} "
                f"adapt_val={adaptation_validation_loss:.6f} selection={selection_loss:.6f}"
            )
        if config.checkpoint_interval > 0 and (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(epoch + 1)
        if stale_epochs >= config.finetune_patience:
            print("[domain_adapted] early stopping")
            break

    model.load_state_dict(best_state)
    save_checkpoint(int(history["epochs_completed"]), completed=True)
    return history


def _evaluate_variant(
    variant: str,
    model: torch.nn.Module,
    datasets: dict[str, list[Data]],
    partitions: dict[str, np.ndarray],
    source_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    target_scaler: StandardScaler,
    config: OODDomainAdaptationConfig,
    output_dir: Path,
    history: dict[str, object],
    feature_scaler: StandardScaler,
) -> dict[str, object]:
    dataset_specs = (
        ("Train", "train"),
        ("Validation", "validation"),
        ("Adaptation_Train", "adaptation_train"),
        ("Adaptation_Validation", "adaptation_validation"),
        ("OOD_Evaluation", "ood_evaluation"),
    )
    metrics: dict[str, dict[str, float]] = {}
    prediction_frames: list[pd.DataFrame] = []
    for dataset_label, key in dataset_specs:
        loader = DataLoader(datasets[key], batch_size=config.batch_size, shuffle=False)
        predictions, actual, dataset_metrics = evaluate_target_normalized_model(
            model,
            loader,
            target_scaler,
            device=config.device,
        )
        sample_ids = source_features.iloc[partitions[key]]["Sample_ID"].tolist()
        frame = _predictions_frame(dataset_label, sample_ids, predictions, actual)
        dataset_metrics["Bias"] = float(frame["Residual"].mean())
        metrics[dataset_label] = dataset_metrics
        prediction_frames.append(frame)

    prediction_results, prediction_metrics = predict_on_prepared_data_target_normalized(
        model,
        datasets["prediction"],
        target_scaler,
        batch_size=config.batch_size,
        device=config.device,
    )
    prediction_frame = _predictions_frame(
        "Prediction",
        prediction_features["Sample_ID"].tolist(),
        prediction_results["Predicted_Stiffness"].to_numpy(dtype=np.float32),
        prediction_results["Actual_Stiffness"].to_numpy(dtype=np.float32),
    )
    prediction_metrics["Bias"] = float(prediction_frame["Residual"].mean())
    metrics["Prediction"] = prediction_metrics
    prediction_frames.append(prediction_frame)

    variant_dir = output_dir / variant
    save_run_artifacts(
        variant_dir,
        model,
        feature_scaler,
        history,
        metrics,
        prediction_results=prediction_results,
    )
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(variant_dir / "all_predictions.csv", index=False)
    (variant_dir / "target_scaler.json").write_text(
        json.dumps(
            {"mean": float(target_scaler.mean_[0]), "scale": float(target_scaler.scale_[0])},
            indent=2,
        ),
        encoding="utf-8",
    )
    row: dict[str, float | int | str] = {
        "Variant": variant,
        "Seed": config.seed,
        "Base_Train_Count": len(datasets["train"]),
        "Adaptation_Train_Count": len(datasets["adaptation_train"]) if variant == "domain_adapted" else 0,
        "Adaptation_Repeats": config.adaptation_repeats if variant == "domain_adapted" else 0,
        "Best_Epoch": int(history.get("best_epoch", 0)),
        "Epochs_Completed": int(history.get("epochs_completed", 0)),
    }
    for dataset_name, dataset_metrics in metrics.items():
        for metric_name in ("RMSE", "MAE", "R2", "Bias"):
            row[f"{dataset_name}_{metric_name}"] = dataset_metrics[metric_name]
    summary = pd.DataFrame([row])
    summary.to_csv(variant_dir / "variant_summary.csv", index=False)
    return {"summary": summary, "predictions": all_predictions, "metrics": metrics}


def _save_adaptation_history(history: dict[str, object], save_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_losses"], label="Replay training", linewidth=2)
    axes[0].plot(history["validation_losses"], label="Ordinary validation", linewidth=2)
    axes[0].plot(
        history["adaptation_validation_losses"],
        label="Adaptation validation",
        linewidth=2,
    )
    axes[0].set(title="Domain adaptation losses", xlabel="Fine-tuning epoch", ylabel="MSE")
    axes[0].legend()
    axes[1].plot(history["selection_losses"], color="#b56576", linewidth=2)
    axes[1].set(title="Balanced selection loss", xlabel="Fine-tuning epoch", ylabel="Selection loss")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_adaptation_partition_figure(assignments: pd.DataFrame, save_path: Path) -> None:
    colors = {
        "Train": "#355070",
        "Validation": "#6d597a",
        "Adaptation_Train": "#e09f3e",
        "Adaptation_Validation": "#9e2a2b",
        "OOD_Evaluation": "#540b0e",
    }
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    for partition_name, frame in assignments.groupby("Partition", sort=True):
        axes[0].scatter(
            frame["Randomness_Value"],
            frame["Graph_Density"],
            color=colors[partition_name],
            alpha=0.65,
            s=28,
            label=partition_name,
        )
    axes[0].set(title="Domain-adaptation partition", xlabel="Randomness", ylabel="Graph density")
    axes[0].legend(fontsize=8)
    order = list(colors)
    axes[1].boxplot(
        [assignments.loc[assignments["Partition"] == name, "Domain_Distance"] for name in order],
        tick_labels=order,
    )
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Distance to external prediction domain", ylabel="Standardized nearest-neighbor distance")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_comparison_figure(comparison: pd.DataFrame, save_path: Path) -> None:
    datasets = ("Validation", "OOD_Evaluation", "Prediction")
    variants = comparison["Variant"].tolist()
    colors = {"control": "#355070", "domain_adapted": "#b56576"}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(datasets))
    width = 0.36
    for variant_index, (_, row) in enumerate(comparison.iterrows()):
        offset = (variant_index - 0.5) * width
        axes[0].bar(x + offset, [row[f"{name}_R2"] for name in datasets], width, color=colors[row["Variant"]], label=row["Variant"])
        axes[1].bar(x + offset, [row[f"{name}_RMSE"] for name in datasets], width, color=colors[row["Variant"]], label=row["Variant"])
        axes[2].bar(x + offset, [row[f"{name}_Bias"] for name in datasets], width, color=colors[row["Variant"]], label=row["Variant"])
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


def run_ood_domain_adaptation_experiment(
    config: OODDomainAdaptationConfig,
    train_root: str | Path,
    predict_root: str | Path,
    output_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_config(config)
    train_root = Path(train_root)
    predict_root = Path(predict_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path(output_root or Path(__file__).resolve().parent.parent / "outputs" / config.output_group)
        / f"run_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "experiment_config.json"
    manifest = {**asdict(config), "train_root": str(train_root), "predict_root": str(predict_root)}
    comparable = {key: value for key, value in manifest.items() if key not in {"device", "resume"}}
    if config.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_comparable = {key: value for key, value in existing.items() if key not in {"device", "resume"}}
        if existing_comparable != json.loads(json.dumps(comparable)):
            raise ValueError(f"Existing run configuration differs in {output_dir}; use a new run directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    source_features = load_lattice_feature_frame(train_root)
    prediction_features = load_lattice_feature_frame(predict_root)
    raw_source_data = load_lattice_dataset(train_root)
    raw_prediction_data = load_lattice_dataset(predict_root)
    assignments, partitions = build_adaptation_partitions(source_features, prediction_features, config)
    datasets, feature_scaler, target_scaler = prepare_data(raw_source_data, raw_prediction_data, partitions)

    assignments.to_csv(output_dir / "ood_adaptation_partition_assignments.csv", index=False)
    partition_summary = pd.DataFrame(
        [
            {
                "Partition": name,
                "Count": len(indices),
                "Mean_Domain_Distance": float(assignments.loc[indices, "Domain_Distance"].mean()),
                "Mean_Stiffness": float(assignments.loc[indices, "Actual_Stiffness"].mean()),
            }
            for name, indices in partitions.items()
        ]
    )
    partition_summary.to_csv(output_dir / "ood_adaptation_partition_summary.csv", index=False)
    _save_adaptation_partition_figure(assignments, output_dir / "ood_adaptation_partition_diagnostics.png")

    set_seed(config.seed)
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=config.batch_size, shuffle=False)
    architecture_config = ArchitectureConfig(
        architecture_name="gcn3",
        architecture_label="GCN-3 OOD Domain Adaptation",
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        seed=config.seed,
        device=config.device,
    )
    control_model = build_model(
        architecture_config,
        input_dim=datasets["train"][0].x.shape[1],
        graph_feature_dim=datasets["train"][0].graph_attr.shape[1],
    )
    base_history = train_ood_model(
        control_model,
        train_loader,
        validation_loader,
        config.base_config(),
        "control",
        output_dir / "checkpoints" / "base_training.pt",
    )
    control_result = _evaluate_variant(
        "control",
        control_model,
        datasets,
        partitions,
        source_features,
        prediction_features,
        target_scaler,
        config,
        output_dir,
        base_history,
        feature_scaler,
    )
    _save_training_history(base_history, config.epochs_phase1, output_dir / "control" / "training_history.png")

    adapted_model = deepcopy(control_model)
    replay_data = list(datasets["train"])
    for _ in range(config.adaptation_repeats):
        replay_data.extend(deepcopy(datasets["adaptation_train"]))
    set_seed(config.seed + 10_000)
    replay_loader = DataLoader(replay_data, batch_size=config.batch_size, shuffle=True)
    adaptation_validation_loader = DataLoader(
        datasets["adaptation_validation"],
        batch_size=config.batch_size,
        shuffle=False,
    )
    adaptation_history = fine_tune_domain_model(
        adapted_model,
        replay_loader,
        validation_loader,
        adaptation_validation_loader,
        config,
        output_dir / "checkpoints" / "domain_adaptation.pt",
    )
    adapted_result = _evaluate_variant(
        "domain_adapted",
        adapted_model,
        datasets,
        partitions,
        source_features,
        prediction_features,
        target_scaler,
        config,
        output_dir,
        adaptation_history,
        feature_scaler,
    )
    _save_adaptation_history(
        adaptation_history,
        output_dir / "domain_adapted" / "adaptation_history.png",
    )

    comparison = pd.concat(
        [control_result["summary"], adapted_result["summary"]],
        ignore_index=True,
    )
    comparison.to_csv(output_dir / "ood_domain_adaptation_comparison.csv", index=False)
    indexed = comparison.set_index("Variant")
    numeric_columns = comparison.select_dtypes(include=[np.number]).columns
    delta = {"Comparison": "domain_adapted_minus_control"}
    delta.update(
        {
            column: float(indexed.loc["domain_adapted", column] - indexed.loc["control", column])
            for column in numeric_columns
            if column != "Seed"
        }
    )
    pd.DataFrame([delta]).to_csv(output_dir / "ood_domain_adaptation_delta.csv", index=False)

    control_predictions = control_result["predictions"].rename(
        columns={"Absolute_Error": "Control_Absolute_Error"}
    )
    adapted_predictions = adapted_result["predictions"].rename(
        columns={"Absolute_Error": "Adapted_Absolute_Error"}
    )
    paired = control_predictions[["Dataset", "Sample_ID", "Control_Absolute_Error"]].merge(
        adapted_predictions[["Dataset", "Sample_ID", "Adapted_Absolute_Error"]],
        on=["Dataset", "Sample_ID"],
        validate="one_to_one",
    )
    paired["Absolute_Error_Reduction"] = paired["Control_Absolute_Error"] - paired["Adapted_Absolute_Error"]
    paired["Adapted_Improved"] = paired["Absolute_Error_Reduction"] > 0.0
    paired.to_csv(output_dir / "ood_domain_adaptation_paired_errors.csv", index=False)
    paired.groupby("Dataset").agg(
        Sample_Count=("Sample_ID", "count"),
        Improved_Count=("Adapted_Improved", "sum"),
        Mean_Absolute_Error_Reduction=("Absolute_Error_Reduction", "mean"),
        Median_Absolute_Error_Reduction=("Absolute_Error_Reduction", "median"),
    ).reset_index().to_csv(output_dir / "ood_domain_adaptation_paired_summary.csv", index=False)
    _save_comparison_figure(comparison, output_dir / "ood_domain_adaptation_comparison.png")
    (output_dir / "experiment_complete.json").write_text(
        json.dumps({"completed": True, "variants": list(VARIANTS)}, indent=2),
        encoding="utf-8",
    )
    print(f"OOD domain-adaptation experiment saved to {output_dir}")
    print(comparison)
    return {
        "output_dir": output_dir,
        "comparison": comparison,
        "partition_summary": partition_summary,
        "control_result": control_result,
        "adapted_result": adapted_result,
    }
