from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from architecture_comparison_runner import ARCHITECTURE_LABELS, ArchitectureConfig, run_architecture_experiment
from colab_gnn_stiffness_prototype import compute_regression_metrics, default_data_roots, find_pipeline_root


@dataclass
class EnsembleConfig:
    architecture_name: str = "gcn3"
    architecture_label: str | None = None
    member_seeds: tuple[int, ...] = (11, 42, 73, 101, 202)
    split_seed: int = 42
    batch_size: int = 16
    hidden_dim: int = 24
    dropout: float = 0.1
    lr_phase1: float = 0.003
    lr_phase2: float = 0.0005
    epochs_phase1: int = 200
    epochs_phase2: int = 700
    patience: int = 999
    weight_decay: float = 1e-5
    loss_name: str = "mse"
    huber_beta: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_group: str = "gcn3_ensemble_fixed_split"

    @property
    def display_name(self) -> str:
        base = self.architecture_label or ARCHITECTURE_LABELS.get(self.architecture_name, self.architecture_name)
        return f"{base} Ensemble"


@dataclass
class MultiSplitEnsembleConfig:
    architecture_name: str = "gcn3"
    architecture_label: str | None = None
    member_seeds: tuple[int, ...] = (11, 42, 73, 101, 202)
    split_seeds: tuple[int, ...] = (11, 42, 73, 101, 202)
    batch_size: int = 16
    hidden_dim: int = 24
    dropout: float = 0.1
    lr_phase1: float = 0.003
    lr_phase2: float = 0.0005
    epochs_phase1: int = 200
    epochs_phase2: int = 700
    patience: int = 999
    weight_decay: float = 1e-5
    loss_name: str = "mse"
    huber_beta: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_group: str = "gcn3_ensemble_multi_split"

    @property
    def display_name(self) -> str:
        base = self.architecture_label or ARCHITECTURE_LABELS.get(self.architecture_name, self.architecture_name)
        return f"{base} Ensemble Multi-Split"


def _ensure_shared_ground_truth(split_name: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    if not np.allclose(reference, candidate, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{split_name} ground truth differs across ensemble members. Check split seeding.")


def _average_predictions(predictions_by_member: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(predictions_by_member, axis=0), axis=0).astype(np.float32)


def build_ensemble_metrics_frame(metrics_by_split: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(metrics_by_split).T


def build_aggregate_frame(summary_frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    aggregate = summary_frame[metric_columns].agg(["mean", "std", "min", "max"]).T.reset_index()
    aggregate.columns = ["Metric", "mean", "std", "min", "max"]
    return aggregate


def build_member_metric_figure(
    member_summary: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
    save_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    metrics = [
        ("Test_R2", "Test R2"),
        ("Test_RMSE", "Test RMSE"),
        ("Prediction_R2", "Prediction R2"),
        ("Validation_R2", "Validation R2"),
    ]

    ensemble_row = ensemble_summary.iloc[0]
    for axis, (metric_key, title) in zip(axes, metrics):
        axis.plot(member_summary["Seed"], member_summary[metric_key], marker="o", linewidth=2, color="#355070")
        axis.axhline(
            float(ensemble_row[metric_key]),
            color="#b56576",
            linestyle="--",
            linewidth=2,
            label="Ensemble",
        )
        axis.set_title(title)
        axis.set_xlabel("Model seed")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_multi_split_metric_figure(summary_frame: pd.DataFrame, save_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    metrics = [
        ("Test_R2", "Test R2"),
        ("Test_RMSE", "Test RMSE"),
        ("Prediction_R2", "Prediction R2"),
        ("Validation_R2", "Validation R2"),
    ]

    for axis, (metric_key, title) in zip(axes, metrics):
        axis.plot(summary_frame["Split_Seed"], summary_frame[metric_key], marker="o", linewidth=2, color="#6d597a")
        axis.set_title(title)
        axis.set_xlabel("Split seed")
        axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_ensemble_prediction_figure(
    split_results: list[tuple[str, np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(split_results), figsize=(6 * len(split_results), 5))
    if len(split_results) == 1:
        axes = [axes]

    for axis, (title, predictions, ground_truth) in zip(axes, split_results):
        axis.scatter(ground_truth, predictions, alpha=0.7, s=40, color="#355070")
        lower = min(float(ground_truth.min()), float(predictions.min()))
        upper = max(float(ground_truth.max()), float(predictions.max()))
        axis.plot([lower, upper], [lower, upper], "r--", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Actual stiffness")
        axis.set_ylabel("Predicted stiffness")
        axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_ensemble_summary(
    config: EnsembleConfig,
    metrics_by_split: dict[str, dict[str, float]],
    prediction_metrics: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Architecture": config.display_name,
                "Architecture_Key": config.architecture_name,
                "Member_Count": len(config.member_seeds),
                "Member_Seeds": ",".join(str(seed) for seed in config.member_seeds),
                "Split_Seed": config.split_seed,
                "Hidden_Dim": config.hidden_dim,
                "Dropout": config.dropout,
                "Weight_Decay": config.weight_decay,
                "LR_Phase1": config.lr_phase1,
                "LR_Phase2": config.lr_phase2,
                "Loss_Name": config.loss_name,
                "Huber_Beta": config.huber_beta,
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


def run_fixed_split_ensemble(
    config: EnsembleConfig,
    train_root: str | Path | None = None,
    predict_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    if train_root is None or predict_root is None:
        default_train_root, default_predict_root = default_data_roots()
        train_root = Path(train_root) if train_root is not None else default_train_root
        predict_root = Path(predict_root) if predict_root is not None else default_predict_root
    else:
        train_root = Path(train_root)
        predict_root = Path(predict_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_root is None:
        output_dir = find_pipeline_root() / "gnn_prototype" / "outputs" / config.output_group / f"run_{timestamp}"
    else:
        output_dir = Path(output_root) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model_output_root = output_dir / "per_model"

    member_results: dict[int, dict[str, object]] = {}
    member_rows: list[dict[str, object]] = []
    val_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    prediction_set_predictions: list[np.ndarray] = []
    val_ground_truth: np.ndarray | None = None
    test_ground_truth: np.ndarray | None = None
    prediction_set_ground_truth: np.ndarray | None = None

    for member_seed in config.member_seeds:
        member_config = ArchitectureConfig(
            architecture_name=config.architecture_name,
            architecture_label=config.architecture_label or ARCHITECTURE_LABELS.get(config.architecture_name),
            batch_size=config.batch_size,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            lr_phase1=config.lr_phase1,
            lr_phase2=config.lr_phase2,
            epochs_phase1=config.epochs_phase1,
            epochs_phase2=config.epochs_phase2,
            patience=config.patience,
            weight_decay=config.weight_decay,
            loss_name=config.loss_name,
            huber_beta=config.huber_beta,
            seed=member_seed,
            split_seed=config.split_seed,
            device=config.device,
            output_group=config.output_group,
        )
        result = run_architecture_experiment(
            member_config,
            train_root=train_root,
            predict_root=predict_root,
            output_root=per_model_output_root,
        )
        member_results[member_seed] = result

        row = result["summary_frame"].iloc[0].to_dict()
        row["Output_Dir"] = str(result["output_dir"])
        member_rows.append(row)

        validation_output = result["split_outputs"]["Validation"]
        test_output = result["split_outputs"]["Test"]
        prediction_frame = result["prediction_results"]

        current_val_ground_truth = validation_output["ground_truth"]
        current_test_ground_truth = test_output["ground_truth"]
        current_prediction_ground_truth = prediction_frame["Actual_Stiffness"].to_numpy(dtype=np.float32)

        if val_ground_truth is None:
            val_ground_truth = current_val_ground_truth
            test_ground_truth = current_test_ground_truth
            prediction_set_ground_truth = current_prediction_ground_truth
        else:
            _ensure_shared_ground_truth("Validation", val_ground_truth, current_val_ground_truth)
            _ensure_shared_ground_truth("Test", test_ground_truth, current_test_ground_truth)
            _ensure_shared_ground_truth("Prediction set", prediction_set_ground_truth, current_prediction_ground_truth)

        val_predictions.append(validation_output["predictions"])
        test_predictions.append(test_output["predictions"])
        prediction_set_predictions.append(prediction_frame["Predicted_Stiffness"].to_numpy(dtype=np.float32))

    assert val_ground_truth is not None
    assert test_ground_truth is not None
    assert prediction_set_ground_truth is not None

    ensemble_val_predictions = _average_predictions(val_predictions)
    ensemble_test_predictions = _average_predictions(test_predictions)
    ensemble_prediction_predictions = _average_predictions(prediction_set_predictions)

    metrics_by_split = {
        "Validation": compute_regression_metrics(ensemble_val_predictions, val_ground_truth),
        "Test": compute_regression_metrics(ensemble_test_predictions, test_ground_truth),
    }
    prediction_metrics = compute_regression_metrics(
        ensemble_prediction_predictions,
        prediction_set_ground_truth,
    )

    member_summary = pd.DataFrame(member_rows).sort_values("Seed").reset_index(drop=True)
    ensemble_summary = build_ensemble_summary(config, metrics_by_split, prediction_metrics)
    ensemble_metrics_frame = build_ensemble_metrics_frame(
        {
            **metrics_by_split,
            "Prediction": prediction_metrics,
        }
    )
    ensemble_prediction_results = pd.DataFrame(
        {
            "Lattice_Index": np.arange(len(ensemble_prediction_predictions)),
            "Predicted_Stiffness": ensemble_prediction_predictions,
            "Actual_Stiffness": prediction_set_ground_truth,
            "Absolute_Error": np.abs(ensemble_prediction_predictions - prediction_set_ground_truth),
            "Percent_Difference": np.abs(ensemble_prediction_predictions - prediction_set_ground_truth)
            / np.clip(np.abs(prediction_set_ground_truth), a_min=np.finfo(np.float32).eps, a_max=None)
            * 100.0,
        }
    )

    member_summary_path = output_dir / "gcn3_ensemble_member_summary.csv"
    ensemble_summary_path = output_dir / "gcn3_ensemble_summary.csv"
    ensemble_metrics_path = output_dir / "gcn3_ensemble_metrics_summary.csv"
    ensemble_predictions_path = output_dir / "gcn3_ensemble_prediction_results.csv"
    metadata_path = output_dir / "gcn3_ensemble_run_metadata.json"
    member_plot_path = output_dir / "gcn3_ensemble_member_metrics.png"
    prediction_plot_path = output_dir / "gcn3_ensemble_prediction_splits.png"

    member_summary.to_csv(member_summary_path, index=False)
    ensemble_summary.to_csv(ensemble_summary_path, index=False)
    ensemble_metrics_frame.to_csv(ensemble_metrics_path, index=True)
    ensemble_prediction_results.to_csv(ensemble_predictions_path, index=False)

    metadata = {
        **asdict(config),
        "display_name": config.display_name,
        "train_root": str(train_root),
        "predict_root": str(predict_root),
        "member_output_dirs": {
            str(seed): str(result["output_dir"])
            for seed, result in member_results.items()
        },
        "ensemble_summary_path": str(ensemble_summary_path),
        "member_summary_path": str(member_summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    build_member_metric_figure(member_summary, ensemble_summary, save_path=member_plot_path)
    build_ensemble_prediction_figure(
        [
            ("Validation", ensemble_val_predictions, val_ground_truth),
            ("Test", ensemble_test_predictions, test_ground_truth),
            ("Prediction Set", ensemble_prediction_predictions, prediction_set_ground_truth),
        ],
        save_path=prediction_plot_path,
    )

    print(f"[{config.display_name}] Member summary")
    print(member_summary)
    print(f"[{config.display_name}] Ensemble summary")
    print(ensemble_summary)
    print(f"[{config.display_name}] Saved artifacts to {output_dir}")

    return {
        "output_dir": output_dir,
        "member_results": member_results,
        "member_summary": member_summary,
        "ensemble_summary": ensemble_summary,
        "ensemble_metrics_frame": ensemble_metrics_frame,
        "ensemble_prediction_results": ensemble_prediction_results,
    }


def run_multi_split_ensemble(
    config: MultiSplitEnsembleConfig,
    train_root: str | Path | None = None,
    predict_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    if train_root is None or predict_root is None:
        default_train_root, default_predict_root = default_data_roots()
        train_root = Path(train_root) if train_root is not None else default_train_root
        predict_root = Path(predict_root) if predict_root is not None else default_predict_root
    else:
        train_root = Path(train_root)
        predict_root = Path(predict_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_root is None:
        output_dir = find_pipeline_root() / "gnn_prototype" / "outputs" / config.output_group / f"run_{timestamp}"
    else:
        output_dir = Path(output_root) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split_output_root = output_dir / "per_split"
    split_results: dict[int, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []

    for split_seed in config.split_seeds:
        split_config = EnsembleConfig(
            architecture_name=config.architecture_name,
            architecture_label=config.architecture_label,
            member_seeds=config.member_seeds,
            split_seed=split_seed,
            batch_size=config.batch_size,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            lr_phase1=config.lr_phase1,
            lr_phase2=config.lr_phase2,
            epochs_phase1=config.epochs_phase1,
            epochs_phase2=config.epochs_phase2,
            patience=config.patience,
            weight_decay=config.weight_decay,
            loss_name=config.loss_name,
            huber_beta=config.huber_beta,
            device=config.device,
            output_group=config.output_group,
        )
        result = run_fixed_split_ensemble(
            split_config,
            train_root=train_root,
            predict_root=predict_root,
            output_root=per_split_output_root,
        )
        split_results[split_seed] = result

        row = result["ensemble_summary"].iloc[0].to_dict()
        row["Output_Dir"] = str(result["output_dir"])
        summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows).sort_values("Split_Seed").reset_index(drop=True)
    metric_columns = [
        "Validation_R2",
        "Test_R2",
        "Test_RMSE",
        "Prediction_R2",
        "Prediction_RMSE",
    ]
    aggregate_frame = build_aggregate_frame(summary_frame, metric_columns=metric_columns)

    summary_path = output_dir / "gcn3_ensemble_multi_split_summary.csv"
    aggregate_path = output_dir / "gcn3_ensemble_multi_split_aggregate.csv"
    best_split_path = output_dir / "gcn3_ensemble_multi_split_best_split_summary.json"
    metadata_path = output_dir / "gcn3_ensemble_multi_split_run_metadata.json"
    metric_plot_path = output_dir / "gcn3_ensemble_multi_split_metric_summary.png"

    summary_frame.to_csv(summary_path, index=False)
    aggregate_frame.to_csv(aggregate_path, index=False)
    build_multi_split_metric_figure(summary_frame, save_path=metric_plot_path)

    best_split_row = summary_frame.sort_values("Test_R2", ascending=False).iloc[0]
    best_split_path.write_text(json.dumps(best_split_row.to_dict(), indent=2), encoding="utf-8")

    metadata = {
        **asdict(config),
        "display_name": config.display_name,
        "train_root": str(train_root),
        "predict_root": str(predict_root),
        "split_output_dirs": {
            str(split_seed): str(result["output_dir"])
            for split_seed, result in split_results.items()
        },
        "summary_path": str(summary_path),
        "aggregate_path": str(aggregate_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[{config.display_name}] Split summary")
    print(summary_frame)
    print(f"[{config.display_name}] Aggregate summary")
    print(aggregate_frame)
    print(f"[{config.display_name}] Saved artifacts to {output_dir}")

    return {
        "output_dir": output_dir,
        "split_results": split_results,
        "summary_frame": summary_frame,
        "aggregate_frame": aggregate_frame,
    }
