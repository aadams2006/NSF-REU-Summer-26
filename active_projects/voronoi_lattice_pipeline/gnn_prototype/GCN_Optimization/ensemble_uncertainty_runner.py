from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, t

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from colab_gnn_stiffness_prototype import compute_regression_metrics  # noqa: E402
from ensemble_runner import EnsembleConfig, run_fixed_split_ensemble  # noqa: E402


@dataclass
class EnsembleUncertaintyConfig:
    member_seeds: tuple[int, ...] = (11, 42, 73, 101, 202)
    split_seed: int = 42
    confidence_levels: tuple[float, ...] = (0.80, 0.90, 0.95)
    batch_size: int = 16
    hidden_dim: int = 24
    dropout: float = 0.10
    lr_phase1: float = 0.003
    lr_phase2: float = 0.0005
    epochs_phase1: int = 200
    epochs_phase2: int = 700
    patience: int = 999
    weight_decay: float = 1e-5
    checkpoint_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = True
    output_group: str = "gcn3_ensemble_uncertainty"

    def ensemble_config(self) -> EnsembleConfig:
        return EnsembleConfig(
            architecture_name="gcn3",
            architecture_label="GCN-3",
            member_seeds=self.member_seeds,
            split_seed=self.split_seed,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            lr_phase1=self.lr_phase1,
            lr_phase2=self.lr_phase2,
            epochs_phase1=self.epochs_phase1,
            epochs_phase2=self.epochs_phase2,
            patience=self.patience,
            weight_decay=self.weight_decay,
            loss_name="mse",
            device=self.device,
            output_group=self.output_group,
            checkpoint_interval=self.checkpoint_interval,
            resume=self.resume,
        )


def validate_config(config: EnsembleUncertaintyConfig) -> None:
    if len(config.member_seeds) < 2:
        raise ValueError("At least two ensemble members are required for uncertainty estimates")
    if len(set(config.member_seeds)) != len(config.member_seeds):
        raise ValueError("member_seeds must be unique")
    if not config.confidence_levels:
        raise ValueError("At least one confidence level is required")
    if any(not 0.0 < level < 1.0 for level in config.confidence_levels):
        raise ValueError("confidence_levels must be between 0 and 1")
    if len(set(config.confidence_levels)) != len(config.confidence_levels):
        raise ValueError("confidence_levels must be unique")


def _level_label(level: float) -> str:
    return str(int(round(level * 100)))


def build_uncertainty_frame(
    dataset_name: str,
    member_seeds: tuple[int, ...],
    member_predictions: np.ndarray,
    actual: np.ndarray,
    confidence_levels: tuple[float, ...],
    sample_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = np.asarray(member_predictions, dtype=np.float64)
    ground_truth = np.asarray(actual, dtype=np.float64).reshape(-1)
    if predictions.ndim != 2:
        raise ValueError("member_predictions must have shape [member, sample]")
    if predictions.shape[0] != len(member_seeds):
        raise ValueError("member prediction rows do not match member_seeds")
    if predictions.shape[1] != len(ground_truth):
        raise ValueError("member predictions and ground truth have different sample counts")
    if sample_ids is None:
        sample_ids = [f"{dataset_name}_{index:04d}" for index in range(len(ground_truth))]
    if len(sample_ids) != len(ground_truth):
        raise ValueError("sample_ids and ground truth have different sample counts")

    member_count = predictions.shape[0]
    mean_prediction = predictions.mean(axis=0)
    median_prediction = np.median(predictions, axis=0)
    member_std = predictions.std(axis=0, ddof=1)
    standard_error = member_std / np.sqrt(member_count)
    residual = mean_prediction - ground_truth
    frame = pd.DataFrame(
        {
            "Dataset": dataset_name,
            "Sample_Index": np.arange(len(ground_truth)),
            "Sample_ID": sample_ids,
            "Actual_Stiffness": ground_truth,
            "Ensemble_Mean": mean_prediction,
            "Ensemble_Median": median_prediction,
            "Residual": residual,
            "Absolute_Error": np.abs(residual),
            "Member_Count": member_count,
            "Member_Std": member_std,
            "Member_Standard_Error": standard_error,
            "Member_Min": predictions.min(axis=0),
            "Member_Max": predictions.max(axis=0),
            "Member_Range": np.ptp(predictions, axis=0),
            "Relative_Member_Std_Percent": member_std
            / np.clip(np.abs(mean_prediction), np.finfo(np.float64).eps, None)
            * 100.0,
        }
    )

    for level in confidence_levels:
        label = _level_label(level)
        critical_value = float(t.ppf((1.0 + level) / 2.0, df=member_count - 1))
        half_width = critical_value * standard_error
        lower = mean_prediction - half_width
        upper = mean_prediction + half_width
        frame[f"Mean_CI_{label}_Lower"] = lower
        frame[f"Mean_CI_{label}_Upper"] = upper
        frame[f"Mean_CI_{label}_Half_Width"] = half_width
        frame[f"Actual_Within_Mean_CI_{label}"] = (ground_truth >= lower) & (ground_truth <= upper)

    deviation_rows: list[dict[str, float | int | str]] = []
    safe_std = np.where(member_std > np.finfo(np.float64).eps, member_std, np.nan)
    for member_index, seed in enumerate(member_seeds):
        member_values = predictions[member_index]
        deviations = member_values - mean_prediction
        frame[f"Member_Seed_{seed}_Prediction"] = member_values
        frame[f"Member_Seed_{seed}_Deviation"] = deviations
        for sample_index, (sample_id, prediction, deviation) in enumerate(
            zip(sample_ids, member_values, deviations, strict=True)
        ):
            deviation_rows.append(
                {
                    "Dataset": dataset_name,
                    "Sample_Index": sample_index,
                    "Sample_ID": sample_id,
                    "Member_Seed": seed,
                    "Member_Prediction": float(prediction),
                    "Ensemble_Mean": float(mean_prediction[sample_index]),
                    "Signed_Deviation": float(deviation),
                    "Absolute_Deviation": float(abs(deviation)),
                    "Standardized_Deviation": float(deviation / safe_std[sample_index]),
                }
            )
    return frame, pd.DataFrame(deviation_rows)


def build_uncertainty_summary(
    uncertainty_frame: pd.DataFrame,
    confidence_levels: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset_name, frame in uncertainty_frame.groupby("Dataset", sort=False):
        predictions = frame["Ensemble_Mean"].to_numpy(dtype=np.float64)
        actual = frame["Actual_Stiffness"].to_numpy(dtype=np.float64)
        spread = frame["Member_Std"].to_numpy(dtype=np.float64)
        absolute_error = frame["Absolute_Error"].to_numpy(dtype=np.float64)
        metrics = compute_regression_metrics(predictions, actual)
        if np.std(spread) > 0.0 and np.std(absolute_error) > 0.0:
            pearson = float(np.corrcoef(spread, absolute_error)[0, 1])
            spearman = float(spearmanr(spread, absolute_error).statistic)
        else:
            pearson = float("nan")
            spearman = float("nan")
        row: dict[str, float | int | str] = {
            "Dataset": dataset_name,
            "Sample_Count": len(frame),
            "Member_Count": int(frame["Member_Count"].iloc[0]),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Bias": float(frame["Residual"].mean()),
            "Mean_Member_Std": float(spread.mean()),
            "Median_Member_Std": float(np.median(spread)),
            "Max_Member_Std": float(spread.max()),
            "Mean_Member_Range": float(frame["Member_Range"].mean()),
            "Pearson_Std_vs_Absolute_Error": pearson,
            "Spearman_Std_vs_Absolute_Error": spearman,
        }
        for level in confidence_levels:
            label = _level_label(level)
            row[f"Nominal_Mean_CI_{label}_Percent"] = level * 100.0
            row[f"Empirical_Mean_CI_{label}_Coverage_Percent"] = float(
                frame[f"Actual_Within_Mean_CI_{label}"].mean() * 100.0
            )
            row[f"Mean_CI_{label}_Mean_Width"] = float(
                2.0 * frame[f"Mean_CI_{label}_Half_Width"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_member_deviation_summary(member_deviations: pd.DataFrame) -> pd.DataFrame:
    return (
        member_deviations.groupby(["Dataset", "Member_Seed"], as_index=False)
        .agg(
            Mean_Signed_Deviation=("Signed_Deviation", "mean"),
            Mean_Absolute_Deviation=("Absolute_Deviation", "mean"),
            Max_Absolute_Deviation=("Absolute_Deviation", "max"),
            Deviation_Std=("Signed_Deviation", "std"),
        )
        .sort_values(["Dataset", "Member_Seed"])
        .reset_index(drop=True)
    )


def _load_member_outputs(
    output_dir: Path,
    member_seeds: tuple[int, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    predictions_by_split: dict[str, list[np.ndarray]] = {
        "Validation": [],
        "Test": [],
        "Prediction": [],
    }
    ground_truth_by_split: dict[str, np.ndarray] = {}
    for seed in member_seeds:
        checkpoint_path = output_dir / "checkpoints" / f"member_seed_{seed}_outputs.npz"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing ensemble member outputs at {checkpoint_path}")
        with np.load(checkpoint_path) as arrays:
            values = {
                "Validation": (
                    arrays["validation_predictions"].astype(np.float64),
                    arrays["validation_ground_truth"].astype(np.float64),
                ),
                "Test": (
                    arrays["test_predictions"].astype(np.float64),
                    arrays["test_ground_truth"].astype(np.float64),
                ),
                "Prediction": (
                    arrays["prediction_predictions"].astype(np.float64),
                    arrays["prediction_ground_truth"].astype(np.float64),
                ),
            }
        for dataset_name, (predictions, ground_truth) in values.items():
            predictions_by_split[dataset_name].append(predictions)
            if dataset_name not in ground_truth_by_split:
                ground_truth_by_split[dataset_name] = ground_truth
            elif not np.allclose(ground_truth_by_split[dataset_name], ground_truth, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Ground truth differs across members for {dataset_name}")
    return {
        dataset_name: (np.stack(predictions, axis=0), ground_truth_by_split[dataset_name])
        for dataset_name, predictions in predictions_by_split.items()
    }


def _plot_confidence_intervals(
    uncertainty_frame: pd.DataFrame,
    confidence_level: float,
    save_path: Path,
) -> None:
    label = _level_label(confidence_level)
    datasets = uncertainty_frame["Dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset_name in zip(axes, datasets, strict=True):
        frame = uncertainty_frame[uncertainty_frame["Dataset"] == dataset_name].sort_values(
            "Actual_Stiffness"
        )
        x = np.arange(len(frame))
        axis.errorbar(
            x,
            frame["Ensemble_Mean"],
            yerr=frame[f"Mean_CI_{label}_Half_Width"],
            fmt="o",
            markersize=3,
            capsize=2,
            color="#355070",
            ecolor="#b56576",
            alpha=0.8,
            label=f"Ensemble mean with {label}% CI",
        )
        axis.scatter(x, frame["Actual_Stiffness"], s=14, color="#e09f3e", label="Actual")
        axis.set(title=dataset_name, xlabel="Samples sorted by actual stiffness", ylabel="Stiffness")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_uncertainty_error_relation(uncertainty_frame: pd.DataFrame, save_path: Path) -> None:
    datasets = uncertainty_frame["Dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset_name in zip(axes, datasets, strict=True):
        frame = uncertainty_frame[uncertainty_frame["Dataset"] == dataset_name]
        correlation = spearmanr(frame["Member_Std"], frame["Absolute_Error"]).statistic
        axis.scatter(frame["Member_Std"], frame["Absolute_Error"], alpha=0.7, color="#6d597a")
        axis.set(
            title=f"{dataset_name}: Spearman={correlation:.3f}",
            xlabel="Ensemble member standard deviation",
            ylabel="Absolute prediction error",
        )
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_prediction_member_deviations(member_deviations: pd.DataFrame, save_path: Path) -> None:
    frame = member_deviations[member_deviations["Dataset"] == "Prediction"]
    pivot = frame.pivot(index="Member_Seed", columns="Sample_ID", values="Signed_Deviation")
    figure, axis = plt.subplots(figsize=(max(12, 0.8 * pivot.shape[1]), 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    axis.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=60, ha="right")
    axis.set_yticks(np.arange(pivot.shape[0]), pivot.index)
    axis.set(title="External prediction deviations from ensemble mean", xlabel="Sample", ylabel="Member seed")
    figure.colorbar(image, ax=axis, label="Member prediction - ensemble mean")
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_interpretation_report(
    config: EnsembleUncertaintyConfig,
    summary: pd.DataFrame,
    save_path: Path,
) -> None:
    lines = [
        "GCN-3 Ensemble Confidence and Deviation Report",
        "==============================================",
        "",
        f"Member seeds: {', '.join(str(seed) for seed in config.member_seeds)}",
        f"Split seed: {config.split_seed}",
        "",
        "Interpretation",
        "--------------",
        "The reported intervals are Student-t confidence intervals for the mean prediction across model seeds.",
        "They measure initialization disagreement, not a calibrated predictive interval for the true stiffness.",
        "Empirical coverage shows how often known targets fall inside each nominal interval.",
        "Low coverage or weak spread/error correlation means the ensemble can agree while remaining systematically wrong.",
        "",
        "Dataset summaries",
        "-----------------",
    ]
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"{row['Dataset']}:",
                f"- R2: {row['R2']:.6f}",
                f"- RMSE: {row['RMSE']:.6f}",
                f"- mean member standard deviation: {row['Mean_Member_Std']:.6f}",
                f"- Spearman spread/error correlation: {row['Spearman_Std_vs_Absolute_Error']:.6f}",
            ]
        )
        for level in config.confidence_levels:
            label = _level_label(level)
            lines.append(
                f"- nominal {label}% mean-CI empirical coverage: "
                f"{row[f'Empirical_Mean_CI_{label}_Coverage_Percent']:.2f}%"
            )
        lines.append("")
    save_path.write_text("\n".join(lines), encoding="utf-8")


def run_ensemble_uncertainty_experiment(
    config: EnsembleUncertaintyConfig,
    train_root: str | Path,
    predict_root: str | Path,
    output_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_config(config)
    ensemble_result = run_fixed_split_ensemble(
        config.ensemble_config(),
        train_root=train_root,
        predict_root=predict_root,
        output_root=output_root,
        run_dir=run_dir,
    )
    output_dir = Path(ensemble_result["output_dir"])
    split_outputs = _load_member_outputs(output_dir, config.member_seeds)
    prediction_ids = sorted(path.name for path in Path(predict_root).glob("randomness_*") if path.is_dir())

    uncertainty_frames: list[pd.DataFrame] = []
    deviation_frames: list[pd.DataFrame] = []
    for dataset_name, (member_predictions, actual) in split_outputs.items():
        sample_ids = prediction_ids if dataset_name == "Prediction" and len(prediction_ids) == len(actual) else None
        uncertainty, deviations = build_uncertainty_frame(
            dataset_name,
            config.member_seeds,
            member_predictions,
            actual,
            config.confidence_levels,
            sample_ids=sample_ids,
        )
        uncertainty_frames.append(uncertainty)
        deviation_frames.append(deviations)

    uncertainty_frame = pd.concat(uncertainty_frames, ignore_index=True)
    member_deviations = pd.concat(deviation_frames, ignore_index=True)
    uncertainty_summary = build_uncertainty_summary(uncertainty_frame, config.confidence_levels)
    deviation_summary = build_member_deviation_summary(member_deviations)

    uncertainty_frame.to_csv(output_dir / "gcn3_ensemble_uncertainty_by_sample.csv", index=False)
    uncertainty_frame[uncertainty_frame["Dataset"] == "Prediction"].to_csv(
        output_dir / "gcn3_ensemble_prediction_confidence.csv",
        index=False,
    )
    member_deviations.to_csv(output_dir / "gcn3_ensemble_member_deviations.csv", index=False)
    deviation_summary.to_csv(output_dir / "gcn3_ensemble_member_deviation_summary.csv", index=False)
    uncertainty_summary.to_csv(output_dir / "gcn3_ensemble_uncertainty_summary.csv", index=False)
    _plot_confidence_intervals(
        uncertainty_frame,
        max(config.confidence_levels),
        output_dir / "gcn3_ensemble_confidence_intervals.png",
    )
    _plot_uncertainty_error_relation(
        uncertainty_frame,
        output_dir / "gcn3_ensemble_uncertainty_vs_error.png",
    )
    _plot_prediction_member_deviations(
        member_deviations,
        output_dir / "gcn3_ensemble_prediction_deviations.png",
    )
    _write_interpretation_report(
        config,
        uncertainty_summary,
        output_dir / "gcn3_ensemble_confidence_report.txt",
    )
    (output_dir / "uncertainty_analysis_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (output_dir / "uncertainty_analysis_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "member_seeds": list(config.member_seeds),
                "confidence_levels": list(config.confidence_levels),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Ensemble uncertainty analysis saved to {output_dir}")
    print(uncertainty_summary)
    return {
        **ensemble_result,
        "uncertainty_frame": uncertainty_frame,
        "prediction_confidence": uncertainty_frame[uncertainty_frame["Dataset"] == "Prediction"],
        "member_deviations": member_deviations,
        "deviation_summary": deviation_summary,
        "uncertainty_summary": uncertainty_summary,
    }
