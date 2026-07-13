from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GRAPH_FEATURE_NAMES = [
    "Node_Count",
    "Edge_Count",
    "Graph_Density",
    "Total_Edge_Weight",
    "Mean_Edge_Weight",
    "Std_Edge_Weight",
    "Mean_Degree",
    "Std_Degree",
    "Mean_Weighted_Degree",
    "Std_Weighted_Degree",
    "Mean_Center_Distance",
    "Std_Center_Distance",
]
ANALYSIS_FEATURES = ["Randomness_Value", "Actual_Stiffness", *GRAPH_FEATURE_NAMES]


@dataclass
class ResidualAnalysisConfig:
    split_seeds: tuple[int, ...] = (11, 42, 73, 101, 202)
    stiffness_bins: int = 5
    feature_bins: int = 5
    uncertainty_bins: int = 5
    worst_case_count: int = 25
    output_group: str = "gcn3_ensemble_residual_analysis"


def _randomness_value(folder_name: str) -> float:
    try:
        return float(folder_name.removeprefix("randomness_"))
    except ValueError:
        return float("nan")


def _derive_graph_features(node_coordinates: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
    connectivity = (adjacency_matrix > 0).astype(np.float32)
    np.fill_diagonal(connectivity, 0.0)

    node_count = float(node_coordinates.shape[0])
    undirected_edges = np.argwhere(np.triu(adjacency_matrix, k=1) > 0)
    edge_count = float(undirected_edges.shape[0])
    density = 0.0 if node_count <= 1 else float(2.0 * edge_count / (node_count * (node_count - 1.0)))
    edge_weights = (
        adjacency_matrix[undirected_edges[:, 0], undirected_edges[:, 1]].astype(np.float32)
        if undirected_edges.size
        else np.empty((0,), dtype=np.float32)
    )

    degree = connectivity.sum(axis=1).astype(np.float32)
    weighted_degree = adjacency_matrix.sum(axis=1).astype(np.float32)
    centroid = node_coordinates.mean(axis=0, keepdims=True)
    center_distance = np.linalg.norm(node_coordinates - centroid, axis=1).astype(np.float32)

    return np.asarray(
        [
            node_count,
            edge_count,
            density,
            float(edge_weights.sum()) if edge_weights.size else 0.0,
            float(edge_weights.mean()) if edge_weights.size else 0.0,
            float(edge_weights.std()) if edge_weights.size else 0.0,
            float(degree.mean()),
            float(degree.std()),
            float(weighted_degree.mean()),
            float(weighted_degree.std()),
            float(center_distance.mean()),
            float(center_distance.std()),
        ],
        dtype=np.float64,
    )


def load_lattice_feature_frame(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root)
    folders = sorted(path for path in root.glob("randomness_*") if path.is_dir())
    if not folders:
        raise FileNotFoundError(f"No randomness folders found in {root}")

    rows: list[dict[str, float | str]] = []
    for folder in folders:
        coordinates = pd.read_csv(folder / "node_features.csv", usecols=["x", "y"]).to_numpy(dtype=np.float64)
        adjacency = pd.read_csv(folder / "adjacency_area.csv", index_col=0).to_numpy(dtype=np.float64)
        stiffness = float(pd.read_csv(folder / "lattice_stiffness.csv").iloc[0, 0])
        features = _derive_graph_features(coordinates, adjacency)
        row: dict[str, float | str] = {
            "Sample_ID": folder.name,
            "Randomness_Value": _randomness_value(folder.name),
            "Actual_Stiffness": stiffness,
        }
        row.update(dict(zip(GRAPH_FEATURE_NAMES, features, strict=True)))
        rows.append(row)

    frame = pd.DataFrame(rows)
    print(f"Loaded {len(frame)} lattice feature records from {root}")
    return frame


def split_indices(sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(sample_count)
    train_indices, remainder = train_test_split(
        indices,
        train_size=0.8,
        shuffle=True,
        random_state=seed,
    )
    validation_indices, test_indices = train_test_split(
        remainder,
        train_size=0.5,
        shuffle=True,
        random_state=seed,
    )
    return train_indices, validation_indices, test_indices


def _member_seed(path: Path) -> int:
    match = re.search(r"member_seed_(\d+)_outputs\.npz$", path.name)
    if match is None:
        raise ValueError(f"Unable to parse member seed from {path}")
    return int(match.group(1))


def load_split_predictions(split_dir: str | Path) -> dict[str, np.ndarray | list[int]]:
    checkpoint_dir = Path(split_dir) / "checkpoints"
    member_paths = sorted(checkpoint_dir.glob("member_seed_*_outputs.npz"), key=_member_seed)
    if not member_paths:
        raise FileNotFoundError(f"No completed member output checkpoints found in {checkpoint_dir}")

    test_predictions: list[np.ndarray] = []
    prediction_predictions: list[np.ndarray] = []
    test_ground_truth: np.ndarray | None = None
    prediction_ground_truth: np.ndarray | None = None
    member_seeds: list[int] = []

    for member_path in member_paths:
        with np.load(member_path) as arrays:
            current_test_ground_truth = arrays["test_ground_truth"].astype(np.float64)
            current_prediction_ground_truth = arrays["prediction_ground_truth"].astype(np.float64)
            if test_ground_truth is None:
                test_ground_truth = current_test_ground_truth
                prediction_ground_truth = current_prediction_ground_truth
            else:
                if not np.allclose(test_ground_truth, current_test_ground_truth, rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Test targets differ across members in {checkpoint_dir}")
                if not np.allclose(prediction_ground_truth, current_prediction_ground_truth, rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Prediction targets differ across members in {checkpoint_dir}")
            test_predictions.append(arrays["test_predictions"].astype(np.float64))
            prediction_predictions.append(arrays["prediction_predictions"].astype(np.float64))
        member_seeds.append(_member_seed(member_path))

    assert test_ground_truth is not None
    assert prediction_ground_truth is not None
    test_stack = np.stack(test_predictions, axis=0)
    prediction_stack = np.stack(prediction_predictions, axis=0)
    return {
        "member_seeds": member_seeds,
        "test_predictions": test_stack.mean(axis=0),
        "test_prediction_std": test_stack.std(axis=0, ddof=0),
        "test_prediction_min": test_stack.min(axis=0),
        "test_prediction_max": test_stack.max(axis=0),
        "test_ground_truth": test_ground_truth,
        "prediction_predictions": prediction_stack.mean(axis=0),
        "prediction_prediction_std": prediction_stack.std(axis=0, ddof=0),
        "prediction_prediction_min": prediction_stack.min(axis=0),
        "prediction_prediction_max": prediction_stack.max(axis=0),
        "prediction_ground_truth": prediction_ground_truth,
    }


def _attach_residual_columns(
    feature_frame: pd.DataFrame,
    predictions: np.ndarray,
    prediction_std: np.ndarray,
    prediction_min: np.ndarray,
    prediction_max: np.ndarray,
    dataset_name: str,
    split_seed: int,
) -> pd.DataFrame:
    frame = feature_frame.reset_index(drop=True).copy()
    if len(frame) != len(predictions):
        raise ValueError(f"{dataset_name} feature and prediction counts do not match for split {split_seed}")
    frame["Dataset"] = dataset_name
    frame["Split_Seed"] = split_seed
    frame["Predicted_Stiffness"] = predictions
    frame["Residual"] = predictions - frame["Actual_Stiffness"].to_numpy()
    frame["Absolute_Error"] = frame["Residual"].abs()
    frame["Squared_Error"] = frame["Residual"] ** 2
    denominator = np.clip(frame["Actual_Stiffness"].abs().to_numpy(), np.finfo(np.float64).eps, None)
    frame["Absolute_Percent_Error"] = frame["Absolute_Error"].to_numpy() / denominator * 100.0
    frame["Ensemble_Std"] = prediction_std
    frame["Member_Prediction_Min"] = prediction_min
    frame["Member_Prediction_Max"] = prediction_max
    frame["Member_Prediction_Range"] = prediction_max - prediction_min
    return frame


def build_residual_frames(
    source_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    ensemble_run_dir: str | Path,
    split_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []
    run_dir = Path(ensemble_run_dir)

    for split_seed in split_seeds:
        split_dir = run_dir / "per_split" / f"split_seed_{split_seed}"
        outputs = load_split_predictions(split_dir)
        train_indices, _, test_indices = split_indices(len(source_features), split_seed)
        train_features = source_features.iloc[train_indices].reset_index(drop=True).copy()
        test_features = source_features.iloc[test_indices].reset_index(drop=True).copy()

        if not np.allclose(
            test_features["Actual_Stiffness"].to_numpy(),
            outputs["test_ground_truth"],
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(f"Reconstructed test split does not align with saved predictions for seed {split_seed}")
        if not np.allclose(
            prediction_features["Actual_Stiffness"].to_numpy(),
            outputs["prediction_ground_truth"],
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(f"Prediction dataset order does not align for seed {split_seed}")

        residual_frames.append(
            _attach_residual_columns(
                test_features,
                outputs["test_predictions"],
                outputs["test_prediction_std"],
                outputs["test_prediction_min"],
                outputs["test_prediction_max"],
                "Test",
                split_seed,
            )
        )
        residual_frames.append(
            _attach_residual_columns(
                prediction_features,
                outputs["prediction_predictions"],
                outputs["prediction_prediction_std"],
                outputs["prediction_prediction_min"],
                outputs["prediction_prediction_max"],
                "Prediction",
                split_seed,
            )
        )

        for dataset_name, frame in (
            ("Train", train_features),
            ("Test", test_features),
            ("Prediction", prediction_features),
        ):
            distribution_frame = frame.copy()
            distribution_frame["Dataset"] = dataset_name
            distribution_frame["Split_Seed"] = split_seed
            distribution_frames.append(distribution_frame)

    return pd.concat(residual_frames, ignore_index=True), pd.concat(distribution_frames, ignore_index=True)


def _r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return float("nan") if denominator == 0.0 else 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator


def _metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["Actual_Stiffness"].to_numpy(dtype=np.float64)
    predicted = frame["Predicted_Stiffness"].to_numpy(dtype=np.float64)
    residual = predicted - actual
    return {
        "Count": len(frame),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "Bias": float(np.mean(residual)),
        "R2": _r2_score(actual, predicted),
        "Mean_Ensemble_Std": float(frame["Ensemble_Std"].mean()),
    }


def build_split_metrics(residual_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (dataset_name, split_seed), frame in residual_frame.groupby(["Dataset", "Split_Seed"], sort=True):
        rows.append({"Dataset": dataset_name, "Split_Seed": split_seed, **_metric_row(frame)})
    return pd.DataFrame(rows)


def add_stiffness_bins(
    residual_frame: pd.DataFrame,
    reference_targets: pd.Series,
    bin_count: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    edges = np.unique(np.quantile(reference_targets.to_numpy(dtype=np.float64), np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) < 3:
        raise ValueError("Not enough unique target values to construct stiffness bins")
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    result = residual_frame.copy()
    result["Stiffness_Bin"] = pd.cut(
        result["Actual_Stiffness"],
        bins=edges,
        labels=labels,
        include_lowest=True,
    )
    return result, edges


def build_stiffness_bin_metrics(residual_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    grouped = residual_frame.groupby(["Dataset", "Stiffness_Bin"], observed=True, sort=True)
    for (dataset_name, bin_name), frame in grouped:
        rows.append(
            {
                "Dataset": dataset_name,
                "Stiffness_Bin": str(bin_name),
                "Target_Min": float(frame["Actual_Stiffness"].min()),
                "Target_Max": float(frame["Actual_Stiffness"].max()),
                **_metric_row(frame),
            }
        )
    return pd.DataFrame(rows)


def build_feature_correlations(residual_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for dataset_name, frame in residual_frame.groupby("Dataset", sort=True):
        for feature_name in ANALYSIS_FEATURES:
            feature = frame[feature_name]
            if feature.nunique(dropna=True) < 2:
                continue
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Feature": feature_name,
                    "Pearson_Abs_Error": float(feature.corr(frame["Absolute_Error"], method="pearson")),
                    "Spearman_Abs_Error": float(feature.corr(frame["Absolute_Error"], method="spearman")),
                    "Pearson_Signed_Residual": float(feature.corr(frame["Residual"], method="pearson")),
                    "Spearman_Signed_Residual": float(feature.corr(frame["Residual"], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def build_feature_bin_metrics(residual_frame: pd.DataFrame, bin_count: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset_name, dataset_frame in residual_frame.groupby("Dataset", sort=True):
        for feature_name in ANALYSIS_FEATURES:
            if dataset_frame[feature_name].nunique(dropna=True) < 2:
                continue
            frame = dataset_frame.copy()
            frame["Feature_Bin"] = pd.qcut(
                frame[feature_name],
                q=min(bin_count, frame[feature_name].nunique()),
                labels=False,
                duplicates="drop",
            )
            for feature_bin, bin_frame in frame.groupby("Feature_Bin", sort=True):
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Feature": feature_name,
                        "Feature_Bin": int(feature_bin) + 1,
                        "Count": len(bin_frame),
                        "Feature_Min": float(bin_frame[feature_name].min()),
                        "Feature_Max": float(bin_frame[feature_name].max()),
                        "MAE": float(bin_frame["Absolute_Error"].mean()),
                        "RMSE": float(np.sqrt(bin_frame["Squared_Error"].mean())),
                        "Bias": float(bin_frame["Residual"].mean()),
                        "Mean_Ensemble_Std": float(bin_frame["Ensemble_Std"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def build_distribution_shift(distribution_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for split_seed, split_frame in distribution_frame.groupby("Split_Seed", sort=True):
        train_frame = split_frame[split_frame["Dataset"] == "Train"]
        for comparison_name in ("Test", "Prediction"):
            comparison_frame = split_frame[split_frame["Dataset"] == comparison_name]
            for feature_name in ANALYSIS_FEATURES:
                train_values = train_frame[feature_name].dropna().to_numpy(dtype=np.float64)
                comparison_values = comparison_frame[feature_name].dropna().to_numpy(dtype=np.float64)
                train_scale = float(train_values.std(ddof=0))
                safe_scale = train_scale if train_scale > np.finfo(np.float64).eps else 1.0
                standardized_difference = float((comparison_values.mean() - train_values.mean()) / safe_scale)
                rows.append(
                    {
                        "Split_Seed": split_seed,
                        "Comparison": f"{comparison_name}_vs_Train",
                        "Feature": feature_name,
                        "Train_Mean": float(train_values.mean()),
                        "Comparison_Mean": float(comparison_values.mean()),
                        "Standardized_Mean_Difference": standardized_difference,
                        "Absolute_Standardized_Mean_Difference": abs(standardized_difference),
                        "Normalized_Wasserstein": float(
                            wasserstein_distance(train_values, comparison_values) / safe_scale
                        ),
                        "KS_Statistic": float(ks_2samp(train_values, comparison_values).statistic),
                    }
                )

    detail_frame = pd.DataFrame(rows)
    summary_frame = (
        detail_frame.groupby(["Comparison", "Feature"], as_index=False)
        .agg(
            Mean_Standardized_Mean_Difference=("Standardized_Mean_Difference", "mean"),
            Mean_Absolute_Standardized_Mean_Difference=("Absolute_Standardized_Mean_Difference", "mean"),
            Max_Absolute_Standardized_Mean_Difference=("Absolute_Standardized_Mean_Difference", "max"),
            Mean_Normalized_Wasserstein=("Normalized_Wasserstein", "mean"),
            Mean_KS_Statistic=("KS_Statistic", "mean"),
        )
        .sort_values(["Comparison", "Mean_Absolute_Standardized_Mean_Difference"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return detail_frame, summary_frame


def build_uncertainty_metrics(
    residual_frame: pd.DataFrame,
    uncertainty_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation_rows: list[dict[str, float | str]] = []
    bin_rows: list[dict[str, float | int | str]] = []
    for dataset_name, frame in residual_frame.groupby("Dataset", sort=True):
        correlation_rows.append(
            {
                "Dataset": dataset_name,
                "Pearson_Uncertainty_vs_Abs_Error": float(
                    frame["Ensemble_Std"].corr(frame["Absolute_Error"], method="pearson")
                ),
                "Spearman_Uncertainty_vs_Abs_Error": float(
                    frame["Ensemble_Std"].corr(frame["Absolute_Error"], method="spearman")
                ),
                "Pearson_Range_vs_Abs_Error": float(
                    frame["Member_Prediction_Range"].corr(frame["Absolute_Error"], method="pearson")
                ),
            }
        )
        ranked = frame.copy()
        ranked["Uncertainty_Bin"] = pd.qcut(
            ranked["Ensemble_Std"].rank(method="first"),
            q=min(uncertainty_bins, len(ranked)),
            labels=False,
            duplicates="drop",
        )
        for uncertainty_bin, bin_frame in ranked.groupby("Uncertainty_Bin", sort=True):
            bin_rows.append(
                {
                    "Dataset": dataset_name,
                    "Uncertainty_Bin": int(uncertainty_bin) + 1,
                    "Count": len(bin_frame),
                    "Mean_Ensemble_Std": float(bin_frame["Ensemble_Std"].mean()),
                    "MAE": float(bin_frame["Absolute_Error"].mean()),
                    "RMSE": float(np.sqrt(bin_frame["Squared_Error"].mean())),
                }
            )
    return pd.DataFrame(correlation_rows), pd.DataFrame(bin_rows)


def build_sample_stability(residual_frame: pd.DataFrame, worst_case_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_aggregations = {feature: (feature, "first") for feature in ANALYSIS_FEATURES}
    stability_frame = (
        residual_frame.groupby(["Dataset", "Sample_ID"], as_index=False)
        .agg(
            Split_Count=("Split_Seed", "nunique"),
            Mean_Predicted_Stiffness=("Predicted_Stiffness", "mean"),
            Std_Predicted_Across_Splits=("Predicted_Stiffness", "std"),
            Mean_Residual=("Residual", "mean"),
            Mean_Absolute_Error=("Absolute_Error", "mean"),
            Max_Absolute_Error=("Absolute_Error", "max"),
            Mean_Ensemble_Std=("Ensemble_Std", "mean"),
            **feature_aggregations,
        )
        .sort_values(["Dataset", "Mean_Absolute_Error"], ascending=[True, False])
        .reset_index(drop=True)
    )
    worst_cases = (
        stability_frame.groupby("Dataset", group_keys=False)
        .head(worst_case_count)
        .reset_index(drop=True)
    )
    return stability_frame, worst_cases


def _save_residual_overview(residual_frame: pd.DataFrame, split_metrics: pd.DataFrame, save_path: Path) -> None:
    colors = {"Test": "#355070", "Prediction": "#b56576"}
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for dataset_name, frame in residual_frame.groupby("Dataset", sort=True):
        color = colors.get(dataset_name, "#6d597a")
        axes[0, 0].scatter(
            frame["Actual_Stiffness"], frame["Predicted_Stiffness"], alpha=0.45, s=24, label=dataset_name, color=color
        )
        axes[0, 1].scatter(
            frame["Actual_Stiffness"], frame["Residual"], alpha=0.45, s=24, label=dataset_name, color=color
        )
    limits = [
        min(residual_frame["Actual_Stiffness"].min(), residual_frame["Predicted_Stiffness"].min()),
        max(residual_frame["Actual_Stiffness"].max(), residual_frame["Predicted_Stiffness"].max()),
    ]
    axes[0, 0].plot(limits, limits, "k--", linewidth=1.5)
    axes[0, 0].set(title="Predicted vs actual", xlabel="Actual stiffness", ylabel="Predicted stiffness")
    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.5)
    axes[0, 1].set(title="Signed residuals", xlabel="Actual stiffness", ylabel="Prediction - actual")
    axes[0, 0].legend()
    axes[0, 1].legend()

    datasets = sorted(residual_frame["Dataset"].unique())
    axes[1, 0].boxplot(
        [residual_frame.loc[residual_frame["Dataset"] == name, "Absolute_Error"] for name in datasets],
        tick_labels=datasets,
        showfliers=True,
    )
    axes[1, 0].set(title="Absolute error distribution", ylabel="Absolute error")
    for dataset_name, frame in split_metrics.groupby("Dataset", sort=True):
        axes[1, 1].plot(frame["Split_Seed"], frame["RMSE"], marker="o", linewidth=2, label=dataset_name)
    axes[1, 1].set(title="RMSE by split", xlabel="Split seed", ylabel="RMSE")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_stiffness_bins(stiffness_metrics: pd.DataFrame, save_path: Path) -> None:
    datasets = sorted(stiffness_metrics["Dataset"].unique())
    bins = list(dict.fromkeys(stiffness_metrics["Stiffness_Bin"]))
    x = np.arange(len(bins))
    width = 0.8 / max(len(datasets), 1)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for index, dataset_name in enumerate(datasets):
        frame = stiffness_metrics[stiffness_metrics["Dataset"] == dataset_name].set_index("Stiffness_Bin").reindex(bins)
        offset = (index - (len(datasets) - 1) / 2) * width
        axes[0].bar(x + offset, frame["MAE"], width=width, label=dataset_name)
        axes[1].bar(x + offset, frame["RMSE"], width=width, label=dataset_name)
    for axis, title, ylabel in (
        (axes[0], "MAE by target-stiffness quantile", "MAE"),
        (axes[1], "RMSE by target-stiffness quantile", "RMSE"),
    ):
        axis.set_xticks(x, bins)
        axis.set_title(title)
        axis.set_xlabel("Training-target quantile")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_feature_correlations(correlation_frame: pd.DataFrame, save_path: Path) -> None:
    datasets = sorted(correlation_frame["Dataset"].unique())
    figure, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 7), squeeze=False)
    for axis, dataset_name in zip(axes.ravel(), datasets):
        frame = correlation_frame[correlation_frame["Dataset"] == dataset_name].copy()
        frame = frame.reindex(frame["Spearman_Abs_Error"].abs().sort_values().index).tail(10)
        colors = np.where(frame["Spearman_Abs_Error"] >= 0.0, "#b56576", "#355070")
        axis.barh(frame["Feature"], frame["Spearman_Abs_Error"], color=colors)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_title(f"{dataset_name}: strongest error associations")
        axis.set_xlabel("Spearman correlation with absolute error")
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_distribution_shift(shift_summary: pd.DataFrame, save_path: Path) -> None:
    pivot = shift_summary.pivot(
        index="Feature",
        columns="Comparison",
        values="Mean_Absolute_Standardized_Mean_Difference",
    ).fillna(0.0)
    figure, axis = plt.subplots(figsize=(8, 9))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_title("Feature distribution shift relative to training data")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(column, row, f"{pivot.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Mean absolute standardized difference")
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_uncertainty(
    residual_frame: pd.DataFrame,
    uncertainty_bins: pd.DataFrame,
    save_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for dataset_name, frame in residual_frame.groupby("Dataset", sort=True):
        axes[0].scatter(frame["Ensemble_Std"], frame["Absolute_Error"], alpha=0.45, s=24, label=dataset_name)
    axes[0].set(title="Ensemble disagreement vs error", xlabel="Member prediction standard deviation", ylabel="Absolute error")
    axes[0].legend()
    for dataset_name, frame in uncertainty_bins.groupby("Dataset", sort=True):
        axes[1].plot(frame["Uncertainty_Bin"], frame["MAE"], marker="o", linewidth=2, label=dataset_name)
    axes[1].set(title="Error by uncertainty quantile", xlabel="Uncertainty bin (low to high)", ylabel="MAE")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_analysis_summary(
    config: ResidualAnalysisConfig,
    split_metrics: pd.DataFrame,
    stiffness_metrics: pd.DataFrame,
    correlations: pd.DataFrame,
    shift_summary: pd.DataFrame,
    uncertainty_correlations: pd.DataFrame,
) -> dict[str, object]:
    dataset_metrics = (
        split_metrics.groupby("Dataset")[["MAE", "RMSE", "Bias", "R2"]].mean().round(8).to_dict(orient="index")
    )
    hardest_bins = {}
    strongest_error_features = {}
    for dataset_name in sorted(stiffness_metrics["Dataset"].unique()):
        bin_row = stiffness_metrics[stiffness_metrics["Dataset"] == dataset_name].sort_values("RMSE", ascending=False).iloc[0]
        hardest_bins[dataset_name] = {
            "stiffness_bin": str(bin_row["Stiffness_Bin"]),
            "rmse": float(bin_row["RMSE"]),
        }
        feature_row = (
            correlations[correlations["Dataset"] == dataset_name]
            .assign(abs_correlation=lambda frame: frame["Spearman_Abs_Error"].abs())
            .sort_values("abs_correlation", ascending=False)
            .iloc[0]
        )
        strongest_error_features[dataset_name] = {
            "feature": str(feature_row["Feature"]),
            "spearman_abs_error": float(feature_row["Spearman_Abs_Error"]),
        }

    prediction_shift = shift_summary[shift_summary["Comparison"] == "Prediction_vs_Train"].iloc[0]
    return {
        "config": asdict(config),
        "mean_metrics_by_dataset": dataset_metrics,
        "hardest_stiffness_bin_by_dataset": hardest_bins,
        "strongest_absolute_error_feature_by_dataset": strongest_error_features,
        "largest_prediction_distribution_shift": {
            "feature": str(prediction_shift["Feature"]),
            "mean_absolute_standardized_difference": float(
                prediction_shift["Mean_Absolute_Standardized_Mean_Difference"]
            ),
        },
        "uncertainty_error_correlations": uncertainty_correlations.to_dict(orient="records"),
    }


def run_residual_error_analysis(
    config: ResidualAnalysisConfig,
    train_root: str | Path,
    predict_root: str | Path,
    ensemble_run_dir: str | Path,
    output_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    train_root = Path(train_root)
    predict_root = Path(predict_root)
    ensemble_run_dir = Path(ensemble_run_dir)
    if not (ensemble_run_dir / "gcn3_ensemble_multi_split_summary.csv").is_file():
        raise FileNotFoundError(f"Completed multi-split ensemble summary not found in {ensemble_run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir is not None:
        output_dir = Path(run_dir)
    elif output_root is not None:
        output_dir = Path(output_root) / f"run_{timestamp}"
    else:
        output_dir = ensemble_run_dir.parents[1] / config.output_group / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_features = load_lattice_feature_frame(train_root)
    prediction_features = load_lattice_feature_frame(predict_root)
    residual_frame, distribution_frame = build_residual_frames(
        source_features,
        prediction_features,
        ensemble_run_dir,
        config.split_seeds,
    )
    residual_frame, stiffness_edges = add_stiffness_bins(
        residual_frame,
        source_features["Actual_Stiffness"],
        config.stiffness_bins,
    )

    split_metrics = build_split_metrics(residual_frame)
    stiffness_metrics = build_stiffness_bin_metrics(residual_frame)
    correlations = build_feature_correlations(residual_frame)
    feature_bin_metrics = build_feature_bin_metrics(residual_frame, config.feature_bins)
    shift_detail, shift_summary = build_distribution_shift(distribution_frame)
    uncertainty_correlations, uncertainty_bins = build_uncertainty_metrics(
        residual_frame,
        config.uncertainty_bins,
    )
    sample_stability, worst_cases = build_sample_stability(residual_frame, config.worst_case_count)
    analysis_summary = build_analysis_summary(
        config,
        split_metrics,
        stiffness_metrics,
        correlations,
        shift_summary,
        uncertainty_correlations,
    )
    analysis_summary["stiffness_bin_edges"] = [None if np.isinf(value) else float(value) for value in stiffness_edges]
    analysis_summary["train_root"] = str(train_root)
    analysis_summary["predict_root"] = str(predict_root)
    analysis_summary["ensemble_run_dir"] = str(ensemble_run_dir)

    tables = {
        "lattice_feature_reference.csv": source_features,
        "lattice_feature_prediction.csv": prediction_features,
        "residual_samples.csv": residual_frame,
        "split_residual_metrics.csv": split_metrics,
        "stiffness_bin_metrics.csv": stiffness_metrics,
        "feature_residual_correlations.csv": correlations,
        "feature_bin_metrics.csv": feature_bin_metrics,
        "distribution_shift_by_split.csv": shift_detail,
        "distribution_shift_summary.csv": shift_summary,
        "uncertainty_error_correlations.csv": uncertainty_correlations,
        "uncertainty_bin_metrics.csv": uncertainty_bins,
        "sample_stability_summary.csv": sample_stability,
        "worst_residual_cases.csv": worst_cases,
    }
    for file_name, frame in tables.items():
        frame.to_csv(output_dir / file_name, index=False)

    (output_dir / "residual_analysis_summary.json").write_text(
        json.dumps(analysis_summary, indent=2),
        encoding="utf-8",
    )
    _save_residual_overview(residual_frame, split_metrics, output_dir / "residual_overview.png")
    _save_stiffness_bins(stiffness_metrics, output_dir / "stiffness_bin_errors.png")
    _save_feature_correlations(correlations, output_dir / "feature_error_correlations.png")
    _save_distribution_shift(shift_summary, output_dir / "feature_distribution_shift.png")
    _save_uncertainty(residual_frame, uncertainty_bins, output_dir / "ensemble_uncertainty_analysis.png")

    print(f"Residual analysis saved to {output_dir}")
    print(json.dumps(analysis_summary, indent=2))
    return {
        "output_dir": output_dir,
        "analysis_summary": analysis_summary,
        "residual_frame": residual_frame,
        "split_metrics": split_metrics,
        "stiffness_metrics": stiffness_metrics,
        "correlations": correlations,
        "feature_bin_metrics": feature_bin_metrics,
        "shift_detail": shift_detail,
        "shift_summary": shift_summary,
        "uncertainty_correlations": uncertainty_correlations,
        "uncertainty_bins": uncertainty_bins,
        "sample_stability": sample_stability,
        "worst_cases": worst_cases,
    }
