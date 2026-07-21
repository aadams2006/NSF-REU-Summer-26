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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
)
from colab_gnn_stiffness_prototype import (  # noqa: E402
    _build_graph_edges,
    _derive_graph_features,
    _derive_node_features,
    compute_regression_metrics,
)


TRANSFORMS = ("original", "mirror_x", "mirror_y", "mirror_xy")


@dataclass
class MirrorSensitivityConfig:
    member_seeds: tuple[int, ...] = (11, 42, 73, 101, 202)
    split_seed: int = 42
    transforms: tuple[str, ...] = TRANSFORMS
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = True
    output_group: str = "gcn3_ensemble_mirror_sensitivity"


def validate_config(config: MirrorSensitivityConfig) -> None:
    unsupported = sorted(set(config.transforms) - set(TRANSFORMS))
    if unsupported:
        raise ValueError(f"Unsupported mirror transforms: {unsupported}")
    if "original" not in config.transforms:
        raise ValueError("The original transform is required as the comparison baseline")
    if len(config.member_seeds) < 2:
        raise ValueError("At least two ensemble members are required")


def mirror_coordinates(coordinates: np.ndarray, transform: str) -> np.ndarray:
    if transform not in TRANSFORMS:
        raise ValueError(f"Unsupported mirror transform: {transform}")
    mirrored = np.asarray(coordinates, dtype=np.float32).copy()
    if transform in {"mirror_x", "mirror_xy"}:
        mirrored[:, 0] = mirrored[:, 0].min() + mirrored[:, 0].max() - mirrored[:, 0]
    if transform in {"mirror_y", "mirror_xy"}:
        mirrored[:, 1] = mirrored[:, 1].min() + mirrored[:, 1].max() - mirrored[:, 1]
    return mirrored


def _load_transformed_sample(folder: Path, transform: str) -> Data:
    coordinates = pd.read_csv(folder / "node_features.csv", usecols=["x", "y"]).to_numpy(dtype=np.float32)
    adjacency = pd.read_csv(folder / "adjacency_area.csv", index_col=0).to_numpy()
    stiffness = float(pd.read_csv(folder / "lattice_stiffness.csv").iloc[0, 0])
    transformed_coordinates = mirror_coordinates(coordinates, transform)
    edge_index, edge_weight = _build_graph_edges(adjacency)
    return Data(
        x=torch.from_numpy(_derive_node_features(transformed_coordinates, adjacency)),
        edge_index=edge_index,
        edge_weight=edge_weight,
        graph_attr=torch.from_numpy(_derive_graph_features(transformed_coordinates, adjacency)).view(1, -1),
        y=torch.tensor([stiffness], dtype=torch.float32),
    )


def _split_test_folders(train_root: Path, split_seed: int) -> list[Path]:
    folders = sorted(path for path in train_root.glob("randomness_*") if path.is_dir())
    _, remainder = train_test_split(folders, train_size=0.8, shuffle=True, random_state=split_seed)
    _, test_folders = train_test_split(remainder, train_size=0.5, shuffle=True, random_state=split_seed)
    return list(test_folders)


def _load_scaler(model_payload: dict[str, object]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(model_payload["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(model_payload["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)
    graph_mean = model_payload.get("graph_scaler_mean")
    graph_scale = model_payload.get("graph_scaler_scale")
    if graph_mean is not None and graph_scale is not None:
        scaler.graph_mean_ = np.asarray(graph_mean, dtype=np.float64)
        scaler.graph_scale_ = np.asarray(graph_scale, dtype=np.float64)
    return scaler


def _load_target_scaler(path: Path) -> StandardScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scaler = StandardScaler()
    scaler.mean_ = np.asarray([payload["mean"]], dtype=np.float64)
    scaler.scale_ = np.asarray([payload["scale"]], dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = 1
    return scaler


def _load_member_model(
    ensemble_run_dir: Path,
    seed: int,
    metadata: dict[str, object],
    device: str,
) -> tuple[torch.nn.Module, StandardScaler, StandardScaler]:
    member_dir = ensemble_run_dir / "per_model" / f"member_seed_{seed}"
    model_path = member_dir / "lattice_gnn_model.pt"
    target_scaler_path = member_dir / "target_scaler.json"
    if not model_path.is_file() or not target_scaler_path.is_file():
        raise FileNotFoundError(
            f"Missing saved model or target scaler for seed {seed} under {member_dir}. "
            "Use the complete Drive output from the final ensemble run."
        )
    payload = torch.load(model_path, map_location=device, weights_only=False)
    feature_scaler = _load_scaler(payload)
    target_scaler = _load_target_scaler(target_scaler_path)
    graph_scaler_mean = payload.get("graph_scaler_mean")
    graph_feature_dim = 0 if graph_scaler_mean is None else len(graph_scaler_mean)
    architecture_config = ArchitectureConfig(
        architecture_name=str(metadata.get("architecture_name", "gcn3")),
        architecture_label=str(metadata.get("architecture_label", "GCN-3")),
        hidden_dim=int(metadata.get("hidden_dim", 24)),
        dropout=float(metadata.get("dropout", 0.1)),
        seed=seed,
        device=device,
    )
    model = build_model(
        architecture_config,
        input_dim=len(feature_scaler.mean_),
        graph_feature_dim=graph_feature_dim,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, feature_scaler, target_scaler


def _predict_member(
    model: torch.nn.Module,
    raw_data: list[Data],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    config: MirrorSensitivityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = deepcopy(raw_data)
    apply_feature_scaler(prepared, feature_scaler)
    apply_target_scaler(prepared, target_scaler)
    loader = DataLoader(prepared, batch_size=config.batch_size, shuffle=False)
    predictions, actual, _ = evaluate_target_normalized_model(
        model,
        loader,
        target_scaler,
        device=config.device,
    )
    return predictions.astype(np.float64), actual.astype(np.float64)


def build_mirror_frames(
    member_predictions: dict[tuple[str, str, int], np.ndarray],
    ground_truth: dict[str, np.ndarray],
    sample_ids: dict[str, list[str]],
    config: MirrorSensitivityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, float | int | str | bool]] = []
    member_rows: list[dict[str, float | int | str]] = []
    for dataset_name, actual in ground_truth.items():
        original_members = np.stack(
            [member_predictions[(dataset_name, "original", seed)] for seed in config.member_seeds],
            axis=0,
        )
        original_mean = original_members.mean(axis=0)
        for transform in config.transforms:
            transformed_members = np.stack(
                [member_predictions[(dataset_name, transform, seed)] for seed in config.member_seeds],
                axis=0,
            )
            transformed_mean = transformed_members.mean(axis=0)
            ensemble_delta = transformed_mean - original_mean
            for index, sample_id in enumerate(sample_ids[dataset_name]):
                sample_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Transform": transform,
                        "Sample_Index": index,
                        "Sample_ID": sample_id,
                        "Actual_Stiffness": float(actual[index]),
                        "Original_Ensemble_Prediction": float(original_mean[index]),
                        "Transformed_Ensemble_Prediction": float(transformed_mean[index]),
                        "Signed_Prediction_Change": float(ensemble_delta[index]),
                        "Absolute_Prediction_Change": float(abs(ensemble_delta[index])),
                        "Relative_Prediction_Change_Percent": float(
                            abs(ensemble_delta[index])
                            / max(abs(original_mean[index]), np.finfo(np.float64).eps)
                            * 100.0
                        ),
                        "Original_Member_Std": float(
                            original_members[:, index].std(ddof=1 if len(config.member_seeds) > 1 else 0)
                        ),
                        "Transformed_Member_Std": float(
                            transformed_members[:, index].std(ddof=1 if len(config.member_seeds) > 1 else 0)
                        ),
                        "Original_Absolute_Error": float(abs(original_mean[index] - actual[index])),
                        "Transformed_Absolute_Error": float(abs(transformed_mean[index] - actual[index])),
                        "Numerically_Unchanged_At_1e_8": bool(abs(ensemble_delta[index]) <= 1e-8),
                    }
                )
                for member_index, seed in enumerate(config.member_seeds):
                    member_delta = transformed_members[member_index, index] - original_members[member_index, index]
                    member_rows.append(
                        {
                            "Dataset": dataset_name,
                            "Transform": transform,
                            "Sample_Index": index,
                            "Sample_ID": sample_id,
                            "Member_Seed": seed,
                            "Original_Prediction": float(original_members[member_index, index]),
                            "Transformed_Prediction": float(transformed_members[member_index, index]),
                            "Signed_Prediction_Change": float(member_delta),
                            "Absolute_Prediction_Change": float(abs(member_delta)),
                        }
                    )
    sample_frame = pd.DataFrame(sample_rows)
    member_frame = pd.DataFrame(member_rows)
    summary_rows: list[dict[str, float | int | str]] = []
    for (dataset_name, transform), frame in sample_frame.groupby(["Dataset", "Transform"], sort=False):
        actual = frame["Actual_Stiffness"].to_numpy(dtype=np.float64)
        transformed = frame["Transformed_Ensemble_Prediction"].to_numpy(dtype=np.float64)
        original_rmse = compute_regression_metrics(
            frame["Original_Ensemble_Prediction"].to_numpy(dtype=np.float64),
            actual,
        )["RMSE"]
        transformed_metrics = compute_regression_metrics(transformed, actual)
        absolute_change = frame["Absolute_Prediction_Change"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "Dataset": dataset_name,
                "Transform": transform,
                "Sample_Count": len(frame),
                "Mean_Signed_Prediction_Change": float(frame["Signed_Prediction_Change"].mean()),
                "Mean_Absolute_Prediction_Change": float(absolute_change.mean()),
                "Median_Absolute_Prediction_Change": float(np.median(absolute_change)),
                "Max_Absolute_Prediction_Change": float(absolute_change.max()),
                "Mirror_Change_RMSE": float(np.sqrt(np.mean(absolute_change**2))),
                "Mean_Relative_Prediction_Change_Percent": float(
                    frame["Relative_Prediction_Change_Percent"].mean()
                ),
                "Numerically_Unchanged_Count_At_1e_8": int(frame["Numerically_Unchanged_At_1e_8"].sum()),
                "Original_RMSE": original_rmse,
                "Transformed_RMSE": transformed_metrics["RMSE"],
                "Transformed_R2": transformed_metrics["R2"],
                "Mean_Change_As_Percent_Of_Original_RMSE": float(
                    absolute_change.mean() / max(original_rmse, np.finfo(np.float64).eps) * 100.0
                ),
            }
        )
    return sample_frame, member_frame, pd.DataFrame(summary_rows)


def _plot_original_vs_mirrored(sample_frame: pd.DataFrame, save_path: Path) -> None:
    transforms = [name for name in TRANSFORMS if name != "original"]
    datasets = sample_frame["Dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(len(datasets), len(transforms), figsize=(6 * len(transforms), 5 * len(datasets)))
    axes = np.asarray(axes).reshape(len(datasets), len(transforms))
    for row, dataset_name in enumerate(datasets):
        for column, transform in enumerate(transforms):
            axis = axes[row, column]
            frame = sample_frame[
                (sample_frame["Dataset"] == dataset_name) & (sample_frame["Transform"] == transform)
            ]
            x = frame["Original_Ensemble_Prediction"]
            y = frame["Transformed_Ensemble_Prediction"]
            lower = float(min(x.min(), y.min()))
            upper = float(max(x.max(), y.max()))
            axis.scatter(x, y, alpha=0.7, color="#355070")
            axis.plot([lower, upper], [lower, upper], "--", color="#b56576")
            axis.set(
                title=f"{dataset_name}: {transform}",
                xlabel="Original prediction",
                ylabel="Mirrored prediction",
            )
            axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_prediction_changes(sample_frame: pd.DataFrame, save_path: Path) -> None:
    frame = sample_frame[
        (sample_frame["Dataset"] == "Prediction") & (sample_frame["Transform"] != "original")
    ]
    pivot = frame.pivot(index="Sample_ID", columns="Transform", values="Signed_Prediction_Change")
    axis = pivot.plot(kind="bar", figsize=(15, 6), color=["#355070", "#b56576", "#e09f3e"])
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        title="External ensemble prediction change after mirroring",
        xlabel="Lattice",
        ylabel="Mirrored prediction - original prediction",
    )
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=55, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def _write_report(summary: pd.DataFrame, save_path: Path) -> None:
    lines = [
        "GCN-3 Ensemble Mirror Sensitivity Report",
        "========================================",
        "",
        "Mirrors are taken about each lattice bounding-box midpoint.",
        "Connectivity and edge weights are unchanged; all coordinate-derived features are recomputed.",
        "For a reflection-invariant scalar stiffness problem, mirrored predictions should ideally equal originals.",
        "",
    ]
    for _, row in summary[summary["Transform"] != "original"].iterrows():
        lines.extend(
            [
                f"{row['Dataset']} / {row['Transform']}:",
                f"- mean absolute prediction change: {row['Mean_Absolute_Prediction_Change']:.8f}",
                f"- maximum absolute prediction change: {row['Max_Absolute_Prediction_Change']:.8f}",
                f"- mean relative prediction change: {row['Mean_Relative_Prediction_Change_Percent']:.4f}%",
                f"- mean change as percent of original RMSE: {row['Mean_Change_As_Percent_Of_Original_RMSE']:.2f}%",
                "",
            ]
        )
    save_path.write_text("\n".join(lines), encoding="utf-8")


def run_mirror_sensitivity_experiment(
    config: MirrorSensitivityConfig,
    ensemble_run_dir: str | Path,
    train_root: str | Path,
    predict_root: str | Path,
    output_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_config(config)
    ensemble_run_dir = Path(ensemble_run_dir)
    train_root = Path(train_root)
    predict_root = Path(predict_root)
    metadata_path = ensemble_run_dir / "gcn3_ensemble_run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Ensemble metadata not found at {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("split_seed", config.split_seed)) != config.split_seed:
        raise ValueError("Configured split seed does not match the saved ensemble")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir is not None:
        output_dir = Path(run_dir)
    elif output_root is not None:
        output_dir = Path(output_root) / f"run_{timestamp}"
    else:
        output_dir = ensemble_run_dir.parent.parent / config.output_group / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "mirror_sensitivity_complete.json"
    if config.resume and completion_path.is_file():
        print(f"Reusing completed mirror sensitivity analysis at {output_dir}")
        return {
            "output_dir": output_dir,
            "sample_frame": pd.read_csv(output_dir / "mirror_sensitivity_by_sample.csv"),
            "member_frame": pd.read_csv(output_dir / "mirror_sensitivity_by_member.csv"),
            "summary": pd.read_csv(output_dir / "mirror_sensitivity_summary.csv"),
        }

    test_folders = _split_test_folders(train_root, config.split_seed)
    prediction_folders = sorted(path for path in predict_root.glob("randomness_*") if path.is_dir())
    folders_by_dataset = {"Test": test_folders, "Prediction": prediction_folders}
    sample_ids = {
        dataset_name: [folder.name for folder in folders]
        for dataset_name, folders in folders_by_dataset.items()
    }
    raw_data: dict[tuple[str, str], list[Data]] = {}
    for dataset_name, folders in folders_by_dataset.items():
        for transform in config.transforms:
            raw_data[(dataset_name, transform)] = [
                _load_transformed_sample(folder, transform) for folder in folders
            ]

    member_predictions: dict[tuple[str, str, int], np.ndarray] = {}
    ground_truth: dict[str, np.ndarray] = {}
    for seed in config.member_seeds:
        print(f"Evaluating mirror sensitivity for member seed {seed}")
        model, feature_scaler, target_scaler = _load_member_model(
            ensemble_run_dir,
            seed,
            metadata,
            config.device,
        )
        for dataset_name in folders_by_dataset:
            for transform in config.transforms:
                predictions, actual = _predict_member(
                    model,
                    raw_data[(dataset_name, transform)],
                    feature_scaler,
                    target_scaler,
                    config,
                )
                member_predictions[(dataset_name, transform, seed)] = predictions
                if dataset_name not in ground_truth:
                    ground_truth[dataset_name] = actual
                elif not np.allclose(ground_truth[dataset_name], actual, rtol=1e-6, atol=1e-6):
                    raise ValueError(f"Ground truth changed across transforms or members for {dataset_name}")

    sample_frame, member_frame, summary = build_mirror_frames(
        member_predictions,
        ground_truth,
        sample_ids,
        config,
    )
    sample_frame.to_csv(output_dir / "mirror_sensitivity_by_sample.csv", index=False)
    sample_frame[sample_frame["Dataset"] == "Prediction"].to_csv(
        output_dir / "mirror_sensitivity_prediction_samples.csv",
        index=False,
    )
    member_frame.to_csv(output_dir / "mirror_sensitivity_by_member.csv", index=False)
    summary.to_csv(output_dir / "mirror_sensitivity_summary.csv", index=False)
    _plot_original_vs_mirrored(sample_frame, output_dir / "mirror_original_vs_transformed.png")
    _plot_prediction_changes(sample_frame, output_dir / "mirror_prediction_changes.png")
    _write_report(summary, output_dir / "mirror_sensitivity_report.txt")
    (output_dir / "mirror_sensitivity_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "ensemble_run_dir": str(ensemble_run_dir),
                "train_root": str(train_root),
                "predict_root": str(predict_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    completion_path.write_text(
        json.dumps({"completed": True, "transforms": list(config.transforms)}, indent=2),
        encoding="utf-8",
    )
    print(f"Mirror sensitivity analysis saved to {output_dir}")
    print(summary)
    return {
        "output_dir": output_dir,
        "sample_frame": sample_frame,
        "member_frame": member_frame,
        "summary": summary,
    }
