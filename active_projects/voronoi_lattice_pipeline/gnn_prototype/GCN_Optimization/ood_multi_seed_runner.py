from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from colab_gnn_stiffness_prototype import load_lattice_dataset  # noqa: E402
from ood_validation_runner import (  # noqa: E402
    DOMAIN_FEATURES,
    OODExperimentConfig,
    _save_domain_partition_figure,
    build_ood_partition,
    run_ood_variant,
    validate_config,
)
from residual_error_analysis import load_lattice_feature_frame  # noqa: E402


EVALUATION_METRICS = (
    "Validation_R2",
    "Validation_RMSE",
    "Validation_MAE",
    "Validation_Bias",
    "OOD_Holdout_R2",
    "OOD_Holdout_RMSE",
    "OOD_Holdout_MAE",
    "OOD_Holdout_Bias",
    "Prediction_R2",
    "Prediction_RMSE",
    "Prediction_MAE",
    "Prediction_Bias",
)


@dataclass
class OODMultiSeedConfig:
    model_seeds: tuple[int, ...] = (11, 42, 73, 101, 202)
    partition_seed: int = 42
    variants: tuple[str, ...] = ("control", "domain_weighted")
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
    output_group: str = "gcn3_ood_domain_weighting_multi_seed"

    def experiment_config(self, model_seed: int) -> OODExperimentConfig:
        return OODExperimentConfig(
            variants=self.variants,
            seed=model_seed,
            ood_fraction=self.ood_fraction,
            validation_fraction=self.validation_fraction,
            weight_strength=self.weight_strength,
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


def validate_multi_seed_config(config: OODMultiSeedConfig) -> None:
    if not config.model_seeds:
        raise ValueError("At least one model seed is required")
    if len(set(config.model_seeds)) != len(config.model_seeds):
        raise ValueError("model_seeds must be unique")
    if set(config.variants) != {"control", "domain_weighted"}:
        raise ValueError("Multi-seed validation requires control and domain_weighted variants")
    validate_config(config.experiment_config(config.model_seeds[0]))


def _write_or_validate_manifest(
    output_dir: Path,
    config: OODMultiSeedConfig,
    train_root: Path,
    predict_root: Path,
) -> None:
    manifest_path = output_dir / "experiment_config.json"
    manifest = {
        **asdict(config),
        "domain_features": DOMAIN_FEATURES,
        "train_root": str(train_root),
        "predict_root": str(predict_root),
    }
    comparable_keys_to_ignore = {"device", "resume"}
    comparable = {key: value for key, value in manifest.items() if key not in comparable_keys_to_ignore}
    if config.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_comparable = {
            key: value for key, value in existing.items() if key not in comparable_keys_to_ignore
        }
        if existing_comparable != json.loads(json.dumps(comparable)):
            raise ValueError(f"Existing run configuration differs in {output_dir}; use a new run directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _save_partition_outputs(
    output_dir: Path,
    assignments: pd.DataFrame,
    partitions: dict[str, np.ndarray],
    prediction_features: pd.DataFrame,
) -> None:
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

    partition_labels = {"train": "Train", "validation": "Validation", "ood": "OOD_Holdout"}
    rows: list[dict[str, float | int | str]] = []
    for partition_name, indices in partitions.items():
        row: dict[str, float | int | str] = {
            "Dataset": partition_labels[partition_name],
            "Count": len(indices),
        }
        row.update(assignments.iloc[indices][DOMAIN_FEATURES].mean().to_dict())
        rows.append(row)
    prediction_row: dict[str, float | int | str] = {
        "Dataset": "Prediction",
        "Count": len(prediction_features),
    }
    prediction_row.update(prediction_features[DOMAIN_FEATURES].mean().to_dict())
    rows.append(prediction_row)
    pd.DataFrame(rows).to_csv(output_dir / "ood_domain_feature_summary.csv", index=False)
    _save_domain_partition_figure(assignments, output_dir / "ood_partition_diagnostics.png")


def _paired_error_rows(
    seed: int,
    control_predictions: pd.DataFrame,
    weighted_predictions: pd.DataFrame,
) -> list[dict[str, float | int | str | bool]]:
    keys = ["Dataset", "Sample_ID"]
    control = control_predictions[keys + ["Absolute_Error"]].rename(
        columns={"Absolute_Error": "Control_Absolute_Error"}
    )
    weighted = weighted_predictions[keys + ["Absolute_Error"]].rename(
        columns={"Absolute_Error": "Weighted_Absolute_Error"}
    )
    paired = control.merge(weighted, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(control) or len(paired) != len(weighted):
        raise ValueError(f"Prediction rows do not align for seed {seed}")
    paired["Absolute_Error_Reduction"] = (
        paired["Control_Absolute_Error"] - paired["Weighted_Absolute_Error"]
    )
    paired["Weighted_Improved"] = paired["Absolute_Error_Reduction"] > 0.0
    paired.insert(0, "Seed", seed)
    return paired.to_dict(orient="records")


def build_aggregate_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, variant_frame in comparison.groupby("Variant", sort=False):
        for metric in EVALUATION_METRICS:
            values = variant_frame[metric].astype(float)
            rows.append(
                {
                    "Variant": variant,
                    "Metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_seed_delta_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for seed, seed_frame in comparison.groupby("Seed", sort=True):
        indexed = seed_frame.set_index("Variant")
        if not {"control", "domain_weighted"}.issubset(indexed.index):
            raise ValueError(f"Seed {seed} is missing a required variant")
        row: dict[str, float | int] = {"Seed": int(seed)}
        for metric in EVALUATION_METRICS:
            row[metric] = float(indexed.loc["domain_weighted", metric] - indexed.loc["control", metric])
        for dataset in ("Validation", "OOD_Holdout", "Prediction"):
            row[f"{dataset}_Absolute_Bias_Reduction"] = float(
                abs(indexed.loc["control", f"{dataset}_Bias"])
                - abs(indexed.loc["domain_weighted", f"{dataset}_Bias"])
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Seed").reset_index(drop=True)


def build_paired_error_summary(paired_errors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (seed, dataset), frame in paired_errors.groupby(["Seed", "Dataset"], sort=True):
        reductions = frame["Absolute_Error_Reduction"].astype(float)
        rows.append(
            {
                "Seed": int(seed),
                "Dataset": dataset,
                "Sample_Count": len(frame),
                "Improved_Count": int((reductions > 0.0).sum()),
                "Worsened_Count": int((reductions < 0.0).sum()),
                "Mean_Absolute_Error_Reduction": float(reductions.mean()),
                "Median_Absolute_Error_Reduction": float(reductions.median()),
            }
        )
    return pd.DataFrame(rows)


def _save_multi_seed_figure(comparison: pd.DataFrame, save_path: Path) -> None:
    metrics = (
        ("Validation_R2", "Validation R2"),
        ("OOD_Holdout_R2", "OOD holdout R2"),
        ("Prediction_R2", "External prediction R2"),
        ("Prediction_RMSE", "External prediction RMSE"),
    )
    colors = {"control": "#355070", "domain_weighted": "#b56576"}
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, (metric, title) in zip(axes.ravel(), metrics, strict=True):
        pivot = comparison.pivot(index="Seed", columns="Variant", values=metric).sort_index()
        for seed, row in pivot.iterrows():
            axis.plot(
                [0, 1],
                [row["control"], row["domain_weighted"]],
                color="#adb5bd",
                linewidth=1.2,
                alpha=0.8,
            )
            axis.scatter(0, row["control"], color=colors["control"], s=42)
            axis.scatter(1, row["domain_weighted"], color=colors["domain_weighted"], s=42)
            axis.annotate(str(seed), (1, row["domain_weighted"]), xytext=(5, 0), textcoords="offset points")
        axis.set_xticks([0, 1], ["Control", "Domain weighted"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _promotion_summary(
    comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> dict[str, object]:
    mean_metrics = comparison.groupby("Variant")[list(EVALUATION_METRICS)].mean()
    win_rules = {
        "Validation_R2": deltas["Validation_R2"] > 0.0,
        "Validation_RMSE": deltas["Validation_RMSE"] < 0.0,
        "OOD_Holdout_R2": deltas["OOD_Holdout_R2"] > 0.0,
        "OOD_Holdout_RMSE": deltas["OOD_Holdout_RMSE"] < 0.0,
        "Prediction_R2": deltas["Prediction_R2"] > 0.0,
        "Prediction_RMSE": deltas["Prediction_RMSE"] < 0.0,
    }
    paired_totals = (
        paired_summary.groupby("Dataset")[["Sample_Count", "Improved_Count", "Worsened_Count"]]
        .sum()
        .astype(int)
        .to_dict(orient="index")
    )
    return {
        "model_seed_count": int(deltas.shape[0]),
        "weighted_seed_wins": {metric: int(wins.sum()) for metric, wins in win_rules.items()},
        "mean_metrics": {
            variant: {metric: float(value) for metric, value in row.items()}
            for variant, row in mean_metrics.iterrows()
        },
        "paired_error_totals": paired_totals,
        "interpretation_rule": (
            "Promote only if OOD and prediction improvements recur across seeds without a material "
            "mean validation regression. Confirm across multiple OOD partitions afterward."
        ),
    }


def run_ood_multi_seed_experiment(
    config: OODMultiSeedConfig,
    train_root: str | Path,
    predict_root: str | Path,
    output_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_multi_seed_config(config)
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
    _write_or_validate_manifest(output_dir, config, train_root, predict_root)

    print("Loading source and prediction graph files once for all model seeds")
    source_features = load_lattice_feature_frame(train_root)
    prediction_features = load_lattice_feature_frame(predict_root)
    raw_source_data = load_lattice_dataset(train_root)
    raw_prediction_data = load_lattice_dataset(predict_root)
    if len(raw_source_data) != len(source_features) or len(raw_prediction_data) != len(prediction_features):
        raise ValueError("Feature records and graph datasets do not align")

    partition_config = config.experiment_config(config.partition_seed)
    assignments, partitions = build_ood_partition(source_features, prediction_features, partition_config)
    _save_partition_outputs(output_dir, assignments, partitions, prediction_features)

    comparison_rows: list[dict[str, object]] = []
    paired_error_rows: list[dict[str, object]] = []
    seed_results: dict[int, dict[str, object]] = {}
    for model_seed in config.model_seeds:
        print(f"=== Model seed {model_seed}; fixed partition seed {config.partition_seed} ===")
        experiment_config = config.experiment_config(model_seed)
        seed_dir = output_dir / "per_seed" / f"seed_{model_seed}"
        variant_results: dict[str, dict[str, object]] = {}
        for variant in config.variants:
            variant_results[variant] = run_ood_variant(
                variant,
                experiment_config,
                raw_source_data,
                raw_prediction_data,
                source_features,
                prediction_features,
                assignments,
                partitions,
                seed_dir,
            )

        seed_comparison = pd.concat(
            [variant_results[variant]["summary"] for variant in config.variants],
            ignore_index=True,
        )
        seed_comparison.insert(1, "Partition_Seed", config.partition_seed)
        seed_comparison.to_csv(seed_dir / "ood_variant_comparison.csv", index=False)
        seed_delta = build_seed_delta_frame(seed_comparison)
        seed_delta.to_csv(seed_dir / "ood_weighting_delta.csv", index=False)
        comparison_rows.extend(seed_comparison.to_dict(orient="records"))
        paired_error_rows.extend(
            _paired_error_rows(
                model_seed,
                variant_results["control"]["predictions"],
                variant_results["domain_weighted"]["predictions"],
            )
        )
        seed_results[model_seed] = variant_results

        progress = pd.DataFrame(comparison_rows).sort_values(["Seed", "Variant"])
        progress.to_csv(output_dir / "multi_seed_progress.csv", index=False)
        (seed_dir / "seed_complete.json").write_text(
            json.dumps({"completed": True, "model_seed": model_seed}, indent=2),
            encoding="utf-8",
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(["Seed", "Variant"]).reset_index(drop=True)
    deltas = build_seed_delta_frame(comparison)
    aggregate = build_aggregate_frame(comparison)
    paired_errors = pd.DataFrame(paired_error_rows).sort_values(["Seed", "Dataset", "Sample_ID"])
    paired_summary = build_paired_error_summary(paired_errors)

    comparison.to_csv(output_dir / "ood_multi_seed_comparison.csv", index=False)
    deltas.to_csv(output_dir / "ood_multi_seed_deltas.csv", index=False)
    aggregate.to_csv(output_dir / "ood_multi_seed_aggregate.csv", index=False)
    paired_errors.to_csv(output_dir / "ood_multi_seed_paired_errors.csv", index=False)
    paired_summary.to_csv(output_dir / "ood_multi_seed_paired_error_summary.csv", index=False)
    _save_multi_seed_figure(comparison, output_dir / "ood_multi_seed_metric_summary.png")

    promotion_summary = _promotion_summary(comparison, deltas, paired_summary)
    (output_dir / "ood_multi_seed_promotion_summary.json").write_text(
        json.dumps(promotion_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "experiment_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "partition_seed": config.partition_seed,
                "model_seeds": list(config.model_seeds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OOD multi-seed experiment saved to {output_dir}")
    print(aggregate)
    return {
        "output_dir": output_dir,
        "assignments": assignments,
        "partitions": partitions,
        "seed_results": seed_results,
        "comparison": comparison,
        "deltas": deltas,
        "aggregate": aggregate,
        "paired_error_summary": paired_summary,
        "promotion_summary": promotion_summary,
    }
