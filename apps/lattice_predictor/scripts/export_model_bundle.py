"""Export the promoted PyTorch GCN ensemble to a browser-readable JSON bundle.

This script intentionally does not import PyTorch. It reads the tensor storages
from trusted ``torch.save`` ZIP archives and exports only the state needed for
inference. That keeps the public predictor small and removes the need for a
Python model server.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parents[1]
RUN_NAME = "run_final_ensemble_uncertainty_v1"
RUN_ROOT = (
    REPOSITORY_ROOT
    / "active_projects"
    / "voronoi_lattice_pipeline"
    / "gnn_prototype"
    / "outputs"
    / "gcn3_ensemble_uncertainty"
    / RUN_NAME
)
OUTPUT_PATH = APP_ROOT / "public" / "model_bundle.json"
MEMBER_SEEDS = (11, 42, 73, 101, 202)

MODEL_STATE_KEYS = (
    "conv1.bias",
    "conv1.lin.weight",
    "bn1.weight",
    "bn1.bias",
    "bn1.running_mean",
    "bn1.running_var",
    "conv2.bias",
    "conv2.lin.weight",
    "bn2.weight",
    "bn2.bias",
    "bn2.running_mean",
    "bn2.running_var",
    "conv3.bias",
    "conv3.lin.weight",
    "bn3.weight",
    "bn3.bias",
    "bn3.running_mean",
    "bn3.running_var",
    "output.weight",
    "output.bias",
)


@dataclass(frozen=True)
class StorageType:
    dtype: np.dtype


@dataclass(frozen=True)
class StorageReference:
    key: str
    dtype: np.dtype
    size: int


class TorchArchiveUnpickler(pickle.Unpickler):
    """Restricted reader for the tensor structures produced by this project."""

    STORAGE_DTYPES = {
        "FloatStorage": np.dtype("<f4"),
        "DoubleStorage": np.dtype("<f8"),
        "LongStorage": np.dtype("<i8"),
        "IntStorage": np.dtype("<i4"),
        "ShortStorage": np.dtype("<i2"),
        "ByteStorage": np.dtype("u1"),
        "BoolStorage": np.dtype("?"),
    }

    ALLOWED_GLOBALS = {
        ("collections", "OrderedDict"): OrderedDict,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("codecs", "encode"): codecs.encode,
        ("_codecs", "encode"): codecs.encode,
    }

    def __init__(self, stream: io.BytesIO, archive: zipfile.ZipFile, archive_root: str):
        super().__init__(stream)
        self.archive = archive
        self.archive_root = archive_root

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name in self.STORAGE_DTYPES:
            return StorageType(self.STORAGE_DTYPES[name])
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return self._rebuild_tensor
        if module == "torch._utils" and name in {
            "_rebuild_parameter",
            "_rebuild_parameter_with_state",
        }:
            return self._rebuild_parameter
        try:
            return self.ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}") from exc

    def persistent_load(self, persistent_id: tuple[Any, ...]) -> StorageReference:
        if len(persistent_id) < 5 or persistent_id[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent ID: {persistent_id!r}")
        _, storage_type, key, _location, size = persistent_id[:5]
        if not isinstance(storage_type, StorageType):
            raise pickle.UnpicklingError("Unknown tensor storage type")
        return StorageReference(str(key), storage_type.dtype, int(size))

    def _rebuild_tensor(
        self,
        storage: StorageReference,
        storage_offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        *_metadata: Any,
    ) -> np.ndarray:
        storage_path = f"{self.archive_root}/data/{storage.key}"
        raw_storage = self.archive.read(storage_path)
        flat = np.frombuffer(raw_storage, dtype=storage.dtype, count=storage.size)
        return np.ndarray(
            shape=tuple(size),
            dtype=storage.dtype,
            buffer=flat,
            offset=int(storage_offset) * storage.dtype.itemsize,
            strides=tuple(int(value) * storage.dtype.itemsize for value in stride),
        ).copy()

    @staticmethod
    def _rebuild_parameter(data: np.ndarray, *_metadata: Any) -> np.ndarray:
        return data


def read_torch_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        archive_root = archive.namelist()[0].split("/", maxsplit=1)[0]
        payload = archive.read(f"{archive_root}/data.pkl")
        return TorchArchiveUnpickler(io.BytesIO(payload), archive, archive_root).load()


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_member(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    member_root = RUN_ROOT / "per_model" / f"member_seed_{seed}"
    checkpoint_path = member_root / "lattice_gnn_model.pt"
    target_scaler_path = member_root / "target_scaler.json"
    payload = read_torch_archive(checkpoint_path)
    state = payload["model_state_dict"]

    missing = [key for key in MODEL_STATE_KEYS if key not in state]
    if missing:
        raise KeyError(f"Member seed {seed} is missing state keys: {', '.join(missing)}")

    member = {
        "seed": seed,
        "state": {key: np.asarray(state[key]).tolist() for key in MODEL_STATE_KEYS},
    }
    metadata = {
        "seed": seed,
        "modelSha256": sha256(checkpoint_path),
        "targetScalerSha256": sha256(target_scaler_path),
    }
    return member, {
        "payload": payload,
        "targetScaler": json.loads(target_scaler_path.read_text(encoding="utf-8")),
        "metadata": metadata,
    }


def build_bundle() -> dict[str, Any]:
    summary = read_csv_row(RUN_ROOT / "gcn3_ensemble_summary.csv")
    members: list[dict[str, Any]] = []
    export_details: list[dict[str, Any]] = []
    shared_payload: dict[str, Any] | None = None
    shared_target_scaler: dict[str, Any] | None = None

    for seed in MEMBER_SEEDS:
        member, details = export_member(seed)
        members.append(member)
        export_details.append(details["metadata"])

        payload = details["payload"]
        target_scaler = details["targetScaler"]
        if shared_payload is None:
            shared_payload = payload
            shared_target_scaler = target_scaler
        else:
            for key in (
                "scaler_mean",
                "scaler_scale",
                "graph_scaler_mean",
                "graph_scaler_scale",
            ):
                if not np.allclose(shared_payload[key], payload[key], rtol=0, atol=0):
                    raise ValueError(f"Feature scaler {key} differs across ensemble members")
            if target_scaler != shared_target_scaler:
                raise ValueError("Target scaler differs across ensemble members")

    assert shared_payload is not None
    assert shared_target_scaler is not None

    return {
        "schemaVersion": 1,
        "model": {
            "name": "GCN-3 Ensemble",
            "run": RUN_NAME,
            "target": "Lattice stiffness",
            "units": "N/mm",
            "memberSeeds": list(MEMBER_SEEDS),
            "splitSeed": int(summary["Split_Seed"]),
            "hiddenDimension": int(summary["Hidden_Dim"]),
            "dropout": float(summary["Dropout"]),
            "metrics": {
                "validation": {
                    "r2": float(summary["Validation_R2"]),
                    "rmse": float(summary["Validation_RMSE"]),
                    "mae": float(summary["Validation_MAE"]),
                },
                "test": {
                    "r2": float(summary["Test_R2"]),
                    "rmse": float(summary["Test_RMSE"]),
                    "mae": float(summary["Test_MAE"]),
                },
                "external": {
                    "r2": float(summary["Prediction_R2"]),
                    "rmse": float(summary["Prediction_RMSE"]),
                    "mae": float(summary["Prediction_MAE"]),
                },
            },
        },
        "preprocessing": {
            "nodeFeatureOrder": [
                "x",
                "y",
                "degree",
                "weighted_degree",
                "center_distance",
                "boundary_indicator",
                "mean_incident_distance",
                "max_incident_distance",
                "mean_incident_weight",
            ],
            "graphFeatureOrder": [
                "node_count",
                "edge_count",
                "density",
                "total_edge_weight",
                "mean_edge_weight",
                "std_edge_weight",
                "mean_degree",
                "std_degree",
                "mean_weighted_degree",
                "std_weighted_degree",
                "mean_center_distance",
                "std_center_distance",
            ],
            "nodeFeatureScaler": {
                "mean": np.asarray(shared_payload["scaler_mean"]).tolist(),
                "scale": np.asarray(shared_payload["scaler_scale"]).tolist(),
            },
            "graphFeatureScaler": {
                "mean": np.asarray(shared_payload["graph_scaler_mean"]).tolist(),
                "scale": np.asarray(shared_payload["graph_scaler_scale"]).tolist(),
            },
            "targetScaler": {
                "mean": float(shared_target_scaler["mean"]),
                "scale": float(shared_target_scaler["scale"]),
            },
            "trainingReference": {
                "source": "800-sample training split, split seed 42",
                "nodeCount": {"min": 117.0, "max": 121.0},
                "edgeCount": {"min": 223.0, "max": 307.0},
                "density": {
                    "min": 0.03168044077134986,
                    "max": 0.04228650137741047,
                },
                "meanDegree": {
                    "min": 3.7478991596638656,
                    "max": 5.074380165289257,
                },
                "stdDegree": {
                    "min": 1.0986452468022392,
                    "max": 1.6235831144835038,
                },
                "meanWeightedDegree": {
                    "min": 3.7478991596638656,
                    "max": 5.074380165289257,
                },
                "meanEdgeWeight": {"min": 1.0, "max": 1.0},
                "xSpan": {"min": 300.0, "max": 300.0},
                "ySpan": {"min": 363.73067, "max": 363.73067},
            },
        },
        "members": members,
        "provenance": {
            "sourceRun": str(RUN_ROOT.relative_to(REPOSITORY_ROOT)),
            "exports": export_details,
            "note": (
                "Tensor values were exported from the promoted torch.save checkpoints. "
                "Inference remains subject to the limitations documented in the model card."
            ),
        },
    }


def main() -> None:
    bundle = build_bundle()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(bundle, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(bundle['members'])} members to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
