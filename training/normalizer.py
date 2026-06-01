from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


STD_EPS = 1e-6
NORMALIZER_VERSION = 1
DEFAULT_NORMALIZATION_REPRESENTATION = "standard"
SUPPORTED_NORMALIZATION_REPRESENTATIONS = frozenset({"standard", "rotvec", "quat", "matrix", "euler"})


def resolve_point_cloud_source_specs(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    visual_cfg = contract.get("robot", {}).get("data_sources", {}).get("visual", {})
    key_entries = visual_cfg.get("keys", [])
    point_cloud_specs: dict[str, dict[str, Any]] = {}

    for entry in key_entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue

        key_name, key_cfg = next(iter(entry.items()))
        if not isinstance(key_cfg, dict):
            continue

        key_name = str(key_name)
        key_type = str(key_cfg.get("type", "rgb")).lower()
        if key_type == "depth":
            point_cloud_specs[key_name] = {"kind": "depth"}
            continue

        if key_type == "scene_points":
            path_value = key_cfg.get("path")
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError(
                    f"`robot.data_sources.visual.keys.{key_name}.path` is required when type is `scene_points`."
                )
            point_cloud_specs[key_name] = {
                "kind": "scene_points",
                "path": path_value,
            }

    return point_cloud_specs


def _resolve_parquet_path(dataset_path: Path) -> Path:
    dummy_path = dataset_path / "dummy.parquet"
    if dummy_path.exists():
        return dummy_path

    dataset_parquet_path = dataset_path / "dataset.parquet"
    if dataset_parquet_path.exists():
        return dataset_parquet_path

    raise FileNotFoundError(
        f"Could not find parquet data in {dataset_path}. Expected `dummy.parquet` or `dataset.parquet`."
    )


def _load_contract(universal_contract: str | Path) -> dict[str, Any]:
    contract_path = Path(universal_contract).expanduser().resolve()
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise ValueError(f"Universal contract `{contract_path}` must parse to a dictionary.")
    return contract


def _resolve_depth_keys(contract: dict[str, Any]) -> set[str]:
    return {
        key_name
        for key_name, key_cfg in resolve_point_cloud_source_specs(contract).items()
        if key_cfg["kind"] == "depth"
    }


def _to_numeric_array(column) -> np.ndarray:
    values = column.combine_chunks().to_numpy(zero_copy_only=False)
    if values.dtype == object and len(values) and isinstance(values[0], (list, tuple, np.ndarray)):
        values = np.stack(values, axis=0)

    return np.asarray(values)


def _feature_shape_from_array(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(array).shape[1:])


def _feature_size(feature_shape: tuple[int, ...]) -> int:
    return int(np.prod(feature_shape, dtype=np.int64)) if feature_shape else 1


def resolve_normalization_config(
    key_cfg: dict[str, Any],
    *,
    key_name: str,
    field_prefix: str,
) -> tuple[bool, str]:
    normalize_value = key_cfg.get("normalize", False)
    if not isinstance(normalize_value, bool):
        raise ValueError(f"`{field_prefix}.normalize` for `{key_name}` must be a boolean.")

    representation = str(
        key_cfg.get(
            "normalization_representation",
            DEFAULT_NORMALIZATION_REPRESENTATION,
        )
    )
    if representation not in SUPPORTED_NORMALIZATION_REPRESENTATIONS:
        raise ValueError(
            f"`{field_prefix}.normalization_representation` for `{key_name}` must be one of "
            f"{sorted(SUPPORTED_NORMALIZATION_REPRESENTATIONS)}; got `{representation}`."
        )

    return bool(normalize_value), representation


def _resolve_normalized_lowdim_keys(
    contract: dict[str, Any],
    parquet_columns: set[str],
) -> dict[str, dict[str, Any]]:
    loader_cfg = contract.get("robot", {}).get("data_loader", {})
    key_entries = loader_cfg.get("keys")
    point_cloud_specs = resolve_point_cloud_source_specs(contract)

    if not isinstance(key_entries, list) or not key_entries:
        raise ValueError("`robot.data_loader.keys` must be a non-empty list.")

    seen_key_configs: dict[str, dict[str, Any]] = {}
    normalized_key_configs: dict[str, dict[str, Any]] = {}
    for entry in key_entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Each entry in `robot.data_loader.keys` must be a single-key mapping.")

        key_name, key_cfg = next(iter(entry.items()))
        if not isinstance(key_cfg, dict):
            raise ValueError(f"`robot.data_loader.keys.{key_name}` must map to a dictionary.")

        key_name = str(key_name)
        normalize, representation = resolve_normalization_config(
            key_cfg,
            key_name=key_name,
            field_prefix=f"robot.data_loader.keys.{key_name}",
        )

        existing_seen_cfg = seen_key_configs.get(key_name)
        if existing_seen_cfg is not None:
            if (
                bool(existing_seen_cfg["normalize"]) != normalize
                or str(existing_seen_cfg["representation"]) != representation
            ):
                raise ValueError(
                    f"Duplicate data-loader key `{key_name}` has conflicting normalization settings: "
                    f"{existing_seen_cfg} vs normalize={normalize}, representation={representation}."
                )
        else:
            seen_key_configs[key_name] = {
                "normalize": normalize,
                "representation": representation,
            }

        if key_name in point_cloud_specs:
            if normalize:
                raise ValueError(
                    f"Normalization is currently supported only for lowdim parquet keys. "
                    f"`{key_name}` is configured as a depth/point-cloud key."
                )
            continue

        if key_name not in parquet_columns:
            raise KeyError(
                f"Data-loader key `{key_name}` is not present in the dataset parquet columns. "
                f"Available columns: {sorted(parquet_columns)}"
            )

        if not normalize:
            continue

        existing_cfg = normalized_key_configs.get(key_name)
        if existing_cfg is not None:
            if existing_cfg["representation"] != representation:
                raise ValueError(
                    f"Duplicate normalized data-loader key `{key_name}` has conflicting normalization "
                    f"representations: `{existing_cfg['representation']}` vs `{representation}`."
                )
            continue

        normalized_key_configs[key_name] = {
            "representation": representation,
        }

    return normalized_key_configs


class DatasetNormalizer:
    def __init__(self, key_stats: dict[str, dict[str, Any]], *, version: int = NORMALIZER_VERSION) -> None:
        self.version = int(version)
        self.key_stats: dict[str, dict[str, Any]] = {}

        for key_name, raw_stats in key_stats.items():
            feature_shape = tuple(int(dim) for dim in raw_stats["feature_shape"])
            feature_size = _feature_size(feature_shape)
            mean = np.asarray(raw_stats["mean"], dtype=np.float32).reshape(-1)
            std = np.asarray(raw_stats["std"], dtype=np.float32).reshape(-1)
            representation = str(raw_stats["representation"])
            count = int(raw_stats["count"])

            if representation not in SUPPORTED_NORMALIZATION_REPRESENTATIONS:
                raise ValueError(
                    f"Key `{key_name}` uses unsupported normalization representation `{representation}`."
                )
            if mean.size != feature_size or std.size != feature_size:
                raise ValueError(
                    f"Key `{key_name}` expected feature size {feature_size}, got mean size {mean.size} "
                    f"and std size {std.size}."
                )

            self.key_stats[str(key_name)] = {
                "mean": mean.astype(np.float32, copy=True),
                "std": np.maximum(std.astype(np.float32, copy=True), STD_EPS),
                "feature_shape": feature_shape,
                "representation": representation,
                "count": count,
            }

    @classmethod
    def load(cls, normalizer_path: str | Path) -> DatasetNormalizer:
        path = Path(normalizer_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing normalizer file: {path}")

        artifact = np.load(path, allow_pickle=True).item()
        if not isinstance(artifact, dict):
            raise ValueError(f"Normalizer artifact `{path}` must be a dictionary.")

        version = int(artifact.get("version", NORMALIZER_VERSION))
        key_stats = artifact.get("keys", {})
        if not isinstance(key_stats, dict):
            raise ValueError(f"Normalizer artifact `{path}` is missing a valid `keys` mapping.")
        return cls(key_stats, version=version)

    def save(self, normalizer_path: str | Path) -> Path:
        path = Path(normalizer_path).expanduser().resolve()
        artifact = {
            "version": self.version,
            "keys": {
                key_name: {
                    "mean": stats["mean"].copy(),
                    "std": stats["std"].copy(),
                    "feature_shape": tuple(stats["feature_shape"]),
                    "representation": stats["representation"],
                    "count": int(stats["count"]),
                }
                for key_name, stats in self.key_stats.items()
            },
        }
        np.save(path, artifact, allow_pickle=True)
        return path

    def has_key(self, key_name: str) -> bool:
        return str(key_name) in self.key_stats

    def require_key(self, key_name: str) -> dict[str, Any]:
        key_name = str(key_name)
        if key_name not in self.key_stats:
            raise KeyError(f"Normalizer does not contain stats for key `{key_name}`.")
        return self.key_stats[key_name]

    def normalize_key(self, key_name: str, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._transform_key(key_name, value, inverse=False)

    def denormalize_key(self, key_name: str, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._transform_key(key_name, value, inverse=True)

    def normalize_modal_dict(self, modal_dict: dict[str, Any]) -> dict[str, Any]:
        return self._transform_mapping(modal_dict, inverse=False)

    def denormalize_modal_dict(self, modal_dict: dict[str, Any]) -> dict[str, Any]:
        return self._transform_mapping(modal_dict, inverse=True)

    def normalize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._transform_nested_mapping(sample, inverse=False)

    def denormalize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._transform_nested_mapping(sample, inverse=True)

    def _transform_mapping(self, modal_dict: dict[str, Any], *, inverse: bool) -> dict[str, Any]:
        if not isinstance(modal_dict, dict):
            raise TypeError("Expected a dictionary of modal tensors/arrays.")

        output: dict[str, Any] = {}
        for key_name, value in modal_dict.items():
            if self.has_key(key_name):
                output[key_name] = self._transform_key(key_name, value, inverse=inverse)
            else:
                output[key_name] = value
        return output

    def _transform_nested_mapping(self, mapping: dict[str, Any], *, inverse: bool) -> dict[str, Any]:
        if not isinstance(mapping, dict):
            raise TypeError("Expected a dictionary sample.")

        output: dict[str, Any] = {}
        for key_name, value in mapping.items():
            if isinstance(value, dict):
                output[key_name] = self._transform_mapping(value, inverse=inverse)
            elif self.has_key(key_name):
                output[key_name] = self._transform_key(key_name, value, inverse=inverse)
            else:
                output[key_name] = value
        return output

    def _transform_key(
        self,
        key_name: str,
        value: np.ndarray | torch.Tensor,
        *,
        inverse: bool,
    ) -> np.ndarray | torch.Tensor:
        stats = self.require_key(key_name)
        if isinstance(value, torch.Tensor):
            return self._transform_tensor(value, stats, inverse=inverse)
        return self._transform_array(value, stats, inverse=inverse)

    def _transform_array(
        self,
        value: np.ndarray,
        stats: dict[str, Any],
        *,
        inverse: bool,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        flattened, leading_shape = self._flatten_value(
            array_shape=array.shape,
            feature_shape=stats["feature_shape"],
        )
        reshaped = array.reshape(flattened)
        if inverse:
            transformed = reshaped * stats["std"] + stats["mean"]
        else:
            transformed = (reshaped - stats["mean"]) / stats["std"]
        return transformed.reshape(self._restore_shape(leading_shape, stats["feature_shape"])).astype(np.float32)

    def _transform_tensor(
        self,
        value: torch.Tensor,
        stats: dict[str, Any],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        tensor = value.to(dtype=torch.float32)
        flattened, leading_shape = self._flatten_value(
            array_shape=tuple(int(dim) for dim in tensor.shape),
            feature_shape=stats["feature_shape"],
        )
        reshaped = tensor.reshape(flattened)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(stats["std"], dtype=torch.float32, device=tensor.device)
        if inverse:
            transformed = reshaped * std + mean
        else:
            transformed = (reshaped - mean) / std
        return transformed.reshape(self._restore_shape(leading_shape, stats["feature_shape"]))

    @staticmethod
    def _flatten_value(
        *,
        array_shape: tuple[int, ...],
        feature_shape: tuple[int, ...],
    ) -> tuple[tuple[int, int], tuple[int, ...]]:
        feature_ndim = len(feature_shape)
        feature_size = _feature_size(feature_shape)

        if feature_ndim:
            if len(array_shape) < feature_ndim:
                raise ValueError(
                    f"Value shape {array_shape} is missing expected trailing feature shape {feature_shape}."
                )
            if tuple(array_shape[-feature_ndim:]) != feature_shape:
                raise ValueError(
                    f"Value shape {array_shape} does not match expected trailing feature shape {feature_shape}."
                )
            leading_shape = array_shape[:-feature_ndim]
        else:
            leading_shape = array_shape

        return (-1, feature_size), leading_shape

    @staticmethod
    def _restore_shape(leading_shape: tuple[int, ...], feature_shape: tuple[int, ...]) -> tuple[int, ...]:
        if feature_shape:
            return tuple(leading_shape) + tuple(feature_shape)
        return tuple(leading_shape)


def build_normalizer(dataset_path: str | Path, universal_contract: str | Path) -> Path:
    import pyarrow.parquet as pq

    dataset_path = Path(dataset_path).expanduser().resolve()
    parquet_path = _resolve_parquet_path(dataset_path)
    parquet_table = pq.read_table(parquet_path)

    normalized_key_configs = _resolve_normalized_lowdim_keys(
        _load_contract(universal_contract),
        set(parquet_table.column_names),
    )

    key_stats: dict[str, dict[str, Any]] = {}
    for key_name, key_cfg in normalized_key_configs.items():
        column_array = _to_numeric_array(parquet_table[key_name]).astype(np.float32, copy=False)
        feature_shape = _feature_shape_from_array(column_array)
        flattened = column_array.reshape(-1, _feature_size(feature_shape))
        column_std = np.maximum(flattened.std(axis=0), STD_EPS)

        key_stats[key_name] = {
            "mean": flattened.mean(axis=0).astype(np.float32),
            "std": column_std.astype(np.float32),
            "feature_shape": feature_shape,
            "representation": key_cfg["representation"],
            "count": int(column_array.shape[0]),
        }

    normalizer = DatasetNormalizer(key_stats, version=NORMALIZER_VERSION)
    output_path = dataset_path / "normalizer.npy"
    return normalizer.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build normalization statistics for a dataset.")
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        required=True,
        type=str,
        help="Path to the extracted dataset directory.",
    )
    parser.add_argument(
        "--universal-contract",
        dest="universal_contract",
        required=True,
        type=str,
        help="Path to the universal contract file.",
    )
    args = parser.parse_args()

    output_path = build_normalizer(args.dataset_path, args.universal_contract)
    print(f"Wrote normalizer to {output_path}")
