from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from parquet_utils import ParquetDatasetReader


POINT_CLOUD_MASK_SUFFIX = "_mask"


def _to_tensor(value: Any, *, dtype: np.dtype[Any] | None = None) -> torch.Tensor:
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    elif array.dtype == np.float64:
        array = array.astype(np.float32, copy=False)
    elif array.dtype == np.int32:
        array = array.astype(np.int64, copy=False)

    array = np.ascontiguousarray(array)
    return torch.from_numpy(array)


class PointCloudChunkReader:
    def __init__(
        self,
        point_cloud_root: Path,
        episode_ends: np.ndarray,
        chunk_size: int,
        key_name: str,
        cache_size: int = 4,
    ) -> None:
        self.point_cloud_root = Path(point_cloud_root)
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
        self.episode_ends_list = self.episode_ends.tolist()
        self.chunk_size = int(chunk_size)
        self.key_name = str(key_name)
        self.cache_size = max(1, int(cache_size))
        self._chunk_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._episode_chunk_files: dict[int, list[Path]] = {}
        self._episode_dirs = self._discover_episode_dirs()

        if self.chunk_size <= 0:
            raise ValueError("Point-cloud chunk size must be positive.")
        if len(self._episode_dirs) != len(self.episode_ends):
            raise ValueError(
                f"Point-cloud root `{self.point_cloud_root}` has {len(self._episode_dirs)} episode directories, "
                f"but metadata declares {len(self.episode_ends)} episodes."
            )

    def _discover_episode_dirs(self) -> list[Path]:
        if not self.point_cloud_root.exists():
            raise FileNotFoundError(f"Missing point-cloud directory: {self.point_cloud_root}")
        if not self.point_cloud_root.is_dir():
            raise NotADirectoryError(f"Point-cloud path is not a directory: {self.point_cloud_root}")

        episode_dirs = sorted(
            path
            for path in self.point_cloud_root.iterdir()
            if path.is_dir() and path.name.startswith("episode_")
        )
        if not episode_dirs:
            raise FileNotFoundError(f"No point-cloud episode directories found under {self.point_cloud_root}")
        return episode_dirs

    def check_idx_oob(self, idx: int) -> None:
        num_rows = int(self.episode_ends[-1]) + 1
        if idx < 0 or idx >= num_rows:
            raise IndexError(f"Index {idx} is out of bounds for point-cloud dataset with {num_rows} rows")

    def get_frame(self, idx: int) -> np.ndarray:
        idx = int(idx)
        self.check_idx_oob(idx)

        episode_idx = bisect.bisect_left(self.episode_ends_list, idx)
        episode_start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1]) + 1
        frame_in_episode = idx - episode_start
        chunk_idx = frame_in_episode // self.chunk_size
        local_frame_idx = frame_in_episode % self.chunk_size

        chunk_files = self._get_episode_chunk_files(episode_idx)
        if chunk_idx >= len(chunk_files):
            raise IndexError(
                f"Frame {idx} maps to missing chunk {chunk_idx} for `{self.key_name}` in {self._episode_dirs[episode_idx].name}."
            )

        chunk = self._get_chunk(chunk_files[chunk_idx])
        if local_frame_idx >= len(chunk):
            raise IndexError(
                f"Frame {idx} maps past the end of chunk `{chunk_files[chunk_idx].name}` for `{self.key_name}`."
            )

        frame = np.asarray(chunk[local_frame_idx], dtype=np.float32)
        if frame.ndim != 2 or frame.shape[-1] != 3:
            raise ValueError(
                f"Point-cloud frame for `{self.key_name}` must have shape (num_points, 3), got {frame.shape}."
            )

        return frame

    def _get_episode_chunk_files(self, episode_idx: int) -> list[Path]:
        cached_chunk_files = self._episode_chunk_files.get(episode_idx)
        if cached_chunk_files is not None:
            return cached_chunk_files

        episode_dir = self._episode_dirs[episode_idx]
        chunk_files = sorted(episode_dir.glob("chunk_*.npy"))
        if not chunk_files:
            raise FileNotFoundError(f"No point-cloud chunks found in {episode_dir}")

        self._episode_chunk_files[episode_idx] = chunk_files
        return chunk_files

    def _get_chunk(self, chunk_path: Path) -> np.ndarray:
        cached_chunk = self._chunk_cache.pop(chunk_path, None)
        if cached_chunk is not None:
            self._chunk_cache[chunk_path] = cached_chunk
            return cached_chunk

        chunk = np.load(chunk_path, allow_pickle=True)
        self._chunk_cache[chunk_path] = chunk

        if len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)

        return chunk


class MultiModalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        universal_contract: str | Path,
        pointcloud_cache_size: int = 4,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.contract_path = Path(universal_contract).expanduser().resolve()

        with self.contract_path.open("r", encoding="utf-8") as handle:
            self.contract = yaml.safe_load(handle)

        metadata_path = self.dataset_path / "metadata.npz"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")

        with np.load(metadata_path) as metadata:
            if "episode_ends" not in metadata:
                raise KeyError(f"`episode_ends` is missing from {metadata_path}")
            self.episode_ends = np.asarray(metadata["episode_ends"], dtype=np.int64).reshape(-1)
            self.chunk_size = int(np.asarray(metadata["chunk_size"]).item()) if "chunk_size" in metadata else None

        if self.episode_ends.ndim != 1 or len(self.episode_ends) == 0:
            raise ValueError("`episode_ends` must be a non-empty 1D array.")
        self.episode_ends_list = self.episode_ends.tolist()

        parquet_path = self.dataset_path / "dummy.parquet"
        if not parquet_path.exists():
            parquet_path = self.dataset_path / "dataset.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing dataset parquet file: {parquet_path}")

        self.parquet_reader = ParquetDatasetReader(parquet_path)
        self.parquet_columns = set(self.parquet_reader.pf.schema_arrow.names)
        self.depth_keys = self._resolve_depth_keys()

        self.key_order: list[str] = []
        self.key_configs: dict[str, dict[str, Any]] = {}
        self.lowdim_keys: list[str] = []
        self.point_cloud_keys: list[str] = []
        self._parse_loader_keys()

        self.point_cloud_readers = self._build_point_cloud_readers(pointcloud_cache_size)
        self.prediction_cfg = self._parse_prediction_cfg()
        self.prediction_shape = (
            self.prediction_cfg["window"],
            *self.prediction_cfg["dim"],
        )
        self._prediction_template = torch.zeros(self.prediction_shape, dtype=torch.float32)

    def _resolve_depth_keys(self) -> set[str]:
        visual_cfg = self.contract.get("robot", {}).get("data_sources", {}).get("visual", {})
        key_entries = visual_cfg.get("keys", [])
        depth_keys: set[str] = set()

        for entry in key_entries:
            if not isinstance(entry, dict) or len(entry) != 1:
                continue
            key_name, key_cfg = next(iter(entry.items()))
            if isinstance(key_cfg, dict) and str(key_cfg.get("type", "rgb")).lower() == "depth":
                depth_keys.add(str(key_name))

        return depth_keys

    def _parse_loader_keys(self) -> None:
        loader_cfg = self.contract.get("robot", {}).get("data_loader", {})
        key_entries = loader_cfg.get("keys")

        if not isinstance(key_entries, list) or not key_entries:
            raise ValueError("`robot.data_loader.keys` must be a non-empty list.")

        for entry in key_entries:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError("Each entry in `robot.data_loader.keys` must be a single-key mapping.")

            key_name, key_cfg = next(iter(entry.items()))
            if not isinstance(key_cfg, dict):
                raise ValueError(f"`robot.data_loader.keys.{key_name}` must map to a dictionary.")

            obs_window = self._require_positive_int(
                key_cfg,
                f"robot.data_loader.keys.{key_name}.obs_window",
            )
            obs_dss = self._require_positive_int(
                key_cfg,
                f"robot.data_loader.keys.{key_name}.obs_dss",
            )

            key_name = str(key_name)
            if key_name in self.depth_keys:
                key_kind = "point_cloud"
                self.point_cloud_keys.append(key_name)
            elif key_name in self.parquet_columns:
                key_kind = "lowdim"
                self.lowdim_keys.append(key_name)
            else:
                raise KeyError(
                    f"Data-loader key `{key_name}` is neither a configured depth stream nor a parquet column. "
                    f"Available parquet columns: {sorted(self.parquet_columns)}"
                )

            self.key_order.append(key_name)
            self.key_configs[key_name] = {
                "kind": key_kind,
                "obs_window": obs_window,
                "obs_dss": obs_dss,
            }

    def _build_point_cloud_readers(self, cache_size: int) -> dict[str, PointCloudChunkReader]:
        if not self.point_cloud_keys:
            return {}
        if self.chunk_size is None:
            raise KeyError("`chunk_size` is required in metadata for point-cloud loading.")

        point_cloud_container = self.dataset_path / "point_clouds"
        if not point_cloud_container.exists():
            raise FileNotFoundError(f"Missing point-cloud root: {point_cloud_container}")

        direct_episode_dirs = sorted(
            path
            for path in point_cloud_container.iterdir()
            if path.is_dir() and path.name.startswith("episode_")
        )

        readers: dict[str, PointCloudChunkReader] = {}
        if direct_episode_dirs:
            if len(self.point_cloud_keys) != 1:
                raise ValueError(
                    "A shared `point_clouds/episode_*` layout only supports one point-cloud key in the data loader."
                )
            key_name = self.point_cloud_keys[0]
            readers[key_name] = PointCloudChunkReader(
                point_cloud_root=point_cloud_container,
                episode_ends=self.episode_ends,
                chunk_size=self.chunk_size,
                key_name=key_name,
                cache_size=cache_size,
            )
            return readers

        for key_name in self.point_cloud_keys:
            key_root = point_cloud_container / key_name
            readers[key_name] = PointCloudChunkReader(
                point_cloud_root=key_root,
                episode_ends=self.episode_ends,
                chunk_size=self.chunk_size,
                key_name=key_name,
                cache_size=cache_size,
            )

        return readers

    def _parse_prediction_cfg(self) -> dict[str, Any]:
        prediction_cfg = self.contract.get("robot", {}).get("data_loader", {}).get("prediction", {})
        if not isinstance(prediction_cfg, dict):
            raise ValueError("`robot.data_loader.prediction` must be a dictionary.")

        window = self._require_positive_int(prediction_cfg, "robot.data_loader.prediction.window")
        dss = self._require_positive_int(prediction_cfg, "robot.data_loader.prediction.dss")
        dim = prediction_cfg.get("dim")
        if not isinstance(dim, (list, tuple)) or not dim:
            raise ValueError("`robot.data_loader.prediction.dim` must be a non-empty list.")

        prediction_dim = tuple(int(axis) for axis in dim)
        if any(axis <= 0 for axis in prediction_dim):
            raise ValueError("All entries in `robot.data_loader.prediction.dim` must be positive.")

        return {
            "window": window,
            "dss": dss,
            "dim": prediction_dim,
            "mode": prediction_cfg.get("mode"),
        }

    @staticmethod
    def _require_positive_int(mapping: dict[str, Any], field_name: str) -> int:
        leaf_name = field_name.rsplit(".", maxsplit=1)[-1]
        if leaf_name not in mapping:
            raise ValueError(f"`{field_name}` is required.")
        value = int(mapping[leaf_name])
        if value <= 0:
            raise ValueError(f"`{field_name}` must be positive.")
        return value

    def get_episode(self, idx: int) -> int:
        idx = int(idx)
        self._check_idx_oob(idx)
        return bisect.bisect_left(self.episode_ends_list, idx)

    def get_episode_bounds(self, idx: int) -> tuple[int, int, int]:
        idx = int(idx)
        episode_idx = self.get_episode(idx)
        episode_start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1]) + 1
        episode_end = int(self.episode_ends[episode_idx])
        return episode_idx, episode_start, episode_end

    def __len__(self) -> int:
        return int(self.episode_ends[-1]) + 1

    def _check_idx_oob(self, idx: int) -> None:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self)} rows")

    @staticmethod
    def _build_obs_indices(idx: int, episode_start: int, obs_window: int, obs_dss: int) -> np.ndarray:
        history = obs_dss * (obs_window - 1)
        indices = np.arange(idx - history, idx + 1, obs_dss, dtype=np.int64)
        indices[indices < episode_start] = episode_start
        return indices

    def _load_lowdim_window(self, key_name: str, obs_indices: np.ndarray) -> torch.Tensor:
        first_index = int(obs_indices[0])
        last_index = int(obs_indices[-1])
        frame = self.parquet_reader.get_idx_range(first_index, last_index, [key_name])
        if key_name not in frame:
            raise KeyError(f"Lowdim key `{key_name}` is missing from the parquet dataset.")

        values = np.asarray(frame[key_name])
        selected_values = values[obs_indices - first_index]
        return _to_tensor(selected_values)

    def _load_point_cloud_window(self, key_name: str, obs_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        reader = self.point_cloud_readers[key_name]
        point_clouds = [reader.get_frame(int(frame_idx)) for frame_idx in obs_indices]
        max_points = max(point_cloud.shape[0] for point_cloud in point_clouds)

        padded = np.zeros((len(point_clouds), max_points, 3), dtype=np.float32)
        mask = np.zeros((len(point_clouds), max_points), dtype=bool)

        for frame_index, point_cloud in enumerate(point_clouds):
            num_points = int(point_cloud.shape[0])
            padded[frame_index, :num_points, :] = point_cloud
            mask[frame_index, :num_points] = True

        return _to_tensor(padded, dtype=np.float32), _to_tensor(mask)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)
        self._check_idx_oob(idx)
        _, episode_start, _ = self.get_episode_bounds(idx)

        obs_dict: dict[str, torch.Tensor] = {}
        for key_name in self.key_order:
            key_cfg = self.key_configs[key_name]
            obs_indices = self._build_obs_indices(
                idx=idx,
                episode_start=episode_start,
                obs_window=int(key_cfg["obs_window"]),
                obs_dss=int(key_cfg["obs_dss"]),
            )

            if key_cfg["kind"] == "point_cloud":
                point_cloud, point_cloud_mask = self._load_point_cloud_window(key_name, obs_indices)
                obs_dict[key_name] = point_cloud
                obs_dict[f"{key_name}{POINT_CLOUD_MASK_SUFFIX}"] = point_cloud_mask
            else:
                obs_dict[key_name] = self._load_lowdim_window(key_name, obs_indices)

        return {
            "obs_dict": obs_dict,
            "prediction": self._prediction_template.clone(),
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return multimodal_collate_fn(batch)


def multimodal_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("`batch` must be non-empty.")

    obs_dicts = [sample["obs_dict"] for sample in batch]
    first_obs_dict = obs_dicts[0]
    expected_keys = list(first_obs_dict.keys())
    for obs_dict in obs_dicts[1:]:
        if list(obs_dict.keys()) != expected_keys:
            raise ValueError("All samples in a batch must expose the same observation keys.")

    collated_obs: dict[str, torch.Tensor] = {}
    handled_keys: set[str] = set()
    point_cloud_keys = [
        key[: -len(POINT_CLOUD_MASK_SUFFIX)]
        for key in expected_keys
        if key.endswith(POINT_CLOUD_MASK_SUFFIX)
        and key[: -len(POINT_CLOUD_MASK_SUFFIX)] in first_obs_dict
    ]

    for key_name in point_cloud_keys:
        mask_key = f"{key_name}{POINT_CLOUD_MASK_SUFFIX}"
        point_cloud_tensors = [sample[key_name] for sample in obs_dicts]
        point_cloud_masks = [sample[mask_key] for sample in obs_dicts]

        max_points = max(int(point_cloud.shape[1]) for point_cloud in point_cloud_tensors)
        time_steps = int(point_cloud_tensors[0].shape[0])
        channel_dim = int(point_cloud_tensors[0].shape[2])

        padded_point_clouds = torch.zeros(
            (len(batch), time_steps, max_points, channel_dim),
            dtype=point_cloud_tensors[0].dtype,
        )
        padded_masks = torch.zeros((len(batch), time_steps, max_points), dtype=torch.bool)

        for batch_index, (point_cloud, mask) in enumerate(zip(point_cloud_tensors, point_cloud_masks, strict=True)):
            if point_cloud.ndim != 3 or point_cloud.shape[0] != time_steps or point_cloud.shape[2] != channel_dim:
                raise ValueError(
                    f"Point-cloud batch entries for `{key_name}` must have shape (T, P, {channel_dim}). "
                    f"Received {tuple(point_cloud.shape)}."
                )
            if tuple(mask.shape) != tuple(point_cloud.shape[:2]):
                raise ValueError(
                    f"Point-cloud mask for `{key_name}` must have shape {tuple(point_cloud.shape[:2])}, "
                    f"got {tuple(mask.shape)}."
                )

            num_points = int(point_cloud.shape[1])
            padded_point_clouds[batch_index, :, :num_points, :] = point_cloud
            padded_masks[batch_index, :, :num_points] = mask

        collated_obs[key_name] = padded_point_clouds
        collated_obs[mask_key] = padded_masks
        handled_keys.add(key_name)
        handled_keys.add(mask_key)

    for key_name in expected_keys:
        if key_name in handled_keys:
            continue
        collated_obs[key_name] = torch.stack([sample[key_name] for sample in obs_dicts], dim=0)

    prediction_batch = torch.stack([sample["prediction"] for sample in batch], dim=0)
    return {
        "obs_dict": collated_obs,
        "prediction": prediction_batch,
    }
