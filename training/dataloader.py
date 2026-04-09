from __future__ import annotations

import argparse
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

try:
    import torch

    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
    _DatasetBase = torch.utils.data.Dataset
except ModuleNotFoundError as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc

    class _DatasetBase:
        pass


@dataclass(frozen=True)
class ObservationKeySpec:
    name: str
    obs_window: int
    obs_dss: int
    dim: tuple[int, ...] | None = None


@dataclass(frozen=True)
class VisualKeySpec:
    name: str
    obs_window: int
    obs_dss: int
    dim: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ActionKeySpec:
    name: str
    source_key: str


@dataclass(frozen=True)
class EpisodeRowRange:
    episode_id: int
    episode_name: str
    camera_name: str
    start_global_index: int
    end_global_index: int
    first_episode_frame: int
    last_episode_frame: int


def _require_torch():
    global torch, _TORCH_IMPORT_ERROR
    if torch is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required for the dataloader. Install the `pytorch` package in the runtime environment."
            ) from exc
        torch = torch_module
        _TORCH_IMPORT_ERROR = None
    return torch


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required to decode chunked videos. Install the `opencv` package in the runtime environment."
        ) from exc
    return cv2


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyArrow is required to load parquet datasets. Install the `pyarrow` package in the runtime environment."
        ) from exc
    return pq


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Matplotlib is required for sample visualization. Install the `matplotlib` package in the runtime environment."
        ) from exc
    return plt


def _default_contract_path() -> Path:
    return Path(__file__).resolve().parents[1] / "universal_contract.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def load_contract(parquet_path: Path, contract_path: str | Path | None = None) -> dict[str, Any]:
    candidate_paths: list[Path] = []
    if contract_path is not None:
        candidate_paths.append(Path(contract_path).expanduser().resolve())
    else:
        candidate_paths.extend(
            [
                parquet_path.with_name("contract.yaml"),
                _default_contract_path(),
            ]
        )

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return _load_yaml(candidate_path)

    pq = _require_pyarrow()
    schema = pq.read_schema(parquet_path)
    metadata = schema.metadata or {}
    source_buffer_dir_raw = metadata.get(b"source_buffer_dir")
    if source_buffer_dir_raw:
        source_buffer_dir = Path(source_buffer_dir_raw.decode("utf-8"))
        if source_buffer_dir.exists() and source_buffer_dir.is_dir():
            for episode_dir in sorted(source_buffer_dir.glob("episode_*")):
                candidate_path = episode_dir / "contract.yaml"
                if candidate_path.exists():
                    return _load_yaml(candidate_path)

    searched = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not locate a contract file for parquet dataset `{parquet_path}`. Searched: {searched}"
    )


def _parse_key_sequence(sequence: Any, section_name: str) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(sequence, list) or not sequence:
        raise ValueError(f"`{section_name}` must be a non-empty sequence.")

    parsed_entries: list[tuple[str, dict[str, Any]]] = []
    for entry in sequence:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                f"Each item in `{section_name}` must be a single-key mapping. Got: {entry!r}"
            )
        key_name, key_cfg = next(iter(entry.items()))
        if not isinstance(key_cfg, dict):
            raise ValueError(f"Configuration for `{key_name}` in `{section_name}` must be a mapping.")
        parsed_entries.append((str(key_name), key_cfg))
    return parsed_entries


def _parse_dim(dim_cfg: Any) -> tuple[int, ...] | None:
    if dim_cfg is None:
        return None
    if not isinstance(dim_cfg, (list, tuple)):
        raise ValueError(f"Expected `dim` to be a sequence, got {dim_cfg!r}")
    return tuple(int(value) for value in dim_cfg)


def parse_lowdim_specs(contract: dict[str, Any]) -> dict[str, ObservationKeySpec]:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        raise ValueError("Contract must contain a top-level `robot` mapping.")

    lowdim_cfg = robot_cfg.get("data_sources", {}).get("lowdim", {})
    if not isinstance(lowdim_cfg, dict):
        raise ValueError("Contract is missing `robot.data_sources.lowdim`.")

    lowdim_specs: dict[str, ObservationKeySpec] = {}
    for key_name, key_cfg in _parse_key_sequence(lowdim_cfg.get("keys", []), "robot.data_sources.lowdim.keys"):
        obs_window = int(key_cfg.get("obs_window", 1))
        obs_dss = int(key_cfg.get("obs_dss", 1))
        if obs_window <= 0 or obs_dss <= 0:
            raise ValueError(f"Lowdim key `{key_name}` must have positive obs_window and obs_dss.")
        lowdim_specs[key_name] = ObservationKeySpec(
            name=key_name,
            obs_window=obs_window,
            obs_dss=obs_dss,
            dim=_parse_dim(key_cfg.get("dim")),
        )

    if not lowdim_specs:
        raise ValueError("No lowdim observation keys were parsed from the contract.")
    return lowdim_specs


def parse_visual_specs(contract: dict[str, Any]) -> dict[str, VisualKeySpec]:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        raise ValueError("Contract must contain a top-level `robot` mapping.")

    visual_cfg = robot_cfg.get("data_sources", {}).get("visual", {})
    if not isinstance(visual_cfg, dict):
        raise ValueError("Contract is missing `robot.data_sources.visual`.")

    visual_specs: dict[str, VisualKeySpec] = {}
    for key_name, key_cfg in _parse_key_sequence(visual_cfg.get("keys", []), "robot.data_sources.visual.keys"):
        obs_window = int(key_cfg.get("obs_window", 1))
        obs_dss_raw = key_cfg.get("obs_dss")
        if obs_dss_raw is None:
            warnings.warn(
                f"Visual key `{key_name}` is missing `obs_dss`; defaulting to 1.",
                stacklevel=2,
            )
            obs_dss = 1
        else:
            obs_dss = int(obs_dss_raw)
        if obs_window <= 0 or obs_dss <= 0:
            raise ValueError(f"Visual key `{key_name}` must have positive obs_window and obs_dss.")
        visual_specs[key_name] = VisualKeySpec(
            name=key_name,
            obs_window=obs_window,
            obs_dss=obs_dss,
            dim=_parse_dim(key_cfg.get("dim")),
        )

    if not visual_specs:
        raise ValueError("No visual observation keys were parsed from the contract.")
    return visual_specs


def parse_action_specs(
    contract: dict[str, Any],
    lowdim_specs: dict[str, ObservationKeySpec],
) -> tuple[int, int, dict[str, ActionKeySpec]]:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        raise ValueError("Contract must contain a top-level `robot` mapping.")

    action_cfg = robot_cfg.get("action", {})
    if not isinstance(action_cfg, dict):
        raise ValueError("Contract is missing `robot.action`.")

    action_window = int(action_cfg.get("window", 0))
    action_dss = int(action_cfg.get("dss", 0))
    if action_window <= 0 or action_dss <= 0:
        raise ValueError("`robot.action.window` and `robot.action.dss` must be positive integers.")

    action_keys = action_cfg.get("keys")
    if not isinstance(action_keys, dict):
        raise ValueError("`robot.action.keys` must be defined as a mapping.")

    required_action_names = ("robot_pos", "robot_ori")
    action_specs: dict[str, ActionKeySpec] = {}
    for action_name in required_action_names:
        action_key_cfg = action_keys.get(action_name)
        if not isinstance(action_key_cfg, dict):
            raise ValueError(f"Missing `{action_name}` in `robot.action.keys`.")
        source_key = str(action_key_cfg.get("source", "")).strip()
        if not source_key:
            raise ValueError(f"`robot.action.keys.{action_name}.source` must be provided.")
        if source_key not in lowdim_specs:
            raise ValueError(
                f"Action source `{source_key}` for `{action_name}` does not match any lowdim key."
            )
        action_specs[action_name] = ActionKeySpec(name=action_name, source_key=source_key)

    return action_window, action_dss, action_specs


def load_parquet_rows(parquet_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    parquet_path = Path(parquet_path).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet dataset does not exist: {parquet_path}")

    pq = _require_pyarrow()
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"Parquet dataset contains no rows: {parquet_path}")

    rows = sorted(rows, key=lambda row: int(row["global_frame_index"]))
    global_indices = [int(row["global_frame_index"]) for row in rows]
    expected_indices = list(range(len(rows)))
    if global_indices != expected_indices:
        raise ValueError(
            "Parquet dataset global_frame_index values must be contiguous from 0 to len(rows)-1."
        )

    metadata = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (table.schema.metadata or {}).items()
    }
    return rows, metadata


def build_episode_ranges(rows: list[dict[str, Any]]) -> dict[int, EpisodeRowRange]:
    episode_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        episode_groups.setdefault(int(row["episode_id"]), []).append(row)

    ranges: dict[int, EpisodeRowRange] = {}
    for episode_id, episode_rows in episode_groups.items():
        episode_rows = sorted(episode_rows, key=lambda row: int(row["global_frame_index"]))
        camera_names = {str(row["camera_name"]) for row in episode_rows}
        if len(camera_names) != 1:
            raise ValueError(
                f"Episode {episode_id} has multiple camera names in the parquet rows: {sorted(camera_names)}"
            )

        episode_frame_indices = [int(row["episode_frame_index"]) for row in episode_rows]
        ranges[episode_id] = EpisodeRowRange(
            episode_id=episode_id,
            episode_name=str(episode_rows[0]["episode_name"]),
            camera_name=str(episode_rows[0]["camera_name"]),
            start_global_index=int(episode_rows[0]["global_frame_index"]),
            end_global_index=int(episode_rows[-1]["global_frame_index"]),
            first_episode_frame=min(episode_frame_indices),
            last_episode_frame=max(episode_frame_indices),
        )
    return ranges


def build_episode_frame_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for row_position, row in enumerate(rows):
        key = (int(row["episode_id"]), int(row["episode_frame_index"]))
        if key in lookup:
            raise ValueError(f"Duplicate episode/frame key found in parquet rows: {key}")
        lookup[key] = row_position
    return lookup


def compute_observation_indices(
    center_episode_frame: int,
    window: int,
    dss: int,
    episode_last_frame: int,
    episode_first_frame: int = 0,
) -> list[int]:
    if window <= 0 or dss <= 0:
        raise ValueError("window and dss must be positive integers.")

    frame_indices: list[int] = []
    for offset in range(window - 1, -1, -1):
        requested_index = center_episode_frame - offset * dss
        frame_indices.append(int(np.clip(requested_index, episode_first_frame, episode_last_frame)))
    return frame_indices


def compute_action_indices(
    center_episode_frame: int,
    window: int,
    dss: int,
    episode_last_frame: int,
    episode_first_frame: int = 0,
) -> list[int]:
    if window <= 0 or dss <= 0:
        raise ValueError("window and dss must be positive integers.")

    frame_indices: list[int] = []
    for step in range(window):
        requested_index = center_episode_frame + step * dss
        frame_indices.append(int(np.clip(requested_index, episode_first_frame, episode_last_frame)))
    return frame_indices


def fetch_lowdim_sequence(
    rows: list[dict[str, Any]],
    frame_lookup: dict[tuple[int, int], int],
    episode_id: int,
    frame_indices: list[int],
    key_name: str,
) -> np.ndarray:
    sequence: list[np.ndarray] = []
    for frame_index in frame_indices:
        row_position = frame_lookup.get((episode_id, frame_index))
        if row_position is None:
            raise KeyError(f"No parquet row exists for episode {episode_id}, frame {frame_index}.")
        lowdim_data = rows[row_position].get("lowdimData")
        if not isinstance(lowdim_data, dict) or key_name not in lowdim_data:
            raise KeyError(f"Lowdim key `{key_name}` is missing from parquet row {row_position}.")
        sequence.append(np.asarray(lowdim_data[key_name], dtype=np.float64))
    return np.stack(sequence, axis=0)


def rotation_matrix_to_rot6d(rotation_matrices: np.ndarray) -> np.ndarray:
    rotation_matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            f"Rotation matrices must end with shape (3, 3). Got {rotation_matrices.shape}."
        )

    if rotation_matrices.ndim == 2:
        return rotation_matrices[:, :2].T.reshape(6)
    return rotation_matrices[:, :, :2].transpose(0, 2, 1).reshape(rotation_matrices.shape[0], 6)


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d representation must end with shape (6,). Got {rot6d.shape}.")

    first_column = rot6d[..., :3]
    second_column = rot6d[..., 3:6]

    first_norm = np.linalg.norm(first_column, axis=-1, keepdims=True)
    if np.any(first_norm <= 0):
        raise ValueError("rot6d first column must have non-zero norm.")
    basis_1 = first_column / first_norm

    second_column = second_column - np.sum(basis_1 * second_column, axis=-1, keepdims=True) * basis_1
    second_norm = np.linalg.norm(second_column, axis=-1, keepdims=True)
    if np.any(second_norm <= 0):
        raise ValueError("rot6d second column must be linearly independent from the first column.")
    basis_2 = second_column / second_norm
    basis_3 = np.cross(basis_1, basis_2)

    return np.stack((basis_1, basis_2, basis_3), axis=-1)


def build_relative_action_targets(
    position_sequence: np.ndarray,
    orientation_sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_sequence = np.asarray(position_sequence, dtype=np.float64)
    orientation_sequence = np.asarray(orientation_sequence, dtype=np.float64)

    if position_sequence.ndim != 2 or position_sequence.shape[1] != 3:
        raise ValueError(
            f"Action position sequence must have shape (T, 3). Got {position_sequence.shape}."
        )
    if orientation_sequence.ndim != 3 or orientation_sequence.shape[1:] != (3, 3):
        raise ValueError(
            f"Action orientation sequence must have shape (T, 3, 3). Got {orientation_sequence.shape}."
        )

    reference_position = position_sequence[0]
    reference_orientation = orientation_sequence[0]
    relative_positions = (reference_orientation.T @ (position_sequence - reference_position).T).T
    relative_orientations = np.einsum("ij,tjk->tik", reference_orientation.T, orientation_sequence)
    relative_rot6d = rotation_matrix_to_rot6d(relative_orientations)
    return relative_positions, relative_rot6d


class ForceWMParquetDataset(_DatasetBase):
    def __init__(
        self,
        parquet_path,
        contract_path=None,
        image_transform: Callable[[Any], Any] | None = None,
        max_open_video_chunks: int = 8,
        enforce_single_camera: bool = True,
    ):
        self.parquet_path = Path(parquet_path).expanduser().resolve()
        self.rows, self.parquet_metadata = load_parquet_rows(self.parquet_path)
        self.contract = load_contract(self.parquet_path, contract_path)
        self.lowdim_specs = parse_lowdim_specs(self.contract)
        self.visual_specs = parse_visual_specs(self.contract)
        self.action_window, self.action_dss, self.action_specs = parse_action_specs(
            self.contract,
            self.lowdim_specs,
        )
        self.image_transform = image_transform
        self.max_open_video_chunks = int(max_open_video_chunks)
        if self.max_open_video_chunks <= 0:
            raise ValueError("max_open_video_chunks must be positive.")

        if enforce_single_camera and len(self.visual_specs) != 1:
            raise ValueError(
                "The current parquet format is one row per camera frame. "
                "This first-pass dataloader supports exactly one visual key."
            )

        row_camera_names = sorted({str(row["camera_name"]) for row in self.rows})
        if enforce_single_camera and len(row_camera_names) != 1:
            raise ValueError(
                "The current parquet format is one row per camera frame. "
                f"Found multiple camera names in the parquet rows: {row_camera_names}"
            )

        self.visual_key_name = next(iter(self.visual_specs))
        if row_camera_names and row_camera_names[0] != self.visual_key_name:
            raise ValueError(
                f"Contract visual key `{self.visual_key_name}` does not match parquet camera name `{row_camera_names[0]}`."
            )

        self.episode_ranges = build_episode_ranges(self.rows)
        self.episode_frame_lookup = build_episode_frame_lookup(self.rows)
        self.parquet_dir = self.parquet_path.parent
        self._video_cache: OrderedDict[str, Any] = OrderedDict()

        self._validate_action_sources()

    def _validate_action_sources(self) -> None:
        first_row_lowdim = self.rows[0].get("lowdimData")
        if not isinstance(first_row_lowdim, dict):
            raise ValueError("Parquet rows are missing `lowdimData` mappings.")

        position_source = self.action_specs["robot_pos"].source_key
        orientation_source = self.action_specs["robot_ori"].source_key
        if position_source not in first_row_lowdim:
            raise ValueError(f"Action position source `{position_source}` is missing from parquet lowdimData.")
        if orientation_source not in first_row_lowdim:
            raise ValueError(f"Action orientation source `{orientation_source}` is missing from parquet lowdimData.")

        position_sample = np.asarray(first_row_lowdim[position_source], dtype=np.float64)
        orientation_sample = np.asarray(first_row_lowdim[orientation_source], dtype=np.float64)
        if position_sample.shape != (3,):
            raise ValueError(
                f"Action position source `{position_source}` must have shape (3,), got {position_sample.shape}."
            )
        if orientation_sample.shape != (3, 3):
            raise ValueError(
                f"Action orientation source `{orientation_source}` must have shape (3, 3), got {orientation_sample.shape}."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _row_for_episode_frame(self, episode_id: int, episode_frame_index: int) -> dict[str, Any]:
        row_position = self.episode_frame_lookup.get((episode_id, episode_frame_index))
        if row_position is None:
            raise KeyError(f"No row found for episode {episode_id}, episode frame {episode_frame_index}.")
        return self.rows[row_position]

    def decode_frame_from_chunk(self, chunk_relative_path: str, frame_index_in_chunk: int) -> np.ndarray:
        cv2 = _require_cv2()

        chunk_path = (self.parquet_dir / chunk_relative_path).resolve()
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk video does not exist: {chunk_path}")

        chunk_key = str(chunk_path)
        capture = self._video_cache.pop(chunk_key, None)
        if capture is None:
            capture = cv2.VideoCapture(chunk_key)
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video chunk `{chunk_path}`.")
        self._video_cache[chunk_key] = capture
        while len(self._video_cache) > self.max_open_video_chunks:
            _, stale_capture = self._video_cache.popitem(last=False)
            stale_capture.release()

        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index_in_chunk))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to decode frame {frame_index_in_chunk} from chunk `{chunk_path}`."
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def fetch_visual_sequence(
        self,
        episode_id: int,
        frame_indices: list[int],
        camera_name: str,
    ) -> np.ndarray:
        frames: list[np.ndarray] = []
        for episode_frame_index in frame_indices:
            row = self._row_for_episode_frame(episode_id, episode_frame_index)
            if str(row["camera_name"]) != camera_name:
                raise ValueError(
                    f"Expected camera `{camera_name}` while decoding frame, got `{row['camera_name']}`."
                )
            try:
                frames.append(
                    self.decode_frame_from_chunk(
                        str(row["video_chunk_path"]),
                        int(row["frame_index_in_chunk"]),
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load visual frame for episode `{row['episode_name']}`, camera `{camera_name}`, "
                    f"chunk `{row['video_chunk_path']}`, frame {row['frame_index_in_chunk']}."
                ) from exc

        return np.stack(frames, axis=0)

    def close(self) -> None:
        for capture in self._video_cache.values():
            capture.release()
        self._video_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        torch_module = _require_torch()

        center_index = int(index)
        if center_index < 0 or center_index >= len(self.rows):
            raise IndexError(f"Sample index {center_index} is out of range for dataset of size {len(self.rows)}.")

        row = self.rows[center_index]
        episode_id = int(row["episode_id"])
        episode_name = str(row["episode_name"])
        camera_name = str(row["camera_name"])
        center_episode_frame = int(row["episode_frame_index"])
        episode_range = self.episode_ranges[episode_id]

        observation: dict[str, Any] = {}
        for key_name, key_spec in self.lowdim_specs.items():
            frame_indices = compute_observation_indices(
                center_episode_frame,
                key_spec.obs_window,
                key_spec.obs_dss,
                episode_range.last_episode_frame,
                episode_range.first_episode_frame,
            )
            sequence = fetch_lowdim_sequence(
                self.rows,
                self.episode_frame_lookup,
                episode_id,
                frame_indices,
                key_name,
            )
            if key_name == self.action_specs["robot_ori"].source_key:
                sequence = rotation_matrix_to_rot6d(sequence)
            observation[key_name] = torch_module.as_tensor(sequence, dtype=torch_module.float32)

        visual_spec = self.visual_specs[self.visual_key_name]
        visual_frame_indices = compute_observation_indices(
            center_episode_frame,
            visual_spec.obs_window,
            visual_spec.obs_dss,
            episode_range.last_episode_frame,
            episode_range.first_episode_frame,
        )
        visual_sequence = self.fetch_visual_sequence(
            episode_id,
            visual_frame_indices,
            camera_name,
        )
        visual_tensor = torch_module.from_numpy(np.ascontiguousarray(visual_sequence)).permute(0, 3, 1, 2).contiguous()
        if self.image_transform is not None:
            visual_tensor = self.image_transform(visual_tensor)
        observation[self.visual_key_name] = visual_tensor

        action_frame_indices = compute_action_indices(
            center_episode_frame,
            self.action_window,
            self.action_dss,
            episode_range.last_episode_frame,
            episode_range.first_episode_frame,
        )
        position_sequence = fetch_lowdim_sequence(
            self.rows,
            self.episode_frame_lookup,
            episode_id,
            action_frame_indices,
            self.action_specs["robot_pos"].source_key,
        )
        orientation_sequence = fetch_lowdim_sequence(
            self.rows,
            self.episode_frame_lookup,
            episode_id,
            action_frame_indices,
            self.action_specs["robot_ori"].source_key,
        )
        relative_positions, relative_rot6d = build_relative_action_targets(
            position_sequence,
            orientation_sequence,
        )

        return {
            "global_index": center_index,
            "episode_id": episode_id,
            "episode_name": episode_name,
            "camera_name": camera_name,
            "observation": observation,
            "action": {
                "robot_pos": torch_module.as_tensor(relative_positions, dtype=torch_module.float32),
                "robot_ori": torch_module.as_tensor(relative_rot6d, dtype=torch_module.float32),
            },
            "frame_ref": {
                "video_chunk_path": str(row["video_chunk_path"]),
                "video_chunk_index": int(row["video_chunk_index"]),
                "frame_index_in_chunk": int(row["frame_index_in_chunk"]),
                "episode_frame_index": center_episode_frame,
            },
            "timestamps": {
                "camera_timestamp_s": float(row["camera_timestamp_s"]),
            },
        }


def sample_random_indices(dataset_or_length: ForceWMParquetDataset | int, num_samples: int, seed: int | None = None) -> list[int]:
    if isinstance(dataset_or_length, int):
        dataset_length = dataset_or_length
    else:
        dataset_length = len(dataset_or_length)

    if dataset_length <= 0:
        raise ValueError("Cannot sample from an empty dataset.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    sample_count = min(int(num_samples), dataset_length)
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_length, size=sample_count, replace=False).tolist()


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return tuple(int(dim) for dim in shape)
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def print_observation_shapes(sample: dict[str, Any]) -> None:
    print(
        f"Loaded sample global_index={sample['global_index']} episode={sample['episode_name']} "
        f"episode_frame={sample['frame_ref']['episode_frame_index']}"
    )
    for key_name, value in sample["observation"].items():
        shape = _shape_tuple(value)
        dtype = getattr(value, "dtype", None)
        if shape is None:
            print(f"  observation[{key_name}]: shape unavailable, dtype={dtype}")
        else:
            print(f"  observation[{key_name}]: shape={shape}, dtype={dtype}")


def _compute_triad_scale(*point_sets: np.ndarray) -> float:
    valid_sets = [np.asarray(points, dtype=np.float64) for points in point_sets if np.asarray(points).size > 0]
    if not valid_sets:
        return 0.01

    all_points = np.concatenate(valid_sets, axis=0)
    extents = np.ptp(all_points, axis=0)
    max_extent = float(np.max(extents)) if extents.size else 0.0
    return max(max_extent * 0.15, 0.01)


def _set_axes_equal_3d(axis, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)

    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)


def _plot_orientation_triads(
    axis,
    positions: np.ndarray,
    rotation_matrices: np.ndarray,
    scale: float,
    alpha: float = 0.75,
) -> None:
    positions = np.asarray(positions, dtype=np.float64)
    rotation_matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Triad positions must have shape (T, 3). Got {positions.shape}.")
    if rotation_matrices.ndim != 3 or rotation_matrices.shape[1:] != (3, 3):
        raise ValueError(
            f"Triad orientations must have shape (T, 3, 3). Got {rotation_matrices.shape}."
        )
    if positions.shape[0] != rotation_matrices.shape[0]:
        raise ValueError(
            "Triad positions and orientation counts must match. "
            f"Got {positions.shape[0]} and {rotation_matrices.shape[0]}."
        )

    axis_colors = ("red", "green", "cyan")
    for position, rotation_matrix in zip(positions, rotation_matrices, strict=False):
        for axis_index, axis_color in enumerate(axis_colors):
            direction = rotation_matrix[:, axis_index] * scale
            axis.quiver(
                position[0],
                position[1],
                position[2],
                direction[0],
                direction[1],
                direction[2],
                color=axis_color,
                alpha=alpha,
                arrow_length_ratio=0.2,
                linewidth=1.0,
            )


def visualize_random_samples(
    parquet_path,
    contract_path=None,
    num_samples: int = 4,
    seed: int | None = None,
    print_sample_shapes: bool = True,
) -> None:
    plt = _require_matplotlib()

    dataset = ForceWMParquetDataset(parquet_path, contract_path=contract_path)
    sample_indices = sample_random_indices(dataset, num_samples, seed)
    if not sample_indices:
        raise ValueError("No sample indices were drawn for visualization.")

    num_plots = len(sample_indices)
    num_cols = min(2, num_plots)
    num_rows = int(math.ceil(num_plots / num_cols))
    figure = plt.figure(figsize=(7 * num_cols, 6 * num_rows))

    try:
        for plot_idx, sample_index in enumerate(sample_indices, start=1):
            sample = dataset[sample_index]
            if print_sample_shapes:
                print_observation_shapes(sample)
            observation_pos = sample["observation"].get("eef_pos")
            observation_ori = sample["observation"].get("eef_ori")
            if observation_pos is None or observation_ori is None:
                raise KeyError(
                    "Visualization requires `eef_pos` and `eef_ori` to be available in the observation dict."
                )

            observation_pos_np = observation_pos.detach().cpu().numpy()
            observation_ori_np = observation_ori.detach().cpu().numpy()
            action_pos_rel_np = sample["action"]["robot_pos"].detach().cpu().numpy()
            action_ori_rel_np = sample["action"]["robot_ori"].detach().cpu().numpy()

            reference_position = observation_pos_np[-1]
            observation_world_orientations = rot6d_to_rotation_matrix(observation_ori_np)
            reference_orientation = observation_world_orientations[-1]
            world_action_positions = reference_position[None, :] + (
                reference_orientation @ action_pos_rel_np.T
            ).T
            action_relative_orientations = rot6d_to_rotation_matrix(action_ori_rel_np)
            world_action_orientations = np.einsum(
                "ij,tjk->tik",
                reference_orientation,
                action_relative_orientations,
            )
            triad_scale = _compute_triad_scale(observation_pos_np, world_action_positions)

            axis = figure.add_subplot(num_rows, num_cols, plot_idx, projection="3d")
            axis.plot(
                observation_pos_np[:, 0],
                observation_pos_np[:, 1],
                observation_pos_np[:, 2],
                color="blue",
                marker="o",
                label="Observation eef_pos",
            )
            axis.plot(
                world_action_positions[:, 0],
                world_action_positions[:, 1],
                world_action_positions[:, 2],
                color="yellow",
                marker="o",
                label="Action robot_pos",
            )
            _plot_orientation_triads(
                axis,
                observation_pos_np,
                observation_world_orientations,
                scale=triad_scale,
                alpha=0.45,
            )
            _plot_orientation_triads(
                axis,
                world_action_positions,
                world_action_orientations,
                scale=triad_scale,
                alpha=0.85,
            )
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
            _set_axes_equal_3d(
                axis,
                np.concatenate((observation_pos_np, world_action_positions), axis=0),
            )
            axis.set_title(
                f"{sample['episode_name']} | global {sample['global_index']} | ep frame {sample['frame_ref']['episode_frame_index']}"
            )
            axis.legend(loc="best")

        figure.tight_layout()
        plt.show()
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parquet-backed ForceWM dataloader utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Sample random dataset indices and visualize observation/action trajectories.",
    )
    visualize_parser.add_argument("parquet_path", help="Path to the extracted parquet dataset.")
    visualize_parser.add_argument(
        "--contract-path",
        default=None,
        help="Optional path to the contract YAML. Defaults to a sibling or repo-level contract.",
    )
    visualize_parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of random samples to visualize.",
    )
    visualize_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    visualize_parser.add_argument(
        "--no-print-observation-shapes",
        action="store_true",
        help="Disable printing observation tensor shapes when samples are loaded.",
    )

    args = parser.parse_args()

    if args.command == "visualize":
        visualize_random_samples(
            parquet_path=args.parquet_path,
            contract_path=args.contract_path,
            num_samples=args.num_samples,
            seed=args.seed,
            print_sample_shapes=not args.no_print_observation_shapes,
        )


if __name__ == "__main__":
    main()
