from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from extractor.point_finder import (
    ContactCylinderSpec,
    SimplePointConfig,
    generate_points_simple,
    load_contact_cylinder_spec,
    load_point_config,
)


DEFAULT_CHUNK_SIZE = 128
DEFAULT_VEL_THRESH = -1
DEFAULT_STATIONARY_WINDOW = 1
DEFAULT_PARQUET_NAME = "dataset.parquet"
DEFAULT_FEMALE_PART_BODY_NAME = "task"


@dataclass(frozen=True)
class ActionLabelConfig:
    current_position_key: str
    current_orientation_key: str
    desired_position_key: str
    desired_orientation_key: str
    desired_force_magnitude_key: str
    frame: str
    orientation_encoding: str
    target_resample: str


@dataclass(frozen=True)
class EpisodeData:
    source_dir: Path
    source_name: str
    timestamps: np.ndarray
    current_positions: np.ndarray
    current_orientations: np.ndarray
    desired_positions: np.ndarray
    desired_orientations: np.ndarray
    desired_force_magnitudes: np.ndarray
    passthrough_lowdim: dict[str, np.ndarray]


@dataclass(frozen=True)
class ProcessedEpisode:
    source_dir: Path
    source_name: str
    timestamps: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray
    action_delta_positions: np.ndarray
    action_delta_rotvecs: np.ndarray
    action_force_magnitudes: np.ndarray
    passthrough_lowdim: dict[str, np.ndarray]


@dataclass(frozen=True)
class EpisodeProcessingResult:
    processed_episode: ProcessedEpisode
    point_clouds: np.ndarray


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyArrow is required for parquet export. Install `pyarrow` in the active environment."
        ) from exc
    return pa, pq


def _require_scipy_rotation():
    try:
        from scipy.spatial.transform import Rotation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciPy is required for orientation processing. Install `scipy` in the active environment."
        ) from exc
    return Rotation


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def load_universal_contract(contract_path: Path) -> dict[str, Any]:
    contract_path = Path(contract_path)
    if not contract_path.exists():
        raise FileNotFoundError(f"Universal contract does not exist: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise ValueError(f"Universal contract at {contract_path} must contain a top-level mapping.")
    return contract


def parse_action_label_config(contract: dict[str, Any]) -> ActionLabelConfig:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        raise KeyError("Universal contract is missing `robot`.")

    action_cfg = robot_cfg.get("action_labels")
    if not isinstance(action_cfg, dict):
        raise KeyError("Universal contract is missing `robot.action_labels`.")

    def require_string(field_name: str) -> str:
        value = action_cfg.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`robot.action_labels.{field_name}` must be a non-empty string.")
        return value.strip()

    config = ActionLabelConfig(
        current_position_key=require_string("current_position_key"),
        current_orientation_key=require_string("current_orientation_key"),
        desired_position_key=require_string("desired_position_key"),
        desired_orientation_key=require_string("desired_orientation_key"),
        desired_force_magnitude_key=require_string("desired_force_magnitude_key"),
        frame=require_string("frame"),
        orientation_encoding=require_string("orientation_encoding"),
        target_resample=require_string("target_resample"),
    )

    if config.frame != "female_part":
        raise ValueError("`robot.action_labels.frame` must currently be `female_part`.")
    if config.orientation_encoding != "rotvec":
        raise ValueError("`robot.action_labels.orientation_encoding` must currently be `rotvec`.")
    return config


def resolve_scene_xml_path(contract: dict[str, Any], contract_path: Path) -> Path | None:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        return None

    raw_xml_path = robot_cfg.get("xml_path")
    if not isinstance(raw_xml_path, str) or not raw_xml_path.strip():
        return None

    candidate_path = Path(raw_xml_path).expanduser()
    if candidate_path.is_absolute():
        return candidate_path.resolve()
    return (contract_path.parent / candidate_path).resolve()


def load_female_part_position_world(
    scene_xml_path: Path | None = None,
    body_name: str = DEFAULT_FEMALE_PART_BODY_NAME,
) -> np.ndarray:
    if scene_xml_path is None:
        scene_xml_path = Path(__file__).resolve().parents[1] / "models" / "parametric_scene.xml"

    scene_xml_path = Path(scene_xml_path)
    if not scene_xml_path.exists():
        raise FileNotFoundError(f"Scene XML does not exist: {scene_xml_path}")

    scene_root = ET.parse(scene_xml_path).getroot()
    search_paths = [scene_xml_path]
    for include_element in scene_root.iter("include"):
        include_path = include_element.attrib.get("file")
        if include_path is None:
            continue
        search_paths.append((scene_xml_path.parent / include_path).resolve())

    for search_path in search_paths:
        if not search_path.exists():
            continue
        xml_root = ET.parse(search_path).getroot()
        for body_element in xml_root.iter("body"):
            if body_element.attrib.get("name") != body_name:
                continue
            raw_position = body_element.attrib.get("pos", "0 0 0")
            position = np.fromstring(raw_position, sep=" ", dtype=np.float32)
            if position.shape != (3,):
                raise ValueError(
                    f"Body `{body_name}` in {search_path} must define a 3D `pos`, got `{raw_position}`."
                )
            return position

    raise ValueError(f"Could not find body `{body_name}` in {scene_xml_path} or its included XML files.")


def discover_episode_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    episode_dirs = sorted(
        path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith("episode_")
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories were found under {input_dir}")
    return episode_dirs


def resolve_output_dir(input_dir: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return input_dir.parent / f"{input_dir.name}_extracted"


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Remove it before extracting again."
        )
    output_dir.mkdir(parents=True, exist_ok=False)


def _require_lowdim_timestamp_key(lowdim_archive: Any, lowdim_path: Path) -> str:
    timestamp_key = "timestamp_s" if "timestamp_s" in lowdim_archive else "ts" if "ts" in lowdim_archive else None
    if timestamp_key is None:
        raise KeyError(f"Expected `timestamp_s` or `ts` in {lowdim_path}")
    return timestamp_key


def _validate_lowdim_timestamps(lowdim_path: Path, timestamps: Any) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        raise ValueError(f"Lowdim timestamps are empty in {lowdim_path}")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"Lowdim timestamps must be finite in {lowdim_path}")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"Lowdim timestamps must be strictly increasing in {lowdim_path}")
    return timestamps


def _validate_lowdim_position_array(lowdim_path: Path, key_name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"`{key_name}` must have shape (T, 3) in {lowdim_path}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{key_name}` must contain only finite values in {lowdim_path}")
    return array


def _validate_lowdim_orientation_array(lowdim_path: Path, key_name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"`{key_name}` must have shape (T, 3, 3) in {lowdim_path}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{key_name}` must contain only finite values in {lowdim_path}")
    return array


def _validate_lowdim_force_array(lowdim_path: Path, key_name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise ValueError(f"`{key_name}` must have shape (T,) or (T, 1) in {lowdim_path}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{key_name}` must contain only finite values in {lowdim_path}")
    if np.any(array < 0.0):
        raise ValueError(f"`{key_name}` must be non-negative in {lowdim_path}")
    return array


def _validate_passthrough_lowdim_array(
    lowdim_path: Path,
    key_name: str,
    value: Any,
    expected_length: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"`{key_name}` must have a leading time dimension in {lowdim_path}")
    if len(array) != expected_length:
        raise ValueError(
            f"`{key_name}` has length {len(array)} but timestamps have length {expected_length} in {lowdim_path}."
        )

    if array.dtype.kind == "f":
        array = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"`{key_name}` must contain only finite values in {lowdim_path}")
        return array
    if array.dtype.kind in {"i", "u"}:
        return np.asarray(array, dtype=np.int64)
    if array.dtype.kind == "b":
        return np.asarray(array, dtype=bool)

    raise ValueError(
        f"`{key_name}` in {lowdim_path} must be numeric or boolean to be exported, got dtype `{array.dtype}`."
    )


def load_lowdim_episode(episode_dir: Path, action_label_config: ActionLabelConfig) -> EpisodeData:
    lowdim_path = episode_dir / "lowdim.npz"
    if not lowdim_path.exists():
        raise FileNotFoundError(f"Missing lowdim file: {lowdim_path}")

    with np.load(lowdim_path) as lowdim_archive:
        timestamp_key = _require_lowdim_timestamp_key(lowdim_archive, lowdim_path)
        required_keys = [
            action_label_config.current_position_key,
            action_label_config.current_orientation_key,
            action_label_config.desired_position_key,
            action_label_config.desired_orientation_key,
            action_label_config.desired_force_magnitude_key,
        ]
        missing_keys = [key_name for key_name in required_keys if key_name not in lowdim_archive]
        if missing_keys:
            raise KeyError(
                f"Missing required lowdim key(s) {missing_keys} in {lowdim_path}. "
                "Update the recorded dataset so it includes the action-label fields."
            )

        timestamps = _validate_lowdim_timestamps(lowdim_path, lowdim_archive[timestamp_key])
        current_positions = _validate_lowdim_position_array(
            lowdim_path,
            action_label_config.current_position_key,
            lowdim_archive[action_label_config.current_position_key],
        )
        current_orientations = _validate_lowdim_orientation_array(
            lowdim_path,
            action_label_config.current_orientation_key,
            lowdim_archive[action_label_config.current_orientation_key],
        )
        desired_positions = _validate_lowdim_position_array(
            lowdim_path,
            action_label_config.desired_position_key,
            lowdim_archive[action_label_config.desired_position_key],
        )
        desired_orientations = _validate_lowdim_orientation_array(
            lowdim_path,
            action_label_config.desired_orientation_key,
            lowdim_archive[action_label_config.desired_orientation_key],
        )
        desired_force_magnitudes = _validate_lowdim_force_array(
            lowdim_path,
            action_label_config.desired_force_magnitude_key,
            lowdim_archive[action_label_config.desired_force_magnitude_key],
        )

        passthrough_lowdim: dict[str, np.ndarray] = {}
        excluded_keys = {
            timestamp_key,
            action_label_config.current_position_key,
            action_label_config.current_orientation_key,
            action_label_config.desired_position_key,
            action_label_config.desired_orientation_key,
            action_label_config.desired_force_magnitude_key,
        }
        for key_name in lowdim_archive.files:
            if key_name in excluded_keys:
                continue
            passthrough_lowdim[key_name] = _validate_passthrough_lowdim_array(
                lowdim_path,
                key_name,
                lowdim_archive[key_name],
                expected_length=len(timestamps),
            )

    expected_length = len(timestamps)
    for key_name, array in [
        (action_label_config.current_position_key, current_positions),
        (action_label_config.current_orientation_key, current_orientations),
        (action_label_config.desired_position_key, desired_positions),
        (action_label_config.desired_orientation_key, desired_orientations),
        (action_label_config.desired_force_magnitude_key, desired_force_magnitudes),
    ]:
        if len(array) != expected_length:
            raise ValueError(
                f"`{key_name}` has length {len(array)} but timestamps have length {expected_length} in {lowdim_path}."
            )

    return EpisodeData(
        source_dir=episode_dir,
        source_name=episode_dir.name,
        timestamps=timestamps,
        current_positions=current_positions,
        current_orientations=current_orientations,
        desired_positions=desired_positions,
        desired_orientations=desired_orientations,
        desired_force_magnitudes=desired_force_magnitudes,
        passthrough_lowdim=passthrough_lowdim,
    )


def apply_edge_trim(length: int, trim_start: int, trim_end: int) -> np.ndarray:
    keep_mask = np.ones(length, dtype=bool)
    trim_start = max(0, int(trim_start))
    trim_end = max(0, int(trim_end))

    if trim_start:
        keep_mask[: min(trim_start, length)] = False
    if trim_end:
        keep_mask[max(length - trim_end, 0) :] = False

    return keep_mask


def build_stationary_mask(
    positions: np.ndarray,
    dt: float,
    vel_thresh: float,
    stationary_window: int,
) -> np.ndarray:
    length = len(positions)
    stationary_mask = np.zeros(length, dtype=bool)
    if length <= 1:
        return stationary_mask

    if dt <= 0.0:
        dt = 1.0

    velocity = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)
    window = max(1, int(stationary_window))

    for frame_index in range(window, length):
        trailing_speed = speed[frame_index - window : frame_index]
        if len(trailing_speed) == window and np.all(trailing_speed < vel_thresh):
            stationary_mask[frame_index] = True

    return stationary_mask


def build_prune_keep_mask(
    timestamps: np.ndarray,
    positions: np.ndarray,
    trim_start: int,
    trim_end: int,
    vel_thresh: float,
    stationary_window: int,
) -> np.ndarray:
    trimmed_keep_mask = apply_edge_trim(len(timestamps), trim_start=trim_start, trim_end=trim_end)
    if not np.any(trimmed_keep_mask):
        return trimmed_keep_mask

    trimmed_timestamps = timestamps[trimmed_keep_mask]
    trimmed_positions = positions[trimmed_keep_mask]
    if len(trimmed_timestamps) <= 1:
        return trimmed_keep_mask

    dt_values = np.diff(trimmed_timestamps)
    positive_dt_values = dt_values[dt_values > 0.0]
    dt = float(np.median(positive_dt_values)) if len(positive_dt_values) else 1.0

    stationary_mask = build_stationary_mask(
        trimmed_positions,
        dt=dt,
        vel_thresh=vel_thresh,
        stationary_window=stationary_window,
    )
    keep_after_stationary = ~stationary_mask

    keep_mask = np.zeros(len(timestamps), dtype=bool)
    trimmed_indices = np.flatnonzero(trimmed_keep_mask)
    keep_mask[trimmed_indices[keep_after_stationary]] = True
    return keep_mask


def express_positions_in_female_part_frame(
    positions_world: np.ndarray,
    female_part_position_world: np.ndarray,
) -> np.ndarray:
    positions_world = np.asarray(positions_world, dtype=np.float32)
    female_part_position_world = np.asarray(female_part_position_world, dtype=np.float32).reshape(3)
    if positions_world.ndim != 2 or positions_world.shape[1] != 3:
        raise ValueError(f"`positions_world` must have shape (T, 3), got {positions_world.shape}.")
    return positions_world - female_part_position_world[None, :]


def compute_action_labels(
    current_positions_world: np.ndarray,
    current_orientations_world: np.ndarray,
    desired_positions_world: np.ndarray,
    desired_orientations_world: np.ndarray,
    desired_force_magnitudes: np.ndarray,
    female_part_position_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_positions_part = express_positions_in_female_part_frame(
        current_positions_world,
        female_part_position_world,
    )
    desired_positions_part = express_positions_in_female_part_frame(
        desired_positions_world,
        female_part_position_world,
    )
    action_delta_positions = (desired_positions_part - current_positions_part).astype(np.float32)

    Rotation = _require_scipy_rotation()
    current_rotations = Rotation.from_matrix(np.asarray(current_orientations_world, dtype=np.float64))
    desired_rotations = Rotation.from_matrix(np.asarray(desired_orientations_world, dtype=np.float64))
    action_delta_rotvecs = (desired_rotations * current_rotations.inv()).as_rotvec().astype(np.float32)
    action_force_magnitudes = np.asarray(desired_force_magnitudes, dtype=np.float32).reshape(-1)

    if len(action_delta_positions) != len(action_delta_rotvecs) or len(action_delta_positions) != len(action_force_magnitudes):
        raise ValueError("Computed action labels must have matching lengths.")

    return action_delta_positions, action_delta_rotvecs, action_force_magnitudes


def express_point_clouds_in_female_part_frame(
    point_clouds_world: np.ndarray,
    eef_positions_world: np.ndarray,
    female_part_position_world: np.ndarray,
) -> np.ndarray:
    point_clouds_world = np.asarray(point_clouds_world, dtype=np.float32)
    eef_positions_world = np.asarray(eef_positions_world, dtype=np.float32)
    female_part_position_world = np.asarray(female_part_position_world, dtype=np.float32).reshape(3)

    if point_clouds_world.ndim != 3 or point_clouds_world.shape[-1] != 3:
        raise ValueError(f"`point_clouds_world` must have shape (T, N, 3), got {point_clouds_world.shape}.")
    if eef_positions_world.ndim != 2 or eef_positions_world.shape[1] != 3:
        raise ValueError(f"`eef_positions_world` must have shape (T, 3), got {eef_positions_world.shape}.")
    if len(point_clouds_world) != len(eef_positions_world):
        raise ValueError("Point-cloud frames and end-effector positions must have matching lengths.")

    point_clouds_part_frame = point_clouds_world - female_part_position_world[None, None, :]
    eef_positions_part_frame = eef_positions_world - female_part_position_world[None, :]
    return np.concatenate(
        [eef_positions_part_frame[:, None, :], point_clouds_part_frame],
        axis=1,
    ).astype(np.float32)


def create_synthetic_point_clouds(
    positions_world: np.ndarray,
    orientations_world: np.ndarray,
    female_part_position_world: np.ndarray,
    contact_spec: ContactCylinderSpec,
    point_config: SimplePointConfig,
) -> np.ndarray:
    point_clouds_world = generate_points_simple(
        positions_world=positions_world,
        orientations_world=orientations_world,
        contact_spec=contact_spec,
        point_config=point_config,
    )
    return express_point_clouds_in_female_part_frame(
        point_clouds_world=point_clouds_world,
        eef_positions_world=positions_world,
        female_part_position_world=female_part_position_world,
    )


def process_episode(
    episode_dir: Path,
    trim_start: int,
    trim_end: int,
    vel_thresh: float,
    stationary_window: int,
    action_label_config: ActionLabelConfig,
    female_part_position_world: np.ndarray,
    contact_spec: ContactCylinderSpec,
    point_config: SimplePointConfig,
) -> EpisodeProcessingResult | None:
    episode = load_lowdim_episode(episode_dir, action_label_config)

    prune_keep_mask = build_prune_keep_mask(
        timestamps=episode.timestamps,
        positions=episode.current_positions,
        trim_start=trim_start,
        trim_end=trim_end,
        vel_thresh=vel_thresh,
        stationary_window=stationary_window,
    )
    if not np.any(prune_keep_mask):
        _warn(f"{episode.source_name}: pruning removed every frame; skipping episode.")
        return None

    pruned_timestamps = episode.timestamps[prune_keep_mask].astype(np.float64)
    pruned_positions = episode.current_positions[prune_keep_mask].astype(np.float32)
    pruned_orientations = episode.current_orientations[prune_keep_mask].astype(np.float32)
    pruned_desired_positions = episode.desired_positions[prune_keep_mask].astype(np.float32)
    pruned_desired_orientations = episode.desired_orientations[prune_keep_mask].astype(np.float32)
    pruned_desired_force_magnitudes = episode.desired_force_magnitudes[prune_keep_mask].astype(np.float32)
    pruned_passthrough_lowdim = {
        key_name: values[prune_keep_mask]
        for key_name, values in episode.passthrough_lowdim.items()
    }

    action_delta_positions, action_delta_rotvecs, action_force_magnitudes = compute_action_labels(
        current_positions_world=pruned_positions,
        current_orientations_world=pruned_orientations,
        desired_positions_world=pruned_desired_positions,
        desired_orientations_world=pruned_desired_orientations,
        desired_force_magnitudes=pruned_desired_force_magnitudes,
        female_part_position_world=female_part_position_world,
    )
    point_clouds = create_synthetic_point_clouds(
        positions_world=pruned_positions,
        orientations_world=pruned_orientations,
        female_part_position_world=female_part_position_world,
        contact_spec=contact_spec,
        point_config=point_config,
    )

    return EpisodeProcessingResult(
        processed_episode=ProcessedEpisode(
            source_dir=episode.source_dir,
            source_name=episode.source_name,
            timestamps=pruned_timestamps,
            positions=pruned_positions,
            orientations=pruned_orientations,
            action_delta_positions=action_delta_positions,
            action_delta_rotvecs=action_delta_rotvecs,
            action_force_magnitudes=action_force_magnitudes,
            passthrough_lowdim=pruned_passthrough_lowdim,
        ),
        point_clouds=point_clouds,
    )


def _numpy_dtype_to_pyarrow_type(pa: Any, array: np.ndarray) -> Any:
    if array.dtype.kind == "f":
        return pa.float32()
    if array.dtype.kind == "i":
        return pa.int64()
    if array.dtype.kind == "u":
        return pa.uint64()
    if array.dtype.kind == "b":
        return pa.bool_()
    raise ValueError(f"Unsupported dtype for parquet export: {array.dtype}")


def _numpy_array_to_parquet_column(pa: Any, key_name: str, array: np.ndarray) -> Any:
    array = np.asarray(array)
    if array.ndim == 0:
        raise ValueError(f"Parquet column `{key_name}` must include a row dimension.")

    element_type = _numpy_dtype_to_pyarrow_type(pa, array)
    if array.ndim == 1:
        return pa.array(array, type=element_type)

    list_size = int(np.prod(array.shape[1:], dtype=np.int64))
    flattened = np.ascontiguousarray(array.reshape(len(array), list_size))
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flattened.reshape(-1), type=element_type),
        list_size,
    )


def write_parquet(output_dir: Path, episodes: list[ProcessedEpisode]) -> int:
    pa, pq = _require_pyarrow()

    all_positions = np.concatenate([episode.positions for episode in episodes], axis=0)
    all_orientations = np.concatenate([episode.orientations for episode in episodes], axis=0).reshape(-1, 9)
    all_action_delta_positions = np.concatenate([episode.action_delta_positions for episode in episodes], axis=0)
    all_action_delta_rotvecs = np.concatenate([episode.action_delta_rotvecs for episode in episodes], axis=0)
    all_action_force_magnitudes = np.concatenate([episode.action_force_magnitudes for episode in episodes], axis=0)

    passthrough_keys = list(episodes[0].passthrough_lowdim.keys())
    passthrough_columns: dict[str, Any] = {}
    for episode in episodes[1:]:
        if list(episode.passthrough_lowdim.keys()) != passthrough_keys:
            raise ValueError("All processed episodes must have the same passthrough lowdim keys.")

    for key_name in passthrough_keys:
        try:
            concatenated = np.concatenate([episode.passthrough_lowdim[key_name] for episode in episodes], axis=0)
        except ValueError as exc:
            raise ValueError(f"Passthrough lowdim key `{key_name}` has inconsistent shapes across episodes.") from exc
        passthrough_columns[key_name] = _numpy_array_to_parquet_column(pa, key_name, concatenated)

    table_columns = {
        "eef_pos": pa.FixedSizeListArray.from_arrays(
            pa.array(all_positions.reshape(-1), type=pa.float32()),
            3,
        ),
        "eef_ori": pa.FixedSizeListArray.from_arrays(
            pa.array(all_orientations.reshape(-1), type=pa.float32()),
            9,
        ),
        "action_delta_pos": pa.FixedSizeListArray.from_arrays(
            pa.array(all_action_delta_positions.reshape(-1), type=pa.float32()),
            3,
        ),
        "action_delta_rotvec": pa.FixedSizeListArray.from_arrays(
            pa.array(all_action_delta_rotvecs.reshape(-1), type=pa.float32()),
            3,
        ),
        "action_force_magnitude": pa.array(all_action_force_magnitudes, type=pa.float32()),
    }
    table_columns.update(passthrough_columns)
    table = pa.table(table_columns)
    parquet_path = output_dir / DEFAULT_PARQUET_NAME
    pq.write_table(table, parquet_path)
    return int(len(all_positions))


def write_metadata(output_dir: Path, episodes: list[ProcessedEpisode], chunk_size: int) -> np.ndarray:
    running_total = 0
    episode_ends = []
    for episode in episodes:
        running_total += len(episode.positions)
        episode_ends.append(running_total - 1)

    episode_ends_array = np.asarray(episode_ends, dtype=np.int64)
    np.savez(
        output_dir / "metadata.npz",
        episode_ends=episode_ends_array,
        chunk_size=np.asarray(chunk_size, dtype=np.int64),
    )
    return episode_ends_array


def write_chunked_point_clouds(
    output_dir: Path,
    output_episode_index: int,
    point_clouds: np.ndarray,
    chunk_size: int,
) -> None:
    point_clouds_dir = output_dir / "point_clouds"
    point_clouds_dir.mkdir(parents=True, exist_ok=True)

    episode_dir = point_clouds_dir / f"episode_{output_episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=False)

    num_chunks = (len(point_clouds) + chunk_size - 1) // chunk_size
    for chunk_index in range(num_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(point_clouds))
        chunk_path = episode_dir / f"chunk_{chunk_index + 1:04d}.npy"
        np.save(chunk_path, point_clouds[chunk_start:chunk_end].astype(np.float32))


def extract_dataset(
    input_dir: Path,
    output_dir: Path,
    universal_contract_path: Path,
    point_config_path: Path | None,
    chunk_size: int,
    trim_start: int,
    trim_end: int,
    vel_thresh: float,
    stationary_window: int,
) -> None:
    contract = load_universal_contract(universal_contract_path)
    action_label_config = parse_action_label_config(contract)
    point_config = load_point_config(point_config_path)
    contact_spec = load_contact_cylinder_spec()
    scene_xml_path = resolve_scene_xml_path(contract, universal_contract_path)
    female_part_position_world = load_female_part_position_world(scene_xml_path=scene_xml_path)

    episode_dirs = discover_episode_dirs(input_dir)
    prepare_output_dir(output_dir)

    processed_episodes: list[ProcessedEpisode] = []
    point_cloud_episode_count = 0
    for episode_dir in episode_dirs:
        processing_result = process_episode(
            episode_dir=episode_dir,
            trim_start=trim_start,
            trim_end=trim_end,
            vel_thresh=vel_thresh,
            stationary_window=stationary_window,
            action_label_config=action_label_config,
            female_part_position_world=female_part_position_world,
            contact_spec=contact_spec,
            point_config=point_config,
        )
        if processing_result is None:
            continue

        output_episode_index = len(processed_episodes) + 1
        write_chunked_point_clouds(
            output_dir=output_dir,
            output_episode_index=output_episode_index,
            point_clouds=processing_result.point_clouds,
            chunk_size=chunk_size,
        )
        point_cloud_episode_count += 1
        processed_episodes.append(processing_result.processed_episode)

    if not processed_episodes:
        raise RuntimeError("No episodes survived extraction. Nothing was written.")

    total_rows = write_parquet(output_dir, processed_episodes)
    episode_ends = write_metadata(output_dir, processed_episodes, chunk_size=chunk_size)
    print(f"Wrote {len(processed_episodes)} episodes and {total_rows} rows to {output_dir}")
    print(f"episode_ends={episode_ends.tolist()}")
    print(f"Wrote chunked synthetic point clouds for {point_cloud_episode_count} episodes.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a lowdim-only parquet and point-cloud dataset.")
    parser.add_argument(
        "--universal-contract",
        required=True,
        type=str,
        help="Path to the universal contract file that defines the action-label mapping.",
    )
    parser.add_argument(
        "--point-config",
        default=None,
        type=str,
        help="Optional YAML file that configures synthetic end-effector point sampling.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing episode_* subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory. Defaults to <input-dir>_extracted.",
    )
    parser.add_argument(
        "--chunk-size",
        default=DEFAULT_CHUNK_SIZE,
        type=int,
        help="Number of frames per output point-cloud chunk.",
    )
    parser.add_argument(
        "--trim-start",
        default=0,
        type=int,
        help="Number of lowdim frames to drop from the start of each episode.",
    )
    parser.add_argument(
        "--trim-end",
        default=0,
        type=int,
        help="Number of lowdim frames to drop from the end of each episode.",
    )
    parser.add_argument(
        "--vel-thresh",
        default=DEFAULT_VEL_THRESH,
        type=float,
        help="Speed threshold used for stationary pruning.",
    )
    parser.add_argument(
        "--stationary-window",
        default=DEFAULT_STATIONARY_WINDOW,
        type=int,
        help="Trailing window size used to classify stationary frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universal_contract_path = Path(args.universal_contract).expanduser().resolve()
    point_config_path = Path(args.point_config).expanduser().resolve() if args.point_config is not None else None
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = resolve_output_dir(input_dir, args.output_dir)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.stationary_window <= 0:
        raise ValueError("--stationary-window must be positive")

    extract_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        universal_contract_path=universal_contract_path,
        point_config_path=point_config_path,
        chunk_size=int(args.chunk_size),
        trim_start=int(args.trim_start),
        trim_end=int(args.trim_end),
        vel_thresh=float(args.vel_thresh),
        stationary_window=int(args.stationary_window),
    )


if __name__ == "__main__":
    main()
