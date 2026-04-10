from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CHUNK_SIZE = 128
DEFAULT_VIDEO_CODEC = "mp4v"
DEFAULT_PARQUET_NAME = "dataset.parquet"
DEFAULT_PARQUET_COMPRESSION = "snappy"
DEFAULT_VISUAL_FPS = 30.0


@dataclass(frozen=True)
class CameraSpec:
    name: str
    fps: float
    dim: tuple[int, int] | None
    video_path: Path
    timestamp_path: Path


@dataclass(frozen=True)
class EpisodeContext:
    episode_dir: Path
    episode_name: str
    episode_id: int
    metadata: dict[str, Any]
    contract: dict[str, Any]
    lowdim_path: Path
    camera_specs: dict[str, CameraSpec]


@dataclass(frozen=True)
class LowdimInterpolator:
    timestamps: np.ndarray
    series: dict[str, np.ndarray]


def discover_episode_dirs(buffer_dir: Path) -> list[Path]:
    buffer_dir = buffer_dir.expanduser().resolve()
    if not buffer_dir.exists():
        raise FileNotFoundError(f"Buffer directory does not exist: {buffer_dir}")
    if not buffer_dir.is_dir():
        raise NotADirectoryError(f"Buffer path is not a directory: {buffer_dir}")

    episode_dirs = sorted(
        path
        for path in buffer_dir.iterdir()
        if path.is_dir() and (path / "metadata.json").exists()
    )
    if not episode_dirs:
        raise FileNotFoundError(
            f"No episode directories with metadata.json were found in {buffer_dir}"
        )
    return episode_dirs


def load_episode_metadata(episode_dir: Path) -> dict[str, Any]:
    metadata_path = episode_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Episode metadata must be a mapping: {metadata_path}")
    return metadata


def load_episode_contract(episode_dir: Path) -> dict[str, Any]:
    contract_path = episode_dir / "contract.yaml"
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise ValueError(f"Episode contract must be a mapping: {contract_path}")
    return contract


def load_camera_specs_from_contract(
    contract: dict[str, Any],
    episode_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, CameraSpec]:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        raise ValueError(
            f"Contract for {episode_dir} must contain a top-level `robot` mapping."
        )

    data_sources = robot_cfg.get("data_sources", {})
    if not isinstance(data_sources, dict):
        raise ValueError(f"Contract for {episode_dir} is missing `robot.data_sources`.")

    visual_cfg = data_sources.get("visual", {})
    if not isinstance(visual_cfg, dict):
        raise ValueError(f"Contract for {episode_dir} is missing `robot.data_sources.visual`.")

    default_fps = float(visual_cfg.get("fps") or metadata.get("camera_fps") or DEFAULT_VISUAL_FPS)
    visual_keys = visual_cfg.get("keys", [])
    if not isinstance(visual_keys, list) or not visual_keys:
        raise ValueError(f"Contract for {episode_dir} does not define any visual keys.")

    visual_dir = episode_dir / "visual"
    camera_specs: dict[str, CameraSpec] = {}
    for entry in visual_keys:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue

        camera_name, camera_cfg = next(iter(entry.items()))
        if not isinstance(camera_cfg, dict):
            raise ValueError(
                f"Visual config for camera `{camera_name}` must be a mapping in {episode_dir}."
            )

        fps = float(camera_cfg.get("fps") or default_fps)
        if fps <= 0.0:
            raise ValueError(
                f"Camera `{camera_name}` has a non-positive fps in {episode_dir}."
            )

        dim_cfg = camera_cfg.get("dim")
        dim: tuple[int, int] | None = None
        if isinstance(dim_cfg, (list, tuple)) and len(dim_cfg) >= 2:
            dim = (int(dim_cfg[0]), int(dim_cfg[1]))

        camera_specs[camera_name] = CameraSpec(
            name=camera_name,
            fps=fps,
            dim=dim,
            video_path=visual_dir / f"{camera_name}.mp4",
            timestamp_path=visual_dir / f"{camera_name}_timestamps.npy",
        )

    if not camera_specs:
        raise ValueError(f"No valid camera specs were parsed from contract in {episode_dir}.")

    return camera_specs


def validate_episode_assets(episode: EpisodeContext) -> None:
    required_paths = [
        episode.episode_dir / "metadata.json",
        episode.episode_dir / "contract.yaml",
        episode.lowdim_path,
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required episode asset: {required_path}")

    if not episode.camera_specs:
        raise ValueError(f"Episode `{episode.episode_name}` has no camera specs.")

    for camera_spec in episode.camera_specs.values():
        if not camera_spec.video_path.exists():
            raise FileNotFoundError(
                f"Missing video file for camera `{camera_spec.name}`: {camera_spec.video_path}"
            )
        if not camera_spec.timestamp_path.exists():
            raise FileNotFoundError(
                f"Missing timestamp file for camera `{camera_spec.name}`: {camera_spec.timestamp_path}"
            )


def build_episode_context(episode_dir: Path) -> EpisodeContext:
    metadata = load_episode_metadata(episode_dir)
    contract = load_episode_contract(episode_dir)
    camera_specs = load_camera_specs_from_contract(contract, episode_dir, metadata)

    episode_name = str(metadata.get("episode_name") or episode_dir.name)
    episode_id = metadata.get("episode_id")
    if episode_id is None:
        try:
            episode_id = int(episode_dir.name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Could not infer episode_id for {episode_dir}; metadata is missing it."
            ) from exc

    episode = EpisodeContext(
        episode_dir=episode_dir,
        episode_name=episode_name,
        episode_id=int(episode_id),
        metadata=metadata,
        contract=contract,
        lowdim_path=episode_dir / "lowdim.npz",
        camera_specs=camera_specs,
    )
    validate_episode_assets(episode)
    return episode


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for extraction. Install the `opencv` package in the runtime environment."
        ) from exc
    return cv2


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyArrow is required to write and read parquet files. Install the `pyarrow` package in the runtime environment."
        ) from exc
    return pa, pq


def _require_tqdm():
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tqdm is required for extractor progress bars. Install the `tqdm` package in the runtime environment."
        ) from exc
    return tqdm


def _require_slerp():
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciPy is required for rotation interpolation. Install the `scipy` package in the runtime environment."
        ) from exc
    return Rotation, Slerp


def _resolve_output_dir(buffer_dir: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return buffer_dir.parent / f"{buffer_dir.name}_extracted"


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Remove it before extracting again."
        )
    output_dir.mkdir(parents=True, exist_ok=False)


def _warning(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def _load_timestamps(timestamp_path: Path) -> np.ndarray:
    timestamps = np.asarray(np.load(timestamp_path), dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError(f"Camera timestamps must be a 1D array: {timestamp_path}")
    return timestamps


def _load_lowdim_archive(lowdim_path: Path) -> dict[str, np.ndarray]:
    with np.load(lowdim_path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _deduplicate_timestamps(
    timestamps: np.ndarray,
    series: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if timestamps.size == 0:
        raise ValueError("Lowdim timestamps are empty.")

    kept_indices: list[int] = []
    kept_timestamps: list[float] = []
    for index, timestamp in enumerate(timestamps):
        timestamp_value = float(timestamp)
        if kept_timestamps and timestamp_value < kept_timestamps[-1]:
            raise ValueError("Lowdim timestamps must be nondecreasing.")
        if kept_timestamps and timestamp_value == kept_timestamps[-1]:
            kept_indices[-1] = index
        else:
            kept_timestamps.append(timestamp_value)
            kept_indices.append(index)

    deduplicated_timestamps = np.asarray(kept_timestamps, dtype=np.float64)
    deduplicated_series = {
        key: np.asarray(values)[kept_indices] for key, values in series.items()
    }
    return deduplicated_timestamps, deduplicated_series


def build_lowdim_interpolator(lowdim_archive: dict[str, np.ndarray]) -> LowdimInterpolator:
    if "timestamp_s" not in lowdim_archive:
        raise KeyError("Lowdim archive is missing `timestamp_s`.")

    timestamps = np.asarray(lowdim_archive["timestamp_s"], dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError("Lowdim `timestamp_s` must be a 1D array.")

    series: dict[str, np.ndarray] = {}
    for key, values in lowdim_archive.items():
        if key == "timestamp_s":
            continue
        if len(values) != len(timestamps):
            raise ValueError(
                f"Lowdim key `{key}` has {len(values)} samples but timestamps have {len(timestamps)} samples."
            )
        series[key] = np.asarray(values, dtype=np.float64)

    deduplicated_timestamps, deduplicated_series = _deduplicate_timestamps(
        timestamps,
        series,
    )
    if deduplicated_timestamps.size == 0:
        raise ValueError("No lowdim timestamps remain after deduplication.")

    return LowdimInterpolator(
        timestamps=deduplicated_timestamps,
        series=deduplicated_series,
    )


def interpolate_numeric_series(
    timestamps: np.ndarray,
    values: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    if timestamps.size == 0:
        raise ValueError("Cannot interpolate an empty numeric series.")

    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)
    if timestamps.size == 1:
        return np.repeat(values[:1], repeats=len(target_timestamps), axis=0)

    flat_values = values.reshape(values.shape[0], -1)
    result = np.empty((len(target_timestamps), flat_values.shape[1]), dtype=np.float64)

    before_mask = target_timestamps <= timestamps[0]
    after_mask = target_timestamps >= timestamps[-1]
    interior_mask = ~(before_mask | after_mask)

    if before_mask.any():
        result[before_mask] = flat_values[0]
    if after_mask.any():
        result[after_mask] = flat_values[-1]

    if interior_mask.any():
        interior_targets = target_timestamps[interior_mask]
        right_indices = np.searchsorted(timestamps, interior_targets, side="right")
        left_indices = right_indices - 1

        left_times = timestamps[left_indices]
        right_times = timestamps[right_indices]
        denominators = right_times - left_times
        weights = ((interior_targets - left_times) / denominators)[:, None]

        left_values = flat_values[left_indices]
        right_values = flat_values[right_indices]
        result[interior_mask] = left_values * (1.0 - weights) + right_values * weights

    return result.reshape((len(target_timestamps),) + values.shape[1:])


def interpolate_rotation_series(
    timestamps: np.ndarray,
    values: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    Rotation, Slerp = _require_slerp()

    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)
    if timestamps.size == 0:
        raise ValueError("Cannot interpolate an empty rotation series.")
    if timestamps.size == 1:
        return np.repeat(values[:1], repeats=len(target_timestamps), axis=0)

    rotations = Rotation.from_matrix(values)
    slerp = Slerp(timestamps, rotations)
    result = np.empty((len(target_timestamps), 3, 3), dtype=np.float64)

    before_mask = target_timestamps <= timestamps[0]
    after_mask = target_timestamps >= timestamps[-1]
    interior_mask = ~(before_mask | after_mask)

    if before_mask.any():
        result[before_mask] = values[0]
    if after_mask.any():
        result[after_mask] = values[-1]
    if interior_mask.any():
        result[interior_mask] = slerp(target_timestamps[interior_mask]).as_matrix()

    return result


def interpolate_lowdim_data(
    interpolator: LowdimInterpolator,
    target_timestamps: np.ndarray,
    *,
    episode_name: str,
    camera_name: str,
) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {}
    for key, values in interpolator.series.items():
        if values.ndim == 3 and values.shape[1:] == (3, 3):
            try:
                aligned[key] = interpolate_rotation_series(
                    interpolator.timestamps,
                    values,
                    target_timestamps,
                )
                continue
            except Exception as exc:
                _warning(
                    f"Falling back to numeric interpolation for `{key}` in episode `{episode_name}` "
                    f"camera `{camera_name}` because rotation interpolation failed: {exc}"
                )

        aligned[key] = interpolate_numeric_series(
            interpolator.timestamps,
            values,
            target_timestamps,
        )
    return aligned


def expand_and_chunk_videos(
    episodes: list[EpisodeContext],
    output_root: Path,
    chunk_size: int,
    video_codec: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    cv2 = _require_cv2()
    tqdm = _require_tqdm()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if len(video_codec) != 4:
        raise ValueError("video_codec must be a four-character code such as `mp4v`.")

    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    frame_manifest: list[dict[str, Any]] = []
    actual_frame_counts: dict[str, dict[str, int]] = defaultdict(dict)
    estimated_total_frames = sum(
        len(_load_timestamps(camera_spec.timestamp_path))
        for episode in episodes
        for camera_spec in episode.camera_specs.values()
    )

    with tqdm(total=estimated_total_frames, desc="Chunking frames", unit="frame") as progress_bar:
        for episode in sorted(episodes, key=lambda ep: ep.episode_id):
            episode_output_dir = output_root / episode.episode_name
            episode_output_dir.mkdir(parents=True, exist_ok=True)

            for camera_name, camera_spec in sorted(episode.camera_specs.items()):
                timestamps = _load_timestamps(camera_spec.timestamp_path)
                capture = cv2.VideoCapture(str(camera_spec.video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"Failed to open camera video: {camera_spec.video_path}")

                writer = None
                chunk_index = -1
                written_frames = 0
                decoded_frames = 0
                expected_frame_count = episode.metadata.get("camera_frame_counts", {}).get(camera_name)
                progress_bar.set_postfix_str(f"{episode.episode_name}/{camera_name}", refresh=False)

                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break

                        if decoded_frames < len(timestamps):
                            if frame.ndim != 3 or frame.shape[2] != 3:
                                raise ValueError(
                                    f"Camera frame for `{camera_name}` in `{episode.episode_name}` is not HxWx3."
                                )

                            height, width = frame.shape[:2]
                            if camera_spec.dim is not None:
                                expected_width, expected_height = camera_spec.dim
                                if (width, height) != (expected_width, expected_height):
                                    _warning(
                                        f"Decoded frame size for `{camera_name}` in `{episode.episode_name}` is "
                                        f"{width}x{height}, which differs from the contract dim {expected_width}x{expected_height}."
                                    )

                            if written_frames % chunk_size == 0:
                                if writer is not None:
                                    writer.release()
                                chunk_index += 1
                                chunk_path = episode_output_dir / f"{camera_name}_chunk_{chunk_index:06d}.mp4"
                                writer = cv2.VideoWriter(
                                    str(chunk_path),
                                    fourcc,
                                    float(camera_spec.fps),
                                    (width, height),
                                )
                                if not writer.isOpened():
                                    raise RuntimeError(
                                        f"Failed to open chunk writer for `{chunk_path}` using codec `{video_codec}`."
                                    )

                            writer.write(frame)
                            frame_manifest.append(
                                {
                                    "episode_id": episode.episode_id,
                                    "episode_name": episode.episode_name,
                                    "camera_name": camera_name,
                                    "episode_frame_index": written_frames,
                                    "camera_timestamp_s": float(timestamps[written_frames]),
                                    "video_chunk_index": chunk_index,
                                    "frame_index_in_chunk": written_frames % chunk_size,
                                    "video_chunk_path": (
                                        episode_output_dir
                                        / f"{camera_name}_chunk_{chunk_index:06d}.mp4"
                                    )
                                    .relative_to(output_root)
                                    .as_posix(),
                                }
                            )
                            written_frames += 1
                            progress_bar.update(1)

                        decoded_frames += 1
                finally:
                    if writer is not None:
                        writer.release()
                    capture.release()

                actual_frame_count = min(decoded_frames, len(timestamps))
                actual_frame_counts[episode.episode_name][camera_name] = actual_frame_count

                if written_frames < len(timestamps):
                    progress_bar.update(len(timestamps) - written_frames)

                if actual_frame_count == 0:
                    raise ValueError(
                        f"Camera `{camera_name}` in episode `{episode.episode_name}` produced zero usable frames."
                    )

                if decoded_frames != len(timestamps):
                    _warning(
                        f"Frame count mismatch for `{camera_name}` in `{episode.episode_name}`: "
                        f"decoded {decoded_frames} frames but found {len(timestamps)} timestamps. "
                        f"Using the first {actual_frame_count} frames."
                    )

                if expected_frame_count is not None and int(expected_frame_count) != actual_frame_count:
                    _warning(
                        f"Episode metadata reports {expected_frame_count} frames for `{camera_name}` in "
                        f"`{episode.episode_name}`, but {actual_frame_count} usable frames were extracted."
                    )

    return frame_manifest, {key: dict(value) for key, value in actual_frame_counts.items()}


def infer_pyarrow_type_from_sample(sample: Any):
    pa, _ = _require_pyarrow()

    if isinstance(sample, dict):
        return pa.struct(
            [
                pa.field(key, infer_pyarrow_type_from_sample(value))
                for key, value in sorted(sample.items())
            ]
        )
    if isinstance(sample, np.ndarray):
        if sample.ndim == 0:
            return infer_pyarrow_type_from_sample(sample.item())
        if sample.shape[0] == 0:
            raise ValueError("Cannot infer a fixed-size list type from an empty array.")
        return pa.list_(infer_pyarrow_type_from_sample(sample[0]), int(sample.shape[0]))
    if isinstance(sample, (list, tuple)):
        if not sample:
            raise ValueError("Cannot infer a fixed-size list type from an empty list.")
        return pa.list_(infer_pyarrow_type_from_sample(sample[0]), len(sample))
    if isinstance(sample, (np.bool_, bool)):
        return pa.bool_()
    if isinstance(sample, (np.integer, int)):
        return pa.int64()
    if isinstance(sample, (np.floating, float)):
        return pa.float64()
    if isinstance(sample, str):
        return pa.string()
    raise TypeError(f"Unsupported sample type for parquet inference: {type(sample)!r}")


def to_nested_python_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_nested_python_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [to_nested_python_value(item) for item in value]
    return value


def _build_schema(
    rows: list[dict[str, Any]],
    dataset_metadata: dict[str, str],
):
    pa, _ = _require_pyarrow()

    lowdim_samples: dict[str, Any] = {}
    for row in rows:
        lowdim_data = row.get("lowdimData", {})
        for key, value in lowdim_data.items():
            if value is not None and key not in lowdim_samples:
                lowdim_samples[key] = value

    if not lowdim_samples:
        raise ValueError("No lowdim samples were found to build the parquet schema.")

    lowdim_fields = [
        pa.field(key, infer_pyarrow_type_from_sample(lowdim_samples[key]))
        for key in sorted(lowdim_samples)
    ]
    metadata_fields = [
        pa.field("start_timestamp_s", pa.float64()),
        pa.field("end_timestamp_s", pa.float64()),
        pa.field("duration_s", pa.float64()),
        pa.field("num_lowdim_samples", pa.int64()),
        pa.field("camera_frame_count", pa.int64()),
        pa.field("lowdim_fps", pa.float64()),
        pa.field("camera_fps", pa.float64()),
    ]

    schema = pa.schema(
        [
            pa.field("episode_id", pa.int64()),
            pa.field("episode_name", pa.string()),
            pa.field("camera_name", pa.string()),
            pa.field("global_frame_index", pa.int64()),
            pa.field("episode_frame_index", pa.int64()),
            pa.field("camera_timestamp_s", pa.float64()),
            pa.field("video_chunk_index", pa.int32()),
            pa.field("frame_index_in_chunk", pa.int32()),
            pa.field("video_chunk_path", pa.string()),
            pa.field("video_chunk_frame_end_index", pa.int64()),
            pa.field("episode_end_index", pa.int64()),
            pa.field("lowdimData", pa.struct(lowdim_fields)),
            pa.field("metadata", pa.struct(metadata_fields)),
        ]
    )
    return schema.with_metadata(
        {key.encode("utf-8"): value.encode("utf-8") for key, value in dataset_metadata.items()}
    )


def create_parquet_file(
    frame_manifest: list[dict[str, Any]],
    episodes: list[EpisodeContext],
    actual_frame_counts: dict[str, dict[str, int]],
    output_root: Path,
    parquet_path: Path,
    *,
    buffer_dir: Path,
    chunk_size: int,
    video_codec: str,
) -> Path:
    pa, pq = _require_pyarrow()
    tqdm = _require_tqdm()

    if not frame_manifest:
        raise ValueError("No frame records were produced by video expansion and chunking.")

    rows = sorted(
        frame_manifest,
        key=lambda row: (row["episode_id"], row["camera_name"], row["episode_frame_index"]),
    )
    episodes_by_name = {episode.episode_name: episode for episode in episodes}
    lowdim_interpolators = {
        episode.episode_name: build_lowdim_interpolator(_load_lowdim_archive(episode.lowdim_path))
        for episode in tqdm(episodes, desc="Loading lowdim", unit="episode")
    }

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["episode_name"], row["camera_name"])].append(row)

    for (episode_name, camera_name), grouped in tqdm(
        grouped_rows.items(),
        total=len(grouped_rows),
        desc="Interpolating lowdim",
        unit="stream",
    ):
        episode = episodes_by_name[episode_name]
        interpolator = lowdim_interpolators[episode_name]
        target_timestamps = np.asarray(
            [row["camera_timestamp_s"] for row in grouped],
            dtype=np.float64,
        )
        aligned_lowdim = interpolate_lowdim_data(
            interpolator,
            target_timestamps,
            episode_name=episode_name,
            camera_name=camera_name,
        )

        camera_frame_count = int(actual_frame_counts[episode_name][camera_name])
        metadata_payload = {
            "start_timestamp_s": float(episode.metadata.get("start_timestamp_s", 0.0)),
            "end_timestamp_s": float(episode.metadata.get("end_timestamp_s", 0.0)),
            "duration_s": float(episode.metadata.get("duration_s", 0.0)),
            "num_lowdim_samples": int(episode.metadata.get("num_lowdim_samples", 0)),
            "camera_frame_count": camera_frame_count,
            "lowdim_fps": float(episode.metadata.get("lowdim_fps", 0.0)),
            "camera_fps": float(episode.metadata.get("camera_fps", 0.0)),
        }

        for index, row in enumerate(grouped):
            row["lowdimData"] = {
                key: np.asarray(values[index]).copy()
                for key, values in aligned_lowdim.items()
            }
            row["metadata"] = dict(metadata_payload)

    for global_index, row in enumerate(rows):
        row["global_frame_index"] = global_index

    chunk_start = 0
    while chunk_start < len(rows):
        chunk_key = (
            rows[chunk_start]["episode_id"],
            rows[chunk_start]["camera_name"],
            rows[chunk_start]["video_chunk_index"],
        )
        chunk_end = chunk_start
        while chunk_end + 1 < len(rows):
            candidate = rows[chunk_end + 1]
            candidate_key = (
                candidate["episode_id"],
                candidate["camera_name"],
                candidate["video_chunk_index"],
            )
            if candidate_key != chunk_key:
                break
            chunk_end += 1
        for row_index in range(chunk_start, chunk_end + 1):
            rows[row_index]["video_chunk_frame_end_index"] = chunk_end
        chunk_start = chunk_end + 1

    episode_start = 0
    while episode_start < len(rows):
        episode_id = rows[episode_start]["episode_id"]
        episode_end = episode_start
        while episode_end + 1 < len(rows) and rows[episode_end + 1]["episode_id"] == episode_id:
            episode_end += 1
        for row_index in range(episode_start, episode_end + 1):
            rows[row_index]["episode_end_index"] = episode_end
        episode_start = episode_end + 1

    python_rows = []
    for row in tqdm(rows, desc="Preparing parquet rows", unit="row"):
        python_rows.append(
            {
                "episode_id": int(row["episode_id"]),
                "episode_name": str(row["episode_name"]),
                "camera_name": str(row["camera_name"]),
                "global_frame_index": int(row["global_frame_index"]),
                "episode_frame_index": int(row["episode_frame_index"]),
                "camera_timestamp_s": float(row["camera_timestamp_s"]),
                "video_chunk_index": int(row["video_chunk_index"]),
                "frame_index_in_chunk": int(row["frame_index_in_chunk"]),
                "video_chunk_path": str(row["video_chunk_path"]),
                "video_chunk_frame_end_index": int(row["video_chunk_frame_end_index"]),
                "episode_end_index": int(row["episode_end_index"]),
                "lowdimData": to_nested_python_value(row["lowdimData"]),
                "metadata": to_nested_python_value(row["metadata"]),
            }
        )

    dataset_metadata = {
        "buffer_name": buffer_dir.name,
        "source_buffer_dir": str(buffer_dir.resolve()),
        "chunk_size": str(chunk_size),
        "video_codec": video_codec,
        "row_format": "per_frame",
        "schema_version": "1",
    }

    schema = _build_schema(rows, dataset_metadata)
    print(f"Writing parquet table to {parquet_path}", flush=True)
    table = pa.Table.from_pylist(python_rows, schema=schema)
    table = table.replace_schema_metadata(schema.metadata)
    pq.write_table(table, parquet_path, compression=DEFAULT_PARQUET_COMPRESSION)
    return parquet_path


def _parse_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("buffer_dir", help="Path to the saved buffer directory to extract.")
    parser.add_argument(
        "--output-dir",
        help="Destination directory for extracted chunks and parquet. Defaults to <buffer_name>_extracted.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Number of frames per output video chunk. Defaults to {DEFAULT_CHUNK_SIZE}.",
    )
    parser.add_argument(
        "--video-codec",
        default=DEFAULT_VIDEO_CODEC,
        help=f"FourCC codec for chunk videos. Defaults to {DEFAULT_VIDEO_CODEC}.",
    )
    parser.add_argument(
        "--parquet-name",
        default=DEFAULT_PARQUET_NAME,
        help=f"Filename for the output parquet file. Defaults to {DEFAULT_PARQUET_NAME}.",
    )


def _run_extract(args: argparse.Namespace) -> Path:
    _require_cv2()
    _require_pyarrow()

    buffer_dir = Path(args.buffer_dir).expanduser().resolve()
    output_dir = _resolve_output_dir(buffer_dir, args.output_dir)
    parquet_path = output_dir / args.parquet_name

    episodes = [build_episode_context(path) for path in discover_episode_dirs(buffer_dir)]
    _prepare_output_dir(output_dir)

    frame_manifest, actual_frame_counts = expand_and_chunk_videos(
        episodes,
        output_dir,
        chunk_size=int(args.chunk_size),
        video_codec=str(args.video_codec),
    )
    create_parquet_file(
        frame_manifest,
        episodes,
        actual_frame_counts,
        output_dir,
        parquet_path,
        buffer_dir=buffer_dir,
        chunk_size=int(args.chunk_size),
        video_codec=str(args.video_codec),
    )
    return parquet_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract saved ForceWM episodes into rechunked videos and a per-frame parquet dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract a saved buffer into parquet.")
    _parse_extract_args(extract_parser)

    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            parquet_path = _run_extract(args)
            print(f"Wrote extracted parquet dataset to {parquet_path}", flush=True)
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
