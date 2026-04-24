from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CHUNK_SIZE = 128
DEFAULT_VEL_THRESH = 1e-3
DEFAULT_STATIONARY_WINDOW = 10
DEFAULT_VIDEO_CODEC = "mp4v"
DEFAULT_PARQUET_NAME = "dataset.parquet"
DEFAULT_CAMERA_KEY = "camera_01"


@dataclass(frozen=True)
class EpisodeData:
    source_dir: Path
    source_name: str
    timestamps: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray
    frames: np.ndarray
    video_fps: float


@dataclass(frozen=True)
class ProcessedEpisode:
    source_name: str
    positions: np.ndarray
    orientations: np.ndarray
    frames: np.ndarray
    video_fps: float


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for extraction. Install `opencv-python` in the active environment."
        ) from exc
    return cv2


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
        from scipy.spatial.transform import Rotation, Slerp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciPy is required for orientation interpolation. Install `scipy` in the active environment."
        ) from exc
    return Rotation, Slerp


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


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
    (output_dir / "videos").mkdir(parents=True, exist_ok=False)


def load_camera_timestamps(episode_dir: Path) -> np.ndarray:
    visual_dir = episode_dir / "visual"
    npy_path = visual_dir / f"{DEFAULT_CAMERA_KEY}_timestamps.npy"
    npz_path = visual_dir / f"{DEFAULT_CAMERA_KEY}_timestamps.npz"

    if npy_path.exists():
        timestamps = np.load(npy_path)
    elif npz_path.exists():
        npz_file = np.load(npz_path)
        if len(npz_file.files) == 0:
            raise ValueError(f"No arrays found in {npz_path}")
        timestamps = npz_file[npz_file.files[0]]
    else:
        raise FileNotFoundError(
            f"Missing camera timestamps for {episode_dir}. Expected {npy_path.name} or {npz_path.name}."
        )

    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        raise ValueError(f"Camera timestamps are empty for {episode_dir}")
    return timestamps


def load_lowdim_arrays(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lowdim_path = episode_dir / "lowdim.npz"
    if not lowdim_path.exists():
        raise FileNotFoundError(f"Missing lowdim file: {lowdim_path}")

    lowdim = np.load(lowdim_path)
    timestamp_key = "timestamp_s" if "timestamp_s" in lowdim else "ts" if "ts" in lowdim else None
    if timestamp_key is None:
        raise KeyError(f"Expected `timestamp_s` or `ts` in {lowdim_path}")
    if "eef_pos" not in lowdim or "eef_ori" not in lowdim:
        raise KeyError(f"Expected `eef_pos` and `eef_ori` in {lowdim_path}")

    timestamps = np.asarray(lowdim[timestamp_key], dtype=np.float64).reshape(-1)
    positions = np.asarray(lowdim["eef_pos"], dtype=np.float64)
    orientations = np.asarray(lowdim["eef_ori"], dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"`eef_pos` must have shape (T, 3) in {lowdim_path}")
    if orientations.ndim != 3 or orientations.shape[1:] != (3, 3):
        raise ValueError(f"`eef_ori` must have shape (T, 3, 3) in {lowdim_path}")
    if not (len(timestamps) == len(positions) == len(orientations)):
        raise ValueError(f"Lowdim arrays in {lowdim_path} do not have matching lengths")

    return timestamps, positions, orientations


def read_video_frames(video_path: Path) -> tuple[np.ndarray, float]:
    cv2 = _require_cv2()
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 30.0

    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise ValueError(f"No frames found in {video_path}")

    return np.stack(frames, axis=0), fps


def load_episode_data(episode_dir: Path) -> EpisodeData:
    timestamps = load_camera_timestamps(episode_dir)
    lowdim_timestamps, positions, orientations = load_lowdim_arrays(episode_dir)
    frames, video_fps = read_video_frames(episode_dir / "visual" / f"{DEFAULT_CAMERA_KEY}.mp4")

    frame_count = min(len(timestamps), len(frames))
    if frame_count == 0:
        raise ValueError(f"No aligned camera frames available for {episode_dir}")
    if len(timestamps) != len(frames):
        _warn(
            f"{episode_dir.name}: camera timestamps ({len(timestamps)}) and video frames ({len(frames)}) "
            f"do not match; truncating both to {frame_count}."
        )

    timestamps = timestamps[:frame_count]
    frames = frames[:frame_count]

    return EpisodeData(
        source_dir=episode_dir,
        source_name=episode_dir.name,
        timestamps=timestamps,
        positions=positions,
        orientations=orientations,
        frames=frames,
        video_fps=video_fps,
    )


def crop_lowdim_to_camera_range(
    lowdim_timestamps: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    camera_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_time = float(camera_timestamps[0])
    end_time = float(camera_timestamps[-1])
    keep_mask = (lowdim_timestamps >= start_time) & (lowdim_timestamps <= end_time)

    return (
        lowdim_timestamps[keep_mask],
        positions[keep_mask],
        orientations[keep_mask],
    )


def sanitize_interpolation_inputs(
    timestamps: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sort_indices = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[sort_indices]
    positions = positions[sort_indices]
    orientations = orientations[sort_indices]

    unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)
    return (
        unique_timestamps,
        positions[unique_indices],
        orientations[unique_indices],
    )


def interpolate_positions(
    source_timestamps: np.ndarray,
    source_positions: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    interpolated = np.empty((len(target_timestamps), 3), dtype=np.float32)
    for axis in range(3):
        interpolated[:, axis] = np.interp(
            target_timestamps,
            source_timestamps,
            source_positions[:, axis],
        ).astype(np.float32)
    return interpolated


def interpolate_orientations(
    source_timestamps: np.ndarray,
    source_orientations: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    Rotation, Slerp = _require_scipy_rotation()

    source_rotations = Rotation.from_matrix(source_orientations)
    slerp = Slerp(source_timestamps, source_rotations)
    target_rotations = slerp(target_timestamps)
    return target_rotations.as_matrix().astype(np.float32)


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


def prune_episode(
    timestamps: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    frames: np.ndarray,
    trim_start: int,
    trim_end: int,
    vel_thresh: float,
    stationary_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_mask = apply_edge_trim(len(timestamps), trim_start=trim_start, trim_end=trim_end)
    if not np.any(keep_mask):
        return (
            positions[:0],
            orientations[:0],
            frames[:0],
        )

    timestamps = timestamps[keep_mask]
    positions = positions[keep_mask]
    orientations = orientations[keep_mask]
    frames = frames[keep_mask]

    if len(timestamps) <= 1:
        return positions, orientations, frames

    dt_values = np.diff(timestamps)
    positive_dt_values = dt_values[dt_values > 0.0]
    dt = float(np.median(positive_dt_values)) if len(positive_dt_values) else 1.0

    stationary_mask = build_stationary_mask(
        positions,
        dt=dt,
        vel_thresh=vel_thresh,
        stationary_window=stationary_window,
    )
    keep_after_stationary = ~stationary_mask
    return (
        positions[keep_after_stationary],
        orientations[keep_after_stationary],
        frames[keep_after_stationary],
    )


def process_episode(
    episode_dir: Path,
    trim_start: int,
    trim_end: int,
    vel_thresh: float,
    stationary_window: int,
) -> ProcessedEpisode | None:
    episode = load_episode_data(episode_dir)
    lowdim_timestamps, lowdim_positions, lowdim_orientations = crop_lowdim_to_camera_range(
        *load_lowdim_arrays(episode_dir),
        episode.timestamps,
    )
    lowdim_timestamps, lowdim_positions, lowdim_orientations = sanitize_interpolation_inputs(
        lowdim_timestamps,
        lowdim_positions,
        lowdim_orientations,
    )

    if len(lowdim_timestamps) < 2:
        _warn(
            f"{episode.source_name}: fewer than 2 lowdim samples remain after camera-range cropping; skipping episode."
        )
        return None

    valid_camera_mask = (
        (episode.timestamps >= lowdim_timestamps[0]) &
        (episode.timestamps <= lowdim_timestamps[-1])
    )
    if not np.any(valid_camera_mask):
        _warn(f"{episode.source_name}: no camera timestamps fall within the lowdim interpolation range; skipping episode.")
        return None

    aligned_timestamps = episode.timestamps[valid_camera_mask]
    aligned_frames = episode.frames[valid_camera_mask]

    aligned_positions = interpolate_positions(
        lowdim_timestamps,
        lowdim_positions,
        aligned_timestamps,
    )
    aligned_orientations = interpolate_orientations(
        lowdim_timestamps,
        lowdim_orientations,
        aligned_timestamps,
    )

    pruned_positions, pruned_orientations, pruned_frames = prune_episode(
        timestamps=aligned_timestamps,
        positions=aligned_positions,
        orientations=aligned_orientations,
        frames=aligned_frames,
        trim_start=trim_start,
        trim_end=trim_end,
        vel_thresh=vel_thresh,
        stationary_window=stationary_window,
    )

    if len(pruned_positions) == 0:
        _warn(f"{episode.source_name}: pruning removed every frame; skipping episode.")
        return None

    return ProcessedEpisode(
        source_name=episode.source_name,
        positions=pruned_positions.astype(np.float32),
        orientations=pruned_orientations.astype(np.float32),
        frames=pruned_frames,
        video_fps=float(episode.video_fps),
    )


def write_parquet(output_dir: Path, episodes: list[ProcessedEpisode]) -> int:
    pa, pq = _require_pyarrow()

    all_positions = np.concatenate([episode.positions for episode in episodes], axis=0)
    all_orientations = np.concatenate([episode.orientations for episode in episodes], axis=0).reshape(-1, 9)

    eef_pos_array = pa.FixedSizeListArray.from_arrays(
        pa.array(all_positions.reshape(-1), type=pa.float32()),
        3,
    )
    eef_ori_array = pa.FixedSizeListArray.from_arrays(
        pa.array(all_orientations.reshape(-1), type=pa.float32()),
        9,
    )

    table = pa.table(
        {
            "eef_pos": eef_pos_array,
            "eef_ori": eef_ori_array,
        }
    )
    parquet_path = output_dir / DEFAULT_PARQUET_NAME
    pq.write_table(table, parquet_path)
    return int(len(all_positions))


def write_metadata(output_dir: Path, episodes: list[ProcessedEpisode], chunk_size: int) -> np.ndarray:
    running_total = 0
    episode_ends = []
    for episode in episodes:
        running_total += len(episode.positions)
        episode_ends.append(running_total - 1)

    metadata_path = output_dir / "metadata.npz"
    np.savez(
        metadata_path,
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        chunk_size=np.asarray(chunk_size, dtype=np.int64),
    )
    return np.asarray(episode_ends, dtype=np.int64)


def write_chunked_videos(output_dir: Path, episodes: list[ProcessedEpisode], chunk_size: int) -> None:
    cv2 = _require_cv2()
    fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
    videos_dir = output_dir / "videos"

    for output_episode_index, episode in enumerate(episodes, start=1):
        episode_dir = videos_dir / f"episode_{output_episode_index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=False)

        frame_height, frame_width = episode.frames.shape[1:3]
        num_chunks = (len(episode.frames) + chunk_size - 1) // chunk_size

        for chunk_index in range(num_chunks):
            chunk_start = chunk_index * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(episode.frames))
            chunk_frames = episode.frames[chunk_start:chunk_end]

            chunk_path = episode_dir / f"chunk_{chunk_index + 1:04d}.mp4"
            writer = cv2.VideoWriter(
                str(chunk_path),
                fourcc,
                episode.video_fps,
                (frame_width, frame_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {chunk_path}")

            for frame in chunk_frames:
                writer.write(frame)
            writer.release()


def extract_dataset(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    trim_start: int,
    trim_end: int,
    vel_thresh: float,
    stationary_window: int,
) -> None:
    episode_dirs = discover_episode_dirs(input_dir)
    prepare_output_dir(output_dir)

    processed_episodes = []
    for episode_dir in episode_dirs:
        processed_episode = process_episode(
            episode_dir,
            trim_start=trim_start,
            trim_end=trim_end,
            vel_thresh=vel_thresh,
            stationary_window=stationary_window,
        )
        if processed_episode is not None:
            processed_episodes.append(processed_episode)

    if not processed_episodes:
        raise RuntimeError("No episodes survived extraction. Nothing was written.")

    total_rows = write_parquet(output_dir, processed_episodes)
    episode_ends = write_metadata(output_dir, processed_episodes, chunk_size=chunk_size)
    write_chunked_videos(output_dir, processed_episodes, chunk_size=chunk_size)

    print(f"Wrote {len(processed_episodes)} episodes and {total_rows} rows to {output_dir}")
    print(f"episode_ends={episode_ends.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a lightweight parquet/video dataset.")
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
        help="Number of frames per output MP4 chunk.",
    )
    parser.add_argument(
        "--trim-start",
        default=0,
        type=int,
        help="Number of aligned frames to drop from the start of each episode.",
    )
    parser.add_argument(
        "--trim-end",
        default=0,
        type=int,
        help="Number of aligned frames to drop from the end of each episode.",
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
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = resolve_output_dir(input_dir, args.output_dir)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.stationary_window <= 0:
        raise ValueError("--stationary-window must be positive")

    extract_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=int(args.chunk_size),
        trim_start=int(args.trim_start),
        trim_end=int(args.trim_end),
        vel_thresh=float(args.vel_thresh),
        stationary_window=int(args.stationary_window),
    )


if __name__ == "__main__":
    main()
