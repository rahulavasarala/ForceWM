from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


class Saver:
    DEFAULT_FPS = 30.0

    def __init__(
        self,
        save_dir,
        robot_observer,
        camera_observer,
        contract_path=None,
        video_codec="mp4v",
    ):
        if robot_observer is None:
            raise ValueError("robot_observer must be provided.")
        if camera_observer is None:
            raise ValueError("camera_observer must be provided.")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.robot_observer = robot_observer
        self.camera_observer = camera_observer
        self.contract_path = Path(contract_path) if contract_path is not None else None
        self.video_codec = str(video_codec)

        self.contract = getattr(self.robot_observer, "contract", None)
        if not isinstance(self.contract, dict):
            raise ValueError("robot_observer must expose a `contract` dictionary.")

        self.lowdim_specs = getattr(self.robot_observer, "lowdim_specs", None)
        if not isinstance(self.lowdim_specs, dict) or not self.lowdim_specs:
            raise ValueError("robot_observer must expose non-empty `lowdim_specs`.")

        self.camera_specs = getattr(self.camera_observer, "camera_specs", None)
        if not isinstance(self.camera_specs, dict) or not self.camera_specs:
            raise ValueError("camera_observer must expose non-empty `camera_specs`.")

        self.lowdim_fps = self._resolve_fps("lowdim", getattr(self.robot_observer, "obs_freq", None))
        self.camera_fps = self._resolve_fps("visual", getattr(self.camera_observer, "camera_freq", None))
        self.lowdim_poll_period_s = 1.0 / self.lowdim_fps
        self.camera_poll_period_s = 1.0 / self.camera_fps

        self._state_lock = threading.Lock()
        self._worker_error_lock = threading.Lock()
        self._worker_error: Exception | None = None
        self._stop_event = threading.Event()

        self._recording = False
        self._lowdim_thread: threading.Thread | None = None
        self._camera_thread: threading.Thread | None = None

        self._episode_id: int | None = None
        self._episode_dir: Path | None = None
        self._visual_dir: Path | None = None
        self._start_timestamp_s: float | None = None
        self._stop_timestamp_s: float | None = None
        self._last_lowdim_timestamp_s: float | None = None
        self._last_camera_timestamp_s: float | None = None
        self._last_completed_episode_summary: dict[str, Any] | None = None
        self._last_camera_source_marker: dict[str, int | float | None] = {}

        self._lowdim_records: dict[str, list[Any]] = {}
        self._camera_timestamps: dict[str, list[float]] = {}
        self._camera_frame_counts: dict[str, int] = {}
        self._camera_duplicate_frame_counts: dict[str, int] = {}
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._video_paths: dict[str, Path] = {}

    def start(self) -> float:
        with self._state_lock:
            if self._recording:
                raise RuntimeError("Saver is already recording an episode.")

            self._worker_error = None
            self._stop_event.clear()
            self._start_timestamp_s = time.time()
            self._stop_timestamp_s = None
            self._last_lowdim_timestamp_s = self._start_timestamp_s
            self._last_camera_timestamp_s = self._start_timestamp_s
            self._last_completed_episode_summary = None

            self._episode_id = self._next_episode_id()
            self._episode_dir = self.save_dir / f"episode_{self._episode_id:06d}"
            self._visual_dir = self._episode_dir / "visual"
            self._episode_dir.mkdir(parents=True, exist_ok=False)
            self._visual_dir.mkdir(parents=True, exist_ok=False)

            self._initialize_episode_storage()
            self._write_contract_snapshot()
            self._initialize_video_writers()

            self._lowdim_thread = threading.Thread(
                target=self._lowdim_worker_loop,
                name="saver-lowdim",
                daemon=True,
            )
            self._camera_thread = threading.Thread(
                target=self._camera_worker_loop,
                name="saver-camera",
                daemon=True,
            )

            self._recording = True
            self._lowdim_thread.start()
            self._camera_thread.start()

            return self._start_timestamp_s

    def stop(self):
        with self._state_lock:
            if not self._recording:
                raise RuntimeError("Saver is not currently recording an episode.")

            stop_timestamp_s = time.time()
            self._stop_timestamp_s = stop_timestamp_s
            self._stop_event.set()
            lowdim_thread = self._lowdim_thread
            camera_thread = self._camera_thread

        if lowdim_thread is not None:
            lowdim_thread.join(timeout=2.0)
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)

        pending_error: Exception | None = None

        try:
            self._drain_lowdim_samples(cutoff_timestamp_s=stop_timestamp_s)
            self._drain_camera_samples(cutoff_timestamp_s=stop_timestamp_s)
        except Exception as exc:
            pending_error = exc

        try:
            self._finalize_episode(stop_timestamp_s)
        except Exception as exc:
            if pending_error is None:
                pending_error = exc

        worker_error = self._consume_worker_error()
        if pending_error is None and worker_error is not None:
            pending_error = worker_error

        if pending_error is not None:
            raise pending_error

    def quit(self):
        with self._state_lock:
            is_recording = self._recording

        if is_recording:
            self.stop()

    @property
    def current_episode_name(self) -> str | None:
        if self._episode_dir is None:
            return None
        return self._episode_dir.name

    @property
    def current_episode_path(self) -> Path | None:
        return self._episode_dir

    @property
    def last_completed_episode_summary(self) -> dict[str, Any] | None:
        if self._last_completed_episode_summary is None:
            return None
        return dict(self._last_completed_episode_summary)

    def _lowdim_worker_loop(self) -> None:
        self._worker_loop(
            loop_name="lowdim",
            period_s=self.lowdim_poll_period_s,
            drain_fn=self._drain_lowdim_samples,
        )

    def _camera_worker_loop(self) -> None:
        self._worker_loop(
            loop_name="camera",
            period_s=self.camera_poll_period_s,
            drain_fn=self._drain_camera_samples,
        )

    def _worker_loop(self, loop_name: str, period_s: float, drain_fn) -> None:
        next_poll_time = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                drain_fn()
            except Exception as exc:
                self._record_worker_error(RuntimeError(f"{loop_name} saver worker failed: {exc}"))
                return

            next_poll_time += period_s
            sleep_duration = next_poll_time - time.perf_counter()
            if sleep_duration > 0.0:
                self._stop_event.wait(sleep_duration)
            else:
                next_poll_time = time.perf_counter()

    def _drain_lowdim_samples(self, cutoff_timestamp_s: float | None = None) -> None:
        if self._last_lowdim_timestamp_s is None:
            return

        snapshot = self._snapshot_observer_buffer(self.robot_observer)
        for sample in snapshot:
            timestamp_s = self._extract_timestamp(sample)
            if timestamp_s <= self._last_lowdim_timestamp_s:
                continue
            if cutoff_timestamp_s is not None and timestamp_s > cutoff_timestamp_s:
                continue

            self._lowdim_records["timestamp_s"].append(timestamp_s)
            for lowdim_name in self.lowdim_specs:
                if lowdim_name not in sample:
                    raise KeyError(f"Lowdim observation is missing key '{lowdim_name}'.")
                self._lowdim_records[lowdim_name].append(np.asarray(sample[lowdim_name]).copy())

            self._last_lowdim_timestamp_s = timestamp_s

    def _drain_camera_samples(self, cutoff_timestamp_s: float | None = None) -> None:
        if self._last_camera_timestamp_s is None:
            return

        snapshot = self._snapshot_observer_buffer(self.camera_observer)
        for sample in snapshot:
            timestamp_s = self._extract_timestamp(sample)
            if timestamp_s <= self._last_camera_timestamp_s:
                continue
            if cutoff_timestamp_s is not None and timestamp_s > cutoff_timestamp_s:
                continue

            for camera_name, camera_spec in self.camera_specs.items():
                if camera_name not in sample:
                    raise KeyError(f"Camera observation is missing key '{camera_name}'.")

                source_marker = self._extract_camera_source_marker(sample, camera_name)
                last_source_marker = self._last_camera_source_marker.get(camera_name)
                if (
                    source_marker is not None
                    and last_source_marker is not None
                    and source_marker <= last_source_marker
                ):
                    self._camera_duplicate_frame_counts[camera_name] += 1
                if source_marker is not None and (
                    last_source_marker is None or source_marker > last_source_marker
                ):
                    self._last_camera_source_marker[camera_name] = source_marker

                frame = self._prepare_frame_for_video(camera_name, sample[camera_name], camera_spec)
                self._video_writers[camera_name].write(frame)
                self._camera_timestamps[camera_name].append(timestamp_s)
                self._camera_frame_counts[camera_name] += 1

            self._last_camera_timestamp_s = timestamp_s

    def _initialize_episode_storage(self) -> None:
        self._lowdim_records = {"timestamp_s": []}
        for lowdim_name in self.lowdim_specs:
            self._lowdim_records[lowdim_name] = []

        self._camera_timestamps = {camera_name: [] for camera_name in self.camera_specs}
        self._camera_frame_counts = {camera_name: 0 for camera_name in self.camera_specs}
        self._camera_duplicate_frame_counts = {camera_name: 0 for camera_name in self.camera_specs}
        self._last_camera_source_marker = {camera_name: None for camera_name in self.camera_specs}
        self._video_writers = {}
        self._video_paths = {}

    def _write_contract_snapshot(self) -> None:
        if self._episode_dir is None:
            raise RuntimeError("Episode directory is not initialized.")

        destination_path = self._episode_dir / "contract.yaml"
        if self.contract_path is not None:
            shutil.copy2(self.contract_path, destination_path)
            return

        with destination_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.contract, handle, sort_keys=False)

    def _initialize_video_writers(self) -> None:
        if self._visual_dir is None:
            raise RuntimeError("Visual directory is not initialized.")

        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        for camera_name, camera_spec in self.camera_specs.items():
            width, height = self._resolve_image_size(camera_spec.get("dim"))
            fps = float(camera_spec.get("fps") or self.camera_fps)
            if fps <= 0.0:
                raise ValueError(f"Camera fps for '{camera_name}' must be positive.")

            video_path = self._visual_dir / f"{camera_name}.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for '{camera_name}'.")

            self._video_writers[camera_name] = writer
            self._video_paths[camera_name] = video_path

    def _finalize_episode(self, stop_timestamp_s: float) -> None:
        try:
            self._release_video_writers()
            self._remove_empty_videos()
            self._write_lowdim_archive()
            self._write_camera_timestamps()
            self._last_completed_episode_summary = self._write_metadata(stop_timestamp_s)
        finally:
            with self._state_lock:
                self._recording = False
                self._lowdim_thread = None
                self._camera_thread = None
                self._stop_event.clear()

    def _release_video_writers(self) -> None:
        for writer in self._video_writers.values():
            writer.release()

    def _remove_empty_videos(self) -> None:
        for camera_name, video_path in self._video_paths.items():
            if self._camera_frame_counts.get(camera_name, 0) != 0:
                continue
            if video_path.exists():
                video_path.unlink()

    def _write_lowdim_archive(self) -> None:
        if self._episode_dir is None:
            raise RuntimeError("Episode directory is not initialized.")

        archive_path = self._episode_dir / "lowdim.npz"
        payload: dict[str, np.ndarray] = {
            "timestamp_s": np.asarray(self._lowdim_records["timestamp_s"], dtype=np.float64)
        }

        for lowdim_name, lowdim_spec in self.lowdim_specs.items():
            payload[lowdim_name] = self._to_numpy_array(
                self._lowdim_records[lowdim_name],
                lowdim_spec.get("dim"),
            )

        np.savez(archive_path, **payload)

    def _write_camera_timestamps(self) -> None:
        if self._visual_dir is None:
            raise RuntimeError("Visual directory is not initialized.")

        for camera_name, timestamps in self._camera_timestamps.items():
            timestamp_path = self._visual_dir / f"{camera_name}_timestamps.npy"
            np.save(timestamp_path, np.asarray(timestamps, dtype=np.float64))

    def _write_metadata(self, stop_timestamp_s: float) -> dict[str, Any]:
        if self._episode_dir is None or self._episode_id is None or self._start_timestamp_s is None:
            raise RuntimeError("Episode metadata cannot be written before recording starts.")

        metadata = {
            "episode_id": self._episode_id,
            "episode_name": self._episode_dir.name,
            "episode_path": str(self._episode_dir),
            "start_timestamp_s": self._start_timestamp_s,
            "end_timestamp_s": stop_timestamp_s,
            "duration_s": stop_timestamp_s - self._start_timestamp_s,
            "num_lowdim_samples": len(self._lowdim_records["timestamp_s"]),
            "camera_frame_counts": dict(self._camera_frame_counts),
            "camera_duplicate_frame_counts": dict(self._camera_duplicate_frame_counts),
            "lowdim_keys": list(self.lowdim_specs.keys()),
            "camera_keys": list(self.camera_specs.keys()),
            "lowdim_fps": self.lowdim_fps,
            "camera_fps": self.camera_fps,
        }

        metadata_path = self._episode_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        return metadata

    def _record_worker_error(self, exc: Exception) -> None:
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = exc
        self._stop_event.set()

    def _consume_worker_error(self) -> Exception | None:
        with self._worker_error_lock:
            error = self._worker_error
            self._worker_error = None
            return error

    @staticmethod
    def _snapshot_observer_buffer(observer) -> list[dict[str, Any]]:
        with observer._lock:
            return list(observer.buffer)

    @staticmethod
    def _extract_timestamp(sample: dict[str, Any]) -> float:
        if "timestamp_s" not in sample:
            raise KeyError("Observation sample is missing `timestamp_s`.")
        return float(sample["timestamp_s"])

    @staticmethod
    def _extract_camera_source_marker(sample: dict[str, Any], camera_name: str) -> int | float | None:
        frame_seqs = sample.get("camera_frame_seqs")
        if isinstance(frame_seqs, dict) and camera_name in frame_seqs:
            return int(frame_seqs[camera_name])

        source_timestamps = sample.get("camera_source_timestamps")
        if isinstance(source_timestamps, dict) and camera_name in source_timestamps:
            return float(source_timestamps[camera_name])

        return None

    def _resolve_fps(self, source_name: str, fallback_fps: Any) -> float:
        if fallback_fps is not None:
            fps = float(fallback_fps)
            if fps > 0.0:
                return fps

        robot_cfg = self.contract.get("robot", {})
        source_cfg = robot_cfg.get("data_sources", {}).get(source_name, {})
        fps = float(source_cfg.get("fps", self.DEFAULT_FPS))
        if fps <= 0.0:
            raise ValueError(f"fps for `{source_name}` must be positive.")
        return fps

    def _next_episode_id(self) -> int:
        largest_episode_id = 0
        for path in self.save_dir.glob("episode_*"):
            if not path.is_dir():
                continue

            try:
                episode_id = int(path.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue

            largest_episode_id = max(largest_episode_id, episode_id)

        return largest_episode_id + 1

    @staticmethod
    def _resolve_image_size(dim: Any) -> tuple[int, int]:
        if not isinstance(dim, (list, tuple)) or len(dim) < 2:
            return 640, 480
        return int(dim[0]), int(dim[1])

    @staticmethod
    def _prepare_frame_for_video(camera_name: str, frame: Any, camera_spec: dict[str, Any]) -> np.ndarray:
        frame_array = np.asarray(frame)
        if frame_array.ndim != 3 or frame_array.shape[2] != 3:
            raise ValueError(
                f"Camera frame for '{camera_name}' must be HxWx3, got shape {frame_array.shape}."
            )

        if frame_array.dtype != np.uint8:
            frame_array = cv2.convertScaleAbs(frame_array)

        expected_width, expected_height = Saver._resolve_image_size(camera_spec.get("dim"))
        if frame_array.shape[1] != expected_width or frame_array.shape[0] != expected_height:
            frame_array = cv2.resize(frame_array, (expected_width, expected_height))

        return np.ascontiguousarray(frame_array)

    @staticmethod
    def _to_numpy_array(values: list[Any], dim: Any) -> np.ndarray:
        if not values:
            if isinstance(dim, (list, tuple)) and dim:
                shape = (0, *[int(axis) for axis in dim])
                return np.empty(shape, dtype=np.float64)
            return np.asarray([], dtype=np.float64)

        try:
            arrays = [np.asarray(value) for value in values]
            return np.stack(arrays, axis=0)
        except Exception:
            return np.asarray(values, dtype=object)
