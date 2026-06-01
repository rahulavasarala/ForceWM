from __future__ import annotations
import json
import shutil
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml
class Saver:
    DEFAULT_FPS = 30.0
    DEFAULT_TASK_BODY_NAME = "task"
    def __init__(self, save_dir, robot_observer, camera_observer, contract_path=None, video_codec="mp4v"):
        if robot_observer is None or camera_observer is None:
            raise ValueError("Both robot_observer and camera_observer must be provided.")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.robot_observer = robot_observer
        self.camera_observer = camera_observer
        self.contract_path = Path(contract_path) if contract_path is not None else None
        self.video_codec = str(video_codec)
        self.contract = getattr(robot_observer, "contract", None)
        self.lowdim_specs = getattr(robot_observer, "lowdim_specs", None)
        self.camera_specs = getattr(camera_observer, "camera_specs", None)
        if not isinstance(self.contract, dict):
            raise ValueError("robot_observer must expose a `contract` dictionary.")
        if not isinstance(self.lowdim_specs, dict) or not self.lowdim_specs:
            raise ValueError("robot_observer must expose non-empty `lowdim_specs`.")
        if not isinstance(self.camera_specs, dict) or not self.camera_specs:
            raise ValueError("camera_observer must expose non-empty `camera_specs`.")
        self.rgb_camera_specs = {name: spec for name, spec in self.camera_specs.items() if not self._is_depth_camera(spec)}
        self.depth_camera_specs = {name: spec for name, spec in self.camera_specs.items() if self._is_depth_camera(spec)}
        if len(self.depth_camera_specs) > 1:
            raise ValueError("Saver currently supports at most one depth camera.")
        for name, spec in self.depth_camera_specs.items():
            if spec.get("align_to") not in self.rgb_camera_specs:
                raise ValueError(f"Depth camera '{name}' must align to a known RGB camera.")
        self.lowdim_fps = self._resolve_fps("lowdim", getattr(robot_observer, "obs_freq", None))
        self.camera_fps = self._resolve_fps("visual", getattr(camera_observer, "camera_freq", None))
        if abs(self.lowdim_fps - self.camera_fps) > 1e-6:
            raise ValueError("Saver requires matching lowdim and visual fps for unified ticking.")
        self.poll_period_s = 1.0 / self.lowdim_fps
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._recording = False
        self._worker_error: Exception | None = None
        self._episode_id: int | None = None
        self._episode_dir: Path | None = None
        self._visual_dir: Path | None = None
        self._depth_frames_dir: Path | None = None
        self._start_timestamp_s: float | None = None
        self._last_completed_episode_summary: dict[str, Any] | None = None
        self._lowdim_records: dict[str, list[Any]] = {}
        self._camera_timestamps: dict[str, list[float]] = {}
        self._camera_frame_counts: dict[str, int] = {}
        self._camera_duplicate_frame_counts: dict[str, int] = {}
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._video_paths: dict[str, Path] = {}
        self._depth_frame_indices: dict[str, int] = {}
        self._last_lowdim_marker: int | None = None
        self._last_camera_markers: dict[str, int | None] = {}
        self._lowdim_timestamp_domain = "sim_time_s" if getattr(robot_observer, "source_type", "") == "sim" else "wall_time_s"
        self._camera_timestamp_domains = {
            name: "sim_time_s" if spec.get("source_type") == "sim" else "wall_time_s"
            for name, spec in self.camera_specs.items()
        }
        self._part_metadata = self._load_part_metadata()
    def start(self) -> float:
        with self._state_lock:
            if self._recording:
                raise RuntimeError("Saver is already recording an episode.")
            self._worker_error = None
            self._stop_event.clear()
            self._start_timestamp_s = time.time()
            self._last_completed_episode_summary = None
            self._episode_id = self._next_episode_id()
            self._episode_dir = self.save_dir / f"episode_{self._episode_id:06d}"
            self._visual_dir = self._episode_dir / "visual"
            self._episode_dir.mkdir(parents=True, exist_ok=False)
            self._visual_dir.mkdir(parents=True, exist_ok=False)
            self._depth_frames_dir = None
            if self.depth_camera_specs:
                self._depth_frames_dir = self._visual_dir / "depth" / "depth_frames"
                self._depth_frames_dir.mkdir(parents=True, exist_ok=False)
            self._reset_episode_storage()
            self._seed_saved_markers()
            contract_path = self._episode_dir / "contract.yaml"
            if self.contract_path is not None:
                shutil.copy2(self.contract_path, contract_path)
            else:
                with contract_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(self.contract, handle, sort_keys=False)
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            for camera_name, camera_spec in self.rgb_camera_specs.items():
                dim = camera_spec.get("dim")
                width, height = (640, 480) if not isinstance(dim, (list, tuple)) or len(dim) < 2 else (int(dim[0]), int(dim[1]))
                fps = float(camera_spec.get("fps") or self.camera_fps)
                video_path = self._visual_dir / f"{camera_name}.mp4"
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for '{camera_name}'.")
                self._video_writers[camera_name] = writer
                self._video_paths[camera_name] = video_path
            self._thread = threading.Thread(target=self._run_loop, name="saver", daemon=True)
            self._recording = True
            self._thread.start()
            return self._start_timestamp_s
    def stop(self):
        with self._state_lock:
            if not self._recording:
                raise RuntimeError("Saver is not currently recording an episode.")
            stop_timestamp_s = time.time()
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        pending_error = self._worker_error
        try:
            self._finalize_episode(stop_timestamp_s)
        except Exception as exc:
            pending_error = pending_error or exc
        if pending_error is not None:
            raise pending_error
    def quit(self):
        if self._recording:
            self.stop()
    current_episode_name = property(lambda self: None if self._episode_dir is None else self._episode_dir.name)
    current_episode_path = property(lambda self: self._episode_dir)
    last_completed_episode_summary = property(
        lambda self: None if self._last_completed_episode_summary is None else dict(self._last_completed_episode_summary)
    )
    def _run_loop(self) -> None:
        next_poll_time = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception as exc:
                self._worker_error = RuntimeError(f"saver worker failed: {exc}")
                self._stop_event.set()
                return
            next_poll_time += self.poll_period_s
            sleep_duration = next_poll_time - time.perf_counter()
            if sleep_duration > 0.0:
                self._stop_event.wait(sleep_duration)
            else:
                next_poll_time = time.perf_counter()
    def _tick_once(self) -> None:
        sample = self.robot_observer.get_latest_obs()
        if sample is not None:
            self._save_lowdim_sample(sample)
        sample = self.camera_observer.get_latest_obs()
        if sample is not None:
            self._save_camera_sample(sample)
    def _save_lowdim_sample(self, sample: dict[str, Any]) -> None:
        key = "source_seq" if getattr(self.robot_observer, "source_type", "") == "sim" else "observer_seq"
        if key not in sample:
            raise KeyError(f"Lowdim observation is missing `{key}`.")
        marker = int(sample[key])
        if self._last_lowdim_marker is not None and marker <= self._last_lowdim_marker:
            return
        self._lowdim_records["timestamp_s"].append(float(sample["timestamp_s"]))
        for lowdim_name in self.lowdim_specs:
            if lowdim_name not in sample:
                raise KeyError(f"Lowdim observation is missing key '{lowdim_name}'.")
            self._lowdim_records[lowdim_name].append(np.asarray(sample[lowdim_name]).copy())
        self._last_lowdim_marker = marker
    def _save_camera_sample(self, sample: dict[str, Any]) -> None:
        if "timestamp_s" not in sample:
            raise KeyError("Camera observation is missing `timestamp_s`.")
        timestamp_s = float(sample["timestamp_s"])
        for camera_name, camera_spec in self.camera_specs.items():
            if camera_name not in sample:
                raise KeyError(f"Camera observation is missing key '{camera_name}'.")
            if camera_spec.get("source_type") == "sim":
                frame_seqs = sample.get("camera_frame_seqs")
                if isinstance(frame_seqs, dict) and camera_name in frame_seqs:
                    marker = int(frame_seqs[camera_name])
                elif "camera_frame_seq" in sample:
                    marker = int(sample["camera_frame_seq"])
                else:
                    raise KeyError(f"Camera observation is missing a frame seq for '{camera_name}'.")
            else:
                if "observer_seq" not in sample:
                    raise KeyError("Camera observation is missing `observer_seq`.")
                marker = int(sample["observer_seq"])
            if self._last_camera_markers.get(camera_name) is not None and marker <= self._last_camera_markers[camera_name]:
                continue
            if self._is_depth_camera(camera_spec):
                self._write_depth_frame(camera_name, sample[camera_name], camera_spec)
            else:
                self._video_writers[camera_name].write(self._prepare_frame_for_video(camera_name, sample[camera_name], camera_spec))
            self._camera_timestamps[camera_name].append(timestamp_s)
            self._camera_frame_counts[camera_name] += 1
            self._last_camera_markers[camera_name] = marker
    def _reset_episode_storage(self) -> None:
        self._lowdim_records = {"timestamp_s": [], **{name: [] for name in self.lowdim_specs}}
        self._camera_timestamps = {name: [] for name in self.camera_specs}
        self._camera_frame_counts = {name: 0 for name in self.camera_specs}
        self._camera_duplicate_frame_counts = {name: 0 for name in self.camera_specs}
        self._video_writers = {}
        self._video_paths = {}
        self._depth_frame_indices = {name: 0 for name in self.depth_camera_specs}
        self._last_lowdim_marker = None
        self._last_camera_markers = {name: None for name in self.camera_specs}
    def _seed_saved_markers(self) -> None:
        sample = self.robot_observer.get_latest_obs()
        key = "source_seq" if getattr(self.robot_observer, "source_type", "") == "sim" else "observer_seq"
        if sample is not None and key in sample:
            self._last_lowdim_marker = int(sample[key])
        sample = self.camera_observer.get_latest_obs()
        if sample is None:
            return
        for camera_name, camera_spec in self.camera_specs.items():
            if camera_name not in sample:
                continue
            if camera_spec.get("source_type") == "sim":
                frame_seqs = sample.get("camera_frame_seqs")
                if isinstance(frame_seqs, dict) and camera_name in frame_seqs:
                    self._last_camera_markers[camera_name] = int(frame_seqs[camera_name])
                elif "camera_frame_seq" in sample:
                    self._last_camera_markers[camera_name] = int(sample["camera_frame_seq"])
            elif "observer_seq" in sample:
                self._last_camera_markers[camera_name] = int(sample["observer_seq"])
    def _finalize_episode(self, stop_timestamp_s: float) -> None:
        try:
            for writer in self._video_writers.values():
                writer.release()
            for camera_name, video_path in self._video_paths.items():
                if self._camera_frame_counts[camera_name] == 0 and video_path.exists():
                    video_path.unlink()
            if self._episode_dir is None or self._visual_dir is None or self._episode_id is None or self._start_timestamp_s is None:
                raise RuntimeError("Episode state is not initialized.")
            np.savez(
                self._episode_dir / "lowdim.npz",
                timestamp_s=np.asarray(self._lowdim_records["timestamp_s"], dtype=np.float64),
                **{name: self._to_numpy_array(self._lowdim_records[name], spec.get("dim")) for name, spec in self.lowdim_specs.items()},
            )
            for camera_name, timestamps in self._camera_timestamps.items():
                if not self._is_depth_camera(self.camera_specs[camera_name]):
                    np.save(self._visual_dir / f"{camera_name}_timestamps.npy", np.asarray(timestamps, dtype=np.float64))
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
                "lowdim_timestamp_domain": self._lowdim_timestamp_domain,
                "camera_timestamp_domains": dict(self._camera_timestamp_domains),
                "lowdim_source_time_range_s": self._timestamp_range(self._lowdim_records["timestamp_s"]),
                "camera_source_time_ranges_s": {name: self._timestamp_range(timestamps) for name, timestamps in self._camera_timestamps.items()},
            }
            if self._part_metadata is not None:
                metadata["part_metadata"] = dict(self._part_metadata)
            with (self._episode_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            self._last_completed_episode_summary = metadata
        finally:
            with self._state_lock:
                self._recording = False
                self._thread = None
                self._stop_event.clear()
    def _resolve_fps(self, source_name: str, fallback_fps: Any) -> float:
        if fallback_fps is not None and float(fallback_fps) > 0.0:
            return float(fallback_fps)
        fps = float(self.contract.get("robot", {}).get("data_sources", {}).get(source_name, {}).get("fps", self.DEFAULT_FPS))
        if fps <= 0.0:
            raise ValueError(f"fps for `{source_name}` must be positive.")
        return fps
    def _next_episode_id(self) -> int:
        largest_episode_id = 0
        for path in self.save_dir.glob("episode_*"):
            if path.is_dir():
                try:
                    largest_episode_id = max(largest_episode_id, int(path.name.split("_", maxsplit=1)[1]))
                except (IndexError, ValueError):
                    pass
        return largest_episode_id + 1
    @staticmethod
    def _prepare_frame_for_video(camera_name: str, frame: Any, camera_spec: dict[str, Any]) -> np.ndarray:
        frame_array = np.asarray(frame)
        if frame_array.ndim != 3 or frame_array.shape[2] != 3:
            raise ValueError(f"Camera frame for '{camera_name}' must be HxWx3, got shape {frame_array.shape}.")
        if frame_array.dtype != np.uint8:
            frame_array = cv2.convertScaleAbs(frame_array)
        dim = camera_spec.get("dim")
        width, height = (640, 480) if not isinstance(dim, (list, tuple)) or len(dim) < 2 else (int(dim[0]), int(dim[1]))
        if frame_array.shape[1] != width or frame_array.shape[0] != height:
            frame_array = cv2.resize(frame_array, (width, height))
        return np.ascontiguousarray(frame_array)
    def _write_depth_frame(self, camera_name: str, frame: Any, camera_spec: dict[str, Any]) -> None:
        if self._depth_frames_dir is None:
            raise RuntimeError("Depth frame directory is not initialized.")
        frame_array = np.asarray(frame)
        if frame_array.ndim != 2:
            raise ValueError(f"Depth frame for '{camera_name}' must be HxW, got shape {frame_array.shape}.")
        if frame_array.dtype != np.uint16:
            frame_array = np.clip(np.rint(frame_array), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        dim = camera_spec.get("dim")
        width, height = (640, 480) if not isinstance(dim, (list, tuple)) or len(dim) < 2 else (int(dim[0]), int(dim[1]))
        if frame_array.shape[1] != width or frame_array.shape[0] != height:
            frame_array = cv2.resize(frame_array, (width, height), interpolation=cv2.INTER_NEAREST)
        output_path = self._depth_frames_dir / f"frame_{self._depth_frame_indices[camera_name]:06d}.png"
        if not cv2.imwrite(str(output_path), np.ascontiguousarray(frame_array)):
            raise RuntimeError(f"Failed to write depth frame for '{camera_name}' to {output_path}.")
        self._depth_frame_indices[camera_name] += 1
    @staticmethod
    def _is_depth_camera(camera_spec: dict[str, Any]) -> bool:
        return str(camera_spec.get("type", "rgb")).lower() == "depth"
    @staticmethod
    def _to_numpy_array(values: list[Any], dim: Any) -> np.ndarray:
        if not values:
            return np.empty((0, *[int(axis) for axis in dim]), dtype=np.float64) if dim else np.asarray([], dtype=np.float64)
        try:
            return np.stack([np.asarray(value) for value in values], axis=0)
        except Exception:
            return np.asarray(values, dtype=object)
    @staticmethod
    def _timestamp_range(values: list[float]) -> dict[str, float] | None:
        return None if not values else {"start": float(values[0]), "end": float(values[-1])}
    def _load_part_metadata(self) -> dict[str, Any] | None:
        scene_xml_path = self._resolve_scene_xml_path()
        if scene_xml_path is None:
            return None
        metadata: dict[str, Any] = {
            "scene_xml_path": str(scene_xml_path),
            "task_body_name": self.DEFAULT_TASK_BODY_NAME,
        }
        try:
            metadata.update(self._infer_part_metadata(scene_xml_path, body_name=self.DEFAULT_TASK_BODY_NAME))
        except Exception as exc:
            metadata["inference_error"] = str(exc)
        return metadata
    def _resolve_scene_xml_path(self) -> Path | None:
        robot_cfg = self.contract.get("robot")
        if not isinstance(robot_cfg, dict):
            return None
        raw_xml_path = robot_cfg.get("xml_path")
        if not isinstance(raw_xml_path, str) or not raw_xml_path.strip():
            return None
        candidate_path = Path(raw_xml_path).expanduser()
        if candidate_path.is_absolute():
            return candidate_path.resolve()
        base_dir = self.contract_path.parent if self.contract_path is not None else Path(__file__).resolve().parents[1]
        return (base_dir / candidate_path).resolve()
    @classmethod
    def _infer_part_metadata(cls, scene_xml_path: Path, body_name: str) -> dict[str, Any]:
        search_paths = cls._collect_xml_search_paths(scene_xml_path)
        for xml_path in search_paths:
            if not xml_path.exists():
                continue
            xml_root = ET.parse(xml_path).getroot()
            body_element = cls._find_body_element(xml_root, body_name)
            if body_element is None:
                continue
            mesh_paths = cls._resolve_mesh_paths(xml_root, xml_path)
            part_name, asset_root = cls._infer_part_identity(xml_path, mesh_paths)
            part_metadata = {
                "part_name": part_name,
                "part_xml_path": str(xml_path),
                "part_position": cls._parse_vector(
                    raw_value=body_element.attrib.get("pos", "0 0 0"),
                    expected_dim=3,
                    field_name="pos",
                    xml_path=xml_path,
                ),
            }
            if asset_root is not None:
                part_metadata["part_asset_root"] = str(asset_root)
            raw_quat = body_element.attrib.get("quat")
            if raw_quat is not None:
                part_metadata["part_orientation_quat"] = cls._parse_vector(
                    raw_value=raw_quat,
                    expected_dim=4,
                    field_name="quat",
                    xml_path=xml_path,
                )
            return part_metadata
        raise ValueError(f"Could not find body `{body_name}` in {scene_xml_path} or its included XML files.")
    @classmethod
    def _collect_xml_search_paths(cls, scene_xml_path: Path) -> list[Path]:
        pending_paths = [scene_xml_path.resolve()]
        search_paths: list[Path] = []
        visited_paths: set[Path] = set()
        while pending_paths:
            xml_path = pending_paths.pop(0)
            if xml_path in visited_paths:
                continue
            visited_paths.add(xml_path)
            search_paths.append(xml_path)
            if not xml_path.exists():
                continue
            xml_root = ET.parse(xml_path).getroot()
            for include_element in xml_root.iter("include"):
                raw_include_path = include_element.attrib.get("file")
                if not isinstance(raw_include_path, str) or not raw_include_path.strip():
                    continue
                pending_paths.append((xml_path.parent / raw_include_path).resolve())
        return search_paths
    @staticmethod
    def _find_body_element(xml_root: ET.Element, body_name: str) -> ET.Element | None:
        for body_element in xml_root.iter("body"):
            if body_element.attrib.get("name") == body_name:
                return body_element
        return None
    @staticmethod
    def _resolve_mesh_paths(xml_root: ET.Element, xml_path: Path) -> list[Path]:
        mesh_paths: list[Path] = []
        for mesh_element in xml_root.iter("mesh"):
            raw_mesh_path = mesh_element.attrib.get("file")
            if not isinstance(raw_mesh_path, str) or not raw_mesh_path.strip():
                continue
            mesh_paths.append((xml_path.parent / raw_mesh_path).resolve())
        return mesh_paths
    @staticmethod
    def _infer_part_identity(part_xml_path: Path, mesh_paths: list[Path]) -> tuple[str, Path | None]:
        for mesh_path in mesh_paths:
            mesh_parts = mesh_path.parts
            if "generated_cad" in mesh_parts:
                generated_cad_index = mesh_parts.index("generated_cad")
                if generated_cad_index + 1 < len(mesh_parts):
                    asset_root = Path(*mesh_parts[: generated_cad_index + 2])
                    return mesh_parts[generated_cad_index + 1], asset_root
        if mesh_paths:
            return part_xml_path.stem, mesh_paths[0].parent
        return part_xml_path.stem, None
    @staticmethod
    def _parse_vector(raw_value: str, expected_dim: int, field_name: str, xml_path: Path) -> list[float]:
        vector = np.fromstring(raw_value, sep=" ", dtype=np.float64)
        if vector.shape != (expected_dim,):
            raise ValueError(
                f"Expected `{field_name}` in {xml_path} to have {expected_dim} values, got `{raw_value}`."
            )
        return vector.tolist()
