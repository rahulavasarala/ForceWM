from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import redis
import yaml


def _default_contract_path() -> Path:
    return Path(__file__).resolve().parents[1] / "universal_contract.yaml"


class CameraObserver:
    def __init__(self, buffer_size, example_obs, camera_freq, robot_data):
        # creates a ring buffer of size buffer_size to store the last buffer_size observations
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")

        self.buffer_size = int(buffer_size)
        self.example_obs = example_obs
        self.contract = self._load_contract(robot_data)
        self.camera_specs = self._parse_camera_specs(self.contract)

        if len(self.camera_specs) == 0:
            raise ValueError("No visual keys were found in the contract.")

        self.camera_freq = float(camera_freq) if camera_freq is not None else self._resolve_default_camera_freq()
        if self.camera_freq <= 0.0:
            raise ValueError("camera_freq must be positive.")

        self.camera_period_s = 1.0 / self.camera_freq
        self.buffer = deque(maxlen=self.buffer_size)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=False)
        self._realsense_streams: dict[str, dict[str, Any]] = {}

    def get_last_k_obs(self, k):
        # create a dictionary that batches the observations in the ring buffer into a single dictionary
        # keyed by visual name, with time stacked along axis 0 when possible
        if k <= 0:
            raise ValueError("k must be positive.")

        with self._lock:
            recent_observations = list(self.buffer)[-k:]

        if not recent_observations:
            return {}

        batched: dict[str, Any] = {}
        batched["timestamp_s"] = np.asarray(
            [obs["timestamp_s"] for obs in recent_observations], dtype=np.float64
        )
        for visual_name in self.camera_specs:
            values = [obs[visual_name] for obs in recent_observations]
            batched[visual_name] = self._stack_or_list(values)

        return batched

    def start_adding_obs(self):
        # starts a thread that continuously adds observations at camera_freq to the ring buffer until stop_adding_obs is called
        if self._thread is not None and self._thread.is_alive():
            return

        self.redis_client.ping()
        self._start_realsense_streams()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._observation_loop, name="camera-observer", daemon=True)
        self._thread.start()

    def stop_adding_obs(self):
        # stops the thread that is adding observations to the ring buffer
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._stop_realsense_streams()

    def _observation_loop(self) -> None:
        next_poll_time = time.perf_counter()

        while not self._stop_event.is_set():
            observation = self._read_observation()
            observation["timestamp_s"] = time.time()
            with self._lock:
                self.buffer.append(observation)

            next_poll_time += self.camera_period_s
            sleep_duration = next_poll_time - time.perf_counter()
            if sleep_duration > 0.0:
                self._stop_event.wait(sleep_duration)
            else:
                next_poll_time = time.perf_counter()

    def _read_observation(self) -> dict[str, Any]:
        observation: dict[str, Any] = {}

        for visual_name, visual_spec in self.camera_specs.items():
            source_type = visual_spec["source_type"]
            if source_type == "sim":
                redis_bytes = self.redis_client.get(visual_spec["redis_key"])
                frame = cv2.imdecode(np.frombuffer(redis_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Failed to decode sim camera frame for '{visual_name}'.")
                observation[visual_name] = frame
            elif source_type == "realsense":
                observation[visual_name] = self._read_realsense_frame(visual_name)
            else:
                observation[visual_name] = np.array(
                    json.loads(self.redis_client.get(visual_spec["redis_key"]))
                )

        return observation

    def _start_realsense_streams(self) -> None:
        realsense_specs = [
            (visual_name, visual_spec)
            for visual_name, visual_spec in self.camera_specs.items()
            if visual_spec["source_type"] == "realsense"
        ]

        if not realsense_specs:
            return

        import pyrealsense2 as rs

        for visual_name, visual_spec in realsense_specs:
            pipeline = rs.pipeline()
            config = rs.config()

            width, height = self._resolve_image_size(visual_spec.get("dim"))
            fps = int(visual_spec.get("fps", self.camera_freq))
            serial_number = visual_spec.get("serial_number")
            if serial_number is not None:
                config.enable_device(str(serial_number))

            visual_type = str(visual_spec.get("type", "rgb")).lower()
            if visual_type == "depth":
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                stream_name = "depth"
            else:
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                stream_name = "color"

            pipeline.start(config)
            self._realsense_streams[visual_name] = {
                "pipeline": pipeline,
                "stream_name": stream_name,
            }

    def _stop_realsense_streams(self) -> None:
        for stream in self._realsense_streams.values():
            stream["pipeline"].stop()
        self._realsense_streams.clear()

    def _read_realsense_frame(self, visual_name: str) -> np.ndarray:
        stream = self._realsense_streams.get(visual_name)
        if stream is None:
            raise RuntimeError(
                f"Realsense stream for '{visual_name}' is not initialized. "
                "Call start_adding_obs() before reading observations."
            )

        frames = stream["pipeline"].wait_for_frames()
        if stream["stream_name"] == "depth":
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError(f"Failed to get depth frame for '{visual_name}'.")
            return np.asanyarray(depth_frame.get_data()).copy()

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"Failed to get color frame for '{visual_name}'.")
        return np.asanyarray(color_frame.get_data()).copy()

    def _resolve_default_camera_freq(self) -> float:
        robot_cfg = self.contract.get("robot", {})
        visual_cfg = robot_cfg.get("data_sources", {}).get("visual", {})
        fps = visual_cfg.get("fps")

        if fps is None:
            raise ValueError("camera_freq was not provided and no visual fps was found in the contract.")
        return float(fps)

    @staticmethod
    def _load_contract(robot_data: Any) -> dict[str, Any]:
        if isinstance(robot_data, dict):
            if "contract" in robot_data:
                robot_data = robot_data["contract"]
            if isinstance(robot_data, dict) and "robot" in robot_data:
                return robot_data
            if isinstance(robot_data, dict):
                return {"robot": robot_data}

        if robot_data is None:
            contract_path = _default_contract_path()
        else:
            contract_path = Path(robot_data)

        with contract_path.open("r", encoding="utf-8") as handle:
            contract = yaml.safe_load(handle)

        if not isinstance(contract, dict):
            raise ValueError("The universal contract must load as a dictionary.")

        return contract

    @staticmethod
    def _parse_camera_specs(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
        robot_cfg = contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("The universal contract must contain a top-level `robot` mapping.")

        prefix = robot_cfg.get("prefix")
        if prefix is None:
            raise ValueError("The robot is missing a prefix in the universal contract.")

        visual_block = robot_cfg.get("data_sources", {}).get("visual", {})
        robot_type = str(robot_cfg.get("type", "")).lower()
        visual_fps = visual_block.get("fps")
        visual_keys = visual_block.get("keys", [])

        parsed_visual: dict[str, dict[str, Any]] = {}
        for visual_entry in visual_keys:
            if not isinstance(visual_entry, dict) or len(visual_entry) != 1:
                continue

            visual_name, visual_cfg = next(iter(visual_entry.items()))
            visual_type = str(visual_cfg.get("type", "rgb")).lower()
            source_type = str(
                visual_cfg.get(
                    "source",
                    "realsense" if visual_type == "realsense" else ("sim" if robot_type == "sim" else "redis"),
                )
            ).lower()
            redis_suffix = visual_cfg.get("redis", visual_name)

            if visual_type == "realsense":
                source_type = "realsense"

            parsed_visual[visual_name] = {
                "source_type": source_type,
                "type": visual_type,
                "redis_key": CameraObserver._make_redis_key(prefix, redis_suffix),
                "dim": visual_cfg.get("dim"),
                "serial_number": visual_cfg.get("serial_number"),
                "fps": visual_cfg.get("fps", visual_fps),
            }

        return parsed_visual

    @staticmethod
    def _resolve_image_size(dim: list[int] | None) -> tuple[int, int]:
        if dim is None or len(dim) < 2:
            return 640, 480
        return int(dim[0]), int(dim[1])

    @staticmethod
    def _make_redis_key(prefix: str, suffix: str) -> str:
        return f"{prefix.rstrip(':')}::{suffix.lstrip(':')}"

    @staticmethod
    def _stack_or_list(values: list[Any]) -> Any:
        try:
            arrays = [np.asarray(value) for value in values]
            return np.stack(arrays, axis=0)
        except Exception:
            return values
