from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import redis
import torch
import yaml
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __package__:
    from .camera_observer import CameraObserver
    from .robot_observer import RobotObserver
else:
    from camera_observer import CameraObserver
    from robot_observer import RobotObserver

from high_level_controller.interpolator import InterpolatorFault, TrajectoryInterpolator
from training.act import ACT


DEFAULT_INTERPOLATOR_FREQUENCY_HZ = 100.0
DEFAULT_BLEND_DURATION = 0.1
DEFAULT_FPS = 30.0
DESIRED_POSITION_SUFFIX = "desired_cartesian_position"
DESIRED_ORIENTATION_SUFFIX = "desired_cartesian_orientation"
WAIT_POLL_PERIOD_S = 0.1


class Inference:
    def __init__(
        self,
        universal_contract: str,
        dataset_path: str,
        checkpoint_path: str,
        act_config: str,
        action_frequency_hz: float,
        num_action_steps: int,
        *,
        interpolator_frequency_hz: float = DEFAULT_INTERPOLATOR_FREQUENCY_HZ,
        blend_duration: float = DEFAULT_BLEND_DURATION,
        device: str | None = None,
    ) -> None:
        self.contract_path = Path(universal_contract).expanduser().resolve()
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.act_config_path = Path(act_config).expanduser().resolve()

        self.contract = self._load_contract(self.contract_path)
        self.lowdim_cfg = self._require_source_cfg("lowdim")
        self.visual_cfg = self._require_source_cfg("visual")
        self.lowdim_fps = self._resolve_fps(self.lowdim_cfg, "lowdim")
        self.camera_fps = self._resolve_fps(self.visual_cfg, "visual")
        self.lowdim_buffer_size = self._resolve_buffer_size(self.lowdim_cfg, "lowdim")
        self.camera_buffer_size = self._resolve_buffer_size(self.visual_cfg, "visual")

        self.lowdim_specs = self._parse_source_specs(self.lowdim_cfg)
        self.visual_specs = self._parse_source_specs(self.visual_cfg)
        self.primary_camera_key = next(iter(self.visual_specs))

        self.obs_window = int(self.lowdim_specs["eef_pos"]["obs_window"])
        self.obs_dss = int(self.lowdim_specs["eef_pos"]["obs_dss"])
        self.lowdim_history_len = self._resolve_history_length(self.lowdim_specs)
        self.visual_history_len = self._resolve_history_length(self.visual_specs)

        self.action_frequency_hz = float(action_frequency_hz)
        self.num_action_steps = int(num_action_steps)
        self.interpolator_frequency_hz = float(interpolator_frequency_hz)
        self.blend_duration = float(blend_duration)
        self.device = self._resolve_device(device)

        if self.action_frequency_hz <= 0.0:
            raise ValueError("action_frequency_hz must be positive.")
        if self.num_action_steps <= 0:
            raise ValueError("num_action_steps must be positive.")
        if self.num_action_steps > self.chunk_size:
            raise ValueError("num_action_steps must be <= robot.action.window.")
        if self.interpolator_frequency_hz <= 0.0:
            raise ValueError("interpolator_frequency_hz must be positive.")
        if self.blend_duration < 0.0:
            raise ValueError("blend_duration must be non-negative.")

        normalizer_path = self.dataset_path / "normalizer.npy"
        if not normalizer_path.exists():
            raise FileNotFoundError(f"Missing normalizer file: {normalizer_path}")
        self.normalizer = np.load(normalizer_path, allow_pickle=True).item()

        self.model = self._load_model()
        self.chunk_size = int(self.model.config.chunk_size)
        self.desired_position_key, self.desired_orientation_key = self._make_desired_pose_keys(self.contract)

        self.robot_observer: RobotObserver | None = None
        self.camera_observer: CameraObserver | None = None

        self.redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=False)
        self.interpolator = TrajectoryInterpolator(
            self.redis_client,
            self.desired_position_key,
            self.desired_orientation_key,
            publish_rate_hz=self.interpolator_frequency_hz,
            blend_duration=self.blend_duration,
        )

        self._shutdown_event = threading.Event()
        self._inference_enabled = threading.Event()
        self._keyboard_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._terminal_settings: Any = None
        self._stdin_fd: int | None = None
        self._background_error: BaseException | None = None
        self._background_error_lock = threading.Lock()

    def run(self) -> None:
        try:
            self._launch_observers()
            self._wait_for_observers()
            self.interpolator.start()
            self._start_keyboard_listener()
            self._start_inference_thread()

            print(f"Inference ready with contract {self.contract_path}", flush=True)
            print(f"Dataset normalizer: {self.dataset_path / 'normalizer.npy'}", flush=True)
            print(f"Checkpoint: {self.checkpoint_path}", flush=True)
            print(
                f"Desired pose keys: position=`{self.desired_position_key}` orientation=`{self.desired_orientation_key}`",
                flush=True,
            )
            print("Press `k` to start inference, `l` to stop future inference, Ctrl-C to exit.", flush=True)

            while not self._shutdown_event.is_set():
                error = self._get_background_error()
                if error is not None:
                    raise error
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nShutting down inference.", flush=True)
        finally:
            self._shutdown_event.set()
            self._inference_enabled.clear()
            self.interpolator.stop()
            self._join_background_threads()
            self._stop_observers()
            self._restore_terminal_settings()

            error = self._get_background_error()
            if error is not None:
                raise error

    def get_obs(self, last_k: int) -> dict[str, dict[str, torch.Tensor]]:
        if self.robot_observer is None or self.camera_observer is None:
            raise RuntimeError("Observers must be launched before building observations.")

        robot_buffer = self.robot_observer.get_last_k_obs(max(last_k, self.lowdim_history_len))
        camera_buffer = self.camera_observer.get_last_k_obs(max(last_k, self.visual_history_len))
        if not robot_buffer or not camera_buffer:
            raise RuntimeError("Observer buffers are empty.")

        obs_indices = self._observation_indices(max(len(robot_buffer["timestamp_s"]), 1))
        camera_indices = self._observation_indices(max(len(camera_buffer["timestamp_s"]), 1))

        eef_pos = np.asarray(robot_buffer["eef_pos"], dtype=np.float32)[obs_indices]
        eef_ori_mats = np.asarray(robot_buffer["eef_ori"], dtype=np.float32)[obs_indices].reshape(-1, 3, 3)
        eef_ori = R.from_matrix(eef_ori_mats).as_quat().astype(np.float32)

        images = np.asarray(camera_buffer[self.primary_camera_key], dtype=np.float32)[camera_indices]
        if images.ndim != 4:
            raise ValueError(f"Expected camera buffer with shape (T, H, W, C), got {images.shape}.")
        images = np.ascontiguousarray(images.transpose(0, 3, 1, 2)) / 255.0

        obs_dict = {
            "obs": {
                "eef_pos": torch.from_numpy(self._normalize_lowdim("eef_pos", eef_pos)),
                "eef_ori": torch.from_numpy(self._normalize_lowdim("eef_ori", eef_ori)),
                "images": torch.from_numpy(self._normalize_images(images)),
            }
        }
        return obs_dict

    def _inference_loop(self) -> None:
        while not self._shutdown_event.is_set():
            if not self._inference_enabled.is_set():
                if self._shutdown_event.wait(0.1):
                    return
                continue

            try:
                inference_start_time = time.monotonic()
                obs_dict = self.get_obs(last_k=max(self.lowdim_history_len, self.visual_history_len))
                with torch.inference_mode():
                    output, _ = self.model(obs_dict)
                action_chunk = self._denormalize_actions(output[0].detach().cpu().numpy())
                ts = inference_start_time + np.arange(self.chunk_size, dtype=np.float64) * (
                    1.0 / self.action_frequency_hz
                )
                self.interpolator.enqueue_chunk(action_chunk, ts)
            except InterpolatorFault as exc:
                self._record_background_error(exc)
                return
            except Exception as exc:
                self._record_background_error(exc)
                return

            wake_time = inference_start_time + self.num_action_steps * (1.0 / self.action_frequency_hz)
            sleep_duration = wake_time - time.monotonic()
            if sleep_duration > 0.0 and self._shutdown_event.wait(sleep_duration):
                return

    def _normalize_lowdim(self, key: str, value: np.ndarray) -> np.ndarray:
        stats = self.normalizer["lowdim"][key]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return ((np.asarray(value, dtype=np.float32) - mean) / std).astype(np.float32)

    def _normalize_images(self, images: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.normalizer["images"]["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(self.normalizer["images"]["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        return ((np.asarray(images, dtype=np.float32) - mean) / std).astype(np.float32)

    def _denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32).reshape(-1, 7)
        pos_stats = self.normalizer["lowdim"]["eef_pos"]
        ori_stats = self.normalizer["lowdim"]["eef_ori"]

        pos = normalized_actions[:, :3] * np.asarray(pos_stats["std"], dtype=np.float32) + np.asarray(
            pos_stats["mean"], dtype=np.float32
        )
        quat = normalized_actions[:, 3:] * np.asarray(ori_stats["std"], dtype=np.float32) + np.asarray(
            ori_stats["mean"], dtype=np.float32
        )
        quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), a_min=1e-8, a_max=None)
        return np.hstack((pos, quat)).astype(np.float32)

    def _launch_observers(self) -> None:
        if self.robot_observer is None:
            self.robot_observer = RobotObserver(
                buffer_size=self.lowdim_buffer_size,
                example_obs={},
                obs_freq=self.lowdim_fps,
                robot_data=self.contract_path,
            )
        if self.camera_observer is None:
            self.camera_observer = CameraObserver(
                buffer_size=self.camera_buffer_size,
                example_obs={},
                camera_freq=self.camera_fps,
                robot_data=self.contract_path,
            )
        self.robot_observer.start_adding_obs()
        self.camera_observer.start_adding_obs()

    def _stop_observers(self) -> None:
        if self.camera_observer is not None:
            self.camera_observer.stop_adding_obs()
        if self.robot_observer is not None:
            self.robot_observer.stop_adding_obs()

    def _wait_for_observers(self) -> None:
        print("Waiting for robot and camera observers...", flush=True)
        last_log_time = 0.0
        while not self._shutdown_event.is_set():
            robot_ready = bool(self.robot_observer and self.robot_observer.get_last_k_obs(1))
            camera_ready = bool(self.camera_observer and self.camera_observer.get_last_k_obs(1))
            if robot_ready and camera_ready:
                return
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                print(f"  waiting... robot_ready={robot_ready} camera_ready={camera_ready}", flush=True)
                last_log_time = now
            time.sleep(WAIT_POLL_PERIOD_S)
        raise RuntimeError("Shutdown requested before observer buffers became ready.")

    def _start_inference_thread(self) -> None:
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="act-inference-loop",
            daemon=True,
        )
        self._inference_thread.start()

    def _start_keyboard_listener(self) -> None:
        if os.name != "posix":
            raise RuntimeError("The inference keyboard listener currently supports POSIX terminals only.")
        if not sys.stdin.isatty():
            raise RuntimeError("inference.py must be run from a real terminal to capture `k`/`l` input.")
        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            return

        import termios
        import tty

        self._stdin_fd = sys.stdin.fileno()
        if self._terminal_settings is None:
            self._terminal_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)

        self._keyboard_thread = threading.Thread(
            target=self._keyboard_listener_loop,
            name="inference-keyboard",
            daemon=True,
        )
        self._keyboard_thread.start()

    def _keyboard_listener_loop(self) -> None:
        while not self._shutdown_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if not key:
                continue
            if key.lower() == "k":
                self._inference_enabled.set()
                print("Started inference.", flush=True)
            elif key.lower() == "l":
                self._inference_enabled.clear()
                print("Stopped future inference.", flush=True)

    def _join_background_threads(self) -> None:
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=1.0)
            self._inference_thread = None
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=1.0)
            self._keyboard_thread = None

    def _restore_terminal_settings(self) -> None:
        if self._stdin_fd is None or self._terminal_settings is None:
            return
        import termios

        termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._terminal_settings)
        self._stdin_fd = None
        self._terminal_settings = None

    def _record_background_error(self, exc: BaseException) -> None:
        with self._background_error_lock:
            if self._background_error is None:
                self._background_error = exc
        self._shutdown_event.set()

    def _get_background_error(self) -> BaseException | None:
        with self._background_error_lock:
            return self._background_error

    def _load_model(self) -> ACT:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")
        if not self.act_config_path.exists():
            raise FileNotFoundError(f"ACT config does not exist: {self.act_config_path}")

        with self.act_config_path.open("r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle)
        config = SimpleNamespace(**config_dict)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint is missing `model_state_dict`.")

        model = ACT(config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _observation_indices(self, available_length: int) -> np.ndarray:
        center_index = max(available_length - 1, 0)
        offsets = np.arange(self.obs_window - 1, -1, -1, dtype=np.int64) * int(self.obs_dss)
        sample_indices = center_index - offsets
        return np.clip(sample_indices, 0, center_index)

    def _resolve_device(self, requested_device: str | None) -> torch.device:
        if requested_device is not None:
            return torch.device(str(requested_device))
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _load_contract(contract_path: Path) -> dict[str, Any]:
        with contract_path.open("r", encoding="utf-8") as handle:
            contract = yaml.safe_load(handle)
        if not isinstance(contract, dict):
            raise ValueError("The universal contract must load as a dictionary.")
        return contract

    def _require_source_cfg(self, source_name: str) -> dict[str, Any]:
        robot_cfg = self.contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("The universal contract must contain a top-level `robot` mapping.")
        data_sources = robot_cfg.get("data_sources")
        if not isinstance(data_sources, dict):
            raise ValueError("The universal contract must contain `robot.data_sources`.")
        source_cfg = data_sources.get(source_name)
        if not isinstance(source_cfg, dict):
            raise ValueError(f"The universal contract must contain `robot.data_sources.{source_name}`.")
        if not isinstance(source_cfg.get("keys"), list) or not source_cfg["keys"]:
            raise ValueError(f"`robot.data_sources.{source_name}.keys` must be a non-empty list.")
        return source_cfg

    def _resolve_fps(self, source_cfg: dict[str, Any], source_name: str) -> float:
        fps = float(source_cfg.get("fps", DEFAULT_FPS))
        if fps <= 0.0:
            raise ValueError(f"`robot.data_sources.{source_name}.fps` must be positive.")
        return fps

    def _resolve_buffer_size(self, source_cfg: dict[str, Any], source_name: str) -> int:
        max_obs_window = 0
        for key_entry in source_cfg["keys"]:
            if not isinstance(key_entry, dict) or len(key_entry) != 1:
                raise ValueError(
                    f"Each entry in `robot.data_sources.{source_name}.keys` must be a single-key mapping."
                )
            key_name, key_cfg = next(iter(key_entry.items()))
            if not isinstance(key_cfg, dict):
                raise ValueError(
                    f"`robot.data_sources.{source_name}.keys.{key_name}` must map to a dictionary."
                )
            obs_window = key_cfg.get("obs_window")
            if obs_window is None:
                raise ValueError(
                    f"`robot.data_sources.{source_name}.keys.{key_name}.obs_window` is required."
                )
            max_obs_window = max(max_obs_window, int(obs_window))
        return max(1, 3 * max_obs_window)

    @staticmethod
    def _parse_source_specs(source_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        parsed = {}
        for entry in source_cfg["keys"]:
            key_name, key_cfg = next(iter(entry.items()))
            parsed[key_name] = {
                "obs_window": int(key_cfg["obs_window"]),
                "obs_dss": int(key_cfg["obs_dss"]),
            }
        return parsed

    @staticmethod
    def _resolve_history_length(source_specs: dict[str, dict[str, Any]]) -> int:
        return max(
            1 + (int(spec["obs_window"]) - 1) * int(spec["obs_dss"])
            for spec in source_specs.values()
        )

    @staticmethod
    def _make_desired_pose_keys(contract: dict[str, Any]) -> tuple[str, str]:
        robot_cfg = contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("Contract must contain a top-level `robot` mapping.")
        prefix = str(robot_cfg.get("prefix", "")).strip()
        if not prefix:
            raise ValueError("Contract robot prefix must be provided.")
        redis_namespace = Inference._normalize_redis_namespace(robot_cfg.get("redis_namespace", "sai"))
        position_key = Inference._make_redis_key(redis_namespace, prefix, DESIRED_POSITION_SUFFIX)
        orientation_key = Inference._make_redis_key(redis_namespace, prefix, DESIRED_ORIENTATION_SUFFIX)
        return position_key, orientation_key

    @staticmethod
    def _normalize_redis_namespace(redis_namespace: Any) -> str:
        if redis_namespace is None:
            return ""
        return str(redis_namespace).strip(":")

    @staticmethod
    def _make_redis_key(redis_namespace: str, prefix: str, suffix: str) -> str:
        redis_key = f"{str(prefix).rstrip(':')}::{str(suffix).lstrip(':')}"
        if not redis_namespace:
            return redis_key
        return f"{redis_namespace}::{redis_key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ACT-based inference runtime.")
    parser.add_argument("--universal-contract", required=True, type=str)
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument("--act-config", required=True, type=str)
    parser.add_argument("--action-frequency-hz", required=True, type=float)
    parser.add_argument("--num-action-steps", required=True, type=int)
    parser.add_argument("--interpolator-frequency-hz", default=DEFAULT_INTERPOLATOR_FREQUENCY_HZ, type=float)
    parser.add_argument("--blend-duration", default=DEFAULT_BLEND_DURATION, type=float)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Inference(
        universal_contract=args.universal_contract,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        act_config=args.act_config,
        action_frequency_hz=args.action_frequency_hz,
        num_action_steps=args.num_action_steps,
        interpolator_frequency_hz=args.interpolator_frequency_hz,
        blend_duration=args.blend_duration,
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()
