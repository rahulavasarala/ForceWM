from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import redis
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__:
    from .camera_observer import CameraObserver
    from .robot_observer import RobotObserver
else:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from camera_observer import CameraObserver
    from robot_observer import RobotObserver

from high_level_controller.interpolator import (
    InterpolatorFault,
    PoseSample,
    TrajectoryInterpolator,
    make_redis_command_sink,
    make_redis_state_source,
)
from training.dataloader import (
    ObservationKeySpec,
    VisualKeySpec,
    load_contract,
    parse_action_specs,
    parse_lowdim_specs,
    parse_visual_specs,
    rot6d_to_rotation_matrix,
    rotation_matrix_to_rot6d,
)


DEFAULT_INTERPOLATOR_FREQUENCY_HZ = 100.0
DEFAULT_MAX_CHUNKS = 2
DEFAULT_FIRST_WAYPOINT_LEAD = 0.0
CURRENT_POSITION_SUFFIX = "current_cartesian_position"
CURRENT_ORIENTATION_SUFFIX = "current_cartesian_orientation"
DESIRED_POSITION_SUFFIX = "desired_cartesian_position"
DESIRED_ORIENTATION_SUFFIX = "desired_cartesian_orientation"
WAIT_POLL_PERIOD_S = 0.1
MODEL_WARMUP_PASSES = 20


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for model-backed inference. Install the `pytorch` package in the runtime environment."
        ) from exc
    return torch


def _require_rotation():
    try:
        from scipy.spatial.transform import Rotation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciPy is required for orientation conversion during model-backed inference."
        ) from exc
    return Rotation


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for online image resizing during model-backed inference."
        ) from exc
    return cv2


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {resolved_path}.")
    return payload


def _resolve_torch_device(requested_device: str | None, torch_module) -> Any:
    if requested_device is not None:
        return torch_module.device(str(requested_device))
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


class Inference:
    def __init__(
        self,
        universal_contract: str,
        action_frequency_hz: float,
        num_action_steps: int,
        *,
        predictor: str = "random",
        model_config: str | None = None,
        model_checkpoint: str | None = None,
        device: str | None = None,
        interpolator_frequency_hz: float = DEFAULT_INTERPOLATOR_FREQUENCY_HZ,
        blend_duration: float = 0.0,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        first_waypoint_lead: float = DEFAULT_FIRST_WAYPOINT_LEAD,
        seed: int = 0,
        position_delta_x: float = 0.04,
        position_delta_y: float = 0.04,
        position_delta_z: float = 0.02,
        rpy_delta_roll: float = 0.10,
        rpy_delta_pitch: float = 0.10,
        rpy_delta_yaw: float = 0.20,
        workspace_min_x: float = 0.25,
        workspace_min_y: float = -0.30,
        workspace_min_z: float = 0.15,
        workspace_max_x: float = 0.65,
        workspace_max_y: float = 0.30,
        workspace_max_z: float = 0.65,
    ) -> None:
        self.contract_path = Path(universal_contract).expanduser().resolve()
        self.contract = load_contract(self.contract_path, self.contract_path)
        self.lowdim_specs: dict[str, ObservationKeySpec] = parse_lowdim_specs(self.contract)
        self.visual_specs: dict[str, VisualKeySpec] = parse_visual_specs(self.contract)
        (
            self.action_window,
            self.action_dss,
            self.action_specs,
        ) = parse_action_specs(self.contract, self.lowdim_specs)

        self.action_frequency_hz = float(action_frequency_hz)
        self.num_action_steps = int(num_action_steps)
        self.predictor = str(predictor).strip().lower()
        self.interpolator_frequency_hz = float(interpolator_frequency_hz)
        self.blend_duration = float(blend_duration)
        self.max_chunks = int(max_chunks)
        self.first_waypoint_lead = float(first_waypoint_lead)
        self.model_config_path = (
            Path(model_config).expanduser().resolve() if model_config is not None else None
        )
        self.model_checkpoint_path = (
            Path(model_checkpoint).expanduser().resolve() if model_checkpoint is not None else None
        )

        if self.action_frequency_hz <= 0.0:
            raise ValueError("action_frequency_hz must be positive.")
        if self.num_action_steps <= 0:
            raise ValueError("num_action_steps must be positive.")
        if self.num_action_steps > self.action_window:
            raise ValueError(
                "num_action_steps must be less than or equal to robot.action.window."
            )
        if self.interpolator_frequency_hz <= 0.0:
            raise ValueError("interpolator_frequency_hz must be positive.")
        if self.blend_duration < 0.0:
            raise ValueError("blend_duration must be non-negative.")
        if self.first_waypoint_lead < 0.0:
            raise ValueError("first_waypoint_lead must be non-negative.")
        if self.max_chunks not in (1, 2):
            raise ValueError("max_chunks must be 1 or 2.")
        if self.predictor not in {"random", "model"}:
            raise ValueError("predictor must be either `random` or `model`.")

        self.action_dt = 1.0 / self.action_frequency_hz
        self.publish_period_s = self.num_action_steps / self.action_frequency_hz
        self.lowdim_buffer_size = self._compute_buffer_size(self.lowdim_specs)
        self.visual_buffer_size = self._compute_buffer_size(self.visual_specs)

        self.position_delta_limits = np.array(
            [position_delta_x, position_delta_y, position_delta_z], dtype=np.float64
        )
        self.rpy_delta_limits = np.array(
            [rpy_delta_roll, rpy_delta_pitch, rpy_delta_yaw], dtype=np.float64
        )
        self.position_noise_std = np.array([0.002, 0.002, 0.001], dtype=np.float64)
        self.rpy_noise_std = np.array([0.005, 0.005, 0.01], dtype=np.float64)
        self.workspace_min = np.array(
            [workspace_min_x, workspace_min_y, workspace_min_z], dtype=np.float64
        )
        self.workspace_max = np.array(
            [workspace_max_x, workspace_max_y, workspace_max_z], dtype=np.float64
        )
        if np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("workspace mins must be strictly less than workspace maxes.")

        self._lowdim_redis_keys = self._parse_lowdim_redis_keys(self.contract)
        self.current_position_key, self.current_orientation_key = self._make_current_pose_keys(
            self.contract
        )
        self.desired_position_key, self.desired_orientation_key = self._make_desired_pose_keys(
            self.contract
        )

        self.robot_observer: RobotObserver | None = None
        self.camera_observer: CameraObserver | None = None

        self.redis_client = redis.Redis(
            host="127.0.0.1",
            port=6379,
            db=0,
            decode_responses=False,
        )
        self.state_source = make_redis_state_source(
            self.redis_client,
            self.current_position_key,
            self.current_orientation_key,
        )
        self.command_sink = make_redis_command_sink(
            self.redis_client,
            self.desired_position_key,
            self.desired_orientation_key,
        )
        self.interpolator = TrajectoryInterpolator(
            command_sink=self.command_sink,
            state_source=self.state_source,
            send_rate_hz=self.interpolator_frequency_hz,
            blend_duration=self.blend_duration,
            max_chunks=self.max_chunks,
        )

        self.rng = np.random.default_rng(seed)

        self._shutdown_event = threading.Event()
        self._publishing_event = threading.Event()
        self._interpolator_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._keyboard_thread: threading.Thread | None = None
        self._terminal_settings: Any = None
        self._stdin_fd: int | None = None
        self._background_error: BaseException | None = None
        self._background_error_lock = threading.Lock()
        self._logged_observation_shapes = False
        self._logged_model_latency = False
        self.model_policy = None
        self.model_normalization_stats: dict[str, Any] | None = None
        self.model_runtime_config: dict[str, Any] | None = None
        self.model_device = None

        if self.predictor == "model":
            self._load_model_predictor(device)

    def get_observation_dict_from_buffers(self) -> dict[str, np.ndarray]:
        if self.robot_observer is None or self.camera_observer is None:
            raise RuntimeError("Observers must be launched before building observations.")

        robot_buffer = self.robot_observer.get_last_k_obs(self.lowdim_buffer_size)
        camera_buffer = self.camera_observer.get_last_k_obs(self.visual_buffer_size)
        if not robot_buffer:
            raise RuntimeError("Robot observer buffer is empty.")
        if not camera_buffer:
            raise RuntimeError("Camera observer buffer is empty.")

        observation: dict[str, np.ndarray] = {}
        orientation_source_key = self.action_specs["robot_ori"].source_key

        for key_name, key_spec in self.lowdim_specs.items():
            if key_name not in robot_buffer:
                raise RuntimeError(f"Lowdim key `{key_name}` is missing from the robot buffer.")
            sequence = self._build_observation_from_source_buffer(
                robot_buffer[key_name],
                key_spec.obs_window,
                key_spec.obs_dss,
            )
            if key_name == orientation_source_key:
                sequence = rotation_matrix_to_rot6d(sequence)
            observation[key_name] = np.asarray(sequence, dtype=np.float32)

        for key_name, key_spec in self.visual_specs.items():
            if key_name not in camera_buffer:
                raise RuntimeError(f"Visual key `{key_name}` is missing from the camera buffer.")
            sequence = self._build_observation_from_source_buffer(
                camera_buffer[key_name],
                key_spec.obs_window,
                key_spec.obs_dss,
            )
            if sequence.ndim == 4:
                sequence = np.ascontiguousarray(sequence.transpose(0, 3, 1, 2))
            elif sequence.ndim == 3:
                sequence = np.ascontiguousarray(sequence[:, None, :, :])
            observation[key_name] = np.asarray(sequence)

        if not self._logged_observation_shapes:
            self._logged_observation_shapes = True
            print("Loaded online observation example from current buffers", flush=True)
            for key_name, value in observation.items():
                print(
                    f"  observation[{key_name}]: shape={value.shape}, dtype={value.dtype}",
                    flush=True,
                )

        return observation

    def launch_inference_loop(self) -> None:
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return

        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="policy-inference-loop",
            daemon=True,
        )
        self._inference_thread.start()

    def _launch_observers(self) -> None:
        if self.robot_observer is None:
            self.robot_observer = RobotObserver(
                buffer_size=self.lowdim_buffer_size,
                example_obs={},
                obs_freq=None,
                robot_data=self.contract_path,
            )
        if self.camera_observer is None:
            self.camera_observer = CameraObserver(
                buffer_size=self.visual_buffer_size,
                example_obs={},
                camera_freq=None,
                robot_data=self.contract_path,
            )

        self.robot_observer.start_adding_obs()
        self.camera_observer.start_adding_obs()

    def _stop_observers(self) -> None:
        if self.camera_observer is not None:
            self.camera_observer.stop_adding_obs()
        if self.robot_observer is not None:
            self.robot_observer.stop_adding_obs()

    def start_keyboard_listener(self) -> None:
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

    def stop_keyboard_listener(self) -> None:
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=1.0)
            self._keyboard_thread = None
        self._restore_terminal_settings()

    def run(self) -> None:
        self._run()

    def _run(self) -> None:
        try:
            self._launch_observers()
            initial_state = self._wait_for_initial_state()
            if self.predictor == "model":
                self.warm_up_inference(MODEL_WARMUP_PASSES)
                initial_state = self.state_source()
            self.interpolator.set_initial_state(initial_state)

            self._interpolator_thread = threading.Thread(
                target=self._run_interpolator,
                name="trajectory-interpolator",
                daemon=True,
            )
            self._interpolator_thread.start()
            self.launch_inference_loop()
            self.start_keyboard_listener()

            if self.predictor == "model":
                self._prepare_observation_for_model(self.get_observation_dict_from_buffers())

            print(
                f"Inference ready with contract {self.contract_path}",
                flush=True,
            )
            print(f"Predictor mode: {self.predictor}", flush=True)
            if self.predictor == "model":
                print(
                    f"Model checkpoint: {self.model_checkpoint_path} on device {self.model_device}",
                    flush=True,
                )
                if self.model_config_path is not None:
                    print(f"Model config: {self.model_config_path}", flush=True)
            print(
                f"Current pose keys: position=`{self.current_position_key}` orientation=`{self.current_orientation_key}`",
                flush=True,
            )
            print(
                f"Desired pose keys: position=`{self.desired_position_key}` orientation=`{self.desired_orientation_key}`",
                flush=True,
            )
            print(
                f"Action window={self.action_window}, action_dss={self.action_dss}, action_hz={self.action_frequency_hz}, "
                f"num_action_steps={self.num_action_steps}, publish_period={self.publish_period_s:.3f}s",
                flush=True,
            )
            print(
                f"Observer buffers: lowdim={self.lowdim_buffer_size}, visual={self.visual_buffer_size}; "
                f"interpolator_hz={self.interpolator_frequency_hz}",
                flush=True,
            )
            print(f"Press `k` to start {self.predictor} inference, `l` to stop it, Ctrl-C to exit.", flush=True)

            while not self._shutdown_event.is_set():
                error = self._get_background_error()
                if error is not None:
                    raise error
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nShutting down inference.", flush=True)
        finally:
            self._shutdown_event.set()
            self._publishing_event.clear()
            self.interpolator.stop()

            if self._inference_thread is not None:
                self._inference_thread.join(timeout=1.0)
                self._inference_thread = None

            if self._interpolator_thread is not None:
                self._interpolator_thread.join(timeout=1.0)
                self._interpolator_thread = None

            self._stop_observers()
            self.stop_keyboard_listener()

            error = self._get_background_error()
            if error is not None:
                raise error

    def warm_up_inference(self, num_passes: int = MODEL_WARMUP_PASSES) -> None:
        if self.predictor != "model":
            return
        if self.model_policy is None:
            raise RuntimeError("Warm-up requested, but no model policy is loaded.")
        if num_passes <= 0:
            return

        print(f"Warming up model inference with {num_passes} dry runs...", flush=True)
        warmup_start_time = time.monotonic()
        for _ in range(int(num_passes)):
            observation = self.get_observation_dict_from_buffers()
            current_state = self.state_source()
            inference_start_time = time.monotonic()
            _ = self._generate_model_action_chunk(
                observation,
                current_state,
                inference_start_time,
            )
        warmup_elapsed_s = time.monotonic() - warmup_start_time
        print(
            f"Model warm-up complete in {warmup_elapsed_s:.2f}s.",
            flush=True,
        )

    def _build_observation_from_source_buffer(
        self,
        values: Any,
        obs_window: int,
        obs_dss: int,
    ) -> np.ndarray:
        return self._slice_and_pad_sequence(values, obs_window, obs_dss)

    @staticmethod
    def _slice_and_pad_sequence(values: Any, obs_window: int, obs_dss: int) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] == 0:
            raise RuntimeError("Cannot build an observation from an empty source buffer.")

        center_index = array.shape[0] - 1
        offsets = np.arange(obs_window - 1, -1, -1, dtype=np.int64) * int(obs_dss)
        sample_indices = center_index - offsets
        sample_indices = np.clip(sample_indices, 0, center_index)
        return np.asarray(array[sample_indices])

    def _generate_dummy_action_chunk(
        self,
        observation_dict: dict[str, np.ndarray],
        current_state: PoseSample,
        now: float,
    ) -> np.ndarray:
        del observation_dict

        current_position = np.asarray(current_state.position, dtype=np.float64).reshape(3)
        current_rpy = np.asarray(current_state.rpy(), dtype=np.float64).reshape(3)

        target_delta = self.rng.uniform(
            low=-self.position_delta_limits,
            high=self.position_delta_limits,
        )
        target_position = np.clip(
            current_position + target_delta,
            self.workspace_min,
            self.workspace_max,
        )

        target_rpy = current_rpy + self.rng.uniform(
            low=-self.rpy_delta_limits,
            high=self.rpy_delta_limits,
        )

        alphas = np.linspace(0.0, 1.0, self.action_window, dtype=np.float64)[:, None]
        positions = (
            (1.0 - alphas) * current_position[None, :]
            + alphas * target_position[None, :]
            + self.rng.normal(
                loc=0.0,
                scale=self.position_noise_std,
                size=(self.action_window, 3),
            )
        )
        positions[0] = current_position
        positions = np.clip(positions, self.workspace_min, self.workspace_max)

        rpy = (
            (1.0 - alphas) * current_rpy[None, :]
            + alphas * target_rpy[None, :]
            + self.rng.normal(
                loc=0.0,
                scale=self.rpy_noise_std,
                size=(self.action_window, 3),
            )
        )
        rpy[0] = current_rpy

        waypoint_times = (
            float(now)
            + self.first_waypoint_lead
            + self.action_dt * np.arange(self.action_window, dtype=np.float64)
        )
        return np.column_stack((positions, rpy, waypoint_times))

    def _load_model_predictor(self, requested_device: str | None) -> None:
        torch = _require_torch()
        if self.model_checkpoint_path is None:
            raise ValueError("`--predictor model` requires `--model-checkpoint` (or `--model-weights`).")
        if not self.model_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint does not exist: {self.model_checkpoint_path}")

        self.model_device = _resolve_torch_device(requested_device, torch)

        if self.model_config_path is None:
            from training.bc_policy import load_policy_from_checkpoint

            policy, normalization_stats, runtime_config = load_policy_from_checkpoint(
                self.model_checkpoint_path,
                device=self.model_device,
                eval_mode=True,
            )
        else:
            if not self.model_config_path.exists():
                raise FileNotFoundError(f"Model config does not exist: {self.model_config_path}")

            from training.bc_policy import DatasetSpec, build_policy_from_config

            runtime_config = _load_yaml_mapping(self.model_config_path)
            checkpoint = torch.load(self.model_checkpoint_path, map_location=self.model_device)
            if "model_state_dict" not in checkpoint:
                raise ValueError("Model checkpoint is missing `model_state_dict`.")
            if "dataset_spec" not in checkpoint:
                raise ValueError("Model checkpoint is missing `dataset_spec`.")
            if "normalization_stats" not in checkpoint:
                raise ValueError("Model checkpoint is missing `normalization_stats`.")
            if "model" not in runtime_config:
                raise ValueError(f"Model config at {self.model_config_path} is missing a `model` section.")

            dataset_spec = DatasetSpec.from_dict(checkpoint["dataset_spec"])
            policy = build_policy_from_config(runtime_config["model"], dataset_spec)
            policy.load_state_dict(checkpoint["model_state_dict"])
            policy.to(self.model_device)
            policy.eval()
            normalization_stats = checkpoint["normalization_stats"]

        self.model_policy = policy
        self.model_normalization_stats = normalization_stats
        self.model_runtime_config = runtime_config
        self._validate_loaded_policy()

    def _validate_loaded_policy(self) -> None:
        if self.model_policy is None:
            raise RuntimeError("Model policy was not loaded.")

        dataset_spec = self.model_policy.dataset_spec
        if dataset_spec.action_window != self.action_window:
            raise ValueError(
                f"Loaded model action window {dataset_spec.action_window} does not match runtime contract action window {self.action_window}."
            )
        if dataset_spec.image_key not in self.visual_specs:
            raise ValueError(
                f"Loaded model expects visual key `{dataset_spec.image_key}`, which is not present in the runtime contract."
            )
        visual_spec = self.visual_specs[dataset_spec.image_key]
        if visual_spec.obs_window != dataset_spec.image_observation_horizon:
            raise ValueError(
                f"Loaded model expects image horizon {dataset_spec.image_observation_horizon} for `{dataset_spec.image_key}`, "
                f"but the runtime contract uses {visual_spec.obs_window}."
            )
        for key_name in dataset_spec.lowdim_keys:
            key_spec = self.lowdim_specs.get(key_name)
            if key_spec is None:
                raise ValueError(
                    f"Loaded model expects lowdim key `{key_name}`, which is not present in the runtime contract."
                )
            if key_spec.obs_window != dataset_spec.lowdim_observation_horizon:
                raise ValueError(
                    f"Loaded model expects lowdim horizon {dataset_spec.lowdim_observation_horizon} for `{key_name}`, "
                    f"but the runtime contract uses {key_spec.obs_window}."
                )
            if key_spec.obs_dss != dataset_spec.lowdim_obs_dss:
                raise ValueError(
                    f"Loaded model expects lowdim obs_dss {dataset_spec.lowdim_obs_dss} for `{key_name}`, "
                    f"but the runtime contract uses {key_spec.obs_dss}."
                )

    def _prepare_observation_for_model(self, observation_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.model_policy is None:
            raise RuntimeError("Model predictor requested before the policy was loaded.")

        dataset_spec = self.model_policy.dataset_spec
        prepared = dict(observation_dict)
        image_key = dataset_spec.image_key
        image_sequence = np.asarray(prepared[image_key])
        expected_channels, expected_height, expected_width = dataset_spec.image_shape
        if image_sequence.ndim != 4:
            raise ValueError(
                f"Model visual observation `{image_key}` must have shape [T, C, H, W]. Got {image_sequence.shape}."
            )
        if image_sequence.shape[0] != dataset_spec.image_observation_horizon:
            raise ValueError(
                f"Model visual observation `{image_key}` must have horizon {dataset_spec.image_observation_horizon}. "
                f"Got {image_sequence.shape[0]}."
            )
        if image_sequence.shape[1] != expected_channels:
            raise ValueError(
                f"Model visual observation `{image_key}` must have {expected_channels} channels. Got {image_sequence.shape[1]}."
            )
        if tuple(int(value) for value in image_sequence.shape[1:]) != dataset_spec.image_shape:
            cv2 = _require_cv2()
            resized_frames = []
            for frame in image_sequence:
                frame_hwc = np.transpose(frame, (1, 2, 0))
                resized_frame = cv2.resize(
                    frame_hwc,
                    (expected_width, expected_height),
                    interpolation=cv2.INTER_AREA,
                )
                resized_frames.append(np.transpose(resized_frame, (2, 0, 1)))
            prepared[image_key] = np.ascontiguousarray(np.stack(resized_frames, axis=0))

        return prepared

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _rotation_matrices_to_rpy(rotation_matrices: np.ndarray) -> np.ndarray:
        Rotation = _require_rotation()
        return Rotation.from_matrix(rotation_matrices).as_euler("xyz")

    def _generate_model_action_chunk(
        self,
        observation_dict: dict[str, np.ndarray],
        current_state: PoseSample,
        inference_start_time: float,
    ) -> np.ndarray:
        del current_state

        if self.model_policy is None or self.model_normalization_stats is None:
            raise RuntimeError("Model predictor was requested, but the policy is not loaded.")

        prepared_observation = self._prepare_observation_for_model(observation_dict)
        position_source_key = self.action_specs["robot_pos"].source_key
        orientation_source_key = self.action_specs["robot_ori"].source_key
        reference_position = np.asarray(prepared_observation[position_source_key][-1], dtype=np.float64).reshape(3)
        reference_rotation = rot6d_to_rotation_matrix(prepared_observation[orientation_source_key][-1])

        torch = _require_torch()
        with torch.inference_mode():
            predictions = self.model_policy.predict_actions(
                prepared_observation,
                self.model_normalization_stats,
            )

        inference_latency_s = time.monotonic() - inference_start_time
        if not self._logged_model_latency:
            self._logged_model_latency = True
            print(
                f"Model inference latency: {1000.0 * inference_latency_s:.1f} ms; waypoint times stay anchored to inference start.",
                flush=True,
            )

        relative_positions = np.asarray(self._to_numpy(predictions["robot_pos"]), dtype=np.float64)
        relative_rot6d = np.asarray(self._to_numpy(predictions["robot_ori"]), dtype=np.float64)
        if relative_positions.shape != (self.action_window, 3):
            raise ValueError(
                f"Model predicted robot_pos with shape {relative_positions.shape}; expected {(self.action_window, 3)}."
            )
        if relative_rot6d.shape != (self.action_window, 6):
            raise ValueError(
                f"Model predicted robot_ori with shape {relative_rot6d.shape}; expected {(self.action_window, 6)}."
            )

        relative_rotations = rot6d_to_rotation_matrix(relative_rot6d)
        world_positions = reference_position[None, :] + (reference_rotation @ relative_positions.T).T
        world_rotations = np.einsum("ij,tjk->tik", reference_rotation, relative_rotations)
        world_rpy = self._rotation_matrices_to_rpy(world_rotations)
        waypoint_times = (
            float(inference_start_time)
            + self.first_waypoint_lead
            + self.action_dt * np.arange(self.action_window, dtype=np.float64)
        )
        return np.column_stack((world_positions, world_rpy, waypoint_times))

    def _predict_action_chunk(
        self,
        observation_dict: dict[str, np.ndarray],
        current_state: PoseSample,
        inference_start_time: float,
    ) -> np.ndarray:
        if self.predictor == "model":
            return self._generate_model_action_chunk(
                observation_dict,
                current_state,
                inference_start_time,
            )
        return self._generate_dummy_action_chunk(observation_dict, current_state, inference_start_time)

    def _keyboard_listener_loop(self) -> None:
        while not self._shutdown_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue

            key = sys.stdin.read(1)
            if not key:
                continue

            try:
                if key.lower() == "k":
                    if not self._publishing_event.is_set():
                        self._publishing_event.set()
                        print(f"Started {self.predictor} inference.", flush=True)
                elif key.lower() == "l":
                    if self._publishing_event.is_set():
                        self._publishing_event.clear()
                        print(f"Stopped {self.predictor} inference.", flush=True)
            except Exception as exc:
                print(f"Keyboard command failed: {exc}", flush=True)

    def _inference_loop(self) -> None:
        next_publish_time = time.monotonic()

        while not self._shutdown_event.is_set():
            if not self._publishing_event.is_set():
                next_publish_time = time.monotonic()
                if self._shutdown_event.wait(0.1):
                    return
                continue

            try:
                observation = self.get_observation_dict_from_buffers()
                current_state = self.state_source()
                inference_start_time = time.monotonic()
                chunk = self._predict_action_chunk(observation, current_state, inference_start_time)
                self.interpolator.enqueue_chunk(chunk)
            except InterpolatorFault as exc:
                print(f"Interpolator fault: {exc}", flush=True)
                self._record_background_error(exc)
                return
            except Exception as exc:
                print(f"Inference loop failed: {exc}", flush=True)
                self._record_background_error(exc)
                return

            next_publish_time += self.publish_period_s
            sleep_duration = next_publish_time - time.monotonic()
            if sleep_duration > 0.0:
                if self._shutdown_event.wait(sleep_duration):
                    return
            else:
                next_publish_time = time.monotonic()

    def _run_interpolator(self) -> None:
        try:
            self.interpolator.run()
        except InterpolatorFault as exc:
            print(f"Interpolator fault: {exc}", flush=True)
            self._record_background_error(exc)
        except Exception as exc:
            print(f"Interpolator runner failed: {exc}", flush=True)
            self._record_background_error(exc)

    @staticmethod
    def _compute_buffer_size(specs: dict[str, ObservationKeySpec | VisualKeySpec]) -> int:
        if not specs:
            return 1
        return max(1, max(2 * spec.obs_window * spec.obs_dss for spec in specs.values()))

    @staticmethod
    def _make_desired_pose_keys(contract: dict[str, Any]) -> tuple[str, str]:
        robot_cfg = contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("Contract must contain a top-level `robot` mapping.")
        prefix = str(robot_cfg.get("prefix", "")).strip()
        if not prefix:
            raise ValueError("Contract robot prefix must be provided.")
        redis_namespace = Inference._normalize_redis_namespace(
            robot_cfg.get("redis_namespace", "sai")
        )
        position_key = Inference._make_redis_key(
            redis_namespace,
            prefix,
            DESIRED_POSITION_SUFFIX,
        )
        orientation_key = Inference._make_redis_key(
            redis_namespace,
            prefix,
            DESIRED_ORIENTATION_SUFFIX,
        )
        return position_key, orientation_key

    @staticmethod
    def _make_current_pose_keys(contract: dict[str, Any]) -> tuple[str, str]:
        robot_cfg = contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("Contract must contain a top-level `robot` mapping.")
        prefix = str(robot_cfg.get("prefix", "")).strip()
        if not prefix:
            raise ValueError("Contract robot prefix must be provided.")
        redis_namespace = Inference._normalize_redis_namespace(
            robot_cfg.get("redis_namespace", "sai")
        )
        position_key = Inference._make_redis_key(
            redis_namespace,
            prefix,
            CURRENT_POSITION_SUFFIX,
        )
        orientation_key = Inference._make_redis_key(
            redis_namespace,
            prefix,
            CURRENT_ORIENTATION_SUFFIX,
        )
        return position_key, orientation_key

    def _wait_for_initial_state(self) -> PoseSample:
        print("Waiting for current pose and observer buffers...", flush=True)
        last_log_time = 0.0

        while not self._shutdown_event.is_set():
            robot_ready = False
            camera_ready = False
            state_sample: PoseSample | None = None

            if self.robot_observer is not None:
                robot_ready = bool(self.robot_observer.get_last_k_obs(1))
            if self.camera_observer is not None:
                camera_ready = bool(self.camera_observer.get_last_k_obs(1))

            try:
                state_sample = self.state_source()
            except Exception:
                state_sample = None

            if robot_ready and camera_ready and state_sample is not None:
                try:
                    self.get_observation_dict_from_buffers()
                except Exception:
                    pass
                return state_sample

            now = time.monotonic()
            if now - last_log_time >= 1.0:
                print(
                    f"  waiting... robot_ready={robot_ready} camera_ready={camera_ready} "
                    f"state_ready={state_sample is not None}",
                    flush=True,
                )
                last_log_time = now
            time.sleep(WAIT_POLL_PERIOD_S)

        raise RuntimeError("Shutdown requested before the initial state became available.")

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

    @classmethod
    def _parse_lowdim_redis_keys(cls, contract: dict[str, Any]) -> dict[str, str]:
        robot_cfg = contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("Contract must contain a top-level `robot` mapping.")

        prefix = str(robot_cfg.get("prefix", "")).strip()
        if not prefix:
            raise ValueError("Contract robot prefix must be provided.")
        redis_namespace = cls._normalize_redis_namespace(robot_cfg.get("redis_namespace", "sai"))

        lowdim_cfg = robot_cfg.get("data_sources", {}).get("lowdim", {})
        if not isinstance(lowdim_cfg, dict):
            raise ValueError("Contract is missing `robot.data_sources.lowdim`.")

        redis_keys: dict[str, str] = {}
        for key_name, key_cfg in cls._parse_key_sequence(
            lowdim_cfg.get("keys", []),
            "robot.data_sources.lowdim.keys",
        ):
            redis_suffix = str(key_cfg.get("redis", key_name)).strip()
            redis_keys[key_name] = cls._make_redis_key(redis_namespace, prefix, redis_suffix)
        return redis_keys

    @staticmethod
    def _parse_key_sequence(
        sequence: Any,
        section_name: str,
    ) -> list[tuple[str, dict[str, Any]]]:
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
                raise ValueError(
                    f"Configuration for `{key_name}` in `{section_name}` must be a mapping."
                )
            parsed_entries.append((str(key_name), key_cfg))
        return parsed_entries

    def _record_background_error(self, exc: BaseException) -> None:
        with self._background_error_lock:
            if self._background_error is None:
                self._background_error = exc
        self._publishing_event.clear()
        self._shutdown_event.set()
        self.interpolator.stop()

    def _get_background_error(self) -> BaseException | None:
        with self._background_error_lock:
            return self._background_error

    def _restore_terminal_settings(self) -> None:
        if self._stdin_fd is None or self._terminal_settings is None:
            return

        import termios

        termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._terminal_settings)
        self._stdin_fd = None
        self._terminal_settings = None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive policy inference runner for Redis-backed ForceWM control."
    )
    parser.add_argument(
        "--universal-contract",
        dest="universal_contract",
        required=True,
        type=str,
        help="Path to the universal contract file.",
    )
    parser.add_argument(
        "--action-frequency-hz",
        dest="action_frequency_hz",
        required=True,
        type=float,
        help="Action waypoint frequency used to time the dummy action chunk.",
    )
    parser.add_argument(
        "--num-action-steps",
        dest="num_action_steps",
        required=True,
        type=int,
        help="Number of action waypoints to execute before publishing a new chunk.",
    )
    parser.add_argument(
        "--predictor",
        choices=("random", "model"),
        default="random",
        help="Predictor mode. `random` keeps the dummy bounded-random walk; `model` loads a trained policy.",
    )
    parser.add_argument(
        "--model-config",
        dest="model_config",
        default=None,
        type=str,
        help="Optional path to the model config YAML. If omitted, the config stored in the checkpoint is used.",
    )
    parser.add_argument(
        "--model-checkpoint",
        "--model-weights",
        dest="model_checkpoint",
        default=None,
        type=str,
        help="Path to the trained model checkpoint (.pt). Required when --predictor model.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        type=str,
        help="Optional torch device override for model-backed inference (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--interpolator-frequency-hz",
        dest="interpolator_frequency_hz",
        default=DEFAULT_INTERPOLATOR_FREQUENCY_HZ,
        type=float,
        help="Send rate for the trajectory interpolator.",
    )
    parser.add_argument(
        "--blend-duration",
        dest="blend_duration",
        default=0.0,
        type=float,
        help="Blend duration for interpolator chunk transitions.",
    )
    parser.add_argument(
        "--max-chunks",
        dest="max_chunks",
        default=DEFAULT_MAX_CHUNKS,
        type=int,
        help="Maximum number of queued chunks allowed by the interpolator.",
    )
    parser.add_argument(
        "--first-waypoint-lead",
        dest="first_waypoint_lead",
        default=DEFAULT_FIRST_WAYPOINT_LEAD,
        type=float,
        help="Additional lead time before the first waypoint in each chunk.",
    )
    parser.add_argument("--seed", dest="seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--position-delta-x", default=0.04, type=float)
    parser.add_argument("--position-delta-y", default=0.04, type=float)
    parser.add_argument("--position-delta-z", default=0.02, type=float)
    parser.add_argument("--rpy-delta-roll", default=0.10, type=float)
    parser.add_argument("--rpy-delta-pitch", default=0.10, type=float)
    parser.add_argument("--rpy-delta-yaw", default=0.20, type=float)
    parser.add_argument("--workspace-min-x", default=0.25, type=float)
    parser.add_argument("--workspace-min-y", default=-0.30, type=float)
    parser.add_argument("--workspace-min-z", default=0.15, type=float)
    parser.add_argument("--workspace-max-x", default=0.65, type=float)
    parser.add_argument("--workspace-max-y", default=0.30, type=float)
    parser.add_argument("--workspace-max-z", default=0.65, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        inference = Inference(
            universal_contract=args.universal_contract,
            action_frequency_hz=args.action_frequency_hz,
            num_action_steps=args.num_action_steps,
            predictor=args.predictor,
            model_config=args.model_config,
            model_checkpoint=args.model_checkpoint,
            device=args.device,
            interpolator_frequency_hz=args.interpolator_frequency_hz,
            blend_duration=args.blend_duration,
            max_chunks=args.max_chunks,
            first_waypoint_lead=args.first_waypoint_lead,
            seed=args.seed,
            position_delta_x=args.position_delta_x,
            position_delta_y=args.position_delta_y,
            position_delta_z=args.position_delta_z,
            rpy_delta_roll=args.rpy_delta_roll,
            rpy_delta_pitch=args.rpy_delta_pitch,
            rpy_delta_yaw=args.rpy_delta_yaw,
            workspace_min_x=args.workspace_min_x,
            workspace_min_y=args.workspace_min_y,
            workspace_min_z=args.workspace_min_z,
            workspace_max_x=args.workspace_max_x,
            workspace_max_y=args.workspace_max_y,
            workspace_max_z=args.workspace_max_z,
        )
        inference.run()
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
