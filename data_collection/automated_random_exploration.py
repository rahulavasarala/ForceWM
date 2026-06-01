from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection.data_collection import DataCollection  # noqa: E402
from policies.random_exploration_policy import effective_replan_after  # noqa: E402
from policies.random_exploration_runtime import (  # noqa: E402
    DEFAULT_CONFIG_PATH as DEFAULT_RUNTIME_CONFIG_PATH,
    RandomExplorationRuntime,
    load_runtime_config,
    world_to_local_xy,
)


DEFAULT_AUTOMATION_CONFIG_PATH = Path(__file__).with_suffix(".yaml")
DEFAULT_HOME_POSITION_WORLD = np.array([0.4, 0.0, 0.39], dtype=float)
DEFAULT_HOME_ORIENTATION_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
DEFAULT_MAX_EPISODE_DURATION_S = 20.0
DEFAULT_POLL_PERIOD_S = 0.02
SENSED_FORCE_SUFFIX = "sensed_force"


@dataclass(frozen=True)
class MotionConfig:
    position_tolerance_m: float
    orientation_tolerance_rad: float
    translation_speed_mps: float
    move_timeout_buffer_s: float
    poll_period_s: float

    def validate(self) -> None:
        if self.position_tolerance_m <= 0.0:
            raise ValueError("motion.position_tolerance_m must be positive.")
        if self.orientation_tolerance_rad <= 0.0:
            raise ValueError("motion.orientation_tolerance_rad must be positive.")
        if self.translation_speed_mps <= 0.0:
            raise ValueError("motion.translation_speed_mps must be positive.")
        if self.move_timeout_buffer_s <= 0.0:
            raise ValueError("motion.move_timeout_buffer_s must be positive.")
        if self.poll_period_s <= 0.0:
            raise ValueError("motion.poll_period_s must be positive.")


@dataclass(frozen=True)
class HomeConfig:
    position_world: np.ndarray
    orientation_world: np.ndarray

    def validate(self) -> None:
        position = np.asarray(self.position_world, dtype=float).reshape(-1)
        orientation = np.asarray(self.orientation_world, dtype=float)
        if position.size != 3:
            raise ValueError("home.position_world must contain exactly three values.")
        if orientation.shape != (3, 3):
            raise ValueError("home.orientation_world must have shape (3, 3).")


@dataclass(frozen=True)
class RandomStartConfig:
    min_distance_from_center_m: float | None
    max_distance_from_center_m: float | None
    max_sampling_attempts: int

    def validate(self) -> None:
        if self.min_distance_from_center_m is not None and self.min_distance_from_center_m < 0.0:
            raise ValueError("random_start.min_distance_from_center_m must be non-negative when provided.")
        if self.max_distance_from_center_m is not None and self.max_distance_from_center_m <= 0.0:
            raise ValueError("random_start.max_distance_from_center_m must be positive when provided.")
        if (
            self.min_distance_from_center_m is not None
            and self.max_distance_from_center_m is not None
            and self.min_distance_from_center_m > self.max_distance_from_center_m
        ):
            raise ValueError(
                "random_start.min_distance_from_center_m cannot exceed random_start.max_distance_from_center_m."
            )
        if self.max_sampling_attempts <= 0:
            raise ValueError("random_start.max_sampling_attempts must be positive.")


@dataclass(frozen=True)
class ContactDescentConfig:
    contact_force_threshold_n: float
    step_size_m: float
    max_descent_distance_m: float

    def validate(self) -> None:
        if self.contact_force_threshold_n < 0.0:
            raise ValueError("contact_descent.contact_force_threshold_n must be non-negative.")
        if self.step_size_m <= 0.0:
            raise ValueError("contact_descent.step_size_m must be positive.")
        if self.max_descent_distance_m <= 0.0:
            raise ValueError("contact_descent.max_descent_distance_m must be positive.")


@dataclass(frozen=True)
class AutomationConfig:
    home: HomeConfig
    random_start: RandomStartConfig
    contact_descent: ContactDescentConfig
    motion: MotionConfig

    def validate(self) -> None:
        self.home.validate()
        self.random_start.validate()
        self.contact_descent.validate()
        self.motion.validate()


def _require_mapping(mapping: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for `{key}` in {context}.")
    return value


def _load_yaml_mapping(path: Path, context: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must load as a dictionary.")
    return mapping


def _normalize_redis_namespace(redis_namespace: Any) -> str:
    if redis_namespace is None:
        return ""
    return str(redis_namespace).strip(":")


def _make_redis_key(redis_namespace: str, prefix: str, suffix: str) -> str:
    redis_key = f"{str(prefix).rstrip(':')}::{str(suffix).lstrip(':')}"
    if not redis_namespace:
        return redis_key
    return f"{redis_namespace}::{redis_key}"


def resolve_sensed_force_key_from_contract(contract: dict[str, Any]) -> str:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        raise ValueError("Contract must contain a top-level `robot` mapping.")

    prefix = str(robot_cfg.get("prefix", "")).strip()
    if not prefix:
        raise ValueError("Contract robot prefix must be provided.")

    redis_namespace = _normalize_redis_namespace(robot_cfg.get("redis_namespace", "sai"))
    return _make_redis_key(redis_namespace, prefix, SENSED_FORCE_SUFFIX)


def _rotation_error_rad(current_rotation: np.ndarray, target_rotation: np.ndarray) -> float:
    current = Rotation.from_matrix(np.asarray(current_rotation, dtype=float).reshape(3, 3))
    target = Rotation.from_matrix(np.asarray(target_rotation, dtype=float).reshape(3, 3))
    relative_rotation = current.inv() * target
    return float(relative_rotation.magnitude())


def _redis_text(value: bytes | str | None) -> str:
    if value is None:
        raise RuntimeError("Requested Redis key is missing.")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_json_value(redis_client, key: str):
    return json.loads(_redis_text(redis_client.get(key)))


def _read_vector(redis_client, key: str) -> np.ndarray:
    vector = np.asarray(_read_json_value(redis_client, key), dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Redis key `{key}` did not contain a 3D vector.")
    return vector.astype(float)


def _read_matrix(redis_client, key: str) -> np.ndarray:
    matrix = np.asarray(_read_json_value(redis_client, key), dtype=float)
    if matrix.size != 9:
        raise ValueError(f"Redis key `{key}` did not contain a 3x3 matrix.")
    return matrix.reshape(3, 3).astype(float)


def _write_scalar(redis_client, key: str, value: float) -> None:
    redis_client.set(key, json.dumps([float(value)]))


def _write_vector(redis_client, key: str, value: np.ndarray) -> None:
    vector = np.asarray(value, dtype=float).reshape(3)
    redis_client.set(key, json.dumps(vector.tolist()))


def _write_matrix(redis_client, key: str, value: np.ndarray) -> None:
    matrix = np.asarray(value, dtype=float).reshape(3, 3)
    redis_client.set(key, json.dumps(matrix.tolist()))


def load_automation_config(config_path: str | Path = DEFAULT_AUTOMATION_CONFIG_PATH) -> AutomationConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Automation config not found: {path}")

    config_dict = _load_yaml_mapping(path, "Automation config")
    home_cfg = _require_mapping(config_dict, "home", "automation config")
    random_start_cfg = _require_mapping(config_dict, "random_start", "automation config")
    contact_descent_cfg = _require_mapping(config_dict, "contact_descent", "automation config")
    motion_cfg = _require_mapping(config_dict, "motion", "automation config")

    automation_config = AutomationConfig(
        home=HomeConfig(
            position_world=np.asarray(
                home_cfg.get("position_world", DEFAULT_HOME_POSITION_WORLD),
                dtype=float,
            ),
            orientation_world=np.asarray(
                home_cfg.get("orientation_world", DEFAULT_HOME_ORIENTATION_WORLD),
                dtype=float,
            ),
        ),
        random_start=RandomStartConfig(
            min_distance_from_center_m=(
                None
                if random_start_cfg.get("min_distance_from_center_m") is None
                else float(random_start_cfg["min_distance_from_center_m"])
            ),
            max_distance_from_center_m=(
                None
                if random_start_cfg.get("max_distance_from_center_m") is None
                else float(random_start_cfg["max_distance_from_center_m"])
            ),
            max_sampling_attempts=int(random_start_cfg.get("max_sampling_attempts", 10000)),
        ),
        contact_descent=ContactDescentConfig(
            contact_force_threshold_n=float(contact_descent_cfg.get("contact_force_threshold_n", 1e-6)),
            step_size_m=float(contact_descent_cfg.get("step_size_m", 0.001)),
            max_descent_distance_m=float(contact_descent_cfg.get("max_descent_distance_m", 0.03)),
        ),
        motion=MotionConfig(
            position_tolerance_m=float(motion_cfg.get("position_tolerance_m", 0.003)),
            orientation_tolerance_rad=float(motion_cfg.get("orientation_tolerance_rad", np.deg2rad(5.0))),
            translation_speed_mps=float(motion_cfg.get("translation_speed_mps", 0.05)),
            move_timeout_buffer_s=float(motion_cfg.get("move_timeout_buffer_s", 5.0)),
            poll_period_s=float(motion_cfg.get("poll_period_s", DEFAULT_POLL_PERIOD_S)),
        ),
    )
    automation_config.validate()
    return automation_config


class AutomatedRandomExplorationCollector:
    def __init__(
        self,
        data_collection: DataCollection,
        runtime: RandomExplorationRuntime,
        automation_config: AutomationConfig,
        *,
        sensed_force_key: str,
        number_of_trials: int,
        max_episode_duration_s: float,
        monotonic_clock: Callable[[], float] | None = None,
        wait_fn: Callable[[float], None] | None = None,
    ) -> None:
        if not str(sensed_force_key).strip():
            raise ValueError("sensed_force_key must be non-empty.")
        if number_of_trials <= 0:
            raise ValueError("number_of_trials must be positive.")
        if max_episode_duration_s <= 0.0:
            raise ValueError("max_episode_duration_s must be positive.")

        automation_config.validate()
        self.data_collection = data_collection
        self.runtime = runtime
        self.automation_config = automation_config
        self.sensed_force_key = str(sensed_force_key)
        self.number_of_trials = int(number_of_trials)
        self.max_episode_duration_s = float(max_episode_duration_s)
        self.monotonic_clock = monotonic_clock or getattr(runtime, "monotonic_clock", time.monotonic)
        self.wait_fn = wait_fn or time.sleep
        self._runtime_active = False

    @property
    def _motion(self) -> MotionConfig:
        return self.automation_config.motion

    def _wait(self, duration_s: float) -> None:
        sleep_duration = max(float(duration_s), 0.0)
        if sleep_duration > 0.0:
            self.wait_fn(sleep_duration)

    def _compute_move_duration_s(
        self,
        current_world_position: np.ndarray,
        target_world_position: np.ndarray,
    ) -> float:
        distance_m = float(
            np.linalg.norm(
                np.asarray(target_world_position, dtype=float).reshape(3)
                - np.asarray(current_world_position, dtype=float).reshape(3)
            )
        )
        return max(1.0, distance_m / self._motion.translation_speed_mps)

    def _start_runtime(self) -> None:
        if self._runtime_active:
            return
        self.runtime.redis_client.ping()
        _write_scalar(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_force_magnitude,
            0.0,
        )
        _write_vector(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_force,
            np.zeros(3, dtype=float),
        )
        self.runtime.interpolator.start()
        self._runtime_active = True

    def _stop_runtime(self) -> None:
        if not self._runtime_active:
            return
        self.runtime.interpolator.stop()
        _write_scalar(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_force_magnitude,
            0.0,
        )
        _write_vector(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_force,
            np.zeros(3, dtype=float),
        )
        self._runtime_active = False

    def _reset_exploration_state(self) -> None:
        self.runtime.global_step_index = 0

    def _read_current_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _read_vector(self.runtime.redis_client, self.runtime.config.pose_keys.current_position),
            _read_matrix(self.runtime.redis_client, self.runtime.config.pose_keys.current_orientation),
        )

    def _read_current_local_xy(self) -> np.ndarray:
        current_world_position, _ = self._read_current_world_pose()
        return world_to_local_xy(
            current_world_position,
            self.runtime.config.runtime.translation_world,
        )

    def _read_sensed_force_world(self) -> np.ndarray:
        return _read_vector(self.runtime.redis_client, self.sensed_force_key)

    def _current_pose_is_in_hole(self) -> bool:
        return bool(self.runtime.params.point_is_in_hole(self._read_current_local_xy()))

    def _sample_valid_local_start_xy(self) -> np.ndarray:
        max_attempts = int(self.automation_config.random_start.max_sampling_attempts)
        min_distance_from_center = self.automation_config.random_start.min_distance_from_center_m
        max_distance_from_center = self.automation_config.random_start.max_distance_from_center_m
        if max_attempts <= 0:
            raise ValueError("random_start.max_sampling_attempts must be positive.")
        if min_distance_from_center is not None and min_distance_from_center < 0.0:
            raise ValueError("random_start.min_distance_from_center_m must be non-negative when provided.")
        if max_distance_from_center is not None and max_distance_from_center <= 0.0:
            raise ValueError("random_start.max_distance_from_center_m must be positive when provided.")
        if (
            min_distance_from_center is not None
            and max_distance_from_center is not None
            and min_distance_from_center > max_distance_from_center
        ):
            raise ValueError(
                "random_start.min_distance_from_center_m cannot exceed random_start.max_distance_from_center_m."
            )

        rectangle = self.runtime.params.rectangle
        for _ in range(max_attempts):
            candidate = np.array(
                [
                    self.runtime.rng.uniform(rectangle.x_min, rectangle.x_max),
                    self.runtime.rng.uniform(rectangle.y_min, rectangle.y_max),
                ],
                dtype=float,
            )
            distance_from_center = float(np.linalg.norm(candidate - self.runtime.params.goal))
            if (
                min_distance_from_center is not None
                and distance_from_center < float(min_distance_from_center)
            ):
                continue
            if (
                max_distance_from_center is not None
                and distance_from_center > float(max_distance_from_center)
            ):
                continue
            if self.runtime.params.contains_workspace(candidate):
                return candidate

        raise RuntimeError("Failed to sample a valid random local start point from the CAD workspace.")

    def _publish_zero_force_command(self) -> None:
        _write_scalar(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_force_magnitude,
            0.0,
        )
        _write_vector(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_force,
            np.zeros(3, dtype=float),
        )

    def _publish_direct_desired_pose(
        self,
        target_world_position: np.ndarray,
        target_world_orientation: np.ndarray,
    ) -> None:
        _write_vector(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_position,
            np.asarray(target_world_position, dtype=float).reshape(3),
        )
        _write_matrix(
            self.runtime.redis_client,
            self.runtime.config.pose_keys.desired_orientation,
            np.asarray(target_world_orientation, dtype=float).reshape(3, 3),
        )
        self._publish_zero_force_command()

    def _wait_until_pose_reached(
        self,
        target_world_position: np.ndarray,
        target_world_orientation: np.ndarray,
        *,
        timeout_s: float,
    ) -> None:
        deadline = float(self.monotonic_clock()) + float(timeout_s)
        target_position = np.asarray(target_world_position, dtype=float).reshape(3)
        target_orientation = np.asarray(target_world_orientation, dtype=float).reshape(3, 3)

        while True:
            current_world_position, current_world_orientation = self._read_current_world_pose()
            position_error = float(np.linalg.norm(current_world_position - target_position))
            orientation_error = _rotation_error_rad(current_world_orientation, target_orientation)
            if (
                position_error <= self._motion.position_tolerance_m
                and orientation_error <= self._motion.orientation_tolerance_rad
            ):
                return

            now = float(self.monotonic_clock())
            if now >= deadline:
                raise TimeoutError("Timed out while waiting for the robot to reach the commanded pose.")
            self._wait(min(self._motion.poll_period_s, deadline - now))

    def _move_to_world_pose(
        self,
        target_world_position: np.ndarray,
        target_world_orientation: np.ndarray,
    ) -> None:
        current_world_position, _ = self._read_current_world_pose()
        move_duration_s = self._compute_move_duration_s(current_world_position, target_world_position)
        self._publish_direct_desired_pose(
            target_world_position,
            target_world_orientation,
        )
        self._wait_until_pose_reached(
            target_world_position,
            target_world_orientation,
            timeout_s=move_duration_s + self._motion.move_timeout_buffer_s,
        )

    def _move_to_home_pose(self) -> None:
        self._move_to_world_pose(
            self.automation_config.home.position_world,
            self.automation_config.home.orientation_world,
        )

    def _move_to_lateral_start_xy(self, local_start_xy: np.ndarray) -> np.ndarray:
        start_world_position = np.array(
            [
                float(self.runtime.config.runtime.translation_world[0] + local_start_xy[0]),
                float(self.runtime.config.runtime.translation_world[1] + local_start_xy[1]),
                float(self.automation_config.home.position_world[2]),
            ],
            dtype=float,
        )
        self._move_to_world_pose(
            start_world_position,
            self.automation_config.home.orientation_world,
        )
        return start_world_position

    def _descend_until_contact(
        self,
        start_world_position: np.ndarray,
    ) -> np.ndarray:
        current_target = np.asarray(start_world_position, dtype=float).reshape(3).copy()
        step_size_m = float(self.automation_config.contact_descent.step_size_m)
        max_descent_distance_m = float(self.automation_config.contact_descent.max_descent_distance_m)
        contact_force_threshold_n = float(self.automation_config.contact_descent.contact_force_threshold_n)
        descended_distance_m = 0.0

        while descended_distance_m <= max_descent_distance_m + 1e-12:
            sensed_force_world = self._read_sensed_force_world()
            if float(np.linalg.norm(sensed_force_world)) > contact_force_threshold_n:
                return current_target

            if descended_distance_m + step_size_m > max_descent_distance_m + 1e-12:
                break

            current_target[2] -= step_size_m
            self._publish_direct_desired_pose(
                current_target,
                self.automation_config.home.orientation_world,
            )
            self._wait(self._motion.poll_period_s)
            descended_distance_m += step_size_m

        raise TimeoutError(
            "Timed out while descending to contact: sensed force stayed below the configured threshold."
        )

    def _prepare_contact_start(self) -> np.ndarray:
        local_start_xy = self._sample_valid_local_start_xy()
        lateral_start_world_position = self._move_to_lateral_start_xy(local_start_xy)
        self._descend_until_contact(lateral_start_world_position)
        return local_start_xy

    def _run_until_hole_entry(self) -> bool:
        deadline = float(self.monotonic_clock()) + float(self.max_episode_duration_s)
        while not self.runtime.shutdown_event.is_set():
            if self._current_pose_is_in_hole():
                return True

            cycle_start = float(self.monotonic_clock())
            if cycle_start >= deadline:
                return False

            self.runtime.run_cycle(cycle_start)
            wake_time = cycle_start + (
                effective_replan_after(self.runtime.params) / self.runtime.params.action_hz_q
            )

            while not self.runtime.shutdown_event.is_set():
                if self._current_pose_is_in_hole():
                    return True

                now = float(self.monotonic_clock())
                if now >= deadline:
                    return False
                if now >= wake_time:
                    break

                sleep_duration = min(
                    float(self._motion.poll_period_s),
                    wake_time - now,
                    deadline - now,
                )
                if sleep_duration > 0.0 and self.runtime.shutdown_event.wait(sleep_duration):
                    return False

        return False

    def _run_single_trial(self, attempt_index: int) -> bool:
        print(f"Starting automated trial attempt {attempt_index}.", flush=True)
        self._reset_exploration_state()

        recording_started = False
        success = False
        try:
            self._move_to_home_pose()
            local_start_xy = self._prepare_contact_start()
            print(
                "Reached pre-contact random start point at local XY "
                f"{np.array2string(local_start_xy, precision=4, suppress_small=True)}.",
                flush=True,
            )

            self.data_collection.start_recording()
            recording_started = True
            self._start_runtime()
            success = bool(self._run_until_hole_entry())
        except Exception as exc:
            print(f"Automated trial attempt {attempt_index} failed: {exc}", flush=True)
            success = False
        finally:
            self._stop_runtime()
            if recording_started and self.data_collection.recording_active:
                try:
                    self.data_collection.stop_recording()
                except Exception as exc:
                    print(
                        f"Failed to finalize the recorded episode for trial {attempt_index}: {exc}",
                        flush=True,
                    )
                    success = False

        if success:
            print(f"Automated trial attempt {attempt_index} succeeded.", flush=True)
            return True

        if recording_started:
            self.data_collection.delete_latest_episode()
            print(f"Discarded failed automated trial attempt {attempt_index}.", flush=True)
        return False

    def run(self) -> None:
        successful_trials = 0
        attempt_index = 0
        self.data_collection.open(enable_keyboard_listener=False, enable_saving_indicator=False)
        try:
            while successful_trials < self.number_of_trials:
                attempt_index += 1
                if self._run_single_trial(attempt_index):
                    successful_trials += 1
                    print(
                        f"Collected {successful_trials}/{self.number_of_trials} successful automated episodes.",
                        flush=True,
                    )
        finally:
            self._stop_runtime()
            self.data_collection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate random-exploration data collection in simulation."
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        required=True,
        type=str,
        help="Directory that contains the named recording buffers.",
    )
    parser.add_argument(
        "--buffer-name",
        dest="buffer_name",
        required=True,
        type=str,
        help="Name of the recording buffer directory.",
    )
    parser.add_argument(
        "--universal-contract",
        dest="universal_contract",
        required=True,
        type=str,
        help="Path to the universal contract file.",
    )
    parser.add_argument(
        "--runtime-config",
        dest="runtime_config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG_PATH,
        help=f"Path to the random exploration runtime config. Default: {DEFAULT_RUNTIME_CONFIG_PATH}",
    )
    parser.add_argument(
        "--automation-config",
        dest="automation_config",
        type=Path,
        default=DEFAULT_AUTOMATION_CONFIG_PATH,
        help=f"Path to the automation YAML config. Default: {DEFAULT_AUTOMATION_CONFIG_PATH}",
    )
    parser.add_argument(
        "--number_of_trials",
        dest="number_of_trials",
        required=True,
        type=int,
        help="Number of successful episodes to collect.",
    )
    parser.add_argument(
        "--max_episode_duration_s",
        dest="max_episode_duration_s",
        default=DEFAULT_MAX_EPISODE_DURATION_S,
        type=float,
        help="Maximum exploration duration per trial before the episode is discarded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_path = Path(args.universal_contract).expanduser().resolve()
    runtime_config = load_runtime_config(args.runtime_config)
    if runtime_config.universal_contract_path != contract_path:
        raise ValueError(
            "--universal-contract must match the universal contract referenced by --runtime-config."
        )

    contract = _load_yaml_mapping(contract_path, "Universal contract")
    sensed_force_key = resolve_sensed_force_key_from_contract(contract)
    automation_config = load_automation_config(args.automation_config)
    data_collection = DataCollection(args.save_dir, args.buffer_name, str(contract_path))
    runtime = RandomExplorationRuntime(runtime_config)
    automation = AutomatedRandomExplorationCollector(
        data_collection,
        runtime,
        automation_config,
        sensed_force_key=sensed_force_key,
        number_of_trials=args.number_of_trials,
        max_episode_duration_s=args.max_episode_duration_s,
    )
    try:
        automation.run()
    except KeyboardInterrupt:
        print("\nStopping automated random-exploration data collection.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
