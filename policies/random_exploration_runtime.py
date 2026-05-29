from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import redis
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from high_level_controller.interpolator import (  # noqa: E402
    InterpolatorFault,
    TrajectoryInterpolator,
)
from policies.random_exploration_policy import (  # noqa: E402
    PlannerParams,
    effective_replan_after,
    plan_action_poses,
    planner_params_from_generation_metadata_defaults,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("random_exploration_runtime.yaml")
CURRENT_POSITION_SUFFIX = "current_cartesian_position"
CURRENT_ORIENTATION_SUFFIX = "current_cartesian_orientation"
DESIRED_POSITION_SUFFIX = "desired_cartesian_position"
DESIRED_ORIENTATION_SUFFIX = "desired_cartesian_orientation"
DESIRED_FORCE_SUFFIX = "desired_force"
DESIRED_FORCE_MAGNITUDE_SUFFIX = "desired_force_magnitude"
FORCE_DIMENSION_SUFFIX = "force_dimension"
FORCE_OR_MOTION_AXIS_SUFFIX = "force_or_motion_axis"


@dataclass(frozen=True)
class RedisConnectionConfig:
    host: str
    port: int
    db: int

    def validate(self) -> None:
        if not str(self.host).strip():
            raise ValueError("redis.host must be non-empty.")
        if int(self.port) < 0:
            raise ValueError("redis.port must be non-negative.")
        if int(self.db) < 0:
            raise ValueError("redis.db must be non-negative.")


@dataclass(frozen=True)
class RuntimeSettings:
    translation_world: np.ndarray
    interpolator_frequency_hz: float
    blend_duration_s: float
    rng_seed: int | None

    def validate(self) -> None:
        translation = np.asarray(self.translation_world, dtype=float).reshape(-1)
        if translation.size != 3:
            raise ValueError("runtime.translation_world must contain exactly three values.")
        if self.interpolator_frequency_hz <= 0.0:
            raise ValueError("runtime.interpolator_frequency_hz must be positive.")
        if self.blend_duration_s < 0.0:
            raise ValueError("runtime.blend_duration_s must be non-negative.")


@dataclass(frozen=True)
class PoseRedisKeys:
    current_position: str
    current_orientation: str
    desired_position: str
    desired_orientation: str
    desired_force: str
    desired_force_magnitude: str
    force_dimension: str
    force_or_motion_axis: str


@dataclass(frozen=True)
class RandomExplorationRuntimeConfig:
    metadata_path: Path
    universal_contract_path: Path
    redis: RedisConnectionConfig
    runtime: RuntimeSettings
    planner_params: PlannerParams
    pose_keys: PoseRedisKeys

    def validate(self) -> None:
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Generation metadata not found: {self.metadata_path}")
        if not self.universal_contract_path.is_file():
            raise FileNotFoundError(
                f"Universal contract not found: {self.universal_contract_path}"
            )
        self.redis.validate()
        self.runtime.validate()
        self.planner_params.validate()
        if self.planner_params.chunk_length < 2:
            raise ValueError(
                "planner.chunk_length must be at least 2 for TrajectoryInterpolator."
            )


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


def _resolve_input_path(raw_value: str | Path, *, config_dir: Path) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative

    repo_relative = (REPO_ROOT / candidate).resolve()
    if repo_relative.exists():
        return repo_relative

    return config_relative


def _normalize_redis_namespace(redis_namespace: Any) -> str:
    if redis_namespace is None:
        return ""
    return str(redis_namespace).strip(":")


def _make_redis_key(redis_namespace: str, prefix: str, suffix: str) -> str:
    redis_key = f"{str(prefix).rstrip(':')}::{str(suffix).lstrip(':')}"
    if not redis_namespace:
        return redis_key
    return f"{redis_namespace}::{redis_key}"


def resolve_pose_redis_keys_from_contract(contract: dict[str, Any]) -> PoseRedisKeys:
    robot_cfg = contract.get("robot")
    if not isinstance(robot_cfg, dict):
        raise ValueError("Contract must contain a top-level `robot` mapping.")

    prefix = str(robot_cfg.get("prefix", "")).strip()
    if not prefix:
        raise ValueError("Contract robot prefix must be provided.")

    redis_namespace = _normalize_redis_namespace(robot_cfg.get("redis_namespace", "sai"))
    return PoseRedisKeys(
        current_position=_make_redis_key(
            redis_namespace, prefix, CURRENT_POSITION_SUFFIX
        ),
        current_orientation=_make_redis_key(
            redis_namespace, prefix, CURRENT_ORIENTATION_SUFFIX
        ),
        desired_position=_make_redis_key(
            redis_namespace, prefix, DESIRED_POSITION_SUFFIX
        ),
        desired_orientation=_make_redis_key(
            redis_namespace, prefix, DESIRED_ORIENTATION_SUFFIX
        ),
        desired_force=_make_redis_key(
            redis_namespace, prefix, DESIRED_FORCE_SUFFIX
        ),
        desired_force_magnitude=_make_redis_key(
            redis_namespace, prefix, DESIRED_FORCE_MAGNITUDE_SUFFIX
        ),
        force_dimension=_make_redis_key(
            redis_namespace, prefix, FORCE_DIMENSION_SUFFIX
        ),
        force_or_motion_axis=_make_redis_key(
            redis_namespace, prefix, FORCE_OR_MOTION_AXIS_SUFFIX
        ),
    )


def load_runtime_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> RandomExplorationRuntimeConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Runtime config not found: {path}")

    config_dict = _load_yaml_mapping(path, "Runtime config")
    metadata_raw = config_dict.get("metadata_path")
    if metadata_raw is None:
        raise ValueError("Runtime config must define `metadata_path`.")
    contract_raw = config_dict.get("universal_contract")
    if contract_raw is None:
        raise ValueError("Runtime config must define `universal_contract`.")

    metadata_path = _resolve_input_path(metadata_raw, config_dir=path.parent)
    universal_contract_path = _resolve_input_path(contract_raw, config_dir=path.parent)
    redis_cfg = _require_mapping(config_dict, "redis", "runtime config")
    runtime_cfg = _require_mapping(config_dict, "runtime", "runtime config")
    planner_cfg = _require_mapping(config_dict, "planner", "runtime config")

    redis_config = RedisConnectionConfig(
        host=str(redis_cfg["host"]),
        port=int(redis_cfg["port"]),
        db=int(redis_cfg["db"]),
    )
    runtime_settings = RuntimeSettings(
        translation_world=np.asarray(runtime_cfg["translation_world"], dtype=float),
        interpolator_frequency_hz=float(runtime_cfg["interpolator_frequency_hz"]),
        blend_duration_s=float(runtime_cfg["blend_duration_s"]),
        rng_seed=None
        if runtime_cfg.get("rng_seed") is None
        else int(runtime_cfg["rng_seed"]),
    )

    generation_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_z_noise_upper_bound = planner_cfg.get("z_noise_upper_bound")
    planner_defaults = {
        "chunk_length": int(planner_cfg["chunk_length"]),
        "step_length_k": float(planner_cfg["step_length_k"]),
        "replan_every_n_chunks": int(planner_cfg["replan_every_n_chunks"]),
        "action_hz_q": float(planner_cfg["action_hz_q"]),
        "step_noise_std": float(planner_cfg["step_noise_std"]),
        "direction_noise_std_deg": float(planner_cfg["direction_noise_std_deg"]),
        "z_noise_std": float(planner_cfg.get("z_noise_std", 0.0 if raw_z_noise_upper_bound is None else raw_z_noise_upper_bound)),
        "z_noise_lower_bound": float(planner_cfg.get("z_noise_lower_bound", 0.0)),
        "step_noise_decay": float(planner_cfg["step_noise_decay"]),
        "direction_noise_decay": float(planner_cfg["direction_noise_decay"]),
        "force_magnitude_lower_bound": float(planner_cfg.get("force_magnitude_lower_bound", 0.0)),
        "force_magnitude_upper_bound": float(planner_cfg.get("force_magnitude_upper_bound", 0.0)),
    }
    if raw_z_noise_upper_bound is not None:
        planner_defaults["z_noise_upper_bound"] = float(raw_z_noise_upper_bound)
    planner_params = planner_params_from_generation_metadata_defaults(
        generation_metadata,
        planner_defaults,
    )

    contract = _load_yaml_mapping(universal_contract_path, "Universal contract")
    pose_keys = resolve_pose_redis_keys_from_contract(contract)

    runtime_config = RandomExplorationRuntimeConfig(
        metadata_path=metadata_path,
        universal_contract_path=universal_contract_path,
        redis=redis_config,
        runtime=runtime_settings,
        planner_params=planner_params,
        pose_keys=pose_keys,
    )
    runtime_config.validate()
    return runtime_config


def world_to_local_xy(
    world_position: np.ndarray,
    translation_world: np.ndarray,
) -> np.ndarray:
    world = np.asarray(world_position, dtype=float).reshape(-1)
    translation = np.asarray(translation_world, dtype=float).reshape(-1)
    if world.size != 3:
        raise ValueError("world_position must contain exactly three values.")
    if translation.size != 3:
        raise ValueError("translation_world must contain exactly three values.")
    return (world - translation)[:2].astype(float)


def local_chunk_to_world(
    chunk_local: np.ndarray,
    translation_world: np.ndarray,
) -> np.ndarray:
    chunk = np.asarray(chunk_local, dtype=float)
    translation = np.asarray(translation_world, dtype=float).reshape(-1)
    if chunk.ndim != 2 or chunk.shape[1] != 8:
        raise ValueError("chunk_local must have shape (N, 8).")
    if translation.size != 3:
        raise ValueError("translation_world must contain exactly three values.")

    world_chunk = np.array(chunk, copy=True)
    world_chunk[:, :3] += translation.reshape(1, 3)
    return world_chunk


def build_chunk_timestamps(
    cycle_start_time: float,
    chunk_length: int,
    action_hz_q: float,
) -> np.ndarray:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive.")
    if action_hz_q <= 0.0:
        raise ValueError("action_hz_q must be positive.")
    return cycle_start_time + np.arange(chunk_length, dtype=np.float64) * (
        1.0 / action_hz_q
    )


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


def _read_int(redis_client, key: str) -> int:
    raw_value = redis_client.get(key)
    if raw_value is None:
        raise RuntimeError(f"Requested Redis key `{key}` is missing.")
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    return int(raw_value)


def _write_scalar(redis_client, key: str, value: float) -> None:
    redis_client.set(key, json.dumps([float(value)]))


class RandomExplorationRuntime:
    def __init__(
        self,
        config: RandomExplorationRuntimeConfig,
        *,
        redis_client=None,
        interpolator=None,
        monotonic_clock=None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.params = config.planner_params
        self.rng = np.random.default_rng(config.runtime.rng_seed)
        self.redis_client = redis_client or redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=False,
        )
        self.interpolator = interpolator or TrajectoryInterpolator(
            self.redis_client,
            config.pose_keys.desired_position,
            config.pose_keys.desired_orientation,
            config.pose_keys.desired_force,
            config.pose_keys.force_dimension,
            config.pose_keys.force_or_motion_axis,
            desired_force_magnitude_key=config.pose_keys.desired_force_magnitude,
            publish_rate_hz=config.runtime.interpolator_frequency_hz,
            blend_duration=config.runtime.blend_duration_s,
        )
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.shutdown_event = shutdown_event or threading.Event()
        self.global_step_index = 0

    def _read_current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = _read_vector(self.redis_client, self.config.pose_keys.current_position)
        orientation = _read_matrix(
            self.redis_client, self.config.pose_keys.current_orientation
        )
        return position, orientation

    def _validate_local_start(self, local_start_xy: np.ndarray) -> None:
        point_xy = np.asarray(local_start_xy, dtype=float).reshape(2)
        if not self.params.rectangle.contains(point_xy):
            raise ValueError(
                "Measured current Cartesian position maps outside the CAD workspace."
            )
        if self.params.point_is_in_hole(point_xy):
            raise ValueError(
                "Measured current Cartesian position maps inside the hole opening."
            )

    def run_cycle(
        self,
        cycle_start_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cycle_start = (
            float(cycle_start_time)
            if cycle_start_time is not None
            else float(self.monotonic_clock())
        )
        current_world_position, _ = self._read_current_pose()
        local_start_xy = world_to_local_xy(
            current_world_position,
            self.config.runtime.translation_world,
        )
        self._validate_local_start(local_start_xy)

        local_positions, local_orientations = plan_action_poses(
            start_xy=local_start_xy,
            global_step_index=self.global_step_index,
            num_points=self.params.chunk_length,
            params=self.params,
            rng=self.rng,
        )
        chunk_force_magnitude = float(
            self.rng.uniform(
                self.params.force_magnitude_lower,
                self.params.force_magnitude_upper,
            )
        )
        force_magnitudes = np.full(
            (local_positions.shape[0], 1),
            chunk_force_magnitude,
            dtype=float,
        )
        local_chunk = np.hstack((local_positions, local_orientations, force_magnitudes))
        world_chunk = local_chunk_to_world(
            local_chunk,
            self.config.runtime.translation_world,
        )
        _write_scalar(
            self.redis_client,
            self.config.pose_keys.desired_force_magnitude,
            float(world_chunk[0, 7]),
        )
        timestamps = build_chunk_timestamps(
            cycle_start,
            self.params.chunk_length,
            self.params.action_hz_q,
        )
        force_dimension = _read_int(self.redis_client, self.config.pose_keys.force_dimension)
        print(
            f"Force dimension == 1: {force_dimension == 1} (value={force_dimension})",
            flush=True,
        )
        self.interpolator.enqueue_chunk(world_chunk, timestamps)
        self.global_step_index += effective_replan_after(self.params)
        return world_chunk, timestamps

    def run(self, max_cycles: int | None = None) -> None:
        self.redis_client.ping()
        _write_scalar(self.redis_client, self.config.pose_keys.desired_force_magnitude, 0.0)
        self.interpolator.start()
        cycle_count = 0
        try:
            while not self.shutdown_event.is_set():
                cycle_start = float(self.monotonic_clock())
                self.run_cycle(cycle_start)
                cycle_count += 1
                if max_cycles is not None and cycle_count >= max_cycles:
                    return

                wake_time = cycle_start + (
                    effective_replan_after(self.params) / self.params.action_hz_q
                )
                sleep_duration = wake_time - float(self.monotonic_clock())
                if sleep_duration > 0.0 and self.shutdown_event.wait(sleep_duration):
                    return
        finally:
            self.interpolator.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run metadata-driven random exploration on the real robot via "
            "TrajectoryInterpolator."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the runtime YAML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args.config)
    runtime = RandomExplorationRuntime(config)
    print(f"Runtime config: {args.config}", flush=True)
    print(f"Metadata: {config.metadata_path}", flush=True)
    print(
        "Redis keys: "
        f"current_pos=`{config.pose_keys.current_position}` "
        f"current_ori=`{config.pose_keys.current_orientation}` "
        f"desired_pos=`{config.pose_keys.desired_position}` "
        f"desired_ori=`{config.pose_keys.desired_orientation}` "
        f"desired_force=`{config.pose_keys.desired_force}` "
        f"desired_force_magnitude=`{config.pose_keys.desired_force_magnitude}` "
        f"force_dimension=`{config.pose_keys.force_dimension}` "
        f"force_axis=`{config.pose_keys.force_or_motion_axis}`",
        flush=True,
    )
    print(
        "Translation world offset: "
        f"{np.array2string(np.asarray(config.runtime.translation_world), precision=4)}",
        flush=True,
    )
    try:
        runtime.run()
    except KeyboardInterrupt:
        print("\nStopping random exploration runtime.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
