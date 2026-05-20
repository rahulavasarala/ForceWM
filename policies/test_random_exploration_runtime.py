from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from high_level_controller.interpolator import InterpolatorFault
from policies.random_exploration_policy import PlannerParams, RectangleConfig
from policies.random_exploration_runtime import (
    PoseRedisKeys,
    RandomExplorationRuntime,
    RandomExplorationRuntimeConfig,
    RedisConnectionConfig,
    RuntimeSettings,
    load_runtime_config,
    local_chunk_to_world,
    resolve_pose_redis_keys_from_contract,
    world_to_local_xy,
)
from policies.surface_models import SurfaceConfig, build_surface_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "generated_cad" / "default_part" / "generation_metadata.json"
)
DEFAULT_CONTRACT_PATH = REPO_ROOT / "universal_contract.yaml"


def _surface_config(
    *,
    base_height: float = 0.03,
    amp: float = 0.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> SurfaceConfig:
    return SurfaceConfig(
        family="default",
        base_height=base_height,
        amp=amp,
        freq_x=8.0,
        freq_y=6.0,
        seed=0,
        gaussian_curvature=0.08,
        gaussian_peak_offset=0.18,
        origin_x=origin_x,
        origin_y=origin_y,
    )


def _planner_params(
    *,
    chunk_length: int = 4,
    replan_every_n_chunks: int = 2,
    step_length_k: float = 0.0,
    hole_radius: float = 0.0,
) -> PlannerParams:
    return PlannerParams(
        rectangle=RectangleConfig(-0.5, 0.5, -0.5, 0.5),
        surface=build_surface_model(_surface_config()),
        chunk_length=chunk_length,
        step_length_k=step_length_k,
        replan_every_n_chunks=replan_every_n_chunks,
        action_hz_q=10.0,
        step_noise_std_0=0.0,
        direction_noise_std_deg_0=0.0,
        z_noise_std=0.0,
        step_noise_decay=1.0,
        direction_noise_decay=1.0,
        goal_xy=(0.0, 0.0),
        hole_center_xy=(0.0, 0.0),
        hole_radius=hole_radius,
    )


def _runtime_config(
    *,
    planner_params: PlannerParams | None = None,
    translation_world: np.ndarray | None = None,
) -> RandomExplorationRuntimeConfig:
    return RandomExplorationRuntimeConfig(
        metadata_path=DEFAULT_METADATA_PATH,
        universal_contract_path=DEFAULT_CONTRACT_PATH,
        redis=RedisConnectionConfig(host="127.0.0.1", port=6379, db=0),
        runtime=RuntimeSettings(
            translation_world=np.asarray(
                [0.4, 0.0, 0.3] if translation_world is None else translation_world,
                dtype=float,
            ),
            interpolator_frequency_hz=100.0,
            blend_duration_s=0.1,
            rng_seed=0,
        ),
        planner_params=_planner_params() if planner_params is None else planner_params,
        pose_keys=PoseRedisKeys(
            current_position="test::current_cartesian_position",
            current_orientation="test::current_cartesian_orientation",
            desired_position="test::desired_cartesian_position",
            desired_orientation="test::desired_cartesian_orientation",
        ),
    )


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = float(start)

    def monotonic(self) -> float:
        return self.now


class _FakeEvent:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.waits: list[float] = []
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True

    def wait(self, duration: float) -> bool:
        self.waits.append(float(duration))
        self.clock.now += max(float(duration), 0.0)
        return self._is_set


class _FakeInterpolator:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.enqueued: list[tuple[np.ndarray, np.ndarray]] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def enqueue_chunk(self, actions, ts) -> None:
        self.enqueued.append(
            (
                np.asarray(actions, dtype=float).copy(),
                np.asarray(ts, dtype=float).copy(),
            )
        )


class _FailingInterpolator(_FakeInterpolator):
    def enqueue_chunk(self, actions, ts) -> None:
        raise InterpolatorFault("boom")


class _SequencedRedisClient:
    def __init__(
        self,
        pose_keys: PoseRedisKeys,
        current_positions: list[np.ndarray],
        current_orientations: list[np.ndarray] | None = None,
    ) -> None:
        self.pose_keys = pose_keys
        self.current_positions = [
            np.asarray(position, dtype=float).reshape(3)
            for position in current_positions
        ]
        orientations = current_orientations or [np.eye(3, dtype=float)]
        self.current_orientations = [
            np.asarray(orientation, dtype=float).reshape(3, 3)
            for orientation in orientations
        ]
        self.position_read_count = 0
        self.orientation_read_count = 0
        self.ping_count = 0

    def ping(self) -> bool:
        self.ping_count += 1
        return True

    def get(self, key: str):
        if key == self.pose_keys.current_position:
            index = min(
                self.position_read_count, len(self.current_positions) - 1
            )
            self.position_read_count += 1
            return json.dumps(self.current_positions[index].tolist()).encode("utf-8")
        if key == self.pose_keys.current_orientation:
            index = min(
                self.orientation_read_count, len(self.current_orientations) - 1
            )
            self.orientation_read_count += 1
            return json.dumps(
                self.current_orientations[index].tolist()
            ).encode("utf-8")
        return None


class RandomExplorationRuntimeTests(unittest.TestCase):
    def test_load_runtime_config_parses_paths_and_planner_params(self) -> None:
        metadata = {
            "block_dimensions": {
                "length": 0.2,
                "width": 0.1,
                "height": 0.04,
            },
            "hole_dimensions": {
                "radius": 0.015,
                "center_x": 0.01,
                "center_y": -0.01,
            },
            "surface": {
                "family": "default",
                "base_height": 0.001,
                "amp": 0.0025,
                "freq_x": 3.0,
                "freq_y": 5.0,
                "seed": 0,
                "gaussian_curvature": 0.08,
                "gaussian_peak_offset": 0.18,
                "origin_x": 0.0,
                "origin_y": 0.0,
                "gaussian_centers_local": [],
                "gaussian_peak_amps": [],
                "gaussian_sigma": 0.08,
            },
        }
        contract_text = """
robot:
  redis_namespace: demo
  prefix: arm::tool
"""
        runtime_text = """
metadata_path: generation_metadata.json
universal_contract: contract.yaml

redis:
  host: 127.0.0.1
  port: 6380
  db: 2

runtime:
  translation_world: [0.4, 0.0, 0.3]
  interpolator_frequency_hz: 120.0
  blend_duration_s: 0.05
  rng_seed: 7

planner:
  chunk_length: 6
  replan_every_n_chunks: 3
  action_hz_q: 15.0
  step_length_k: 0.02
  step_noise_std: 0.01
  direction_noise_std_deg: 12.0
  z_noise_std: 0.0004
  step_noise_decay: 0.91
  direction_noise_decay: 0.82
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metadata_path = tmp_path / "generation_metadata.json"
            contract_path = tmp_path / "contract.yaml"
            config_path = tmp_path / "runtime.yaml"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            contract_path.write_text(contract_text, encoding="utf-8")
            config_path.write_text(runtime_text, encoding="utf-8")

            config = load_runtime_config(config_path)

        self.assertEqual(config.metadata_path, metadata_path.resolve())
        self.assertEqual(config.universal_contract_path, contract_path.resolve())
        self.assertEqual(config.redis.host, "127.0.0.1")
        self.assertEqual(config.redis.port, 6380)
        self.assertEqual(config.redis.db, 2)
        np.testing.assert_allclose(
            config.runtime.translation_world, np.array([0.4, 0.0, 0.3], dtype=float)
        )
        self.assertAlmostEqual(config.runtime.interpolator_frequency_hz, 120.0)
        self.assertAlmostEqual(config.runtime.blend_duration_s, 0.05)
        self.assertEqual(config.runtime.rng_seed, 7)
        self.assertEqual(config.planner_params.chunk_length, 6)
        self.assertEqual(config.planner_params.replan_every_n_chunks, 3)
        self.assertAlmostEqual(config.planner_params.step_length_k, 0.02)
        self.assertAlmostEqual(config.planner_params.step_noise_std_0, 0.01)
        self.assertAlmostEqual(config.planner_params.direction_noise_std_deg_0, 12.0)
        self.assertAlmostEqual(config.planner_params.z_noise_std, 0.0004)
        self.assertAlmostEqual(config.planner_params.rectangle.x_min, -0.1)
        self.assertAlmostEqual(config.planner_params.rectangle.x_max, 0.1)
        self.assertAlmostEqual(config.planner_params.goal[0], 0.01)
        self.assertAlmostEqual(config.planner_params.goal[1], -0.01)
        self.assertEqual(
            config.pose_keys.current_position,
            "demo::arm::tool::current_cartesian_position",
        )
        self.assertEqual(
            config.pose_keys.desired_orientation,
            "demo::arm::tool::desired_cartesian_orientation",
        )

    def test_resolve_pose_redis_keys_from_contract_uses_namespace_and_prefix(self) -> None:
        pose_keys = resolve_pose_redis_keys_from_contract(
            {
                "robot": {
                    "redis_namespace": "sai",
                    "prefix": "sim::franka",
                }
            }
        )

        self.assertEqual(
            pose_keys.current_position,
            "sai::sim::franka::current_cartesian_position",
        )
        self.assertEqual(
            pose_keys.current_orientation,
            "sai::sim::franka::current_cartesian_orientation",
        )
        self.assertEqual(
            pose_keys.desired_position,
            "sai::sim::franka::desired_cartesian_position",
        )
        self.assertEqual(
            pose_keys.desired_orientation,
            "sai::sim::franka::desired_cartesian_orientation",
        )

    def test_local_world_transform_helpers_translate_positions_only(self) -> None:
        translation = np.array([0.4, 0.0, 0.3], dtype=float)
        local_chunk = np.array(
            [
                [0.02, -0.01, 0.03, 0.1, 0.2, 0.3, 0.9],
                [0.04, 0.05, 0.04, -0.1, 0.0, 0.4, 0.91],
            ],
            dtype=float,
        )

        world_chunk = local_chunk_to_world(local_chunk, translation)
        local_xy = world_to_local_xy(world_chunk[0, :3], translation)

        np.testing.assert_allclose(world_chunk[:, :3], local_chunk[:, :3] + translation)
        np.testing.assert_allclose(world_chunk[:, 3:], local_chunk[:, 3:])
        np.testing.assert_allclose(local_xy, local_chunk[0, :2])

    def test_runtime_run_schedules_chunks_and_replans_from_measured_current_pose(self) -> None:
        config = _runtime_config(planner_params=_planner_params())
        current_positions = [
            np.array([0.42, 0.01, 0.33], dtype=float),
            np.array([0.44, 0.02, 0.33], dtype=float),
        ]
        fake_redis = _SequencedRedisClient(config.pose_keys, current_positions)
        fake_interpolator = _FakeInterpolator()
        fake_clock = _FakeClock(start=100.0)
        fake_event = _FakeEvent(fake_clock)

        runtime = RandomExplorationRuntime(
            config,
            redis_client=fake_redis,
            interpolator=fake_interpolator,
            monotonic_clock=fake_clock.monotonic,
            shutdown_event=fake_event,
        )
        runtime.run(max_cycles=2)

        self.assertTrue(fake_interpolator.started)
        self.assertTrue(fake_interpolator.stopped)
        self.assertEqual(fake_redis.ping_count, 1)
        self.assertEqual(len(fake_interpolator.enqueued), 2)
        self.assertEqual(len(fake_event.waits), 1)
        self.assertAlmostEqual(fake_event.waits[0], 0.2)

        first_chunk, first_ts = fake_interpolator.enqueued[0]
        second_chunk, second_ts = fake_interpolator.enqueued[1]
        np.testing.assert_allclose(
            first_ts,
            100.0 + np.arange(config.planner_params.chunk_length, dtype=float) * 0.1,
        )
        np.testing.assert_allclose(
            second_ts,
            100.2 + np.arange(config.planner_params.chunk_length, dtype=float) * 0.1,
        )
        np.testing.assert_allclose(first_chunk[0, :3], current_positions[0], atol=1e-9)
        np.testing.assert_allclose(second_chunk[0, :3], current_positions[1], atol=1e-9)
        self.assertEqual(runtime.global_step_index, 4)

    def test_runtime_stops_and_surfaces_interpolator_fault(self) -> None:
        config = _runtime_config(planner_params=_planner_params())
        fake_redis = _SequencedRedisClient(
            config.pose_keys,
            [np.array([0.42, 0.01, 0.33], dtype=float)],
        )
        failing_interpolator = _FailingInterpolator()

        runtime = RandomExplorationRuntime(
            config,
            redis_client=fake_redis,
            interpolator=failing_interpolator,
        )

        with self.assertRaisesRegex(InterpolatorFault, "boom"):
            runtime.run(max_cycles=1)

        self.assertTrue(failing_interpolator.started)
        self.assertTrue(failing_interpolator.stopped)

    def test_runtime_rejects_local_start_outside_workspace(self) -> None:
        config = _runtime_config(planner_params=_planner_params())
        fake_redis = _SequencedRedisClient(
            config.pose_keys,
            [np.array([1.2, 0.0, 0.33], dtype=float)],
        )
        runtime = RandomExplorationRuntime(
            config,
            redis_client=fake_redis,
            interpolator=_FakeInterpolator(),
        )

        with self.assertRaisesRegex(ValueError, "outside the CAD workspace"):
            runtime.run_cycle(10.0)

    def test_runtime_rejects_local_start_inside_hole(self) -> None:
        config = _runtime_config(
            planner_params=_planner_params(hole_radius=0.05),
        )
        fake_redis = _SequencedRedisClient(
            config.pose_keys,
            [np.array([0.4, 0.0, 0.33], dtype=float)],
        )
        runtime = RandomExplorationRuntime(
            config,
            redis_client=fake_redis,
            interpolator=_FakeInterpolator(),
        )

        with self.assertRaisesRegex(ValueError, "inside the hole opening"):
            runtime.run_cycle(10.0)


if __name__ == "__main__":
    unittest.main()
