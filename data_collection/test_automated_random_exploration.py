from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_collection.automated_random_exploration import (
    AutomatedRandomExplorationCollector,
    AutomationConfig,
    ContactDescentConfig,
    DEFAULT_HOME_ORIENTATION_WORLD,
    DEFAULT_HOME_POSITION_WORLD,
    HomeConfig,
    MotionConfig,
    RandomStartConfig,
    load_automation_config,
)
from policies.random_exploration_policy import PlannerParams, RectangleConfig
from policies.surface_models import SurfaceConfig, build_surface_model


def _automation_config() -> AutomationConfig:
    return AutomationConfig(
        home=HomeConfig(
            position_world=np.array(DEFAULT_HOME_POSITION_WORLD, copy=True),
            orientation_world=np.array(DEFAULT_HOME_ORIENTATION_WORLD, copy=True),
        ),
        random_start=RandomStartConfig(
            min_distance_from_center_m=0.02,
            max_distance_from_center_m=0.05,
            max_sampling_attempts=123,
        ),
        contact_descent=ContactDescentConfig(
            contact_force_threshold_n=0.1,
            step_size_m=0.002,
            max_descent_distance_m=0.01,
        ),
        motion=MotionConfig(
            position_tolerance_m=0.003,
            orientation_tolerance_rad=float(np.deg2rad(5.0)),
            translation_speed_mps=0.05,
            move_timeout_buffer_s=5.0,
            poll_period_s=0.02,
        ),
    )


def _planner_params() -> PlannerParams:
    return PlannerParams(
        rectangle=RectangleConfig(-0.1, 0.1, -0.1, 0.1),
        surface=build_surface_model(
            SurfaceConfig(
                family="default",
                base_height=0.03,
                amp=0.0,
                freq_x=8.0,
                freq_y=6.0,
                seed=0,
                gaussian_curvature=0.08,
                gaussian_peak_offset=0.18,
                origin_x=0.0,
                origin_y=0.0,
            )
        ),
        chunk_length=4,
        step_length_k=0.0,
        replan_every_n_chunks=1,
        action_hz_q=10.0,
        step_noise_std_0=0.0,
        direction_noise_std_deg_0=0.0,
        z_noise_std=0.0,
        step_noise_decay=1.0,
        direction_noise_decay=1.0,
        force_magnitude_lower_bound=0.0,
        force_magnitude_upper_bound=0.0,
        goal_xy=(0.0, 0.0),
        hole_center_xy=(0.0, 0.0),
        hole_radius=0.01,
    )


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = float(start)

    def monotonic(self) -> float:
        return self.now

    def advance(self, duration: float) -> None:
        self.now += max(float(duration), 0.0)


class _FakeShutdownEvent:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True

    def wait(self, duration: float) -> bool:
        self.clock.advance(duration)
        return self._is_set


class _FakeRng:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def uniform(self, _low: float, _high: float) -> float:
        if not self.values:
            raise AssertionError("Fake RNG ran out of values.")
        return float(self.values.pop(0))


class _FakeDataCollection:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._recording_active = False

    @property
    def recording_active(self) -> bool:
        return bool(self._recording_active)

    def open(self, *, enable_keyboard_listener: bool, enable_saving_indicator: bool) -> None:
        self.events.append(
            f"collector:open:{int(enable_keyboard_listener)}:{int(enable_saving_indicator)}"
        )

    def close(self) -> None:
        self.events.append("collector:close")

    def start_recording(self) -> None:
        self._recording_active = True
        self.events.append("collector:start_recording")

    def stop_recording(self) -> None:
        self._recording_active = False
        self.events.append("collector:stop_recording")

    def delete_latest_episode(self) -> None:
        self.events.append("collector:delete_latest_episode")


class _FakeRedisClient:
    def __init__(
        self,
        events: list[str],
        pose_keys,
        sensed_force_key: str,
        *,
        initial_position: np.ndarray,
        initial_orientation: np.ndarray,
        sensed_force_sequences: list[list[np.ndarray]],
    ) -> None:
        self.events = events
        self.pose_keys = pose_keys
        self.sensed_force_key = sensed_force_key
        self.current_position = np.asarray(initial_position, dtype=float).reshape(3)
        self.current_orientation = np.asarray(initial_orientation, dtype=float).reshape(3, 3)
        self.sensed_force_sequences = [
            [np.asarray(force, dtype=float).reshape(3) for force in sequence]
            for sequence in sensed_force_sequences
        ]
        self.sensed_force_attempt_index = 0
        self.sensed_force_read_index = 0
        self.values: dict[str, str] = {}
        self.ping_count = 0

    def ping(self) -> bool:
        self.ping_count += 1
        return True

    def start_attempt(self) -> None:
        self.sensed_force_read_index = 0

    def finish_attempt(self) -> None:
        self.sensed_force_attempt_index += 1
        self.sensed_force_read_index = 0

    def set_current_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self.current_position = np.asarray(position, dtype=float).reshape(3)
        self.current_orientation = np.asarray(orientation, dtype=float).reshape(3, 3)

    def get(self, key: str):
        if key == self.pose_keys.current_position:
            return json.dumps(self.current_position.tolist()).encode("utf-8")
        if key == self.pose_keys.current_orientation:
            return json.dumps(self.current_orientation.tolist()).encode("utf-8")
        if key == self.sensed_force_key:
            sequence_index = min(
                self.sensed_force_attempt_index,
                len(self.sensed_force_sequences) - 1,
            )
            sequence = self.sensed_force_sequences[sequence_index]
            force_index = min(self.sensed_force_read_index, len(sequence) - 1)
            self.sensed_force_read_index += 1
            return json.dumps(sequence[force_index].tolist()).encode("utf-8")
        stored_value = self.values.get(key)
        if stored_value is None:
            return None
        return stored_value.encode("utf-8")

    def set(self, key: str, value) -> bool:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        self.values[key] = str(value)
        if key == self.pose_keys.desired_position:
            target_position = np.asarray(json.loads(str(value)), dtype=float).reshape(3)
            if np.allclose(target_position, DEFAULT_HOME_POSITION_WORLD):
                self.events.append("runtime:direct_home_command")
            elif np.isclose(target_position[2], DEFAULT_HOME_POSITION_WORLD[2]):
                self.events.append("runtime:direct_lateral_command")
            else:
                self.events.append("runtime:direct_descent_command")
            self.current_position = target_position
        elif key == self.pose_keys.desired_orientation:
            self.current_orientation = np.asarray(json.loads(str(value)), dtype=float).reshape(3, 3)
        return True


class _FakeInterpolator:
    def __init__(self, redis_client: _FakeRedisClient, events: list[str]) -> None:
        self.redis_client = redis_client
        self.events = events
        self.started = False
        self.stopped = False
        self.enqueued: list[tuple[np.ndarray, np.ndarray]] = []

    def start(self) -> None:
        self.started = True
        self.events.append("runtime:start")

    def stop(self) -> None:
        self.stopped = True
        self.events.append("runtime:stop")

    def enqueue_chunk(self, actions, ts) -> None:
        actions_array = np.asarray(actions, dtype=float).copy()
        timestamps = np.asarray(ts, dtype=float).copy()
        self.enqueued.append((actions_array, timestamps))
        self.events.append("runtime:enqueue_policy_chunk")


class _FakeRuntime:
    def __init__(
        self,
        events: list[str],
        *,
        outcomes: list[bool],
        rng_values: list[float],
        sensed_force_key: str,
        sensed_force_sequences: list[list[np.ndarray]],
    ) -> None:
        self.events = events
        self.outcomes = list(outcomes)
        self.params = _planner_params()
        self.global_step_index = 0
        self.config = SimpleNamespace(
            runtime=SimpleNamespace(
                translation_world=np.array([0.4, 0.0, 0.3], dtype=float),
            ),
            pose_keys=SimpleNamespace(
                current_position="test::current_cartesian_position",
                current_orientation="test::current_cartesian_orientation",
                desired_position="test::desired_cartesian_position",
                desired_orientation="test::desired_cartesian_orientation",
                desired_force="test::desired_force",
                desired_force_magnitude="test::desired_force_magnitude",
            ),
        )
        self.clock = _FakeClock()
        self.shutdown_event = _FakeShutdownEvent(self.clock)
        self.rng = _FakeRng(rng_values)
        self.redis_client = _FakeRedisClient(
            events,
            self.config.pose_keys,
            sensed_force_key,
            initial_position=np.array([0.41, 0.0, 0.36], dtype=float),
            initial_orientation=np.array(DEFAULT_HOME_ORIENTATION_WORLD, copy=True),
            sensed_force_sequences=sensed_force_sequences,
        )
        self.interpolator = _FakeInterpolator(self.redis_client, events)

    def run_cycle(self, _cycle_start_time: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.events.append("runtime:run_cycle")
        if not self.outcomes:
            raise AssertionError("No more fake outcomes were configured.")
        outcome = bool(self.outcomes.pop(0))
        if outcome:
            self.redis_client.set_current_pose(
                self.config.runtime.translation_world + np.array([0.0, 0.0, 0.33], dtype=float),
                np.array(DEFAULT_HOME_ORIENTATION_WORLD, copy=True),
            )
        self.global_step_index += self.params.chunk_length
        return np.zeros((self.params.chunk_length, 8), dtype=float), np.linspace(0.0, 0.3, self.params.chunk_length)


class AutomatedRandomExplorationCollectorTests(unittest.TestCase):
    def test_load_automation_config_parses_home_and_random_start_settings(self) -> None:
        config_text = """
home:
  position_world: [0.5, 0.1, 0.42]
  orientation_world:
    - [1.0, 0.0, 0.0]
    - [0.0, -1.0, 0.0]
    - [0.0, 0.0, -1.0]

random_start:
  min_distance_from_center_m: 0.03
  max_distance_from_center_m: 0.07
  max_sampling_attempts: 456

contact_descent:
  contact_force_threshold_n: 0.2
  step_size_m: 0.003
  max_descent_distance_m: 0.05

motion:
  position_tolerance_m: 0.004
  orientation_tolerance_rad: 0.1
  translation_speed_mps: 0.06
  move_timeout_buffer_s: 6.0
  poll_period_s: 0.03
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "automation.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            config = load_automation_config(config_path)

        np.testing.assert_allclose(config.home.position_world, np.array([0.5, 0.1, 0.42], dtype=float))
        self.assertAlmostEqual(config.random_start.min_distance_from_center_m, 0.03)
        self.assertAlmostEqual(config.random_start.max_distance_from_center_m, 0.07)
        self.assertEqual(config.random_start.max_sampling_attempts, 456)
        self.assertAlmostEqual(config.contact_descent.contact_force_threshold_n, 0.2)

    def test_sample_valid_local_start_xy_respects_distance_limit(self) -> None:
        events: list[str] = []
        runtime = _FakeRuntime(
            events,
            outcomes=[True],
            rng_values=[0.0, 0.0, 0.06, 0.0, 0.04, -0.02],
            sensed_force_key="test::sensed_force",
            sensed_force_sequences=[[np.zeros(3, dtype=float)]],
        )
        collector = AutomatedRandomExplorationCollector(
            _FakeDataCollection(events),
            runtime,
            _automation_config(),
            sensed_force_key="test::sensed_force",
            number_of_trials=1,
            max_episode_duration_s=1.0,
            monotonic_clock=runtime.clock.monotonic,
            wait_fn=runtime.clock.advance,
        )

        sampled_xy = collector._sample_valid_local_start_xy()

        np.testing.assert_allclose(sampled_xy, np.array([0.04, -0.02], dtype=float))
        self.assertGreaterEqual(float(np.linalg.norm(sampled_xy - runtime.params.goal)), 0.02 - 1e-9)
        self.assertLessEqual(float(np.linalg.norm(sampled_xy - runtime.params.goal)), 0.05 + 1e-9)
        self.assertTrue(runtime.params.contains_workspace(sampled_xy))

    def test_successful_trial_records_only_after_lateral_move_and_contact_descent(self) -> None:
        events: list[str] = []
        collector = _FakeDataCollection(events)
        runtime = _FakeRuntime(
            events,
            outcomes=[True],
            rng_values=[0.04, -0.02],
            sensed_force_key="test::sensed_force",
            sensed_force_sequences=[
                [np.zeros(3, dtype=float), np.array([0.0, 0.0, 0.2], dtype=float)]
            ],
        )
        automation = AutomatedRandomExplorationCollector(
            collector,
            runtime,
            _automation_config(),
            sensed_force_key="test::sensed_force",
            number_of_trials=1,
            max_episode_duration_s=0.05,
            monotonic_clock=runtime.clock.monotonic,
            wait_fn=runtime.clock.advance,
        )

        automation.run()

        self.assertEqual(events[0], "collector:open:0:0")
        self.assertLess(events.index("runtime:direct_home_command"), events.index("runtime:direct_lateral_command"))
        self.assertLess(events.index("runtime:direct_lateral_command"), events.index("runtime:direct_descent_command"))
        self.assertLess(events.index("runtime:direct_descent_command"), events.index("collector:start_recording"))
        self.assertLess(events.index("collector:start_recording"), events.index("runtime:start"))
        self.assertIn("runtime:run_cycle", events)
        self.assertIn("collector:stop_recording", events)
        self.assertNotIn("collector:delete_latest_episode", events)
        self.assertEqual(events[-1], "collector:close")

    def test_failed_trial_is_deleted_and_retried_until_success_target(self) -> None:
        events: list[str] = []
        collector = _FakeDataCollection(events)
        runtime = _FakeRuntime(
            events,
            outcomes=[False, True],
            rng_values=[0.04, -0.02, 0.03, 0.01],
            sensed_force_key="test::sensed_force",
            sensed_force_sequences=[
                [np.zeros(3, dtype=float), np.array([0.0, 0.0, 0.2], dtype=float)],
                [np.zeros(3, dtype=float), np.array([0.0, 0.0, 0.2], dtype=float)],
            ],
        )
        automation = AutomatedRandomExplorationCollector(
            collector,
            runtime,
            _automation_config(),
            sensed_force_key="test::sensed_force",
            number_of_trials=1,
            max_episode_duration_s=0.05,
            monotonic_clock=runtime.clock.monotonic,
            wait_fn=runtime.clock.advance,
        )

        original_run_single_trial = automation._run_single_trial

        def wrapped_run_single_trial(attempt_index: int) -> bool:
            try:
                return original_run_single_trial(attempt_index)
            finally:
                runtime.redis_client.finish_attempt()

        automation._run_single_trial = wrapped_run_single_trial  # type: ignore[method-assign]
        automation.run()

        self.assertEqual(events.count("collector:start_recording"), 2)
        self.assertEqual(events.count("collector:stop_recording"), 2)
        self.assertEqual(events.count("collector:delete_latest_episode"), 1)
        self.assertGreaterEqual(events.count("runtime:run_cycle"), 2)
        self.assertEqual(events[-1], "collector:close")


if __name__ == "__main__":
    unittest.main()
