from __future__ import annotations

import json
import time
import unittest

import numpy as np

from high_level_controller.interpolator import InterpolatorFault, TrajectoryInterpolator, _Plan


DESIRED_POSITION_KEY = "test::desired_cartesian_position"
DESIRED_ORIENTATION_KEY = "test::desired_cartesian_orientation"
DESIRED_FORCE_KEY = "test::desired_force"
DESIRED_FORCE_MAGNITUDE_KEY = "test::desired_force_magnitude"
FORCE_DIMENSION_KEY = "test::force_dimension"
FORCE_OR_MOTION_AXIS_KEY = "test::force_or_motion_axis"


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {
            FORCE_DIMENSION_KEY: "0",
            FORCE_OR_MOTION_AXIS_KEY: json.dumps([1.0, 0.0, 0.0]),
        }

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value) -> bool:
        self.values[key] = value
        return True


class TrajectoryInterpolatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis_client = _FakeRedis()
        self.interpolator = TrajectoryInterpolator(
            self.redis_client,
            DESIRED_POSITION_KEY,
            DESIRED_ORIENTATION_KEY,
            DESIRED_FORCE_KEY,
            FORCE_DIMENSION_KEY,
            FORCE_OR_MOTION_AXIS_KEY,
            desired_force_magnitude_key=DESIRED_FORCE_MAGNITUDE_KEY,
            publish_rate_hz=100.0,
            blend_duration=1.0,
        )
        self.identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    def test_enqueue_chunk_rejects_non_8d_actions(self) -> None:
        now = time.monotonic()
        actions = np.zeros((2, 7), dtype=float)
        ts = now + np.array([0.1, 0.2], dtype=float)

        with self.assertRaisesRegex(InterpolatorFault, r"shape \(N, 8\)"):
            self.interpolator.enqueue_chunk(actions, ts)

    def test_enqueue_chunk_rejects_negative_force_magnitude(self) -> None:
        now = time.monotonic()
        actions = np.array(
            [
                [0.0, 0.0, 0.0, *self.identity_quat, 0.0],
                [0.1, 0.0, 0.0, *self.identity_quat, -0.5],
            ],
            dtype=float,
        )
        ts = now + np.array([0.1, 0.2], dtype=float)

        with self.assertRaisesRegex(InterpolatorFault, "magnitude_force must be non-negative"):
            self.interpolator.enqueue_chunk(actions, ts)

    def test_plan_sample_uses_adjacent_waypoint_dx_and_linear_force_interpolation(self) -> None:
        actions = np.array(
            [
                [0.0, 0.0, 0.0, *self.identity_quat, 1.0],
                [1.0, 0.0, 0.0, *self.identity_quat, 3.0],
                [1.0, 2.0, 0.0, *self.identity_quat, 5.0],
            ],
            dtype=float,
        )
        plan = _Plan(actions, np.array([0.0, 1.0, 2.0], dtype=float))

        first_segment_sample = plan.sample(0.5)
        second_segment_sample = plan.sample(1.5)
        before_start_sample = plan.sample(-0.1)
        after_end_sample = plan.sample(3.0)

        np.testing.assert_allclose(first_segment_sample.dx_world, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(second_segment_sample.dx_world, np.array([0.0, 2.0, 0.0]))
        self.assertAlmostEqual(first_segment_sample.force_magnitude, 2.0)
        self.assertAlmostEqual(second_segment_sample.force_magnitude, 4.0)
        np.testing.assert_allclose(before_start_sample.dx_world, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(after_end_sample.dx_world, np.array([0.0, 2.0, 0.0]))

    def test_blended_sample_blends_dx_world_and_force_magnitude(self) -> None:
        old_plan = _Plan(
            np.array(
                [
                    [0.0, 0.0, 0.0, *self.identity_quat, 2.0],
                    [1.0, 0.0, 0.0, *self.identity_quat, 2.0],
                ],
                dtype=float,
            ),
            np.array([0.0, 1.0], dtype=float),
        )
        new_plan = _Plan(
            np.array(
                [
                    [1.0, 0.0, 0.0, *self.identity_quat, 4.0],
                    [1.0, 1.0, 0.0, *self.identity_quat, 4.0],
                ],
                dtype=float,
            ),
            np.array([0.0, 1.0], dtype=float),
        )
        self.interpolator._blend = (old_plan, new_plan, 0.0, 1.0)

        sample = self.interpolator._sample(0.5)

        self.assertIsNotNone(sample)
        assert sample is not None
        np.testing.assert_allclose(sample.dx_world, np.array([0.5, 0.5, 0.0]))
        self.assertAlmostEqual(sample.force_magnitude, 3.0)

    def test_desired_force_is_zero_for_force_dimension_zero(self) -> None:
        self.redis_client.values[FORCE_DIMENSION_KEY] = "0"

        desired_force = self.interpolator._desired_force_from_sample(
            2.0,
            np.array([1.0, 0.0, 0.0], dtype=float),
        )

        np.testing.assert_allclose(desired_force, np.zeros(3, dtype=float))

    def test_desired_force_matches_signed_axis_projection_for_force_dimension_one(self) -> None:
        self.redis_client.values[FORCE_DIMENSION_KEY] = "1"
        self.redis_client.values[FORCE_OR_MOTION_AXIS_KEY] = json.dumps([0.0, 2.0, 0.0])

        desired_force = self.interpolator._desired_force_from_sample(
            2.5,
            np.array([0.0, -3.0, 1.0], dtype=float),
        )

        np.testing.assert_allclose(desired_force, np.array([0.0, -2.5, 0.0], dtype=float))

    def test_desired_force_is_zero_when_projection_is_zero(self) -> None:
        self.redis_client.values[FORCE_DIMENSION_KEY] = "1"
        self.redis_client.values[FORCE_OR_MOTION_AXIS_KEY] = json.dumps([1.0, 0.0, 0.0])

        desired_force = self.interpolator._desired_force_from_sample(
            1.5,
            np.array([0.0, 1.0, 0.0], dtype=float),
        )

        np.testing.assert_allclose(desired_force, np.zeros(3, dtype=float))

    def test_desired_force_flips_when_it_points_away_from_negative_z(self) -> None:
        self.redis_client.values[FORCE_DIMENSION_KEY] = "1"
        self.redis_client.values[FORCE_OR_MOTION_AXIS_KEY] = json.dumps([0.0, 0.0, 1.0])

        desired_force = self.interpolator._desired_force_from_sample(
            1.5,
            np.array([0.0, 0.0, 2.0], dtype=float),
        )

        np.testing.assert_allclose(desired_force, np.array([0.0, 0.0, -1.5], dtype=float))

    def test_publish_sample_writes_zero_force_after_plan_finishes(self) -> None:
        now = time.monotonic()
        actions = np.array(
            [
                [0.0, 0.0, 0.0, *self.identity_quat, 1.0],
                [0.2, 0.0, 0.0, *self.identity_quat, 1.0],
            ],
            dtype=float,
        )
        ts = now + np.array([0.1, 0.2], dtype=float)
        self.interpolator.enqueue_chunk(actions, ts)

        sample = self.interpolator._sample(ts[-1] + 0.5)
        self.assertIsNone(sample)

        self.interpolator._publish_sample(sample)

        np.testing.assert_allclose(
            np.asarray(json.loads(self.redis_client.values[DESIRED_FORCE_KEY]), dtype=float),
            np.zeros(3, dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(json.loads(self.redis_client.values[DESIRED_FORCE_MAGNITUDE_KEY]), dtype=float),
            np.array([0.0], dtype=float),
        )

    def test_publish_sample_writes_force_magnitude_scalar_key(self) -> None:
        sample = _Plan(
            np.array(
                [
                    [0.0, 0.0, 0.0, *self.identity_quat, 2.0],
                    [1.0, 0.0, 0.0, *self.identity_quat, 4.0],
                ],
                dtype=float,
            ),
            np.array([0.0, 1.0], dtype=float),
        ).sample(0.5)

        self.interpolator._publish_sample(sample)

        np.testing.assert_allclose(
            np.asarray(json.loads(self.redis_client.values[DESIRED_FORCE_MAGNITUDE_KEY]), dtype=float),
            np.array([3.0], dtype=float),
        )


if __name__ == "__main__":
    unittest.main()
