from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from policies.random_exploration_policy import (
    PlannerParams,
    RectangleConfig,
    direction_noise_std_deg,
    load_planner_params,
    load_planner_params_from_generation_metadata,
    plan_action_points,
    plan_action_poses,
    plan_chunks,
    step_noise_std,
)
from policies.surface_models import SurfaceConfig, build_surface_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "generated_cad" / "default_part" / "generation_metadata.json"
)


def _surface_config(
    *,
    family: str = "default",
    base_height: float = 0.03,
    amp: float = 0.0,
    freq_x: float = 8.0,
    freq_y: float = 6.0,
    seed: int = 0,
    gaussian_curvature: float = 0.08,
    gaussian_peak_offset: float = 0.18,
    origin_x: float = 0.5,
    origin_y: float = 0.5,
) -> SurfaceConfig:
    return SurfaceConfig(
        family=family,
        base_height=base_height,
        amp=amp,
        freq_x=freq_x,
        freq_y=freq_y,
        seed=seed,
        gaussian_curvature=gaussian_curvature,
        gaussian_peak_offset=gaussian_peak_offset,
        origin_x=origin_x,
        origin_y=origin_y,
    )


class RandomExplorationPolicyTests(unittest.TestCase):
    def test_load_planner_params_rejects_invalid_rectangle(self) -> None:
        yaml_text = """
rectangle:
  x_min: 1.0
  x_max: 1.0
  y_min: 0.0
  y_max: 1.0
surface:
  family: default
  base_height: 0.03
  amp: 0.003
  freq_x: 8.0
  freq_y: 6.0
  seed: 0
  gaussian_curvature: 0.08
  gaussian_peak_offset: 0.18
defaults:
  chunk_length: 4
  step_length_k: 0.1
  replan_every_n_chunks: 2
  action_hz_q: 10.0
  step_noise_std: 0.0
  direction_noise_std_deg: 0.0
  z_noise_std: 0.0
  step_noise_decay: 1.0
  direction_noise_decay: 1.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "planner.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_planner_params(config_path)

    def test_load_planner_params_rejects_missing_surface_family_params(self) -> None:
        yaml_text = """
rectangle:
  x_min: 0.0
  x_max: 1.0
  y_min: 0.0
  y_max: 1.0
surface:
  family: random_gaussian_two_peak
  base_height: 0.03
  amp: 0.003
  seed: 0
  gaussian_curvature: 0.08
defaults:
  chunk_length: 4
  step_length_k: 0.1
  replan_every_n_chunks: 2
  action_hz_q: 10.0
  step_noise_std: 0.0
  direction_noise_std_deg: 0.0
  z_noise_std: 0.0
  step_noise_decay: 1.0
  direction_noise_decay: 1.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "planner.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_planner_params(config_path)

    def test_load_planner_params_reads_surface_defaults(self) -> None:
        yaml_text = """
rectangle:
  x_min: -1.0
  x_max: 2.0
  y_min: -0.5
  y_max: 0.5
surface:
  family: default
  base_height: 0.02
  amp: 0.004
  freq_x: 5.0
  freq_y: 7.0
  seed: 0
  gaussian_curvature: 0.08
  gaussian_peak_offset: 0.18
defaults:
  chunk_length: 6
  step_length_k: 0.07
  replan_every_n_chunks: 3
  action_hz_q: 12.0
  step_noise_std: 0.02
  direction_noise_std_deg: 15.0
  z_noise_lower_bound: 0.0001
  z_noise_upper_bound: 0.0008
  step_noise_decay: 0.91
  direction_noise_decay: 0.83
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "planner.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            params = load_planner_params(config_path)

        self.assertEqual(params.chunk_length, 6)
        self.assertAlmostEqual(params.step_length_k, 0.07)
        self.assertEqual(params.replan_every_n_chunks, 3)
        self.assertAlmostEqual(params.action_hz_q, 12.0)
        self.assertAlmostEqual(params.rectangle.x_min, -1.0)
        self.assertAlmostEqual(params.rectangle.y_max, 0.5)
        self.assertAlmostEqual(params.surface.config.base_height, 0.02)
        self.assertAlmostEqual(params.surface.config.freq_y, 7.0)
        self.assertAlmostEqual(params.z_noise_lower, 0.0001)
        self.assertAlmostEqual(params.z_noise_upper, 0.0008)

    def test_noise_decay_is_monotonic(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
            surface=build_surface_model(_surface_config()),
            chunk_length=4,
            step_length_k=0.1,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.05,
            direction_noise_std_deg_0=30.0,
            z_noise_std=0.0,
            step_noise_decay=0.9,
            direction_noise_decay=0.8,
        )
        step_stds = [step_noise_std(step, params) for step in range(5)]
        dir_stds = [direction_noise_std_deg(step, params) for step in range(5)]
        self.assertEqual(step_stds, sorted(step_stds, reverse=True))
        self.assertEqual(dir_stds, sorted(dir_stds, reverse=True))

    def test_validate_rejects_z_noise_lower_bound_above_upper_bound(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
            surface=build_surface_model(_surface_config()),
            chunk_length=4,
            step_length_k=0.1,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.05,
            direction_noise_std_deg_0=30.0,
            z_noise_std=0.0,
            step_noise_decay=0.9,
            direction_noise_decay=0.8,
            z_noise_lower_bound=0.002,
            z_noise_upper_bound=0.001,
        )

        with self.assertRaisesRegex(ValueError, "z_noise_lower_bound must be <= z_noise_upper_bound"):
            params.validate()

    def test_validate_rejects_force_magnitude_lower_bound_above_upper_bound(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
            surface=build_surface_model(_surface_config()),
            chunk_length=4,
            step_length_k=0.1,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.05,
            direction_noise_std_deg_0=30.0,
            z_noise_std=0.0,
            step_noise_decay=0.9,
            direction_noise_decay=0.8,
            force_magnitude_lower_bound=2.0,
            force_magnitude_upper_bound=1.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "force_magnitude_lower_bound must be <= force_magnitude_upper_bound",
        ):
            params.validate()

    def test_plan_action_poses_preserves_xy_and_sets_surface_z_without_noise(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 2.0, -1.0, 1.0),
            surface=build_surface_model(
                _surface_config(base_height=0.01, amp=0.002, origin_x=1.0, origin_y=0.0)
            ),
            chunk_length=4,
            step_length_k=0.5,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.0,
            direction_noise_std_deg_0=0.0,
            z_noise_std=0.0,
            step_noise_decay=1.0,
            direction_noise_decay=1.0,
        )
        xy_rng = np.random.default_rng(0)
        pose_rng = np.random.default_rng(0)
        points_xy = plan_action_points(
            start_xy=np.array([0.0, 0.0]),
            global_step_index=0,
            num_points=4,
            params=params,
            rng=xy_rng,
        )
        positions_xyz, _ = plan_action_poses(
            start_xy=np.array([0.0, 0.0]),
            global_step_index=0,
            num_points=4,
            params=params,
            rng=pose_rng,
        )

        np.testing.assert_allclose(positions_xyz[:, :2], points_xy, atol=1e-9)
        expected_z = params.surface.height(points_xy[:, 0], points_xy[:, 1])
        np.testing.assert_allclose(positions_xyz[:, 2], expected_z, atol=1e-9)

    def test_plan_action_poses_applies_downward_uniform_z_offset(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 2.0, -1.0, 1.0),
            surface=build_surface_model(
                _surface_config(base_height=0.01, amp=0.002, origin_x=1.0, origin_y=0.0)
            ),
            chunk_length=4,
            step_length_k=0.5,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.0,
            direction_noise_std_deg_0=0.0,
            z_noise_std=0.0008,
            step_noise_decay=1.0,
            direction_noise_decay=1.0,
            z_noise_lower_bound=0.0002,
            z_noise_upper_bound=0.0008,
        )
        pose_rng = np.random.default_rng(123)
        positions_xyz, _ = plan_action_poses(
            start_xy=np.array([0.0, 0.0]),
            global_step_index=0,
            num_points=4,
            params=params,
            rng=pose_rng,
        )

        expected_surface_z = params.surface.height(positions_xyz[:, 0], positions_xyz[:, 1])
        z_offsets = expected_surface_z - positions_xyz[:, 2]

        self.assertTrue(np.all(z_offsets >= params.z_noise_lower - 1e-12))
        self.assertTrue(np.all(z_offsets <= params.z_noise_upper + 1e-12))

    def test_load_planner_params_from_generation_metadata_uses_part_bounds(self) -> None:
        params = load_planner_params_from_generation_metadata(DEFAULT_METADATA_PATH)
        self.assertAlmostEqual(params.rectangle.x_min, -0.05)
        self.assertAlmostEqual(params.rectangle.x_max, 0.05)
        self.assertAlmostEqual(params.goal[0], 0.0)
        self.assertAlmostEqual(params.goal[1], 0.0)
        self.assertGreater(params.hole_radius, 0.0)

    def test_plan_action_points_rejects_start_inside_hole(self) -> None:
        params = load_planner_params_from_generation_metadata(DEFAULT_METADATA_PATH)
        with self.assertRaisesRegex(ValueError, "outside the hole opening"):
            plan_action_points(
                start_xy=np.array(params.goal, copy=True),
                global_step_index=0,
                num_points=3,
                params=params,
                rng=np.random.default_rng(0),
            )

    def test_plan_action_points_stops_at_hole_rim(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
            surface=build_surface_model(_surface_config(base_height=0.0, amp=0.0)),
            chunk_length=4,
            step_length_k=0.25,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.0,
            direction_noise_std_deg_0=0.0,
            z_noise_std=0.0,
            step_noise_decay=1.0,
            direction_noise_decay=1.0,
            goal_xy=(0.5, 0.5),
            hole_center_xy=(0.5, 0.5),
            hole_radius=0.1,
        )
        start_xy = np.array([0.72, 0.5], dtype=float)

        points_xy = plan_action_points(
            start_xy=start_xy,
            global_step_index=0,
            num_points=4,
            params=params,
            rng=np.random.default_rng(0),
        )

        expected_rim_point = np.array([0.6, 0.5], dtype=float)
        np.testing.assert_allclose(points_xy[0], expected_rim_point, atol=1e-9)
        np.testing.assert_allclose(points_xy, np.tile(expected_rim_point, (4, 1)), atol=1e-9)

    def test_plan_action_points_stay_within_rectangle_and_outside_hole(self) -> None:
        params = load_planner_params_from_generation_metadata(DEFAULT_METADATA_PATH)
        points_xy = plan_action_points(
            start_xy=np.array([params.rectangle.x_max, 0.0], dtype=float),
            global_step_index=0,
            num_points=12,
            params=params,
            rng=np.random.default_rng(2),
        )

        for point_xy in points_xy:
            self.assertTrue(params.rectangle.contains(point_xy))
            self.assertFalse(params.point_is_in_hole(point_xy))

    def test_random_gaussian_surface_is_deterministic_for_fixed_seed(self) -> None:
        surface_a = build_surface_model(
            _surface_config(
                family="random_gaussian_two_peak",
                amp=0.004,
                seed=7,
                gaussian_curvature=0.09,
                gaussian_peak_offset=0.15,
            )
        )
        surface_b = build_surface_model(
            _surface_config(
                family="random_gaussian_two_peak",
                amp=0.004,
                seed=7,
                gaussian_curvature=0.09,
                gaussian_peak_offset=0.15,
            )
        )

        sample_x = np.array([0.2, 0.5, 0.8])
        sample_y = np.array([0.1, 0.3, 0.7])
        np.testing.assert_allclose(
            surface_a.height(sample_x, sample_y),
            surface_b.height(sample_x, sample_y),
            atol=1e-12,
        )
        grad_a = surface_a.gradient(sample_x, sample_y)
        grad_b = surface_b.gradient(sample_x, sample_y)
        np.testing.assert_allclose(grad_a[0], grad_b[0], atol=1e-12)
        np.testing.assert_allclose(grad_a[1], grad_b[1], atol=1e-12)

    def test_quaternion_orientation_is_normalized_and_matches_surface_underside_normal(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
            surface=build_surface_model(
                _surface_config(base_height=0.02, amp=0.003, origin_x=0.5, origin_y=0.5)
            ),
            chunk_length=3,
            step_length_k=0.18,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.0,
            direction_noise_std_deg_0=0.0,
            z_noise_std=0.0,
            step_noise_decay=1.0,
            direction_noise_decay=1.0,
        )
        positions_xyz, orientations_xyzw = plan_action_poses(
            start_xy=np.array([0.15, 0.2]),
            global_step_index=0,
            num_points=3,
            params=params,
            rng=np.random.default_rng(3),
        )

        norms = np.linalg.norm(orientations_xyzw, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-9)
        for point_xyz, quaternion_xyzw in zip(
            positions_xyz, orientations_xyzw, strict=True
        ):
            rotation_matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()
            z_axis = rotation_matrix[:, 2]
            dzdx, dzdy = params.surface.gradient(point_xyz[0], point_xyz[1])
            expected_normal = np.array(
                [float(dzdx), float(dzdy), -1.0], dtype=float
            )
            expected_normal = expected_normal / np.linalg.norm(expected_normal)
            self.assertGreater(float(np.dot(z_axis, expected_normal)), 0.999)

    def test_flat_surface_orientation_matches_requested_base_matrix(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
            surface=build_surface_model(_surface_config(base_height=0.0, amp=0.0)),
            chunk_length=2,
            step_length_k=0.2,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.0,
            direction_noise_std_deg_0=0.0,
            z_noise_std=0.0,
            step_noise_decay=1.0,
            direction_noise_decay=1.0,
        )
        positions_xyz, orientations_xyzw = plan_action_poses(
            start_xy=np.array([0.0, 0.0]),
            global_step_index=0,
            num_points=2,
            params=params,
            rng=np.random.default_rng(1),
        )

        first_rotation = Rotation.from_quat(orientations_xyzw[0]).as_matrix()
        expected_rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(first_rotation, expected_rotation, atol=1e-9)

    def test_surface_orientation_uses_projected_world_x_not_travel_direction(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
            surface=build_surface_model(
                _surface_config(base_height=0.03, amp=0.002, origin_x=0.5, origin_y=0.5)
            ),
            chunk_length=2,
            step_length_k=0.2,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.0,
            direction_noise_std_deg_0=0.0,
            z_noise_std=0.0,
            step_noise_decay=1.0,
            direction_noise_decay=1.0,
        )
        positions_xyz, orientations_xyzw = plan_action_poses(
            start_xy=np.array([0.5, 0.9]),
            global_step_index=0,
            num_points=2,
            params=params,
            rng=np.random.default_rng(1),
        )

        point_xyz = positions_xyz[0]
        rotation_matrix = Rotation.from_quat(orientations_xyzw[0]).as_matrix()
        dzdx, dzdy = params.surface.gradient(point_xyz[0], point_xyz[1])
        expected_z_axis = np.array([float(dzdx), float(dzdy), -1.0], dtype=float)
        expected_z_axis = expected_z_axis / np.linalg.norm(expected_z_axis)

        expected_x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        expected_x_axis = expected_x_axis - expected_z_axis * float(
            np.dot(expected_x_axis, expected_z_axis)
        )
        expected_x_axis = expected_x_axis / np.linalg.norm(expected_x_axis)

        np.testing.assert_allclose(rotation_matrix[:, 2], expected_z_axis, atol=1e-9)
        np.testing.assert_allclose(rotation_matrix[:, 0], expected_x_axis, atol=1e-9)

    def test_plan_chunks_returns_single_pose_per_chunk(self) -> None:
        params = PlannerParams(
            rectangle=RectangleConfig(0.0, 2.0, 0.0, 2.0),
            surface=build_surface_model(_surface_config(base_height=0.03, origin_x=1.0, origin_y=1.0)),
            chunk_length=8,
            step_length_k=0.25,
            replan_every_n_chunks=1,
            action_hz_q=10.0,
            step_noise_std_0=0.0,
            direction_noise_std_deg_0=0.0,
            z_noise_std=0.0,
            step_noise_decay=1.0,
            direction_noise_decay=1.0,
        )
        chunks = plan_chunks(
            start_xy=np.array([0.0, 0.0]),
            global_step_index=0,
            num_chunks=8,
            params=params,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(len(chunks), 8)
        for chunk in chunks:
            self.assertEqual(chunk.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
