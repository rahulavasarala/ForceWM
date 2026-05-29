from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from extractor import point_finder as mod


class PointFinderTests(unittest.TestCase):
    def _camera_calibration(self) -> mod.CameraCalibration:
        return mod.CameraCalibration(
            fovy_degrees=60.0,
            camera_position_world=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            camera_right_world=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            camera_down_world=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            camera_forward_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    def _aligned_episode(
        self,
        frame_bgr: np.ndarray,
        depth_frames_mm: np.ndarray | None = None,
    ) -> SimpleNamespace:
        episode = SimpleNamespace(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            orientations=np.array([np.eye(3, dtype=np.float32)]),
            frames=frame_bgr[None],
        )
        if depth_frames_mm is not None:
            episode.depth_frames_mm = np.asarray(depth_frames_mm, dtype=np.uint16)
        return episode

    def test_load_contact_cylinder_spec_reads_default_ee_contact(self) -> None:
        spec = mod.load_contact_cylinder_spec()
        self.assertAlmostEqual(spec.radius_m, 0.015)
        self.assertAlmostEqual(spec.half_height_m, 0.05)

    def test_load_camera_calibration_parses_xyaxes_into_camera_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene.xml"
            scene_path.write_text(
                (
                    "<mujoco>"
                    "<worldbody>"
                    '<camera name="stationary_camera" pos="0.5 0.0 0.5" '
                    'xyaxes="0 1 0 -1 0 1" fovy="60"/>'
                    "</worldbody>"
                    "</mujoco>"
                ),
                encoding="utf-8",
            )

            calibration = mod.load_camera_calibration(scene_xml_path=scene_path)

        expected_up = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        expected_up /= np.linalg.norm(expected_up)
        expected_down = -expected_up
        expected_forward = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
        expected_forward /= np.linalg.norm(expected_forward)

        np.testing.assert_allclose(calibration.camera_position_world, np.array([0.5, 0.0, 0.5], dtype=np.float32))
        np.testing.assert_allclose(calibration.camera_right_world, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(calibration.camera_down_world, expected_down, atol=1e-6)
        np.testing.assert_allclose(calibration.camera_forward_world, expected_forward, atol=1e-6)
        self.assertAlmostEqual(calibration.fovy_degrees, 60.0)

    def test_generate_contact_candidate_points_returns_30_bottom_half_samples(self) -> None:
        spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        local_points, local_normals = mod.generate_contact_candidate_points(spec)

        self.assertEqual(local_points.shape, (30, 3))
        self.assertEqual(local_normals.shape, (30, 3))
        np.testing.assert_allclose(np.linalg.norm(local_points[:, :2], axis=1), 0.1, atol=1e-6)
        self.assertTrue(np.all(local_points[:, 2] <= 0.0))
        self.assertTrue(np.all(local_points[:, 2] >= -0.2))
        unique_heights = np.unique(local_points[:, 2])
        np.testing.assert_allclose(
            unique_heights,
            np.array([-0.2, -0.1, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(np.linalg.norm(local_normals[:, :2], axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(local_normals[:, 2], 0.0, atol=1e-6)

    def test_generate_simple_point_template_matches_default_layout(self) -> None:
        spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        template = mod.generate_simple_point_template(spec)

        self.assertEqual(template.shape, (31, 3))
        np.testing.assert_allclose(template[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.assertGreater(np.count_nonzero(np.isclose(template[:, 2], 0.0)), 1)
        self.assertEqual(np.count_nonzero(np.isclose(template[:, 2], -0.2)), 8)
        self.assertEqual(np.count_nonzero(np.isclose(template[:, 2], -0.4)), 8)

    def test_generate_simple_point_template_anchors_bottom_surface_at_eef_point(self) -> None:
        spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        template = mod.generate_simple_point_template(spec)

        bottom_surface = template[:15]
        middle_ring = template[15:23]
        upper_ring = template[23:]

        np.testing.assert_allclose(bottom_surface[:, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(middle_ring[:, 2], -0.2, atol=1e-6)
        np.testing.assert_allclose(upper_ring[:, 2], -0.4, atol=1e-6)

    def test_generate_points_simple_rotates_template_with_orientation(self) -> None:
        spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        theta = np.pi / 2.0
        rotation_z = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )[None]

        point_clouds = mod.generate_points_simple(
            positions_world=positions,
            orientations_world=rotation_z,
            contact_spec=spec,
        )
        template = mod.generate_simple_point_template(spec)

        expected_first_surface_point = template[1] @ rotation_z[0].T + positions[0]
        self.assertEqual(point_clouds.shape, (1, 31, 3))
        np.testing.assert_allclose(point_clouds[0, 1], expected_first_surface_point, atol=1e-6)

    def test_load_point_config_parses_explicit_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "points.yaml"
            config_path.write_text(
                (
                    "bottom_surface:\n"
                    "  include_center: true\n"
                    "  concentric_rings:\n"
                    "    - radius_scale: 0.25\n"
                    "      num_points: 4\n"
                    "middle_ring:\n"
                    "  height_fraction: 0.5\n"
                    "  num_points: 6\n"
                    "upper_ring:\n"
                    "  height_fraction: 1.0\n"
                    "  num_points: 8\n"
                ),
                encoding="utf-8",
            )

            point_config = mod.load_point_config(config_path)

        self.assertTrue(point_config.bottom_surface.include_center)
        self.assertEqual(len(point_config.bottom_surface.concentric_rings), 1)
        self.assertEqual(point_config.bottom_surface.concentric_rings[0].num_points, 4)
        self.assertEqual(point_config.middle_ring.num_points, 6)
        self.assertEqual(point_config.upper_ring.height_fraction, 1.0)

    def test_project_world_points_to_pixels_projects_camera_center(self) -> None:
        calibration = self._camera_calibration()
        projected_pixels, depths = mod.project_world_points_to_pixels(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            camera_calibration=calibration,
            frame_shape=(100, 100),
        )

        self.assertAlmostEqual(projected_pixels[0, 0], 49.5, places=5)
        self.assertAlmostEqual(projected_pixels[0, 1], 49.5, places=5)
        self.assertAlmostEqual(depths[0], 1.0, places=5)

    def test_suppress_occluded_pixels_keeps_nearest_projection(self) -> None:
        keep_indices = mod.suppress_occluded_pixels(
            projected_pixels_xy=np.array(
                [[10.0, 10.0], [10.5, 10.5], [25.0, 25.0]],
                dtype=np.float32,
            ),
            camera_depths=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

        np.testing.assert_array_equal(keep_indices, np.array([0, 2], dtype=np.int64))

    def test_select_points_to_track_accepts_nearby_white_patch_without_snapping(self) -> None:
        calibration = self._camera_calibration()
        contact_spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        frame_bgr = np.zeros((160, 160, 3), dtype=np.uint8)

        local_points, local_normals = mod.generate_contact_candidate_points(contact_spec)
        projected_pixels, depths = mod.project_world_points_to_pixels(
            local_points,
            camera_calibration=calibration,
            frame_shape=frame_bgr.shape[:2],
        )
        rounded_pixels = np.rint(projected_pixels).astype(np.int64)
        facing_scores = np.einsum(
            "ij,ij->i",
            local_normals,
            calibration.camera_position_world[None, :] - local_points,
        )
        valid_mask = depths > mod.MIN_CAMERA_DEPTH
        valid_mask &= facing_scores > 0.0
        valid_mask &= rounded_pixels[:, 0] >= 1
        valid_mask &= rounded_pixels[:, 0] + 1 < frame_bgr.shape[1]
        valid_mask &= rounded_pixels[:, 1] >= 0
        valid_mask &= rounded_pixels[:, 1] < frame_bgr.shape[0]
        candidate_index = int(np.flatnonzero(valid_mask)[0])
        candidate_pixel = rounded_pixels[candidate_index]
        frame_bgr[candidate_pixel[1], candidate_pixel[0] + 1] = np.array([190, 195, 200], dtype=np.uint8)

        sampled_pixels = mod.select_points_to_track(
            aligned_episode=self._aligned_episode(frame_bgr),
            camera_calibration=calibration,
            contact_spec=contact_spec,
        )

        self.assertEqual(sampled_pixels.shape, (1, 2))
        np.testing.assert_array_equal(sampled_pixels[0], candidate_pixel.astype(np.int32))
        self.assertFalse(
            np.array_equal(
                sampled_pixels[0],
                np.array([candidate_pixel[0] + 1, candidate_pixel[1]], dtype=np.int32),
            )
        )

    def test_diagnose_point_selection_reports_kept_and_not_contact_color_reasons(self) -> None:
        calibration = self._camera_calibration()
        contact_spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        frame_bgr = np.zeros((160, 160, 3), dtype=np.uint8)

        local_points, local_normals = mod.generate_contact_candidate_points(contact_spec)
        projected_pixels, depths = mod.project_world_points_to_pixels(
            local_points,
            camera_calibration=calibration,
            frame_shape=frame_bgr.shape[:2],
        )
        rounded_pixels = np.rint(projected_pixels).astype(np.int64)
        facing_scores = np.einsum(
            "ij,ij->i",
            local_normals,
            calibration.camera_position_world[None, :] - local_points,
        )
        valid_mask = depths > mod.MIN_CAMERA_DEPTH
        valid_mask &= facing_scores > 0.0
        valid_mask &= rounded_pixels[:, 0] >= 1
        valid_mask &= rounded_pixels[:, 0] + 1 < frame_bgr.shape[1]
        valid_mask &= rounded_pixels[:, 1] >= 0
        valid_mask &= rounded_pixels[:, 1] < frame_bgr.shape[0]
        candidate_index = int(np.flatnonzero(valid_mask)[0])
        candidate_pixel = rounded_pixels[candidate_index]
        frame_bgr[candidate_pixel[1], candidate_pixel[0] + 1] = np.array([190, 195, 200], dtype=np.uint8)

        diagnostics = mod.diagnose_point_selection(
            aligned_episode=self._aligned_episode(frame_bgr),
            camera_calibration=calibration,
            contact_spec=contact_spec,
        )

        self.assertEqual(diagnostics.rejection_reasons[candidate_index], "kept")
        self.assertTrue(diagnostics.final_keep_mask[candidate_index])
        self.assertGreater(sum(reason == "not_contact_color" for reason in diagnostics.rejection_reasons), 0)
        self.assertEqual(len(diagnostics.rejection_reasons), 30)

    def test_select_points_to_track_rejects_nonwhite_pixels(self) -> None:
        calibration = self._camera_calibration()
        contact_spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        sampled_pixels = mod.select_points_to_track(
            aligned_episode=self._aligned_episode(np.zeros((160, 160, 3), dtype=np.uint8)),
            camera_calibration=calibration,
            contact_spec=contact_spec,
        )

        self.assertEqual(sampled_pixels.shape, (0, 2))

    def test_diagnose_point_selection_rejects_large_reprojection_error(self) -> None:
        calibration = self._camera_calibration()
        contact_spec = mod.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        frame_bgr = np.zeros((160, 160, 3), dtype=np.uint8)
        depth_frame_mm = np.zeros((160, 160), dtype=np.uint16)

        local_points, local_normals = mod.generate_contact_candidate_points(contact_spec)
        projected_pixels, depths = mod.project_world_points_to_pixels(
            local_points,
            camera_calibration=calibration,
            frame_shape=frame_bgr.shape[:2],
        )
        rounded_pixels = np.rint(projected_pixels).astype(np.int64)
        facing_scores = np.einsum(
            "ij,ij->i",
            local_normals,
            calibration.camera_position_world[None, :] - local_points,
        )
        valid_mask = depths > mod.MIN_CAMERA_DEPTH
        valid_mask &= facing_scores > 0.0
        valid_mask &= rounded_pixels[:, 0] >= 0
        valid_mask &= rounded_pixels[:, 0] < frame_bgr.shape[1]
        valid_mask &= rounded_pixels[:, 1] >= 0
        valid_mask &= rounded_pixels[:, 1] < frame_bgr.shape[0]
        candidate_index = int(np.flatnonzero(valid_mask)[0])
        candidate_pixel = rounded_pixels[candidate_index]
        frame_bgr[candidate_pixel[1], candidate_pixel[0]] = np.array([160, 160, 160], dtype=np.uint8)
        depth_frame_mm[candidate_pixel[1], candidate_pixel[0]] = np.uint16(1000)

        diagnostics = mod.diagnose_point_selection(
            aligned_episode=self._aligned_episode(frame_bgr, depth_frames_mm=depth_frame_mm[None]),
            camera_calibration=calibration,
            contact_spec=contact_spec,
        )

        self.assertTrue(diagnostics.depth_filter_applied)
        self.assertEqual(diagnostics.rejection_reasons[candidate_index], "large_reprojection_error")
        self.assertFalse(diagnostics.final_keep_mask[candidate_index])
        self.assertGreater(
            float(diagnostics.reprojection_error_m[candidate_index]),
            mod.DEFAULT_MAX_REPROJECTION_ERROR_M,
        )


if __name__ == "__main__":
    unittest.main()
