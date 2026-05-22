from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualization_tools import point_cloud_generator as mod


class PointCloudGeneratorTests(unittest.TestCase):
    def test_load_camera_model_parses_xyaxes(self) -> None:
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

            camera_model = mod.load_camera_model(scene_path, "stationary_camera")

        expected_up = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        expected_up /= np.linalg.norm(expected_up)
        expected_down = -expected_up
        expected_forward = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
        expected_forward /= np.linalg.norm(expected_forward)

        np.testing.assert_allclose(camera_model.position_world, np.array([0.5, 0.0, 0.5], dtype=np.float32))
        np.testing.assert_allclose(camera_model.right_world, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(camera_model.down_world, expected_down, atol=1e-6)
        np.testing.assert_allclose(camera_model.forward_world, expected_forward, atol=1e-6)
        self.assertAlmostEqual(camera_model.fovy_degrees, 60.0)

    def test_build_dense_point_cloud_axial_projects_center_pixel_forward(self) -> None:
        camera_model = mod.CameraModel(
            fovy_degrees=60.0,
            position_world=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            right_world=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            down_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        depth_frame_mm = np.zeros((3, 3), dtype=np.uint16)
        depth_frame_mm[1, 1] = 1000

        world_points, pixels_xy, depth_m = mod.build_dense_point_cloud(
            depth_frame_mm,
            camera_model=camera_model,
            depth_mode="axial",
            stride=1,
        )

        np.testing.assert_allclose(world_points, np.array([[0.0, 0.0, 1.0]], dtype=np.float32), atol=1e-6)
        np.testing.assert_array_equal(pixels_xy, np.array([[1, 1]], dtype=np.int32))
        np.testing.assert_allclose(depth_m, np.array([1.0], dtype=np.float32), atol=1e-6)

    def test_build_dense_point_cloud_range_differs_off_axis(self) -> None:
        camera_model = mod.CameraModel(
            fovy_degrees=60.0,
            position_world=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            right_world=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            down_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        depth_frame_mm = np.zeros((5, 5), dtype=np.uint16)
        depth_frame_mm[0, 4] = 1000

        axial_points, _, _ = mod.build_dense_point_cloud(
            depth_frame_mm,
            camera_model=camera_model,
            depth_mode="axial",
            stride=1,
        )
        range_points, _, _ = mod.build_dense_point_cloud(
            depth_frame_mm,
            camera_model=camera_model,
            depth_mode="range",
            stride=1,
        )

        self.assertEqual(axial_points.shape, (1, 3))
        self.assertEqual(range_points.shape, (1, 3))
        self.assertGreater(float(axial_points[0, 2]), float(range_points[0, 2]))
        self.assertGreater(abs(float(axial_points[0, 0])), abs(float(range_points[0, 0])))


if __name__ == "__main__":
    unittest.main()
