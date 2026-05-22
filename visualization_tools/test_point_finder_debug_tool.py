from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from extractor import point_finder
from visualization_tools import point_finder_debug_tool as mod


class PointFinderDebugToolTests(unittest.TestCase):
    def test_reconstruct_depth_points_matches_expected_world_point(self) -> None:
        frame_bgr = np.full((100, 100, 3), 255, dtype=np.uint8)
        aligned_episode = SimpleNamespace(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            orientations=np.array([np.eye(3, dtype=np.float32)]),
            frames=frame_bgr[None],
        )
        calibration = point_finder.CameraCalibration(
            fovy_degrees=60.0,
            camera_position_world=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            camera_right_world=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            camera_down_world=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            camera_forward_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        contact_spec = point_finder.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)
        diagnostics = point_finder.diagnose_point_selection(
            aligned_episode=aligned_episode,
            camera_calibration=calibration,
            contact_spec=contact_spec,
        )

        keep_indices = np.flatnonzero(diagnostics.final_keep_mask)
        self.assertGreater(len(keep_indices), 0)
        depth_frame_mm = np.zeros(diagnostics.frame_shape, dtype=np.uint16)
        for point_index in keep_indices.tolist():
            pixel_x, pixel_y = diagnostics.rounded_pixels[point_index]
            depth_frame_mm[pixel_y, pixel_x] = np.uint16(round(float(diagnostics.camera_depths[point_index]) * 1000.0))

        world_points, observed_depth_m, point_errors_m = mod.reconstruct_depth_points(
            diagnostics=diagnostics,
            depth_frame_mm=depth_frame_mm,
            camera_calibration=calibration,
        )

        np.testing.assert_allclose(
            observed_depth_m,
            diagnostics.camera_depths[keep_indices],
            atol=1e-3,
        )
        self.assertTrue(np.nanmax(point_errors_m) < 0.008)


if __name__ == "__main__":
    unittest.main()
