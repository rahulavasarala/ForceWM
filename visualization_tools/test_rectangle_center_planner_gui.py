from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from policies.random_exploration_policy import PlannerParams, RectangleConfig  # noqa: E402
from policies.surface_models import SurfaceConfig, build_surface_model  # noqa: E402
from visualization_tools.rectangle_center_planner_gui import (  # noqa: E402
    RectangleCenterPlannerGui,
    effective_replan_after,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "generated_cad" / "default_part" / "generation_metadata.json"
)


def _planner_params(chunk_length: int, replan_after: int) -> PlannerParams:
    return PlannerParams(
        rectangle=RectangleConfig(0.0, 1.0, 0.0, 1.0),
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
                origin_x=0.5,
                origin_y=0.5,
            )
        ),
        chunk_length=chunk_length,
        step_length_k=0.1,
        replan_every_n_chunks=replan_after,
        action_hz_q=10.0,
        step_noise_std_0=0.0,
        direction_noise_std_deg_0=0.0,
        z_noise_std=0.0,
        step_noise_decay=1.0,
        direction_noise_decay=1.0,
    )


class RectangleCenterPlannerGuiTests(unittest.TestCase):
    def _make_gui(self) -> RectangleCenterPlannerGui:
        gui = RectangleCenterPlannerGui(DEFAULT_METADATA_PATH)
        self.addCleanup(gui.close)
        return gui

    def _project_point_to_event(self, gui: RectangleCenterPlannerGui, point_xyz: np.ndarray) -> SimpleNamespace:
        point_xyz = np.asarray(point_xyz, dtype=float).reshape(1, 3)
        pixel_xy = gui._project_world_points_to_pixels(point_xyz)[0]
        return SimpleNamespace(
            inaxes=gui.surface_axis,
            x=float(pixel_xy[0]),
            y=float(pixel_xy[1]),
            xdata=None,
            ydata=None,
        )

    def test_effective_replan_after_clamps_to_chunk_length(self) -> None:
        self.assertEqual(effective_replan_after(_planner_params(8, 12)), 8)

    def test_effective_replan_after_keeps_valid_value(self) -> None:
        self.assertEqual(effective_replan_after(_planner_params(8, 3)), 3)

    def test_gui_loads_metadata_surface_with_motion_sliders(self) -> None:
        gui = self._make_gui()
        self.assertFalse(hasattr(gui, "selection_axis"))
        self.assertAlmostEqual(gui.params.rectangle.x_min, -0.05)
        self.assertAlmostEqual(gui.params.rectangle.x_max, 0.05)
        self.assertEqual(set(gui.buttons.keys()), {"pause", "reset", "replan_now"})
        self.assertEqual(
            set(gui.sliders.keys()),
            {
                "step_length_k",
                "step_noise_std_0",
                "direction_noise_std_deg_0",
                "z_noise_std",
            },
        )

    def test_motion_sliders_update_runtime_planner_params(self) -> None:
        gui = self._make_gui()

        gui.sliders["step_length_k"].set_val(0.025)
        gui.sliders["step_noise_std_0"].set_val(0.01)
        gui.sliders["direction_noise_std_deg_0"].set_val(22.0)
        gui.sliders["z_noise_std"].set_val(0.0012)

        self.assertAlmostEqual(gui.params.step_length_k, 0.025)
        self.assertAlmostEqual(gui.params.step_noise_std_0, 0.01)
        self.assertAlmostEqual(gui.params.direction_noise_std_deg_0, 22.0)
        self.assertAlmostEqual(gui.params.z_noise_std, 0.0012)
        self.assertIn("Updated step/noise sliders", gui.last_status_message)

    def test_surface_axis_limits_match_part_dimensions(self) -> None:
        gui = self._make_gui()
        gui.figure.canvas.draw()
        x_limits = gui.surface_axis.get_xlim()
        y_limits = gui.surface_axis.get_ylim()

        self.assertAlmostEqual(x_limits[0], gui.params.rectangle.x_min)
        self.assertAlmostEqual(x_limits[1], gui.params.rectangle.x_max)
        self.assertAlmostEqual(y_limits[0], gui.params.rectangle.y_min)
        self.assertAlmostEqual(y_limits[1], gui.params.rectangle.y_max)

    def test_clicking_valid_surface_point_starts_run(self) -> None:
        gui = self._make_gui()
        point_xyz = gui.surface_pick_points[len(gui.surface_pick_points) // 2]
        event = self._project_point_to_event(gui, point_xyz)

        gui._on_click(event)

        self.assertIsNotNone(gui.start_position)
        self.assertTrue(gui.running)

    def test_clicking_hole_opening_is_ignored(self) -> None:
        gui = self._make_gui()
        hole_center = gui.params.goal
        hole_center_xyz = np.array(
            [
                hole_center[0],
                hole_center[1],
                float(gui.params.surface.height(hole_center[0], hole_center[1])),
            ],
            dtype=float,
        )
        event = self._project_point_to_event(gui, hole_center_xyz)

        gui._on_click(event)

        self.assertIsNone(gui.start_position)
        self.assertIn("Ignored click", gui.last_status_message)

    def test_info_panel_reflects_metadata_surface_family(self) -> None:
        gui = self._make_gui()
        gui._draw_info()
        info_text = gui.info_axis.texts[0].get_text()

        self.assertIn("Part: default_part", info_text)
        self.assertIn("Surface family: default", info_text)


if __name__ == "__main__":
    unittest.main()
