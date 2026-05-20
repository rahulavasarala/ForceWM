from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

from policies.random_exploration_policy import PlannerParams, RectangleConfig
from policies.surface_models import SurfaceConfig, build_surface_model
from visualization_tools.rectangle_center_planner_gui import effective_replan_after


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
    def test_effective_replan_after_clamps_to_chunk_length(self) -> None:
        self.assertEqual(effective_replan_after(_planner_params(8, 12)), 8)

    def test_effective_replan_after_keeps_valid_value(self) -> None:
        self.assertEqual(effective_replan_after(_planner_params(8, 3)), 3)


if __name__ == "__main__":
    unittest.main()
