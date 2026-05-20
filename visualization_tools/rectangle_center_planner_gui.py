from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policies.random_exploration_policy import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PlannerParams,
    compute_surface_tangent_quaternion,
    direction_noise_std_deg,
    load_planner_params_from_generation_metadata,
    plan_action_poses,
    step_noise_std,
    surface_position_at_xy,
)


SURFACE_MESH_RESOLUTION = 65
SURFACE_PICK_RESOLUTION = 95
SURFACE_PICK_PIXEL_THRESHOLD = 18.0


def _format_vector(vector: np.ndarray | None) -> str:
    if vector is None:
        return "none"
    return np.array2string(
        np.asarray(vector, dtype=float).reshape(-1),
        precision=4,
        suppress_small=True,
        floatmode="fixed",
    )


def effective_replan_after(params: PlannerParams) -> int:
    return max(1, min(params.replan_every_n_chunks, params.chunk_length))


class RectangleCenterPlannerGui:
    def __init__(
        self,
        metadata_path: Path,
        planner_config_path: Path = DEFAULT_CONFIG_PATH,
        seed: int | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Generation metadata not found: {self.metadata_path}")
        self.planner_config_path = Path(planner_config_path).expanduser().resolve()
        self.params = load_planner_params_from_generation_metadata(
            self.metadata_path,
            planner_config_path=self.planner_config_path,
        )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.rng = np.random.default_rng(seed)

        self.start_xy: np.ndarray | None = None
        self.start_position: np.ndarray | None = None
        self.current_position: np.ndarray | None = None
        self.current_orientation_xyzw: np.ndarray | None = None
        self.display_position: np.ndarray | None = None
        self.display_orientation_xyzw: np.ndarray | None = None
        self.executed_positions: list[np.ndarray] = []
        self.replan_positions: list[np.ndarray] = []

        self.planned_positions = np.zeros((0, 3), dtype=float)
        self.planned_orientations = np.zeros((0, 4), dtype=float)
        self.plan_anchor_positions = np.zeros((0, 3), dtype=float)
        self.plan_anchor_orientations = np.zeros((0, 4), dtype=float)
        self.position_splines: tuple[CubicSpline, CubicSpline, CubicSpline] | None = None
        self.orientation_slerp: Slerp | None = None
        self.plan_key_times = np.zeros(0, dtype=float)
        self.plan_cycle_start_wall_time: float | None = None
        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0

        self.running = False
        self.next_step_wall_time: float | None = None
        self.paused_progress_steps: float | None = None
        self.last_status_message = "Click on the 3D surface to start."
        self._last_draw_time = time.perf_counter()

        self.surface_mesh = self._build_surface_mesh(self.params)
        self.surface_pick_points = self._build_surface_pick_points(self.params)

        self.figure = plt.figure("CAD Metadata Surface Planner", figsize=(16.0, 9.0))
        self.surface_axis = self.figure.add_axes([0.05, 0.10, 0.60, 0.82], projection="3d")
        self.info_axis = self.figure.add_axes([0.69, 0.43, 0.27, 0.46])
        self.info_axis.axis("off")

        self.buttons: dict[str, Button] = {}
        self.sliders: dict[str, Slider] = {}
        self._build_controls()

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self.timer = self.figure.canvas.new_timer(interval=30)
        self.timer.add_callback(self._on_timer)
        self._draw()

    def _workspace_mask(
        self,
        mesh_x: np.ndarray,
        mesh_y: np.ndarray,
        params: PlannerParams,
    ) -> np.ndarray:
        mask = np.ones_like(mesh_x, dtype=bool)
        if params.hole_radius > 0.0:
            hole_center = params.hole_center
            hole_distance_sq = (mesh_x - hole_center[0]) ** 2 + (mesh_y - hole_center[1]) ** 2
            mask &= hole_distance_sq >= params.hole_radius * params.hole_radius
        return mask

    def _build_surface_mesh(
        self,
        params: PlannerParams,
        resolution: int = SURFACE_MESH_RESOLUTION,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rectangle = params.rectangle
        x_grid = np.linspace(rectangle.x_min, rectangle.x_max, resolution)
        y_grid = np.linspace(rectangle.y_min, rectangle.y_max, resolution)
        mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="xy")
        mesh_z = np.asarray(params.surface.height(mesh_x, mesh_y), dtype=float)
        valid_mask = self._workspace_mask(mesh_x, mesh_y, params)
        return mesh_x, mesh_y, np.where(valid_mask, mesh_z, np.nan)

    def _build_surface_pick_points(
        self,
        params: PlannerParams,
        resolution: int = SURFACE_PICK_RESOLUTION,
    ) -> np.ndarray:
        rectangle = params.rectangle
        x_grid = np.linspace(rectangle.x_min, rectangle.x_max, resolution)
        y_grid = np.linspace(rectangle.y_min, rectangle.y_max, resolution)
        mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="xy")
        mesh_z = np.asarray(params.surface.height(mesh_x, mesh_y), dtype=float)
        valid_mask = self._workspace_mask(mesh_x, mesh_y, params)
        return np.column_stack((mesh_x[valid_mask], mesh_y[valid_mask], mesh_z[valid_mask]))

    def _build_controls(self) -> None:
        rectangle = self.params.rectangle
        x_span = max(rectangle.x_max - rectangle.x_min, 1e-6)
        y_span = max(rectangle.y_max - rectangle.y_min, 1e-6)
        diagonal_span = float(np.hypot(x_span, y_span))

        step_length_max = max(self.params.step_length_k * 2.5, 0.5 * diagonal_span, 1e-4)
        step_noise_max = max(self.params.step_noise_std_0 * 2.5, step_length_max, 1e-4)
        z_noise_max = max(self.params.z_noise_std * 4.0, 0.1 * max(x_span, y_span), 1e-5)

        slider_specs = [
            ("step_length_k", "Step size", [0.69, 0.33, 0.24, 0.03], 0.0, step_length_max, self.params.step_length_k, "%.4f"),
            ("step_noise_std_0", "XY noise", [0.69, 0.27, 0.24, 0.03], 0.0, step_noise_max, self.params.step_noise_std_0, "%.4f"),
            (
                "direction_noise_std_deg_0",
                "Dir noise (deg)",
                [0.69, 0.21, 0.24, 0.03],
                0.0,
                180.0,
                self.params.direction_noise_std_deg_0,
                "%.1f",
            ),
            ("z_noise_std", "Z noise", [0.69, 0.15, 0.24, 0.03], 0.0, z_noise_max, self.params.z_noise_std, "%.4f"),
        ]
        for slider_key, label, axes_rect, minimum, maximum, initial, value_format in slider_specs:
            slider_axis = self.figure.add_axes(axes_rect)
            slider = Slider(
                slider_axis,
                label,
                minimum,
                maximum,
                valinit=initial,
                valfmt=value_format,
            )
            slider.on_changed(self._on_motion_slider_changed)
            self.sliders[slider_key] = slider

        pause_axis = self.figure.add_axes([0.69, 0.06, 0.08, 0.05])
        reset_axis = self.figure.add_axes([0.79, 0.06, 0.08, 0.05])
        replan_axis = self.figure.add_axes([0.69, 0.01, 0.18, 0.05])

        self.buttons["pause"] = Button(pause_axis, "Pause")
        self.buttons["pause"].on_clicked(self._toggle_pause)
        self.buttons["reset"] = Button(reset_axis, "Reset")
        self.buttons["reset"].on_clicked(self._reset)
        self.buttons["replan_now"] = Button(replan_axis, "Replan Now")
        self.buttons["replan_now"].on_clicked(self._replan_now)

    def _on_motion_slider_changed(self, _) -> None:
        self.params = replace(
            self.params,
            step_length_k=float(self.sliders["step_length_k"].val),
            step_noise_std_0=float(self.sliders["step_noise_std_0"].val),
            direction_noise_std_deg_0=float(
                self.sliders["direction_noise_std_deg_0"].val
            ),
            z_noise_std=float(self.sliders["z_noise_std"].val),
        )
        self.params.validate()

        if self.current_position is not None:
            self._plan_from_current_pose()
            if self.running:
                self.next_step_wall_time = time.perf_counter() + 1.0 / self.params.action_hz_q
                self.last_status_message = (
                    "Updated step/noise sliders and replanned from the current pose."
                )
            else:
                self.last_status_message = (
                    "Updated step/noise sliders. Resume to follow the refreshed plan."
                )
        else:
            self.last_status_message = (
                "Updated step/noise sliders. Click on the 3D surface to start."
            )
        self._draw()

    def _surface_position(self, point_xy: np.ndarray) -> np.ndarray:
        return surface_position_at_xy(point_xy, self.params)

    def _current_progress_steps(self, now: float) -> float:
        if self.plan_cycle_start_wall_time is None or self.plan_anchor_positions.shape[0] == 0:
            return float(self.plan_point_index)

        max_progress = float(max(self.plan_anchor_positions.shape[0] - 1, 0))
        return float(
            np.clip(
                (now - self.plan_cycle_start_wall_time) * self.params.action_hz_q,
                0.0,
                max_progress,
            )
        )

    def _update_display_pose_from_progress(self, progress_steps: float) -> None:
        if self.plan_anchor_positions.shape[0] == 0:
            self.display_position = None if self.current_position is None else np.array(
                self.current_position, copy=True
            )
            self.display_orientation_xyzw = (
                None
                if self.current_orientation_xyzw is None
                else np.array(self.current_orientation_xyzw, copy=True)
            )
            return

        clipped_progress = float(
            np.clip(progress_steps, 0.0, max(self.plan_anchor_positions.shape[0] - 1, 0))
        )

        if self.position_splines is None:
            index = min(
                int(round(clipped_progress)), self.plan_anchor_positions.shape[0] - 1
            )
            self.display_position = np.array(self.plan_anchor_positions[index], copy=True)
        else:
            self.display_position = np.array(
                [
                    float(self.position_splines[0](clipped_progress)),
                    float(self.position_splines[1](clipped_progress)),
                    float(self.position_splines[2](clipped_progress)),
                ],
                dtype=float,
            )

        if self.orientation_slerp is None or self.plan_anchor_orientations.shape[0] == 0:
            index = min(
                int(round(clipped_progress)), self.plan_anchor_orientations.shape[0] - 1
            )
            self.display_orientation_xyzw = np.array(
                self.plan_anchor_orientations[index], copy=True
            )
        else:
            self.display_orientation_xyzw = (
                self.orientation_slerp([clipped_progress]).as_quat()[0].astype(float)
            )

    def _sample_plan_spline(self, start_progress: float = 0.0) -> np.ndarray:
        if self.plan_anchor_positions.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        if self.position_splines is None or self.plan_anchor_positions.shape[0] < 2:
            return self.plan_anchor_positions.copy()

        end_progress = float(self.plan_anchor_positions.shape[0] - 1)
        if start_progress >= end_progress:
            return self.plan_anchor_positions[-1:].copy()

        progress = np.linspace(start_progress, end_progress, 220)
        return np.column_stack(
            [
                self.position_splines[0](progress),
                self.position_splines[1](progress),
                self.position_splines[2](progress),
            ]
        )

    def _current_plan_point_counter(self) -> tuple[int, int]:
        total_points = self.params.chunk_length
        if self.planned_positions.size == 0:
            return 0, total_points

        current_point_index = min(total_points, self.plan_point_index + 1)
        return current_point_index, total_points

    def _points_until_replan(self) -> int:
        replan_after = effective_replan_after(self.params)
        if self.planned_positions.size == 0:
            return 0
        return max(0, replan_after - self.plan_point_index)

    def _plan_from_current_pose(self) -> None:
        if self.current_position is None:
            return

        start_xy = np.asarray(self.current_position[:2], dtype=float)
        planned_positions, planned_orientations = plan_action_poses(
            start_xy=start_xy,
            global_step_index=self.global_step_index,
            num_points=self.params.chunk_length,
            params=self.params,
            rng=self.rng,
        )

        start_anchor_position = self._surface_position(start_xy)
        if self.current_orientation_xyzw is None:
            start_direction_xy = planned_positions[0, :2] - start_xy
            start_orientation_xyzw, _ = compute_surface_tangent_quaternion(
                start_xy,
                start_direction_xy,
                self.params.surface,
            )
            self.current_orientation_xyzw = np.array(start_orientation_xyzw, copy=True)
        start_anchor_orientation = np.array(self.current_orientation_xyzw, copy=True)

        self.planned_positions = planned_positions
        self.planned_orientations = planned_orientations
        self.plan_anchor_positions = np.vstack((start_anchor_position, planned_positions))
        self.plan_anchor_orientations = np.vstack(
            (start_anchor_orientation, planned_orientations)
        )

        self.plan_key_times = np.arange(self.plan_anchor_positions.shape[0], dtype=float)
        if self.plan_anchor_positions.shape[0] >= 2:
            self.position_splines = (
                CubicSpline(self.plan_key_times, self.plan_anchor_positions[:, 0]),
                CubicSpline(self.plan_key_times, self.plan_anchor_positions[:, 1]),
                CubicSpline(self.plan_key_times, self.plan_anchor_positions[:, 2]),
            )
            self.orientation_slerp = Slerp(
                self.plan_key_times,
                Rotation.from_quat(self.plan_anchor_orientations),
            )
        else:
            self.position_splines = None
            self.orientation_slerp = None

        self.plan_point_index = 0
        self.plan_cycle_start_wall_time = time.perf_counter()
        self.replan_count += 1
        self.replan_positions.append(np.array(start_anchor_position, copy=True))
        self.paused_progress_steps = None
        self.last_status_message = (
            f"Replanned batch {self.replan_count} from {_format_vector(start_anchor_position)} "
            f"for {self.params.chunk_length} planned pose(s); "
            f"replan after {effective_replan_after(self.params)} executed pose(s)."
        )
        self._update_display_pose_from_progress(0.0)

    def _start_from_xy(self, point_xy: np.ndarray) -> None:
        point_xy = np.asarray(point_xy, dtype=float).reshape(2)
        if not self.params.contains_workspace(point_xy):
            self.last_status_message = "Ignored click outside the valid top surface."
            return

        start_position = self._surface_position(point_xy)
        self.start_xy = np.array(point_xy, copy=True)
        self.start_position = np.array(start_position, copy=True)
        self.current_position = np.array(start_position, copy=True)
        self.current_orientation_xyzw = None
        self.display_position = np.array(start_position, copy=True)
        self.display_orientation_xyzw = None
        self.executed_positions = [np.array(start_position, copy=True)]
        self.replan_positions = []

        self.planned_positions = np.zeros((0, 3), dtype=float)
        self.planned_orientations = np.zeros((0, 4), dtype=float)
        self.plan_anchor_positions = np.zeros((0, 3), dtype=float)
        self.plan_anchor_orientations = np.zeros((0, 4), dtype=float)
        self.position_splines = None
        self.orientation_slerp = None
        self.plan_key_times = np.zeros(0, dtype=float)
        self.plan_cycle_start_wall_time = None

        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0
        self.running = True
        self.paused_progress_steps = None

        self._plan_from_current_pose()
        self.next_step_wall_time = time.perf_counter() + 1.0 / self.params.action_hz_q

    def _step_once(self) -> None:
        if self.current_position is None or self.planned_positions.size == 0:
            return

        total_points = self.params.chunk_length
        replan_after = effective_replan_after(self.params)
        if self.plan_point_index >= min(total_points, replan_after):
            self._plan_from_current_pose()
            return

        next_position = self.planned_positions[self.plan_point_index]
        next_orientation = self.planned_orientations[self.plan_point_index]
        self.current_position = np.array(next_position, copy=True)
        self.current_orientation_xyzw = np.array(next_orientation, copy=True)
        self.display_position = np.array(next_position, copy=True)
        self.display_orientation_xyzw = np.array(next_orientation, copy=True)
        self.executed_positions.append(np.array(next_position, copy=True))
        self.plan_point_index += 1
        self.global_step_index += 1

        if self.plan_point_index >= min(total_points, replan_after):
            self._plan_from_current_pose()
        else:
            poses_left = max(0, replan_after - self.plan_point_index)
            current_point, total_plan_points = self._current_plan_point_counter()
            self.last_status_message = (
                f"Executed step {self.global_step_index}. "
                f"Planned pose {current_point}/{total_plan_points}, "
                f"{poses_left} pose(s) until replanning."
            )

    def _toggle_pause(self, _) -> None:
        if not self.running:
            self.running = True
            self.buttons["pause"].label.set_text("Pause")
            self.last_status_message = "Simulation resumed."

            now = time.perf_counter()
            if self.paused_progress_steps is not None:
                self.plan_cycle_start_wall_time = (
                    now - self.paused_progress_steps / self.params.action_hz_q
                )
                remaining_until_next_point = max(
                    0.0,
                    (self.plan_point_index + 1 - self.paused_progress_steps)
                    / self.params.action_hz_q,
                )
                self.next_step_wall_time = now + remaining_until_next_point
            else:
                self.next_step_wall_time = now + 1.0 / self.params.action_hz_q
            self.paused_progress_steps = None
        else:
            self.running = False
            self.buttons["pause"].label.set_text("Resume")
            self.last_status_message = "Simulation paused."
            self.paused_progress_steps = self._current_progress_steps(time.perf_counter())

        self.figure.canvas.draw_idle()

    def _reset(self, _) -> None:
        self.start_xy = None
        self.start_position = None
        self.current_position = None
        self.current_orientation_xyzw = None
        self.display_position = None
        self.display_orientation_xyzw = None
        self.executed_positions = []
        self.replan_positions = []

        self.planned_positions = np.zeros((0, 3), dtype=float)
        self.planned_orientations = np.zeros((0, 4), dtype=float)
        self.plan_anchor_positions = np.zeros((0, 3), dtype=float)
        self.plan_anchor_orientations = np.zeros((0, 4), dtype=float)
        self.position_splines = None
        self.orientation_slerp = None
        self.plan_key_times = np.zeros(0, dtype=float)
        self.plan_cycle_start_wall_time = None

        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0
        self.running = False
        self.next_step_wall_time = None
        self.paused_progress_steps = None
        self.last_status_message = "Reset. Click on the 3D surface to start."
        self.buttons["pause"].label.set_text("Pause")
        self._draw()

    def _replan_now(self, _) -> None:
        if self.current_position is None:
            self.last_status_message = "No active start point to replan from."
            self._draw()
            return

        self._plan_from_current_pose()
        if self.running:
            self.next_step_wall_time = time.perf_counter() + 1.0 / self.params.action_hz_q
        self._draw()

    def _on_close(self, _) -> None:
        if self.timer is not None:
            self.timer.stop()

    def _project_world_points_to_pixels(self, points_xyz: np.ndarray) -> np.ndarray:
        self.figure.canvas.draw()
        projected_x, projected_y, _ = proj3d.proj_transform(
            points_xyz[:, 0],
            points_xyz[:, 1],
            points_xyz[:, 2],
            self.surface_axis.get_proj(),
        )
        return self.surface_axis.transData.transform(
            np.column_stack((projected_x, projected_y))
        )

    def _pick_surface_point_from_pixels(
        self,
        pixel_x: float,
        pixel_y: float,
        pixel_threshold: float = SURFACE_PICK_PIXEL_THRESHOLD,
    ) -> np.ndarray | None:
        if self.surface_pick_points.size == 0:
            return None

        projected_pixels = self._project_world_points_to_pixels(self.surface_pick_points)
        deltas = projected_pixels - np.array([pixel_x, pixel_y], dtype=float)
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        min_index = int(np.argmin(distances_sq))
        if float(np.sqrt(distances_sq[min_index])) > pixel_threshold:
            return None
        return np.array(self.surface_pick_points[min_index], copy=True)

    def _on_click(self, event) -> None:
        if event.inaxes is not self.surface_axis or event.x is None or event.y is None:
            return

        picked_point = self._pick_surface_point_from_pixels(float(event.x), float(event.y))
        if picked_point is None:
            self.last_status_message = "Ignored click away from the valid top surface."
            self._draw()
            return

        self.buttons["pause"].label.set_text("Pause")
        self._start_from_xy(picked_point[:2])
        self._draw()

    def _draw_orientation_triad(
        self,
        axis,
        position_xyz: np.ndarray | None,
        quaternion_xyzw: np.ndarray | None,
        length: float,
    ) -> None:
        if position_xyz is None or quaternion_xyzw is None:
            return

        rotation_matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()
        origin = np.asarray(position_xyz, dtype=float).reshape(3)
        colors = ("red", "green", "blue")
        for column_index, color in enumerate(colors):
            direction = rotation_matrix[:, column_index] * length
            axis.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                linewidth=2.0,
                arrow_length_ratio=0.18,
            )

    def _draw_surface_axis(self) -> None:
        axis = self.surface_axis
        axis.clear()

        rectangle = self.params.rectangle
        mesh_x, mesh_y, mesh_z = self.surface_mesh
        x_span = max(rectangle.x_max - rectangle.x_min, 1e-6)
        y_span = max(rectangle.y_max - rectangle.y_min, 1e-6)
        triad_length = 0.08 * max(x_span, y_span)

        axis.plot_surface(
            mesh_x,
            mesh_y,
            np.ma.masked_invalid(mesh_z),
            color="lightsteelblue",
            alpha=0.62,
            linewidth=0,
            antialiased=True,
        )

        if self.params.hole_radius > 0.0:
            theta = np.linspace(0.0, 2.0 * np.pi, 220)
            rim_x = self.params.hole_center[0] + self.params.hole_radius * np.cos(theta)
            rim_y = self.params.hole_center[1] + self.params.hole_radius * np.sin(theta)
            rim_z = np.asarray(self.params.surface.height(rim_x, rim_y), dtype=float)
            axis.plot(
                rim_x,
                rim_y,
                rim_z,
                color="crimson",
                linewidth=2.0,
                label="Hole rim",
            )

        goal_xy = self.params.goal
        goal_z = float(self.params.surface.height(goal_xy[0], goal_xy[1]))
        axis.scatter(
            [goal_xy[0]],
            [goal_xy[1]],
            [goal_z],
            color="crimson",
            s=70,
            marker="x",
            label="Hole center / goal",
        )

        if self.start_position is not None:
            axis.scatter(
                [self.start_position[0]],
                [self.start_position[1]],
                [self.start_position[2]],
                color="goldenrod",
                s=70,
                marker="o",
                label="Start",
                zorder=5,
            )

        if self.executed_positions:
            executed = np.vstack(self.executed_positions)
            axis.plot(
                executed[:, 0],
                executed[:, 1],
                executed[:, 2],
                color="black",
                linewidth=2.0,
                label="Executed poses",
            )

        if self.replan_positions:
            replans = np.vstack(self.replan_positions)
            axis.scatter(
                replans[:, 0],
                replans[:, 1],
                replans[:, 2],
                color="darkorange",
                s=42,
                marker="D",
                label="Replan point",
                zorder=5,
            )

        if self.display_position is not None:
            axis.scatter(
                [self.display_position[0]],
                [self.display_position[1]],
                [self.display_position[2]],
                color="royalblue",
                s=80,
                marker="o",
                label="Current",
                zorder=6,
            )
            self._draw_orientation_triad(
                axis,
                self.display_position,
                self.display_orientation_xyzw,
                triad_length,
            )

        if self.planned_positions.size > 0:
            smooth_path = self._sample_plan_spline(0.0)
            if smooth_path.size > 0:
                axis.plot(
                    smooth_path[:, 0],
                    smooth_path[:, 1],
                    smooth_path[:, 2],
                    linestyle="--",
                    linewidth=1.4,
                    color="teal",
                    alpha=0.9,
                    label="Planned spline",
                )

            axis.scatter(
                self.planned_positions[:, 0],
                self.planned_positions[:, 1],
                self.planned_positions[:, 2],
                color="teal",
                s=28,
                marker="o",
                alpha=0.9,
                label="Planned poses",
            )

        current_point, total_points = self._current_plan_point_counter()
        replan_after = effective_replan_after(self.params)
        axis.text2D(
            0.02,
            0.98,
            f"Planned Pose: {current_point}/{total_points}\nReplan After: {replan_after}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.5",
                "alpha": 0.9,
            },
        )

        finite_mesh_z = mesh_z[np.isfinite(mesh_z)]
        z_candidates = [float(finite_mesh_z.min()), float(finite_mesh_z.max())]
        for pose_set in (self.executed_positions, self.replan_positions):
            if pose_set:
                stacked = np.vstack(pose_set)
                z_candidates.extend([float(stacked[:, 2].min()), float(stacked[:, 2].max())])
        if self.planned_positions.size > 0:
            z_candidates.extend(
                [
                    float(self.planned_positions[:, 2].min()),
                    float(self.planned_positions[:, 2].max()),
                ]
            )
        if self.display_position is not None:
            z_candidates.append(float(self.display_position[2]))
        z_candidates.append(goal_z)

        z_min = min(z_candidates)
        z_max = max(z_candidates)
        z_span = max(z_max - z_min, 0.12 * max(x_span, y_span))
        axis.set_xlim(rectangle.x_min, rectangle.x_max)
        axis.set_ylim(rectangle.y_min, rectangle.y_max)
        axis.set_zlim(z_min - 0.1 * z_span, z_max + 0.15 * z_span)
        axis.set_box_aspect((x_span, y_span, z_span))
        axis.set_title(f"CAD Surface Planner - {self.metadata.get('part_name', 'part')}")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.legend(loc="upper right", fontsize=8, frameon=True)

    def _draw_info(self) -> None:
        self.info_axis.clear()
        self.info_axis.axis("off")

        next_step_sigma = step_noise_std(self.global_step_index, self.params)
        next_dir_sigma = direction_noise_std_deg(self.global_step_index, self.params)
        current_point_number, total_plan_points = self._current_plan_point_counter()
        replan_after = effective_replan_after(self.params)
        points_until_replan = self._points_until_replan()

        info_lines = [
            "Controls",
            "  Click on the 3D top surface to start a new run",
            "  Motion sliders adjust step size and planner noise only",
            "  Pause: stop/resume smooth playback",
            "  Reset: clear the current run",
            "  Replan Now: rebuild future poses from current XY",
            "",
            f"Status: {self.last_status_message}",
            f"Part: {self.metadata.get('part_name', 'unknown')}",
            f"Metadata: {self.metadata_path.name}",
            f"Planner defaults: {self.planner_config_path.name}",
            f"Surface family: {self.params.surface.config.family}",
            f"Goal XY: {_format_vector(self.params.goal)}",
            f"Hole radius: {self.params.hole_radius:.6f}",
            f"Current XYZ: {_format_vector(self.display_position)}",
            f"Current quat xyzw: {_format_vector(self.display_orientation_xyzw)}",
            f"Current anchor XYZ: {_format_vector(self.current_position)}",
            f"Start XYZ: {_format_vector(self.start_position)}",
            f"Global step index: {self.global_step_index}",
            f"Replan count: {self.replan_count}",
            f"Current planned pose: {current_point_number}/{total_plan_points}",
            f"Planned poses per cycle: {self.params.chunk_length}",
            f"Replan after: {replan_after}",
            f"Poses until replan: {points_until_replan}",
            f"Z noise std: {self.params.z_noise_std:.6f}",
            f"Next XY step noise std: {next_step_sigma:.6f}",
            f"Next dir noise std (deg): {next_dir_sigma:.6f}",
            f"Configured action Hz q: {self.params.action_hz_q:.2f}",
        ]
        self.info_axis.text(
            0.0,
            1.0,
            "\n".join(info_lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )

    def _draw(self) -> None:
        self._draw_surface_axis()
        self._draw_info()
        self.figure.canvas.draw_idle()

    def _on_timer(self) -> None:
        now = time.perf_counter()

        if self.running and self.current_position is not None:
            if self.next_step_wall_time is None:
                self.next_step_wall_time = now + 1.0 / self.params.action_hz_q
            step_period = 1.0 / self.params.action_hz_q
            while self.next_step_wall_time is not None and now >= self.next_step_wall_time:
                self._step_once()
                self.next_step_wall_time += step_period

            progress = self._current_progress_steps(now)
            self._update_display_pose_from_progress(progress)

        if now - getattr(self, "_last_draw_time", 0.0) >= 1.0 / 30.0:
            self._last_draw_time = now
            self._draw()

    def show(self) -> None:
        self.timer.start()
        plt.show()

    def close(self) -> None:
        self._on_close(None)
        plt.close(self.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize random exploration over a CAD part top surface loaded from generation metadata."
        )
    )
    parser.add_argument(
        "metadata_json",
        type=Path,
        help="Path to the CAD part generation_metadata.json file.",
    )
    parser.add_argument(
        "--planner-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "YAML planner defaults file used for motion/noise parameters. "
            f"Default: {DEFAULT_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for repeatable planning.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gui = RectangleCenterPlannerGui(
        metadata_path=args.metadata_json,
        planner_config_path=args.planner_config,
        seed=args.seed,
    )
    gui.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
