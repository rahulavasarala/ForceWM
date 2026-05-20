from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.widgets import Button, Slider
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
    load_planner_params,
    plan_action_poses,
    step_noise_std,
    surface_position_at_xy,
)


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
    def __init__(self, config_path: Path, seed: int | None = None) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.initial_params = load_planner_params(self.config_path)
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
        self.active_plan_params: PlannerParams | None = None
        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0

        self.running = False
        self.next_step_wall_time: float | None = None
        self.paused_progress_steps: float | None = None
        self.last_status_message = "Click inside the XY selector to start."
        self.last_draw_time = time.perf_counter()

        self.surface_mesh = self._build_surface_mesh(self.initial_params)

        self.figure = plt.figure("Rectangle-Center 3D Planner", figsize=(16.0, 9.0))
        self.surface_axis = self.figure.add_axes([0.05, 0.10, 0.58, 0.82], projection="3d")
        self.selection_axis = self.figure.add_axes([0.67, 0.77, 0.28, 0.16])
        self.info_axis = self.figure.add_axes([0.67, 0.51, 0.28, 0.21])
        self.info_axis.axis("off")

        self.sliders: dict[str, Slider] = {}
        self.buttons: dict[str, Button] = {}
        self._build_controls()

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self.timer = self.figure.canvas.new_timer(interval=30)
        self.timer.add_callback(self._on_timer)
        self._draw()

    def _build_surface_mesh(
        self, params: PlannerParams, resolution: int = 45
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rectangle = params.rectangle
        x_grid = np.linspace(rectangle.x_min, rectangle.x_max, resolution)
        y_grid = np.linspace(rectangle.y_min, rectangle.y_max, resolution)
        mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="xy")
        mesh_z = params.surface.height(mesh_x, mesh_y)
        return mesh_x, mesh_y, np.asarray(mesh_z, dtype=float)

    def _build_controls(self) -> None:
        slider_specs = [
            (
                "chunk_length",
                "Action Chunk Size",
                1,
                32,
                self.initial_params.chunk_length,
                1,
            ),
            (
                "replan_every_n_chunks",
                "Replan After",
                1,
                32,
                self.initial_params.replan_every_n_chunks,
                1,
            ),
            (
                "step_length_k",
                "Step Length k",
                0.001,
                0.25,
                self.initial_params.step_length_k,
                None,
            ),
            ("action_hz_q", "Action Hz q", 1, 60, self.initial_params.action_hz_q, 1),
            (
                "step_noise_std_0",
                "Step Noise Std",
                0.0,
                0.15,
                self.initial_params.step_noise_std_0,
                None,
            ),
            (
                "direction_noise_std_deg_0",
                "Dir Noise Std (deg)",
                0.0,
                180.0,
                self.initial_params.direction_noise_std_deg_0,
                None,
            ),
            (
                "z_noise_std",
                "Z Noise Std",
                0.0,
                0.01,
                self.initial_params.z_noise_std,
                None,
            ),
            (
                "step_noise_decay",
                "Step Noise Decay",
                0.0,
                1.0,
                self.initial_params.step_noise_decay,
                None,
            ),
            (
                "direction_noise_decay",
                "Dir Noise Decay",
                0.0,
                1.0,
                self.initial_params.direction_noise_decay,
                None,
            ),
        ]

        slider_left = 0.69
        slider_width = 0.24
        slider_height = 0.03
        slider_top = 0.45
        slider_gap = 0.037

        for index, (key, label, vmin, vmax, init, valstep) in enumerate(slider_specs):
            axis = self.figure.add_axes(
                [slider_left, slider_top - index * slider_gap, slider_width, slider_height]
            )
            slider = Slider(
                ax=axis,
                label=label,
                valmin=vmin,
                valmax=vmax,
                valinit=init,
                valstep=valstep,
            )
            slider.on_changed(self._on_slider_change)
            self.sliders[key] = slider

        pause_axis = self.figure.add_axes([0.69, 0.07, 0.08, 0.05])
        reset_axis = self.figure.add_axes([0.79, 0.07, 0.08, 0.05])
        replan_axis = self.figure.add_axes([0.69, 0.01, 0.18, 0.05])

        self.buttons["pause"] = Button(pause_axis, "Pause")
        self.buttons["pause"].on_clicked(self._toggle_pause)
        self.buttons["reset"] = Button(reset_axis, "Reset")
        self.buttons["reset"].on_clicked(self._reset)
        self.buttons["replan_now"] = Button(replan_axis, "Replan Now")
        self.buttons["replan_now"].on_clicked(self._replan_now)

    def _slider_int(self, key: str) -> int:
        return int(round(float(self.sliders[key].val)))

    def _current_params(self) -> PlannerParams:
        params = PlannerParams(
            rectangle=self.initial_params.rectangle,
            surface=self.initial_params.surface,
            chunk_length=self._slider_int("chunk_length"),
            replan_every_n_chunks=self._slider_int("replan_every_n_chunks"),
            step_length_k=float(self.sliders["step_length_k"].val),
            action_hz_q=float(self.sliders["action_hz_q"].val),
            step_noise_std_0=float(self.sliders["step_noise_std_0"].val),
            direction_noise_std_deg_0=float(
                self.sliders["direction_noise_std_deg_0"].val
            ),
            z_noise_std=float(self.sliders["z_noise_std"].val),
            step_noise_decay=float(self.sliders["step_noise_decay"].val),
            direction_noise_decay=float(self.sliders["direction_noise_decay"].val),
            center_tolerance=self.initial_params.center_tolerance,
        )
        params.validate()
        return params

    def _surface_position(
        self, point_xy: np.ndarray, params: PlannerParams | None = None
    ) -> np.ndarray:
        active_params = params or self.active_plan_params or self._current_params()
        return surface_position_at_xy(point_xy, active_params)

    def _current_progress_steps(self, now: float) -> float:
        if (
            self.plan_cycle_start_wall_time is None
            or self.active_plan_params is None
            or self.plan_anchor_positions.shape[0] == 0
        ):
            return float(self.plan_point_index)

        max_progress = float(max(self.plan_anchor_positions.shape[0] - 1, 0))
        return float(
            np.clip(
                (now - self.plan_cycle_start_wall_time) * self.active_plan_params.action_hz_q,
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
        params = self.active_plan_params or self._current_params()
        total_points = params.chunk_length
        if self.active_plan_params is None and self.planned_positions.size == 0:
            return 0, total_points

        current_point_index = min(total_points, self.plan_point_index + 1)
        return current_point_index, total_points

    def _points_until_replan(self, params: PlannerParams) -> int:
        replan_after = effective_replan_after(params)
        if self.active_plan_params is None and self.planned_positions.size == 0:
            return 0
        return max(0, replan_after - self.plan_point_index)

    def _plan_from_current_pose(self) -> None:
        if self.current_position is None:
            return

        params = self._current_params()
        start_xy = np.asarray(self.current_position[:2], dtype=float)
        planned_positions, planned_orientations = plan_action_poses(
            start_xy=start_xy,
            global_step_index=self.global_step_index,
            num_points=params.chunk_length,
            params=params,
            rng=self.rng,
        )

        start_anchor_position = self._surface_position(start_xy, params=params)
        if self.current_orientation_xyzw is None:
            start_direction_xy = planned_positions[0, :2] - start_xy
            start_orientation_xyzw, _ = compute_surface_tangent_quaternion(
                start_xy,
                start_direction_xy,
                params.surface,
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

        self.active_plan_params = params
        self.plan_point_index = 0
        self.plan_cycle_start_wall_time = time.perf_counter()
        self.replan_count += 1
        self.replan_positions.append(np.array(start_anchor_position, copy=True))
        self.paused_progress_steps = None
        self.last_status_message = (
            f"Replanned batch {self.replan_count} from {_format_vector(start_anchor_position)} "
            f"for {params.chunk_length} planned pose(s); "
            f"replan after {effective_replan_after(params)} executed pose(s)."
        )
        self._update_display_pose_from_progress(0.0)

    def _start_from_xy(self, point_xy: np.ndarray) -> None:
        rectangle = self.initial_params.rectangle
        clamped_xy = rectangle.clamp(point_xy)
        if not rectangle.contains(clamped_xy):
            return

        start_position = self._surface_position(clamped_xy, params=self.initial_params)
        self.start_xy = np.array(clamped_xy, copy=True)
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
        self.active_plan_params = None

        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0
        self.running = True
        self.paused_progress_steps = None

        self._plan_from_current_pose()
        params = self.active_plan_params or self._current_params()
        self.next_step_wall_time = time.perf_counter() + 1.0 / params.action_hz_q

    def _step_once(self) -> None:
        if (
            self.current_position is None
            or self.active_plan_params is None
            or self.planned_positions.size == 0
        ):
            return

        total_points = self.active_plan_params.chunk_length
        replan_after = effective_replan_after(self.active_plan_params)
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

            params = self.active_plan_params or self._current_params()
            now = time.perf_counter()
            if self.paused_progress_steps is not None and self.active_plan_params is not None:
                self.plan_cycle_start_wall_time = (
                    now - self.paused_progress_steps / self.active_plan_params.action_hz_q
                )
                remaining_until_next_point = max(
                    0.0,
                    (self.plan_point_index + 1 - self.paused_progress_steps)
                    / self.active_plan_params.action_hz_q,
                )
                self.next_step_wall_time = now + remaining_until_next_point
            else:
                self.next_step_wall_time = now + 1.0 / params.action_hz_q
            self.paused_progress_steps = None
        else:
            self.running = False
            self.buttons["pause"].label.set_text("Resume")
            self.last_status_message = "Simulation paused."
            if self.active_plan_params is not None:
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
        self.active_plan_params = None

        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0
        self.running = False
        self.next_step_wall_time = None
        self.paused_progress_steps = None
        self.last_status_message = "Reset. Click inside the XY selector to start."
        self.buttons["pause"].label.set_text("Pause")
        self._draw()

    def _replan_now(self, _) -> None:
        if self.current_position is None:
            self.last_status_message = "No active start point to replan from."
            self._draw()
            return

        self._plan_from_current_pose()
        if self.running and self.active_plan_params is not None:
            self.next_step_wall_time = time.perf_counter() + 1.0 / self.active_plan_params.action_hz_q
        self._draw()

    def _on_slider_change(self, _) -> None:
        self.last_status_message = (
            "Planner parameters updated. Changes apply on next replan."
        )
        self.figure.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if (
            event.inaxes is not self.selection_axis
            or event.xdata is None
            or event.ydata is None
        ):
            return

        point_xy = np.array([float(event.xdata), float(event.ydata)], dtype=float)
        if not self.initial_params.rectangle.contains(point_xy):
            self.last_status_message = "Ignored click outside rectangle."
            self._draw()
            return

        self.buttons["pause"].label.set_text("Pause")
        self._start_from_xy(point_xy)
        self._draw()

    def _on_close(self, _) -> None:
        if self.timer is not None:
            self.timer.stop()

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

        rectangle = self.initial_params.rectangle
        mesh_x, mesh_y, mesh_z = self.surface_mesh
        x_span = max(rectangle.x_max - rectangle.x_min, 1e-6)
        y_span = max(rectangle.y_max - rectangle.y_min, 1e-6)
        triad_length = 0.08 * max(x_span, y_span)

        axis.plot_surface(
            mesh_x,
            mesh_y,
            mesh_z,
            color="lightsteelblue",
            alpha=0.55,
            linewidth=0,
            antialiased=True,
        )
        axis.set_title("3D Surface Planner")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")

        center_xy = rectangle.center
        center_xyz = self._surface_position(center_xy, params=self.initial_params)
        axis.scatter(
            [center_xyz[0]],
            [center_xyz[1]],
            [center_xyz[2]],
            color="crimson",
            s=70,
            marker="x",
            label="Center",
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
        replan_after = (
            effective_replan_after(self.active_plan_params)
            if self.active_plan_params is not None
            else effective_replan_after(self._current_params())
        )
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

        z_candidates = [float(mesh_z.min()), float(mesh_z.max())]
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

        z_min = min(z_candidates)
        z_max = max(z_candidates)
        z_span = max(z_max - z_min, 0.12 * max(x_span, y_span))
        axis.set_xlim(rectangle.x_min, rectangle.x_max)
        axis.set_ylim(rectangle.y_min, rectangle.y_max)
        axis.set_zlim(z_min - 0.1 * z_span, z_max + 0.15 * z_span)
        axis.set_box_aspect((x_span, y_span, z_span))
        axis.legend(loc="upper right", fontsize=8, frameon=True)

    def _draw_selection_axis(self) -> None:
        axis = self.selection_axis
        axis.clear()

        rectangle = self.initial_params.rectangle
        axis.set_title("XY Start Selector")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_aspect("equal", adjustable="box")

        x_pad = 0.08 * max(rectangle.x_max - rectangle.x_min, 1e-3)
        y_pad = 0.08 * max(rectangle.y_max - rectangle.y_min, 1e-3)
        axis.set_xlim(rectangle.x_min - x_pad, rectangle.x_max + x_pad)
        axis.set_ylim(rectangle.y_min - y_pad, rectangle.y_max + y_pad)

        axis.add_patch(
            patches.Rectangle(
                (rectangle.x_min, rectangle.y_min),
                rectangle.x_max - rectangle.x_min,
                rectangle.y_max - rectangle.y_min,
                fill=False,
                linewidth=2.0,
                edgecolor="black",
            )
        )

        center_xy = rectangle.center
        axis.scatter(
            [center_xy[0]],
            [center_xy[1]],
            color="crimson",
            s=45,
            marker="x",
            label="Center",
        )

        if self.start_xy is not None:
            axis.scatter(
                [self.start_xy[0]],
                [self.start_xy[1]],
                color="goldenrod",
                s=50,
                marker="o",
                label="Start",
                zorder=5,
            )

        if self.executed_positions:
            executed = np.vstack(self.executed_positions)
            axis.plot(
                executed[:, 0],
                executed[:, 1],
                color="black",
                linewidth=1.8,
                label="Executed XY",
            )

        if self.replan_positions:
            replans = np.vstack(self.replan_positions)
            axis.scatter(
                replans[:, 0],
                replans[:, 1],
                color="darkorange",
                s=32,
                marker="D",
                label="Replan XY",
            )

        if self.display_position is not None:
            axis.scatter(
                [self.display_position[0]],
                [self.display_position[1]],
                color="royalblue",
                s=58,
                marker="o",
                label="Current XY",
                zorder=6,
            )

        if self.planned_positions.size > 0:
            smooth_path = self._sample_plan_spline(0.0)
            if smooth_path.size > 0:
                axis.plot(
                    smooth_path[:, 0],
                    smooth_path[:, 1],
                    linestyle="--",
                    linewidth=1.2,
                    color="teal",
                    alpha=0.9,
                    label="Planned spline XY",
                )
            axis.scatter(
                self.planned_positions[:, 0],
                self.planned_positions[:, 1],
                color="teal",
                s=20,
                marker="o",
                alpha=0.9,
                label="Planned XY",
            )

        axis.legend(loc="upper right", fontsize=7, frameon=True)

    def _draw_info(self) -> None:
        params = self.active_plan_params or self._current_params()
        self.info_axis.clear()
        self.info_axis.axis("off")

        next_step_sigma = step_noise_std(self.global_step_index, params)
        next_dir_sigma = direction_noise_std_deg(self.global_step_index, params)
        current_point_number, total_plan_points = self._current_plan_point_counter()
        replan_after = effective_replan_after(params)
        points_until_replan = self._points_until_replan(params)

        info_lines = [
            "Controls",
            "  Click inside the XY selector to start a new run",
            "  Pause: stop/resume smooth playback",
            "  Reset: clear the current run",
            "  Replan Now: rebuild future poses from current XY",
            "",
            f"Status: {self.last_status_message}",
            f"Running: {'yes' if self.running else 'no'}",
            f"Current XYZ: {_format_vector(self.display_position)}",
            f"Current quat xyzw: {_format_vector(self.display_orientation_xyzw)}",
            f"Current anchor XYZ: {_format_vector(self.current_position)}",
            f"Start XYZ: {_format_vector(self.start_position)}",
            f"Global step index: {self.global_step_index}",
            f"Replan count: {self.replan_count}",
            f"Current planned pose: {current_point_number}/{total_plan_points}",
            f"Planned poses per cycle: {params.chunk_length}",
            f"Replan after: {replan_after}",
            f"Poses until replan: {points_until_replan}",
            f"Surface family: {params.surface.config.family}",
            f"Z noise std: {params.z_noise_std:.6f}",
            f"Next XY step noise std: {next_step_sigma:.6f}",
            f"Next dir noise std (deg): {next_dir_sigma:.6f}",
            f"Configured action Hz q: {params.action_hz_q:.2f}",
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
        self._draw_selection_axis()
        self._draw_info()
        self.figure.canvas.draw_idle()

    def _on_timer(self) -> None:
        now = time.perf_counter()
        params = self.active_plan_params or self._current_params()

        if self.running and self.current_position is not None:
            if self.next_step_wall_time is None:
                self.next_step_wall_time = now + 1.0 / params.action_hz_q
            step_period = 1.0 / params.action_hz_q
            while self.next_step_wall_time is not None and now >= self.next_step_wall_time:
                self._step_once()
                self.next_step_wall_time += step_period

            progress = self._current_progress_steps(now)
            self._update_display_pose_from_progress(progress)

        if now - self.last_draw_time >= 1.0 / 20.0:
            self.last_draw_time = now
            self._draw()

    def run(self) -> None:
        self.timer.start()
        plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline 3D rectangle-center planner simulator. The planner emits "
            "discrete XYZ poses on an analytic surface, and the GUI visualizes "
            "cubic position interpolation with quaternion slerp."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the planner YAML config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for repeatable noisy trajectories.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    simulator = RectangleCenterPlannerGui(config_path=args.config, seed=args.seed)
    simulator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
