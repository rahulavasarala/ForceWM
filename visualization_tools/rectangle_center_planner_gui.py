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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policies.random_exploration_policy import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PlannerParams,
    direction_noise_std_deg,
    load_planner_params,
    plan_action_points,
    step_noise_std,
)


def _format_point(point_xy: np.ndarray | None) -> str:
    if point_xy is None:
        return "none"
    return np.array2string(
        np.asarray(point_xy, dtype=float).reshape(2),
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

        self.start_point: np.ndarray | None = None
        self.current_point: np.ndarray | None = None
        self.display_point: np.ndarray | None = None
        self.executed_points: list[np.ndarray] = []
        self.replan_points: list[np.ndarray] = []

        self.planned_points = np.zeros((0, 2), dtype=float)
        self.plan_anchor_points = np.zeros((0, 2), dtype=float)
        self.plan_splines: tuple[CubicSpline, CubicSpline] | None = None
        self.plan_cycle_start_wall_time: float | None = None
        self.active_plan_params: PlannerParams | None = None
        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0

        self.running = False
        self.next_step_wall_time: float | None = None
        self.paused_progress_steps: float | None = None
        self.last_status_message = "Click inside the rectangle to start."
        self.last_draw_time = time.perf_counter()

        self.figure = plt.figure("Rectangle-Center Path Planner", figsize=(14.0, 8.0))
        self.plot_axis = self.figure.add_axes([0.06, 0.10, 0.55, 0.82])
        self.info_axis = self.figure.add_axes([0.64, 0.58, 0.32, 0.34])
        self.info_axis.axis("off")

        self.sliders: dict[str, Slider] = {}
        self.buttons: dict[str, Button] = {}
        self._build_controls()

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self.timer = self.figure.canvas.new_timer(interval=30)
        self.timer.add_callback(self._on_timer)
        self._draw()

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

        slider_left = 0.68
        slider_width = 0.25
        slider_height = 0.03
        slider_top = 0.50
        slider_gap = 0.045

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

        pause_axis = self.figure.add_axes([0.68, 0.07, 0.10, 0.05])
        reset_axis = self.figure.add_axes([0.80, 0.07, 0.10, 0.05])
        replan_axis = self.figure.add_axes([0.68, 0.01, 0.22, 0.05])

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
            chunk_length=self._slider_int("chunk_length"),
            replan_every_n_chunks=self._slider_int("replan_every_n_chunks"),
            step_length_k=float(self.sliders["step_length_k"].val),
            action_hz_q=float(self.sliders["action_hz_q"].val),
            step_noise_std_0=float(self.sliders["step_noise_std_0"].val),
            direction_noise_std_deg_0=float(
                self.sliders["direction_noise_std_deg_0"].val
            ),
            step_noise_decay=float(self.sliders["step_noise_decay"].val),
            direction_noise_decay=float(self.sliders["direction_noise_decay"].val),
            center_tolerance=self.initial_params.center_tolerance,
        )
        params.validate()
        return params

    def _current_progress_steps(self, now: float) -> float:
        if (
            self.plan_cycle_start_wall_time is None
            or self.active_plan_params is None
            or self.plan_anchor_points.shape[0] == 0
        ):
            return float(self.plan_point_index)

        max_progress = float(max(self.plan_anchor_points.shape[0] - 1, 0))
        return float(
            np.clip(
                (now - self.plan_cycle_start_wall_time) * self.active_plan_params.action_hz_q,
                0.0,
                max_progress,
            )
        )

    def _update_display_point_from_progress(self, progress_steps: float) -> None:
        if self.plan_anchor_points.shape[0] == 0:
            self.display_point = None if self.current_point is None else np.array(
                self.current_point, copy=True
            )
            return

        if self.plan_splines is None:
            index = min(int(round(progress_steps)), self.plan_anchor_points.shape[0] - 1)
            self.display_point = np.array(self.plan_anchor_points[index], copy=True)
            return

        x_spline, y_spline = self.plan_splines
        self.display_point = np.array(
            [float(x_spline(progress_steps)), float(y_spline(progress_steps))],
            dtype=float,
        )

    def _sample_plan_spline(self, start_progress: float) -> np.ndarray:
        if self.plan_anchor_points.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        if self.plan_splines is None or self.plan_anchor_points.shape[0] < 2:
            return self.plan_anchor_points.copy()

        end_progress = float(self.plan_anchor_points.shape[0] - 1)
        if start_progress >= end_progress:
            return self.plan_anchor_points[-1:].copy()

        progress = np.linspace(start_progress, end_progress, 200)
        x_spline, y_spline = self.plan_splines
        return np.column_stack((x_spline(progress), y_spline(progress)))

    def _current_plan_point_counter(self) -> tuple[int, int]:
        params = self.active_plan_params or self._current_params()
        total_points = params.chunk_length
        if self.active_plan_params is None and self.planned_points.size == 0:
            return 0, total_points

        current_point_index = min(total_points, self.plan_point_index + 1)
        return current_point_index, total_points

    def _points_until_replan(self, params: PlannerParams) -> int:
        replan_after = effective_replan_after(params)
        if self.active_plan_params is None and self.planned_points.size == 0:
            return 0
        return max(0, replan_after - self.plan_point_index)

    def _current_plan_points(self) -> np.ndarray:
        if self.planned_points.size == 0:
            return np.zeros((0, 2), dtype=float)
        return self.planned_points

    def _plan_from_current_point(self) -> None:
        if self.current_point is None:
            return

        params = self._current_params()
        self.planned_points = plan_action_points(
            start_xy=self.current_point,
            global_step_index=self.global_step_index,
            num_points=params.chunk_length,
            params=params,
            rng=self.rng,
        )
        self.plan_anchor_points = np.vstack((self.current_point, self.planned_points))
        if self.plan_anchor_points.shape[0] >= 2:
            anchor_t = np.arange(self.plan_anchor_points.shape[0], dtype=float)
            self.plan_splines = (
                CubicSpline(anchor_t, self.plan_anchor_points[:, 0]),
                CubicSpline(anchor_t, self.plan_anchor_points[:, 1]),
            )
        else:
            self.plan_splines = None

        self.active_plan_params = params
        self.plan_point_index = 0
        self.plan_cycle_start_wall_time = time.perf_counter()
        self.replan_count += 1
        self.replan_points.append(np.array(self.current_point, copy=True))
        self.paused_progress_steps = None
        self.last_status_message = (
            f"Replanned batch {self.replan_count} from {_format_point(self.current_point)} "
            f"for {params.chunk_length} planned point(s); "
            f"replan after {effective_replan_after(params)} executed point(s)."
        )
        self._update_display_point_from_progress(0.0)

    def _start_from_point(self, point_xy: np.ndarray) -> None:
        rectangle = self.initial_params.rectangle
        clamped = rectangle.clamp(point_xy)
        if not rectangle.contains(clamped):
            return

        self.start_point = clamped
        self.current_point = np.array(clamped, copy=True)
        self.display_point = np.array(clamped, copy=True)
        self.executed_points = [np.array(clamped, copy=True)]
        self.replan_points = []

        self.planned_points = np.zeros((0, 2), dtype=float)
        self.plan_anchor_points = np.zeros((0, 2), dtype=float)
        self.plan_splines = None
        self.plan_cycle_start_wall_time = None
        self.active_plan_params = None

        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0
        self.running = True
        self.paused_progress_steps = None

        self._plan_from_current_point()
        params = self.active_plan_params or self._current_params()
        self.next_step_wall_time = time.perf_counter() + 1.0 / params.action_hz_q

    def _step_once(self) -> None:
        if (
            self.current_point is None
            or self.active_plan_params is None
            or self.planned_points.size == 0
        ):
            return

        total_points = self.active_plan_params.chunk_length
        replan_after = effective_replan_after(self.active_plan_params)
        if self.plan_point_index >= min(total_points, replan_after):
            self._plan_from_current_point()
            return

        next_point = self.planned_points[self.plan_point_index]
        self.current_point = np.array(next_point, copy=True)
        self.display_point = np.array(next_point, copy=True)
        self.executed_points.append(np.array(next_point, copy=True))
        self.plan_point_index += 1
        self.global_step_index += 1

        if self.plan_point_index >= min(total_points, replan_after):
            self._plan_from_current_point()
        else:
            points_left = max(0, replan_after - self.plan_point_index)
            current_point, total_plan_points = self._current_plan_point_counter()
            self.last_status_message = (
                f"Executed step {self.global_step_index}. "
                f"Planned point {current_point}/{total_plan_points}, "
                f"{points_left} action point(s) until replanning."
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
        self.start_point = None
        self.current_point = None
        self.display_point = None
        self.executed_points = []
        self.replan_points = []

        self.planned_points = np.zeros((0, 2), dtype=float)
        self.plan_anchor_points = np.zeros((0, 2), dtype=float)
        self.plan_splines = None
        self.plan_cycle_start_wall_time = None
        self.active_plan_params = None

        self.plan_point_index = 0
        self.global_step_index = 0
        self.replan_count = 0
        self.running = False
        self.next_step_wall_time = None
        self.paused_progress_steps = None
        self.last_status_message = "Reset. Click inside the rectangle to start."
        self.buttons["pause"].label.set_text("Pause")
        self._draw()

    def _replan_now(self, _) -> None:
        if self.current_point is None:
            self.last_status_message = "No active start point to replan from."
            self._draw()
            return

        self._plan_from_current_point()
        if self.running and self.active_plan_params is not None:
            self.next_step_wall_time = time.perf_counter() + 1.0 / self.active_plan_params.action_hz_q
        self._draw()

    def _on_slider_change(self, _) -> None:
        self.last_status_message = (
            "Planner parameters updated. Changes apply on next replan."
        )
        self.figure.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.plot_axis or event.xdata is None or event.ydata is None:
            return

        point_xy = np.array([float(event.xdata), float(event.ydata)], dtype=float)
        if not self.initial_params.rectangle.contains(point_xy):
            self.last_status_message = "Ignored click outside rectangle."
            self._draw()
            return

        self.buttons["pause"].label.set_text("Pause")
        self._start_from_point(point_xy)
        self._draw()

    def _on_close(self, _) -> None:
        if self.timer is not None:
            self.timer.stop()

    def _draw_plot(self) -> None:
        axis = self.plot_axis
        axis.clear()

        rectangle = self.initial_params.rectangle
        axis.set_title("Rectangle-Center Planner")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_aspect("equal", adjustable="box")

        x_pad = 0.1 * max(rectangle.x_max - rectangle.x_min, 1e-3)
        y_pad = 0.1 * max(rectangle.y_max - rectangle.y_min, 1e-3)
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

        center = rectangle.center
        axis.scatter(
            [center[0]],
            [center[1]],
            color="crimson",
            s=70,
            marker="x",
            label="Center",
        )

        if self.start_point is not None:
            axis.scatter(
                [self.start_point[0]],
                [self.start_point[1]],
                color="goldenrod",
                s=70,
                marker="o",
                label="Start",
                zorder=5,
            )

        if self.executed_points:
            executed = np.vstack(self.executed_points)
            axis.plot(
                executed[:, 0],
                executed[:, 1],
                color="black",
                linewidth=2.0,
                label="Executed anchors",
            )

        if self.replan_points:
            replans = np.vstack(self.replan_points)
            axis.scatter(
                replans[:, 0],
                replans[:, 1],
                color="darkorange",
                s=45,
                marker="D",
                label="Replan point",
                zorder=5,
            )

        if self.display_point is not None:
            axis.scatter(
                [self.display_point[0]],
                [self.display_point[1]],
                color="royalblue",
                s=80,
                marker="o",
                label="Current",
                zorder=6,
            )

        planned_points = self._current_plan_points()
        if planned_points.size > 0:
            smooth_path = self._sample_plan_spline(0.0)
            if smooth_path.size > 0:
                axis.plot(
                    smooth_path[:, 0],
                    smooth_path[:, 1],
                    linestyle="--",
                    linewidth=1.6,
                    color="teal",
                    alpha=0.9,
                    label="Planned spline",
                )

            axis.scatter(
                planned_points[:, 0],
                planned_points[:, 1],
                color="teal",
                s=28,
                marker="o",
                alpha=0.9,
                label="Planned points",
            )

        current_point, total_points = self._current_plan_point_counter()
        replan_after = (
            effective_replan_after(self.active_plan_params)
            if self.active_plan_params is not None
            else effective_replan_after(self._current_params())
        )
        axis.text(
            0.02,
            0.98,
            f"Planned Point: {current_point}/{total_points}\nReplan After: {replan_after}",
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

        axis.legend(loc="upper right", fontsize=9, frameon=True)

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
            "  Click inside the rectangle to start a new run",
            "  Pause: stop/resume smooth playback",
            "  Reset: clear the current run",
            "  Replan Now: rebuild future action points from current anchor",
            "",
            f"Status: {self.last_status_message}",
            f"Running: {'yes' if self.running else 'no'}",
            f"Current XY: {_format_point(self.display_point)}",
            f"Current anchor XY: {_format_point(self.current_point)}",
            f"Start XY: {_format_point(self.start_point)}",
            f"Global step index: {self.global_step_index}",
            f"Replan count: {self.replan_count}",
            f"Current planned point: {current_point_number}/{total_plan_points}",
            f"Planned points per cycle: {params.chunk_length}",
            f"Replan after: {replan_after}",
            f"Action points until replan: {points_until_replan}",
            f"Next step noise std: {next_step_sigma:.6f}",
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
            fontsize=10,
        )

    def _draw(self) -> None:
        self._draw_plot()
        self._draw_info()
        self.figure.canvas.draw_idle()

    def _on_timer(self) -> None:
        now = time.perf_counter()
        params = self.active_plan_params or self._current_params()

        if self.running and self.current_point is not None:
            if self.next_step_wall_time is None:
                self.next_step_wall_time = now + 1.0 / params.action_hz_q
            step_period = 1.0 / params.action_hz_q
            while self.next_step_wall_time is not None and now >= self.next_step_wall_time:
                self._step_once()
                self.next_step_wall_time += step_period

            progress = self._current_progress_steps(now)
            self._update_display_point_from_progress(progress)

        if now - self.last_draw_time >= 1.0 / 30.0:
            self.last_draw_time = now
            self._draw()

    def run(self) -> None:
        self.timer.start()
        plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline rectangle-center path planner simulator. The planner emits "
            "discrete XY action points, and the GUI visualizes smooth cubic "
            "interpolation between them."
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
