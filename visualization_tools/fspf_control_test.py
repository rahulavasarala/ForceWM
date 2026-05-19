from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import redis
from matplotlib.lines import Line2D


DEFAULT_REDIS_HOST = "127.0.0.1"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_UPDATE_HZ = 30.0
DEFAULT_POSITION_SPEED_MPS = 0.06
DEFAULT_AXIS_LENGTH = 0.9
REDIS_KEY_PREFIXES = ("sim::franka", "sai::sim::franka")


@dataclass(frozen=True)
class RedisKeys:
    current_position: str
    current_orientation: str
    desired_position: str
    desired_orientation: str
    desired_force: str
    force_dimension: str
    force_or_motion_axis: str
    reset: str


def _build_key_candidates() -> tuple[RedisKeys, ...]:
    candidates: list[RedisKeys] = []
    for prefix in REDIS_KEY_PREFIXES:
        candidates.append(
            RedisKeys(
                current_position=f"{prefix}::current_cartesian_position",
                current_orientation=f"{prefix}::current_cartesian_orientation",
                desired_position=f"{prefix}::desired_cartesian_position",
                desired_orientation=f"{prefix}::desired_cartesian_orientation",
                desired_force=f"{prefix}::desired_force",
                force_dimension=f"{prefix}::force_dimension",
                force_or_motion_axis=f"{prefix}::force_or_motion_axis",
                reset=f"{prefix}::reset",
            )
        )
    return tuple(candidates)


KEY_CANDIDATES = _build_key_candidates()


def _redis_text(value: bytes | str | None) -> str:
    if value is None:
        raise RuntimeError("Requested Redis key is missing.")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_json_value(redis_client: redis.Redis, key: str):
    raw_value = redis_client.get(key)
    return json.loads(_redis_text(raw_value))


def _read_vector(redis_client: redis.Redis, key: str) -> np.ndarray:
    vector = np.asarray(_read_json_value(redis_client, key), dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Redis key `{key}` did not contain a 3D vector.")
    return vector


def _read_matrix(redis_client: redis.Redis, key: str) -> np.ndarray:
    matrix = np.asarray(_read_json_value(redis_client, key), dtype=float)
    if matrix.size != 9:
        raise ValueError(f"Redis key `{key}` did not contain a 3x3 matrix.")
    return matrix.reshape(3, 3)


def _read_int(redis_client: redis.Redis, key: str) -> int:
    raw_value = redis_client.get(key)
    if raw_value is None:
        raise RuntimeError(f"Requested Redis key `{key}` is missing.")
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    return int(raw_value)


def _write_vector(redis_client: redis.Redis, key: str, value: np.ndarray) -> None:
    redis_client.set(key, json.dumps(np.asarray(value, dtype=float).reshape(3).tolist()))


def _write_matrix(redis_client: redis.Redis, key: str, value: np.ndarray) -> None:
    matrix = np.asarray(value, dtype=float).reshape(3, 3)
    redis_client.set(key, json.dumps(matrix.tolist()))


def _set_reset(redis_client: redis.Redis, key: str) -> None:
    redis_client.set(key, "1")


def _resolve_keys(redis_client: redis.Redis) -> RedisKeys:
    for candidate in KEY_CANDIDATES:
        if redis_client.exists(candidate.current_position):
            return candidate
    return KEY_CANDIDATES[0]


def _project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float).reshape(3, 3))
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return axis / norm


def _orthonormal_complement(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _normalize_axis(axis)
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    basis_1 = reference - float(np.dot(reference, axis)) * axis
    basis_1 = _normalize_axis(basis_1)
    basis_2 = _normalize_axis(np.cross(axis, basis_1))
    return basis_1, basis_2


def _sigma_from_particle_filter(force_dimension: int, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=float).reshape(3)
    sigma_force = np.zeros((3, 3), dtype=float)
    sigma_motion = np.eye(3, dtype=float)

    if force_dimension == 0:
        sigma_force = np.zeros((3, 3), dtype=float)
        sigma_motion = np.eye(3, dtype=float)
    elif force_dimension == 1:
        sigma_force = np.outer(axis, axis)
        sigma_motion = np.eye(3, dtype=float) - sigma_force
    elif force_dimension == 2:
        sigma_motion = np.outer(axis, axis)
        sigma_force = np.eye(3, dtype=float) - sigma_motion
    elif force_dimension == 3:
        sigma_force = np.eye(3, dtype=float)
        sigma_motion = np.zeros((3, 3), dtype=float)

    return sigma_motion, sigma_force


def _axis_visualization(force_dimension: int, axis: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    world_axes = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]

    if force_dimension <= 0:
        return world_axes, []
    if force_dimension >= 3:
        return [], world_axes

    axis = _normalize_axis(axis)
    orthogonal_1, orthogonal_2 = _orthonormal_complement(axis)
    if force_dimension == 1:
        return [orthogonal_1, orthogonal_2], [axis]
    return [axis], [orthogonal_1, orthogonal_2]


def _format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(
        np.asarray(matrix, dtype=float),
        precision=3,
        suppress_small=True,
        floatmode="fixed",
    )


def _format_vector(vector: np.ndarray) -> str:
    return np.array2string(
        np.asarray(vector, dtype=float).reshape(3),
        precision=4,
        suppress_small=True,
        floatmode="fixed",
    )


class FspfControlTest:
    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_db: int,
        position_speed_mps: float,
        update_hz: float,
        axis_length: float,
    ) -> None:
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
        )
        self.redis_client.ping()
        self.keys = _resolve_keys(self.redis_client)

        self.position_speed_mps = float(position_speed_mps)
        self.poll_period_s = 1.0 / float(update_hz)
        self.axis_length = float(axis_length)

        self.held_keys: set[str] = set()
        self.last_tick_time = time.perf_counter()
        self.last_status_message = "Ready"

        self.desired_position = self._read_vector_fallback(
            self.keys.desired_position,
            fallback_key=self.keys.current_position,
        )
        self.desired_orientation = self._read_matrix_fallback(
            self.keys.desired_orientation,
            fallback_key=self.keys.current_orientation,
        )
        self.current_position = self._read_vector_fallback(
            self.keys.current_position,
            fallback_value=self.desired_position,
        )
        self.current_orientation = self._read_matrix_fallback(
            self.keys.current_orientation,
            fallback_value=self.desired_orientation,
        )
        self.force_dimension = 0
        self.force_or_motion_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        self.sigma_motion = np.eye(3, dtype=float)
        self.sigma_force = np.zeros((3, 3), dtype=float)
        self.desired_force = np.zeros(3, dtype=float)

        self._write_desired_commands()
        self._poll_state_from_redis()

        self.figure = plt.figure("FSPF Control Test", figsize=(13, 8))
        grid = self.figure.add_gridspec(
            2,
            3,
            width_ratios=[1.55, 1.0, 1.0],
            height_ratios=[1.0, 1.0],
        )
        self.axis_plot = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.sigma_motion_plot = self.figure.add_subplot(grid[0, 1])
        self.sigma_force_plot = self.figure.add_subplot(grid[0, 2])
        self.info_plot = self.figure.add_subplot(grid[1, 1:])
        self.info_plot.axis("off")

        # Disable Matplotlib's default figure shortcuts on this window so keys
        # like `s` and backspace are reserved for robot control.
        manager = getattr(self.figure.canvas, "manager", None)
        key_press_handler_id = getattr(manager, "key_press_handler_id", None)
        if key_press_handler_id is not None:
            self.figure.canvas.mpl_disconnect(key_press_handler_id)

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._timer = self.figure.canvas.new_timer(
            interval=max(1, int(round(self.poll_period_s * 1000.0)))
        )
        self._timer.add_callback(self._on_timer)
        self.figure.tight_layout()

    def _read_vector_fallback(
        self,
        primary_key: str,
        fallback_key: str | None = None,
        fallback_value: np.ndarray | None = None,
    ) -> np.ndarray:
        try:
            return _read_vector(self.redis_client, primary_key)
        except Exception:
            if fallback_key is not None:
                return _read_vector(self.redis_client, fallback_key)
            if fallback_value is not None:
                return np.asarray(fallback_value, dtype=float).reshape(3)
            raise

    def _read_matrix_fallback(
        self,
        primary_key: str,
        fallback_key: str | None = None,
        fallback_value: np.ndarray | None = None,
    ) -> np.ndarray:
        try:
            return _project_to_rotation_matrix(_read_matrix(self.redis_client, primary_key))
        except Exception:
            if fallback_key is not None:
                return _project_to_rotation_matrix(_read_matrix(self.redis_client, fallback_key))
            if fallback_value is not None:
                return _project_to_rotation_matrix(fallback_value)
            raise

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key in {"escape", "x"}:
            plt.close(self.figure)
            return
        if key in {"backspace", "r"}:
            _set_reset(self.redis_client, self.keys.reset)
            self.last_status_message = "Sent reset request to simulation."
            self.figure.canvas.draw_idle()
            return
        if key:
            self.held_keys.add(key)

    def _on_key_release(self, event) -> None:
        key = (event.key or "").lower()
        if key:
            self.held_keys.discard(key)

    def _on_close(self, _) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _position_command_from_keys(self) -> np.ndarray:
        position_delta = np.zeros(3, dtype=float)
        if "w" in self.held_keys:
            position_delta[0] += 1.0
        if "s" in self.held_keys:
            position_delta[0] -= 1.0
        if "a" in self.held_keys:
            position_delta[1] += 1.0
        if "d" in self.held_keys:
            position_delta[1] -= 1.0
        if "q" in self.held_keys:
            position_delta[2] += 1.0
        if "e" in self.held_keys:
            position_delta[2] -= 1.0
        return position_delta

    def _update_desired_pose_from_keys(self, dt: float) -> bool:
        position_delta = self._position_command_from_keys()

        next_desired_position = np.array(self.current_position, copy=True)
        position_norm = float(np.linalg.norm(position_delta))
        if position_norm > 0.0:
            next_desired_position = self.current_position + (
                self.position_speed_mps * dt * position_delta / position_norm
            )

        next_desired_force = np.zeros(3, dtype=float)
        if self.force_dimension == 1 and position_norm > 0.0:
            dx_world = next_desired_position - self.current_position
            force_axis = _normalize_axis(self.force_or_motion_axis)
            projected_component = float(np.dot(dx_world, force_axis))
            if abs(projected_component) > 1e-12:
                next_desired_force = 2.0 * np.sign(projected_component) * force_axis

        commands_changed = (
            not np.allclose(next_desired_position, self.desired_position, atol=1e-9)
            or not np.allclose(next_desired_force, self.desired_force, atol=1e-9)
        )
        self.desired_position = next_desired_position
        self.desired_force = next_desired_force

        return commands_changed

    def _write_desired_commands(self) -> None:
        _write_vector(self.redis_client, self.keys.desired_position, self.desired_position)
        _write_matrix(self.redis_client, self.keys.desired_orientation, self.desired_orientation)
        _write_vector(self.redis_client, self.keys.desired_force, self.desired_force)

    def _poll_state_from_redis(self) -> None:
        self.current_position = self._read_vector_fallback(
            self.keys.current_position,
            fallback_value=self.current_position,
        )
        self.current_orientation = self._read_matrix_fallback(
            self.keys.current_orientation,
            fallback_value=self.current_orientation,
        )
        self.desired_position = self._read_vector_fallback(
            self.keys.desired_position,
            fallback_value=self.desired_position,
        )
        self.desired_orientation = self._read_matrix_fallback(
            self.keys.desired_orientation,
            fallback_value=self.desired_orientation,
        )
        self.desired_force = self._read_vector_fallback(
            self.keys.desired_force,
            fallback_value=self.desired_force,
        )
        self.force_dimension = _read_int(self.redis_client, self.keys.force_dimension)
        self.force_or_motion_axis = self._read_vector_fallback(
            self.keys.force_or_motion_axis,
            fallback_value=self.force_or_motion_axis,
        )
        self.sigma_motion, self.sigma_force = _sigma_from_particle_filter(
            self.force_dimension,
            self.force_or_motion_axis,
        )

    def _draw_axis_panel(self) -> None:
        self.axis_plot.clear()
        self.axis_plot.set_title("Force/Motion Axes")
        self.axis_plot.set_xlabel("X")
        self.axis_plot.set_ylabel("Y")
        self.axis_plot.set_zlabel("Z")
        self.axis_plot.set_xlim(-1.0, 1.0)
        self.axis_plot.set_ylim(-1.0, 1.0)
        self.axis_plot.set_zlim(-1.0, 1.0)
        self.axis_plot.set_box_aspect((1.0, 1.0, 1.0))

        for axis_index, basis_axis in enumerate(np.eye(3)):
            self.axis_plot.plot(
                [0.0, basis_axis[0]],
                [0.0, basis_axis[1]],
                [0.0, basis_axis[2]],
                linestyle="--",
                linewidth=1.0,
                color="0.7",
            )
            self.axis_plot.text(
                1.08 * basis_axis[0],
                1.08 * basis_axis[1],
                1.08 * basis_axis[2],
                f"e{axis_index + 1}",
                color="0.45",
            )

        motion_axes, force_axes = _axis_visualization(
            self.force_dimension,
            self.force_or_motion_axis,
        )

        for axis in motion_axes:
            self.axis_plot.quiver(
                0.0,
                0.0,
                0.0,
                axis[0],
                axis[1],
                axis[2],
                length=self.axis_length,
                normalize=True,
                color="royalblue",
                linewidth=2.8,
            )
            self.axis_plot.quiver(
                0.0,
                0.0,
                0.0,
                -axis[0],
                -axis[1],
                -axis[2],
                length=self.axis_length,
                normalize=True,
                color="royalblue",
                linewidth=1.6,
                alpha=0.45,
            )

        for axis in force_axes:
            self.axis_plot.quiver(
                0.0,
                0.0,
                0.0,
                axis[0],
                axis[1],
                axis[2],
                length=self.axis_length,
                normalize=True,
                color="crimson",
                linewidth=2.8,
            )
            self.axis_plot.quiver(
                0.0,
                0.0,
                0.0,
                -axis[0],
                -axis[1],
                -axis[2],
                length=self.axis_length,
                normalize=True,
                color="crimson",
                linewidth=1.6,
                alpha=0.45,
            )

        legend_handles = [
            Line2D([0], [0], color="0.7", linestyle="--", linewidth=1.4, label="World basis"),
            Line2D([0], [0], color="royalblue", linewidth=2.8, label="Motion axis"),
            Line2D([0], [0], color="crimson", linewidth=2.8, label="Force axis"),
        ]
        self.axis_plot.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=8,
            frameon=True,
        )

    def _draw_sigma_matrix(self, axis, matrix: np.ndarray, title: str, cmap: str) -> None:
        axis.clear()
        axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap=cmap)
        axis.set_title(title)
        axis.set_xticks(range(3))
        axis.set_yticks(range(3))
        axis.set_xticklabels(["x", "y", "z"])
        axis.set_yticklabels(["x", "y", "z"])

        for row in range(3):
            for col in range(3):
                axis.text(
                    col,
                    row,
                    f"{matrix[row, col]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[row, col]) > 0.55 else "black",
                    fontsize=10,
                )

    def _draw_info_panel(self) -> None:
        self.info_plot.clear()
        self.info_plot.axis("off")

        held_keys_text = ", ".join(sorted(self.held_keys)) if self.held_keys else "none"
        info_lines = [
            "Controls",
            "  Position: W/S -> +/-X, A/D -> +/-Y, Q/E -> +/-Z",
            "  Desired force: signed projection of dx_world onto force_or_motion_axis, scaled to 2.0 when fdim == 1",
            "  Reset sim: Backspace or R",
            "  Quit: X or Escape",
            "",
            f"Held keys: {held_keys_text}",
            f"Status: {self.last_status_message}",
            "",
            f"Force dimension: {self.force_dimension}",
            f"Force/motion axis: {_format_vector(self.force_or_motion_axis)}",
            f"Desired force: {_format_vector(self.desired_force)}",
            "",
            f"Desired position: {_format_vector(self.desired_position)}",
            f"Current position: {_format_vector(self.current_position)}",
            "",
            "sigma_m",
            _format_matrix(self.sigma_motion),
            "",
            "sigma_f",
            _format_matrix(self.sigma_force),
        ]
        self.info_plot.text(
            0.01,
            0.99,
            "\n".join(info_lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

    def _draw(self) -> None:
        self._draw_axis_panel()
        self._draw_sigma_matrix(
            self.sigma_motion_plot,
            self.sigma_motion,
            "Sigma Motion",
            "Blues",
        )
        self._draw_sigma_matrix(
            self.sigma_force_plot,
            self.sigma_force,
            "Sigma Force",
            "Reds",
        )
        self._draw_info_panel()
        self.figure.canvas.draw_idle()

    def _on_timer(self) -> None:
        now = time.perf_counter()
        dt = max(1e-3, now - self.last_tick_time)
        self.last_tick_time = now

        try:
            self._poll_state_from_redis()
            commands_changed = self._update_desired_pose_from_keys(dt)
            if commands_changed:
                self._write_desired_commands()
                self.last_status_message = "Streaming Cartesian/force test commands."
            else:
                self.last_status_message = "Holding desired pose and current force command."
            self._draw()
        except Exception as exception:
            self.last_status_message = f"Redis/update error: {exception}"
            self._draw()

    def run(self) -> None:
        self._draw()
        self._timer.start()
        plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Redis control test for the force-space particle-filter "
            "simulation. Hold keys to move the desired Cartesian pose, and "
            "when the filter reports force dimension 1 the script applies a "
            "2N desired force along the estimated force axis. "
            "visualize sigma_m / sigma_f in real time."
        )
    )
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST)
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT)
    parser.add_argument("--redis-db", type=int, default=DEFAULT_REDIS_DB)
    parser.add_argument("--update-hz", type=float, default=DEFAULT_UPDATE_HZ)
    parser.add_argument(
        "--position-speed",
        type=float,
        default=DEFAULT_POSITION_SPEED_MPS,
        help="Desired-position speed in meters per second while a motion key is held.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=DEFAULT_AXIS_LENGTH,
        help="Rendered length of the red/blue force-motion axes.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    controller = FspfControlTest(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        position_speed_mps=args.position_speed,
        update_hz=args.update_hz,
        axis_length=args.axis_length,
    )
    controller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
