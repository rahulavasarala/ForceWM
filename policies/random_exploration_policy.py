from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from policies.surface_models import (
    AnalyticSurface,
    build_surface_model,
    surface_model_from_generation_metadata,
    surface_config_from_mapping,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".yaml")
CENTER_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RectangleConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def validate(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError("Rectangle requires x_min < x_max.")
        if self.y_min >= self.y_max:
            raise ValueError("Rectangle requires y_min < y_max.")

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.x_min + self.x_max),
                0.5 * (self.y_min + self.y_max),
            ],
            dtype=float,
        )

    def contains(self, point_xy: np.ndarray) -> bool:
        point = np.asarray(point_xy, dtype=float).reshape(2)
        return (
            self.x_min <= point[0] <= self.x_max
            and self.y_min <= point[1] <= self.y_max
        )

    def clamp(self, point_xy: np.ndarray) -> np.ndarray:
        point = np.asarray(point_xy, dtype=float).reshape(2)
        return np.array(
            [
                np.clip(point[0], self.x_min, self.x_max),
                np.clip(point[1], self.y_min, self.y_max),
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class PlannerParams:
    rectangle: RectangleConfig
    surface: AnalyticSurface
    chunk_length: int
    step_length_k: float
    replan_every_n_chunks: int
    action_hz_q: float
    step_noise_std_0: float
    direction_noise_std_deg_0: float
    z_noise_std: float
    step_noise_decay: float
    direction_noise_decay: float
    goal_xy: tuple[float, float] | None = None
    hole_center_xy: tuple[float, float] | None = None
    hole_radius: float = 0.0
    center_tolerance: float = CENTER_TOLERANCE

    def validate(self) -> None:
        self.rectangle.validate()
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive.")
        if self.replan_every_n_chunks <= 0:
            raise ValueError("replan_every_n_chunks must be positive.")
        if self.action_hz_q <= 0.0:
            raise ValueError("action_hz_q must be positive.")
        if self.step_length_k < 0.0:
            raise ValueError("step_length_k must be non-negative.")
        if self.step_noise_std_0 < 0.0:
            raise ValueError("step_noise_std_0 must be non-negative.")
        if self.direction_noise_std_deg_0 < 0.0:
            raise ValueError("direction_noise_std_deg_0 must be non-negative.")
        if self.z_noise_std < 0.0:
            raise ValueError("z_noise_std must be non-negative.")
        if not 0.0 <= self.step_noise_decay <= 1.0:
            raise ValueError("step_noise_decay must lie in [0, 1].")
        if not 0.0 <= self.direction_noise_decay <= 1.0:
            raise ValueError("direction_noise_decay must lie in [0, 1].")
        if self.goal_xy is not None and len(self.goal_xy) != 2:
            raise ValueError("goal_xy must contain exactly two coordinates.")
        if self.hole_center_xy is not None and len(self.hole_center_xy) != 2:
            raise ValueError("hole_center_xy must contain exactly two coordinates.")
        if self.hole_radius < 0.0:
            raise ValueError("hole_radius must be non-negative.")
        if self.center_tolerance <= 0.0:
            raise ValueError("center_tolerance must be positive.")

    @property
    def goal(self) -> np.ndarray:
        if self.goal_xy is None:
            return self.rectangle.center
        return np.asarray(self.goal_xy, dtype=float).reshape(2)

    @property
    def hole_center(self) -> np.ndarray:
        if self.hole_center_xy is None:
            return np.array(self.goal, copy=True)
        return np.asarray(self.hole_center_xy, dtype=float).reshape(2)

    def point_is_in_hole(self, point_xy: np.ndarray) -> bool:
        if self.hole_radius <= 0.0:
            return False
        point = np.asarray(point_xy, dtype=float).reshape(2)
        distance = float(np.linalg.norm(point - self.hole_center))
        return distance < max(self.hole_radius - 1e-12, 0.0)

    def contains_workspace(self, point_xy: np.ndarray) -> bool:
        point = np.asarray(point_xy, dtype=float).reshape(2)
        return self.rectangle.contains(point) and not self.point_is_in_hole(point)

    def _segment_circle_intersection_parameters(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
    ) -> tuple[float, float] | None:
        if self.hole_radius <= 0.0:
            return None

        start = np.asarray(start_xy, dtype=float).reshape(2)
        end = np.asarray(end_xy, dtype=float).reshape(2)
        delta = end - start
        a = float(np.dot(delta, delta))
        if a <= self.center_tolerance * self.center_tolerance:
            return None

        relative_start = start - self.hole_center
        b = 2.0 * float(np.dot(relative_start, delta))
        c = float(np.dot(relative_start, relative_start) - self.hole_radius * self.hole_radius)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None

        sqrt_discriminant = float(np.sqrt(max(discriminant, 0.0)))
        t0 = (-b - sqrt_discriminant) / (2.0 * a)
        t1 = (-b + sqrt_discriminant) / (2.0 * a)
        return (min(t0, t1), max(t0, t1))

    def clip_segment_to_workspace(
        self,
        start_xy: np.ndarray,
        proposed_xy: np.ndarray,
    ) -> np.ndarray:
        start = np.asarray(start_xy, dtype=float).reshape(2)
        end = self.rectangle.clamp(proposed_xy)

        if self.hole_radius <= 0.0:
            return end

        intersections = self._segment_circle_intersection_parameters(start, end)
        if intersections is None:
            return end

        overlap_start = max(intersections[0], 0.0)
        overlap_end = min(intersections[1], 1.0)
        if overlap_end > overlap_start + self.center_tolerance:
            entry_t = max(intersections[0], 0.0)
            if entry_t <= self.center_tolerance:
                return np.array(start, copy=True)
            return start + entry_t * (end - start)

        if self.point_is_in_hole(end):
            direction = end - self.hole_center
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= self.center_tolerance:
                fallback = start - self.hole_center
                fallback_norm = float(np.linalg.norm(fallback))
                direction = fallback if fallback_norm > self.center_tolerance else np.array([1.0, 0.0], dtype=float)
                direction_norm = float(np.linalg.norm(direction))
            return self.hole_center + self.hole_radius * direction / direction_norm
        return end


def _require_mapping(config_dict: dict, key: str) -> dict:
    value = config_dict.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for `{key}` in planner config.")
    return value


def _load_config_dict(config_path: str | Path) -> tuple[Path, dict]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Planner config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle) or {}
    return path, config_dict


def _planner_params_from_defaults(
    defaults_cfg: dict,
    *,
    rectangle: RectangleConfig,
    surface: AnalyticSurface,
    goal_xy: tuple[float, float] | None = None,
    hole_center_xy: tuple[float, float] | None = None,
    hole_radius: float = 0.0,
) -> PlannerParams:
    params = PlannerParams(
        rectangle=rectangle,
        surface=surface,
        chunk_length=int(defaults_cfg["chunk_length"]),
        step_length_k=float(defaults_cfg["step_length_k"]),
        replan_every_n_chunks=int(defaults_cfg["replan_every_n_chunks"]),
        action_hz_q=float(defaults_cfg["action_hz_q"]),
        step_noise_std_0=float(defaults_cfg["step_noise_std"]),
        direction_noise_std_deg_0=float(defaults_cfg["direction_noise_std_deg"]),
        z_noise_std=float(defaults_cfg["z_noise_std"]),
        step_noise_decay=float(defaults_cfg["step_noise_decay"]),
        direction_noise_decay=float(defaults_cfg["direction_noise_decay"]),
        goal_xy=goal_xy,
        hole_center_xy=hole_center_xy,
        hole_radius=float(hole_radius),
    )
    params.validate()
    return params


def planner_params_from_generation_metadata_defaults(
    generation_metadata: dict,
    defaults_cfg: dict,
) -> PlannerParams:
    block_cfg = _require_mapping(generation_metadata, "block_dimensions")
    hole_cfg = _require_mapping(generation_metadata, "hole_dimensions")

    length = float(block_cfg["length"])
    width = float(block_cfg["width"])
    rectangle = RectangleConfig(
        x_min=-0.5 * length,
        x_max=0.5 * length,
        y_min=-0.5 * width,
        y_max=0.5 * width,
    )
    rectangle.validate()

    surface = surface_model_from_generation_metadata(generation_metadata)
    goal_xy = (float(hole_cfg["center_x"]), float(hole_cfg["center_y"]))
    hole_center_xy = goal_xy
    hole_radius = float(hole_cfg["radius"])
    return _planner_params_from_defaults(
        defaults_cfg,
        rectangle=rectangle,
        surface=surface,
        goal_xy=goal_xy,
        hole_center_xy=hole_center_xy,
        hole_radius=hole_radius,
    )


def load_planner_params(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PlannerParams:
    _, config_dict = _load_config_dict(config_path)

    rectangle_cfg = _require_mapping(config_dict, "rectangle")
    defaults_cfg = _require_mapping(config_dict, "defaults")
    surface_cfg = _require_mapping(config_dict, "surface")

    rectangle = RectangleConfig(
        x_min=float(rectangle_cfg["x_min"]),
        x_max=float(rectangle_cfg["x_max"]),
        y_min=float(rectangle_cfg["y_min"]),
        y_max=float(rectangle_cfg["y_max"]),
    )
    rectangle.validate()

    surface = build_surface_model(
        surface_config_from_mapping(surface_cfg, origin_xy=tuple(rectangle.center))
    )
    return _planner_params_from_defaults(
        defaults_cfg,
        rectangle=rectangle,
        surface=surface,
    )


def load_planner_params_from_generation_metadata(
    metadata_path: str | Path,
    planner_config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> PlannerParams:
    metadata_file = Path(metadata_path).expanduser().resolve()
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Generation metadata not found: {metadata_file}")

    with metadata_file.open("r", encoding="utf-8") as handle:
        generation_metadata = json.load(handle)

    _, config_dict = _load_config_dict(planner_config_path)
    defaults_cfg = _require_mapping(config_dict, "defaults")
    return planner_params_from_generation_metadata_defaults(
        generation_metadata,
        defaults_cfg,
    )


def effective_replan_after(params: PlannerParams) -> int:
    return max(1, min(params.replan_every_n_chunks, params.chunk_length))


def step_noise_std(step_index: int, params: PlannerParams) -> float:
    params.validate()
    if step_index < 0:
        raise ValueError("step_index must be non-negative.")
    return float(params.step_noise_std_0 * (params.step_noise_decay**step_index))


def direction_noise_std_deg(step_index: int, params: PlannerParams) -> float:
    params.validate()
    if step_index < 0:
        raise ValueError("step_index must be non-negative.")
    return float(
        params.direction_noise_std_deg_0
        * (params.direction_noise_decay**step_index)
    )


def surface_position_at_xy(point_xy: np.ndarray, params: PlannerParams) -> np.ndarray:
    point = np.asarray(point_xy, dtype=float).reshape(2)
    return np.array(
        [point[0], point[1], float(params.surface.height(point[0], point[1]))],
        dtype=float,
    )


def _nominal_direction(current_xy: np.ndarray, params: PlannerParams) -> np.ndarray:
    goal = params.goal
    to_center = goal - current_xy
    distance = float(np.linalg.norm(to_center))
    if distance <= params.center_tolerance:
        return np.zeros(2, dtype=float)
    return to_center / distance


def _normalize_vector(vector: np.ndarray, *, tol: float = 1e-10) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= tol:
        return None
    return vector / norm


def compute_surface_tangent_quaternion(
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    surface: AnalyticSurface,
    previous_x_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point_xy, dtype=float).reshape(2)
    # The additional arguments are retained for call-site compatibility, but
    # the frame is now determined purely by the local surface slope.
    del direction_xy, previous_x_axis

    dzdx, dzdy = surface.gradient(point[0], point[1])
    # Build the contact frame from the underside normal of the analytic surface.
    # On a flat surface this exactly reproduces the requested base orientation:
    # [[ 1,  0,  0],
    #  [ 0, -1,  0],
    #  [ 0,  0, -1]]
    z_axis = _normalize_vector(np.array([float(dzdx), float(dzdy), -1.0]))
    if z_axis is None:
        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)

    candidate_vectors: list[np.ndarray] = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, -1.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ]

    x_axis = None
    for candidate in candidate_vectors:
        projected = candidate - z_axis * float(np.dot(candidate, z_axis))
        x_axis = _normalize_vector(projected)
        if x_axis is not None:
            break

    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    y_axis = _normalize_vector(np.cross(z_axis, x_axis))
    if y_axis is None:
        y_axis = np.array([0.0, -1.0, 0.0], dtype=float)
    x_axis = _normalize_vector(np.cross(y_axis, z_axis))
    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return quaternion_xyzw.astype(float), x_axis.astype(float)


def plan_chunks(
    start_xy: np.ndarray,
    global_step_index: int,
    num_chunks: int,
    params: PlannerParams,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    positions_xyz, orientations_xyzw = plan_action_poses(
        start_xy=start_xy,
        global_step_index=global_step_index,
        num_points=num_chunks,
        params=params,
        rng=rng,
    )
    poses = np.hstack((positions_xyz, orientations_xyzw))
    return [pose.reshape(1, 7) for pose in poses]


def plan_action_points(
    start_xy: np.ndarray,
    global_step_index: int,
    num_points: int,
    params: PlannerParams,
    rng: np.random.Generator,
) -> np.ndarray:
    params.validate()
    if global_step_index < 0:
        raise ValueError("global_step_index must be non-negative.")
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    start = np.asarray(start_xy, dtype=float).reshape(2)
    if not params.rectangle.contains(start):
        raise ValueError("start_xy must lie inside the configured rectangle.")
    if params.point_is_in_hole(start):
        raise ValueError("start_xy must lie on the valid part surface and outside the hole opening.")

    current_xy = np.array(start, copy=True)
    planned_points = np.zeros((num_points, 2), dtype=float)

    for point_index in range(num_points):
        absolute_step_index = global_step_index + point_index
        direction = _nominal_direction(current_xy, params)
        if np.allclose(direction, 0.0, atol=params.center_tolerance):
            if params.hole_radius > 0.0:
                current_xy = params.clip_segment_to_workspace(current_xy, params.goal)
            else:
                current_xy = np.array(params.goal, copy=True)
            planned_points[point_index] = current_xy
            continue

        sigma_step = step_noise_std(absolute_step_index, params)
        sigma_dir_deg = direction_noise_std_deg(absolute_step_index, params)

        nominal_heading = float(np.arctan2(direction[1], direction[0]))
        heading_noise = float(rng.normal(0.0, np.deg2rad(sigma_dir_deg)))
        step_magnitude = max(
            0.0, params.step_length_k + float(rng.normal(0.0, sigma_step))
        )

        stepped_xy = current_xy + step_magnitude * np.array(
            [
                np.cos(nominal_heading + heading_noise),
                np.sin(nominal_heading + heading_noise),
            ],
            dtype=float,
        )
        current_xy = params.clip_segment_to_workspace(current_xy, stepped_xy)
        if (
            params.hole_radius <= 0.0
            and np.linalg.norm(current_xy - params.goal) <= params.center_tolerance
        ):
            current_xy = np.array(params.goal, copy=True)

        planned_points[point_index] = current_xy

    return planned_points


def plan_action_poses(
    start_xy: np.ndarray,
    global_step_index: int,
    num_points: int,
    params: PlannerParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    points_xy = plan_action_points(
        start_xy=start_xy,
        global_step_index=global_step_index,
        num_points=num_points,
        params=params,
        rng=rng,
    )

    heights = params.surface.height(points_xy[:, 0], points_xy[:, 1])
    z_noise = rng.normal(0.0, params.z_noise_std, size=num_points)
    positions_xyz = np.column_stack((points_xy, heights + z_noise)).astype(float)

    orientations_xyzw = np.zeros((num_points, 4), dtype=float)
    previous_x_axis = None
    previous_xy = np.asarray(start_xy, dtype=float).reshape(2)
    for point_index, point_xy in enumerate(points_xy):
        direction_xy = point_xy - previous_xy
        quaternion_xyzw, previous_x_axis = compute_surface_tangent_quaternion(
            point_xy,
            direction_xy,
            params.surface,
            previous_x_axis=previous_x_axis,
        )
        orientations_xyzw[point_index] = quaternion_xyzw
        previous_xy = point_xy

    return positions_xyz, orientations_xyzw
