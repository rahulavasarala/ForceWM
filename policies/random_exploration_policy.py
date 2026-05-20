from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


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
    chunk_length: int
    step_length_k: float
    replan_every_n_chunks: int
    action_hz_q: float
    step_noise_std_0: float
    direction_noise_std_deg_0: float
    step_noise_decay: float
    direction_noise_decay: float
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
        if not 0.0 <= self.step_noise_decay <= 1.0:
            raise ValueError("step_noise_decay must lie in [0, 1].")
        if not 0.0 <= self.direction_noise_decay <= 1.0:
            raise ValueError("direction_noise_decay must lie in [0, 1].")
        if self.center_tolerance <= 0.0:
            raise ValueError("center_tolerance must be positive.")


def _require_mapping(config_dict: dict, key: str) -> dict:
    value = config_dict.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for `{key}` in planner config.")
    return value


def load_planner_params(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PlannerParams:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Planner config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle) or {}

    rectangle_cfg = _require_mapping(config_dict, "rectangle")
    defaults_cfg = _require_mapping(config_dict, "defaults")

    params = PlannerParams(
        rectangle=RectangleConfig(
            x_min=float(rectangle_cfg["x_min"]),
            x_max=float(rectangle_cfg["x_max"]),
            y_min=float(rectangle_cfg["y_min"]),
            y_max=float(rectangle_cfg["y_max"]),
        ),
        chunk_length=int(defaults_cfg["chunk_length"]),
        step_length_k=float(defaults_cfg["step_length_k"]),
        replan_every_n_chunks=int(defaults_cfg["replan_every_n_chunks"]),
        action_hz_q=float(defaults_cfg["action_hz_q"]),
        step_noise_std_0=float(defaults_cfg["step_noise_std"]),
        direction_noise_std_deg_0=float(defaults_cfg["direction_noise_std_deg"]),
        step_noise_decay=float(defaults_cfg["step_noise_decay"]),
        direction_noise_decay=float(defaults_cfg["direction_noise_decay"]),
    )
    params.validate()
    return params


def step_noise_std(step_index: int, params: PlannerParams) -> float:
    params.validate()
    if step_index < 0:
        raise ValueError("step_index must be non-negative.")
    return float(params.step_noise_std_0 * (params.step_noise_decay ** step_index))


def direction_noise_std_deg(step_index: int, params: PlannerParams) -> float:
    params.validate()
    if step_index < 0:
        raise ValueError("step_index must be non-negative.")
    return float(
        params.direction_noise_std_deg_0
        * (params.direction_noise_decay ** step_index)
    )


def _nominal_direction(current_xy: np.ndarray, params: PlannerParams) -> np.ndarray:
    center = params.rectangle.center
    to_center = center - current_xy
    distance = float(np.linalg.norm(to_center))
    if distance <= params.center_tolerance:
        return np.zeros(2, dtype=float)
    return to_center / distance


def plan_chunks(
    start_xy: np.ndarray,
    global_step_index: int,
    num_chunks: int,
    params: PlannerParams,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    points = plan_action_points(
        start_xy=start_xy,
        global_step_index=global_step_index,
        num_points=num_chunks,
        params=params,
        rng=rng,
    )
    return [point.reshape(1, 2) for point in points]


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

    current_xy = np.array(start, copy=True)
    planned_points = np.zeros((num_points, 2), dtype=float)

    for point_index in range(num_points):
        absolute_step_index = global_step_index + point_index
        direction = _nominal_direction(current_xy, params)
        if np.allclose(direction, 0.0, atol=params.center_tolerance):
            current_xy = np.array(params.rectangle.center, copy=True)
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
        current_xy = params.rectangle.clamp(stepped_xy)
        if np.linalg.norm(current_xy - params.rectangle.center) <= params.center_tolerance:
            current_xy = np.array(params.rectangle.center, copy=True)

        planned_points[point_index] = current_xy

    return planned_points
