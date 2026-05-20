from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SURFACE_FAMILIES = ("default", "random_gaussian_two_peak")


@dataclass(frozen=True)
class SurfaceConfig:
    family: str
    base_height: float
    amp: float
    freq_x: float
    freq_y: float
    seed: int
    gaussian_curvature: float
    gaussian_peak_offset: float
    origin_x: float = 0.0
    origin_y: float = 0.0

    def validate(self) -> None:
        if self.family not in SURFACE_FAMILIES:
            raise ValueError(
                f"Unsupported surface family `{self.family}`. "
                f"Expected one of {SURFACE_FAMILIES}."
            )
        if self.amp < 0.0:
            raise ValueError("surface amp must be non-negative.")
        if self.gaussian_curvature <= 0.0:
            raise ValueError("gaussian_curvature must be positive.")
        if self.gaussian_peak_offset < 0.0:
            raise ValueError("gaussian_peak_offset must be non-negative.")


@dataclass(frozen=True)
class AnalyticSurface:
    config: SurfaceConfig
    gaussian_centers_local: np.ndarray
    gaussian_peak_amps: np.ndarray
    gaussian_sigma: float

    def _local_coords(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        return x_arr - self.config.origin_x, y_arr - self.config.origin_y

    def height(self, x: Any, y: Any) -> np.ndarray:
        x_local, y_local = self._local_coords(x, y)
        if self.config.family == "default":
            return self.config.base_height + self.config.amp * np.sin(
                self.config.freq_x * x_local
            ) * np.cos(self.config.freq_y * y_local)

        z = np.full_like(x_local, self.config.base_height, dtype=float)
        sigma_sq = self.gaussian_sigma * self.gaussian_sigma
        for center, peak_amp in zip(
            self.gaussian_centers_local, self.gaussian_peak_amps, strict=True
        ):
            dx = x_local - center[0]
            dy = y_local - center[1]
            z += peak_amp * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_sq))
        return z

    def gradient(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        x_local, y_local = self._local_coords(x, y)
        if self.config.family == "default":
            dzdx = self.config.amp * self.config.freq_x * np.cos(
                self.config.freq_x * x_local
            ) * np.cos(self.config.freq_y * y_local)
            dzdy = -self.config.amp * self.config.freq_y * np.sin(
                self.config.freq_x * x_local
            ) * np.sin(self.config.freq_y * y_local)
            return dzdx, dzdy

        dzdx = np.zeros_like(x_local, dtype=float)
        dzdy = np.zeros_like(y_local, dtype=float)
        sigma_sq = self.gaussian_sigma * self.gaussian_sigma
        for center, peak_amp in zip(
            self.gaussian_centers_local, self.gaussian_peak_amps, strict=True
        ):
            dx = x_local - center[0]
            dy = y_local - center[1]
            gaussian = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_sq))
            dzdx += peak_amp * gaussian * (-dx / sigma_sq)
            dzdy += peak_amp * gaussian * (-dy / sigma_sq)
        return dzdx, dzdy


def _require_keys(mapping: dict[str, Any], keys: list[str], section_name: str) -> None:
    missing_keys = [key for key in keys if key not in mapping]
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `{section_name}`: {', '.join(missing_keys)}"
        )


def surface_config_from_mapping(
    surface_cfg: dict[str, Any],
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> SurfaceConfig:
    if not isinstance(surface_cfg, dict):
        raise ValueError("Expected mapping for `surface` in planner config.")

    _require_keys(surface_cfg, ["family", "base_height", "amp"], "surface")
    family = str(surface_cfg["family"])
    if family == "default":
        _require_keys(surface_cfg, ["freq_x", "freq_y"], "surface")
    elif family == "random_gaussian_two_peak":
        _require_keys(
            surface_cfg,
            ["seed", "gaussian_curvature", "gaussian_peak_offset"],
            "surface",
        )
    else:
        raise ValueError(
            f"Unsupported surface family `{family}`. Expected one of {SURFACE_FAMILIES}."
        )

    return SurfaceConfig(
        family=family,
        base_height=float(surface_cfg["base_height"]),
        amp=float(surface_cfg["amp"]),
        freq_x=float(surface_cfg.get("freq_x", 0.0)),
        freq_y=float(surface_cfg.get("freq_y", 0.0)),
        seed=int(surface_cfg.get("seed", 0)),
        gaussian_curvature=float(surface_cfg.get("gaussian_curvature", 1.0)),
        gaussian_peak_offset=float(surface_cfg.get("gaussian_peak_offset", 0.0)),
        origin_x=float(origin_xy[0]),
        origin_y=float(origin_xy[1]),
    )


def build_surface_model(config: SurfaceConfig) -> AnalyticSurface:
    config.validate()
    if config.family == "default":
        return AnalyticSurface(
            config=config,
            gaussian_centers_local=np.zeros((0, 2), dtype=float),
            gaussian_peak_amps=np.zeros(0, dtype=float),
            gaussian_sigma=max(config.gaussian_curvature, 1e-4),
        )

    rng = np.random.default_rng(config.seed)
    peak_offset = config.gaussian_peak_offset
    centers = rng.uniform(-peak_offset, peak_offset, size=(2, 2))
    weights = rng.uniform(0.7, 1.3, size=2)
    weights = weights / np.sum(weights)
    peak_amps = config.amp * weights
    return AnalyticSurface(
        config=config,
        gaussian_centers_local=centers,
        gaussian_peak_amps=peak_amps,
        gaussian_sigma=max(config.gaussian_curvature, 1e-4),
    )


def build_surface_model_from_argparse(
    args: Any,
    *,
    base_height: float = 0.0,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> AnalyticSurface:
    config = SurfaceConfig(
        family=str(args.surface_family),
        base_height=float(base_height),
        amp=float(args.surface_amp),
        freq_x=float(args.surface_freq_x),
        freq_y=float(args.surface_freq_y),
        seed=int(args.surface_seed),
        gaussian_curvature=float(args.gaussian_curvature),
        gaussian_peak_offset=float(args.gaussian_peak_offset),
        origin_x=float(origin_xy[0]),
        origin_y=float(origin_xy[1]),
    )
    return build_surface_model(config)
