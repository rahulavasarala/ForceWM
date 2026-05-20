from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from trimesh.decomposition import convex_decomposition


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policies.surface_models import (  # noqa: E402
    AnalyticSurface,
    SurfaceConfig,
    build_surface_model,
)


DEFAULT_PART_NAME = "default_part"
DEFAULT_BLOCK_LENGTH = 0.10
DEFAULT_BLOCK_WIDTH = 0.08
DEFAULT_BLOCK_HEIGHT = 0.03
DEFAULT_HOLE_RADIUS = 0.018
DEFAULT_NX = 160
DEFAULT_NY = 160
DEFAULT_BODY_POS = (0.4, 0.0, 0.3)
DEFAULT_BODY_QUAT = (1.0, 0.0, 0.0, 0.0)
SAFE_PART_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def default_surface_config() -> SurfaceConfig:
    return SurfaceConfig(
        family="default",
        base_height=0.0,
        amp=0.002,
        freq_x=80.0,
        freq_y=60.0,
        seed=0,
        gaussian_curvature=0.015,
        gaussian_peak_offset=0.02,
        origin_x=0.0,
        origin_y=0.0,
    )


def make_default_cad_part_config(part_name: str = DEFAULT_PART_NAME) -> "CadPartConfig":
    return CadPartConfig(
        part_name=part_name,
        length=DEFAULT_BLOCK_LENGTH,
        width=DEFAULT_BLOCK_WIDTH,
        height=DEFAULT_BLOCK_HEIGHT,
        hole_radius=DEFAULT_HOLE_RADIUS,
        nx=DEFAULT_NX,
        ny=DEFAULT_NY,
        surface=default_surface_config(),
        body_pos=DEFAULT_BODY_POS,
        body_quat=DEFAULT_BODY_QUAT,
    )


@dataclass(frozen=True)
class CadPartConfig:
    part_name: str
    length: float
    width: float
    height: float
    hole_radius: float
    nx: int
    ny: int
    surface: SurfaceConfig = field(default_factory=default_surface_config)
    body_pos: tuple[float, float, float] = DEFAULT_BODY_POS
    body_quat: tuple[float, float, float, float] = DEFAULT_BODY_QUAT

    def normalized_part_name(self) -> str:
        return self.part_name.strip()

    def validate_part_name(self) -> str:
        normalized = self.normalized_part_name()
        if not normalized:
            raise ValueError("Part name must be non-empty.")
        if SAFE_PART_NAME_RE.fullmatch(normalized) is None:
            raise ValueError(
                "Part name may only contain letters, numbers, underscores, hyphens, and periods."
            )
        return normalized

    def validate_geometry(self) -> AnalyticSurface:
        if self.length <= 0.0:
            raise ValueError("Block length must be positive.")
        if self.width <= 0.0:
            raise ValueError("Block width must be positive.")
        if self.height <= 0.0:
            raise ValueError("Block height must be positive.")
        if self.hole_radius <= 0.0:
            raise ValueError("Hole radius must be positive.")
        if int(self.nx) < 4 or int(self.ny) < 4:
            raise ValueError("nx and ny must each be at least 4.")
        if self.hole_radius >= 0.5 * min(self.length, self.width):
            raise ValueError("Hole radius must be smaller than half of the block width and length.")

        self.surface.validate()
        surface_model = build_surface_model(self.surface)
        _, _, sampled_top_surface = sample_top_surface(
            self,
            surface_model=surface_model,
            resolution_x=max(11, int(self.nx) + 1),
            resolution_y=max(11, int(self.ny) + 1),
        )
        if float(np.min(sampled_top_surface)) <= 0.0:
            raise ValueError(
                "Sampled top surface dips to or below the base plane; increase block height or reduce curvature."
            )
        return surface_model

    def validate(self) -> AnalyticSurface:
        self.validate_part_name()
        return self.validate_geometry()

    @property
    def block_dimensions(self) -> dict[str, float]:
        return {
            "length": float(self.length),
            "width": float(self.width),
            "height": float(self.height),
        }

    @property
    def hole_dimensions(self) -> dict[str, float]:
        return {
            "radius": float(self.hole_radius),
            "diameter": float(2.0 * self.hole_radius),
            "center_x": 0.0,
            "center_y": 0.0,
        }

    @property
    def discretization(self) -> dict[str, int]:
        return {
            "nx": int(self.nx),
            "ny": int(self.ny),
        }


@dataclass(frozen=True)
class GeneratedPartArtifacts:
    part_name: str
    output_dir: Path
    visual_mesh_path: Path
    convex_dir: Path
    metadata_path: Path
    surface_function_path: Path
    xml_path: Path


@dataclass(frozen=True)
class WireframePatch:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class PreviewWireframeData:
    top_surface: WireframePatch
    bottom_surface: WireframePatch
    side_walls: tuple[WireframePatch, ...]
    hole_wall: WireframePatch
    z_min: float
    z_max: float


def surface_config_from_args(args: argparse.Namespace) -> SurfaceConfig:
    return SurfaceConfig(
        family=str(args.surface_family),
        base_height=0.0,
        amp=float(args.surface_amp),
        freq_x=float(args.surface_freq_x),
        freq_y=float(args.surface_freq_y),
        seed=int(args.surface_seed),
        gaussian_curvature=float(args.gaussian_curvature),
        gaussian_peak_offset=float(args.gaussian_peak_offset),
        origin_x=0.0,
        origin_y=0.0,
    )


def cad_part_config_from_args(args: argparse.Namespace) -> CadPartConfig:
    return CadPartConfig(
        part_name=str(args.part_name).strip(),
        length=float(args.block_length),
        width=float(args.block_width),
        height=float(args.block_height),
        hole_radius=float(args.hole_radius),
        nx=int(args.nx),
        ny=int(args.ny),
        surface=surface_config_from_args(args),
        body_pos=DEFAULT_BODY_POS,
        body_quat=DEFAULT_BODY_QUAT,
    )


def build_surface_function(args: argparse.Namespace) -> AnalyticSurface:
    return build_surface_model(surface_config_from_args(args))


def default_surface(x, y, amp=0.003, freq=8.0):
    surface_model = build_surface_model(
        SurfaceConfig(
            family="default",
            base_height=0.0,
            amp=amp,
            freq_x=freq,
            freq_y=freq,
            seed=0,
            gaussian_curvature=0.015,
            gaussian_peak_offset=0.02,
        )
    )
    return surface_model.height(x, y)


def build_surface_model_for_part(config: CadPartConfig) -> AnalyticSurface:
    return build_surface_model(config.surface)


def sample_top_surface(
    config: CadPartConfig,
    *,
    surface_model: AnalyticSurface | None = None,
    resolution_x: int | None = None,
    resolution_y: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_surface_model = surface_model or build_surface_model_for_part(config)
    sample_x = np.linspace(
        -0.5 * config.length,
        0.5 * config.length,
        max(2, int(resolution_x or config.nx) + 1),
        dtype=float,
    )
    sample_y = np.linspace(
        -0.5 * config.width,
        0.5 * config.width,
        max(2, int(resolution_y or config.ny) + 1),
        dtype=float,
    )
    sample_xx, sample_yy = np.meshgrid(sample_x, sample_y, indexing="xy")
    sampled_top_surface = config.height + active_surface_model.height(sample_xx, sample_yy)
    return sample_xx, sample_yy, np.asarray(sampled_top_surface, dtype=float)


def surface_model_to_metadata(surface_model: AnalyticSurface) -> dict[str, object]:
    config = surface_model.config
    metadata: dict[str, object] = {
        "family": config.family,
        "base_height": float(config.base_height),
        "amp": float(config.amp),
        "freq_x": float(config.freq_x),
        "freq_y": float(config.freq_y),
        "seed": int(config.seed),
        "gaussian_curvature": float(config.gaussian_curvature),
        "gaussian_peak_offset": float(config.gaussian_peak_offset),
        "origin_x": float(config.origin_x),
        "origin_y": float(config.origin_y),
    }

    if surface_model.gaussian_centers_local.size:
        metadata["gaussian_centers_local"] = surface_model.gaussian_centers_local.tolist()
    else:
        metadata["gaussian_centers_local"] = []

    if surface_model.gaussian_peak_amps.size:
        metadata["gaussian_peak_amps"] = surface_model.gaussian_peak_amps.tolist()
    else:
        metadata["gaussian_peak_amps"] = []

    metadata["gaussian_sigma"] = float(surface_model.gaussian_sigma)
    return metadata


def build_surface_function_source(surface_metadata: dict[str, object], block_height: float) -> str:
    family = surface_metadata["family"]
    if family == "default":
        return f"""import numpy as np

SURFACE_METADATA = {json.dumps(surface_metadata, indent=4)}
BLOCK_HEIGHT = {block_height:.16g}


def surface_height(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_local = x_arr - SURFACE_METADATA["origin_x"]
    y_local = y_arr - SURFACE_METADATA["origin_y"]
    return SURFACE_METADATA["base_height"] + SURFACE_METADATA["amp"] * np.sin(
        SURFACE_METADATA["freq_x"] * x_local
    ) * np.cos(SURFACE_METADATA["freq_y"] * y_local)


def top_surface_z(x, y):
    return BLOCK_HEIGHT + surface_height(x, y)
"""

    return f"""import numpy as np

SURFACE_METADATA = {json.dumps(surface_metadata, indent=4)}
BLOCK_HEIGHT = {block_height:.16g}


def surface_height(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_local = x_arr - SURFACE_METADATA["origin_x"]
    y_local = y_arr - SURFACE_METADATA["origin_y"]
    z = np.full_like(x_local, SURFACE_METADATA["base_height"], dtype=float)
    sigma_sq = SURFACE_METADATA["gaussian_sigma"] * SURFACE_METADATA["gaussian_sigma"]
    for center, peak_amp in zip(
        SURFACE_METADATA["gaussian_centers_local"],
        SURFACE_METADATA["gaussian_peak_amps"],
        strict=True,
    ):
        dx = x_local - center[0]
        dy = y_local - center[1]
        z += peak_amp * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_sq))
    return z


def top_surface_z(x, y):
    return BLOCK_HEIGHT + surface_height(x, y)
"""


def write_generation_metadata(
    out_dir: Path | str,
    *,
    config: CadPartConfig,
    surface_model: AnalyticSurface,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    surface_metadata = surface_model_to_metadata(surface_model)
    _, _, sampled_top_surface = sample_top_surface(
        config,
        surface_model=surface_model,
        resolution_x=max(2, int(config.nx)),
        resolution_y=max(2, int(config.ny)),
    )
    generation_metadata = {
        "part_name": config.normalized_part_name(),
        "block_dimensions": config.block_dimensions,
        "hole_dimensions": config.hole_dimensions,
        "discretization": config.discretization,
        "surface": surface_metadata,
        "top_surface_definition": "z(x, y) = block_height + surface_height(x, y)",
        "sampled_top_surface_z_range": {
            "min": float(np.min(sampled_top_surface)),
            "max": float(np.max(sampled_top_surface)),
        },
    }

    metadata_path = out_dir / "generation_metadata.json"
    metadata_path.write_text(json.dumps(generation_metadata, indent=2) + "\n")

    surface_function_path = out_dir / "surface_function.py"
    surface_function_path.write_text(
        build_surface_function_source(surface_metadata, config.height)
    )

    return metadata_path, surface_function_path


def build_preview_wireframe_data(
    config: CadPartConfig,
    *,
    preview_resolution_x: int = 40,
    preview_resolution_y: int = 36,
    wall_samples: int = 24,
    hole_theta_samples: int = 72,
    hole_height_samples: int = 10,
) -> PreviewWireframeData:
    surface_model = config.validate_geometry()
    top_x, top_y, top_z = sample_top_surface(
        config,
        surface_model=surface_model,
        resolution_x=preview_resolution_x,
        resolution_y=preview_resolution_y,
    )
    hole_mask = (top_x * top_x + top_y * top_y) < (config.hole_radius * config.hole_radius)
    top_surface = WireframePatch(
        x=top_x,
        y=top_y,
        z=np.where(hole_mask, np.nan, top_z),
    )
    bottom_surface = WireframePatch(
        x=top_x,
        y=top_y,
        z=np.where(hole_mask, np.nan, np.zeros_like(top_z)),
    )

    def wall_at_constant_x(x_value: float) -> WireframePatch:
        y_values = np.linspace(-0.5 * config.width, 0.5 * config.width, wall_samples, dtype=float)
        x_values = np.full_like(y_values, x_value, dtype=float)
        top_values = config.height + surface_model.height(x_values, y_values)
        return WireframePatch(
            x=np.vstack((x_values, x_values)),
            y=np.vstack((y_values, y_values)),
            z=np.vstack((np.zeros_like(top_values), top_values)),
        )

    def wall_at_constant_y(y_value: float) -> WireframePatch:
        x_values = np.linspace(-0.5 * config.length, 0.5 * config.length, wall_samples, dtype=float)
        y_values = np.full_like(x_values, y_value, dtype=float)
        top_values = config.height + surface_model.height(x_values, y_values)
        return WireframePatch(
            x=np.vstack((x_values, x_values)),
            y=np.vstack((y_values, y_values)),
            z=np.vstack((np.zeros_like(top_values), top_values)),
        )

    theta = np.linspace(0.0, 2.0 * np.pi, hole_theta_samples, endpoint=True, dtype=float)
    hole_x = config.hole_radius * np.cos(theta)
    hole_y = config.hole_radius * np.sin(theta)
    hole_top = config.height + surface_model.height(hole_x, hole_y)
    z_scale = np.linspace(0.0, 1.0, hole_height_samples, dtype=float)[:, None]
    hole_wall = WireframePatch(
        x=np.tile(hole_x, (hole_height_samples, 1)),
        y=np.tile(hole_y, (hole_height_samples, 1)),
        z=z_scale * hole_top[None, :],
    )

    side_walls = (
        wall_at_constant_x(-0.5 * config.length),
        wall_at_constant_x(0.5 * config.length),
        wall_at_constant_y(-0.5 * config.width),
        wall_at_constant_y(0.5 * config.width),
    )
    return PreviewWireframeData(
        top_surface=top_surface,
        bottom_surface=bottom_surface,
        side_walls=side_walls,
        hole_wall=hole_wall,
        z_min=0.0,
        z_max=float(np.nanmax(top_surface.z)),
    )


def _add_quad(faces: list[list[int]], a: int, b: int, c: int, d: int) -> None:
    faces.append([a, b, c])
    faces.append([a, c, d])


def make_hole_block(
    L=0.08,
    W=0.08,
    H=0.03,
    hole_r=0.015,
    nx=160,
    ny=160,
    ntheta=128,
    surface_fn=default_surface,
):
    dx = L / nx
    dy = W / ny

    x_edges = np.linspace(-L / 2, L / 2, nx + 1)
    y_edges = np.linspace(-W / 2, W / 2, ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    top_surface = H + surface_fn(Xc, Yc)
    z_max = float(np.max(top_surface))

    dz = min(dx, dy)
    nz = max(2, int(np.ceil(z_max / dz)))
    z_edges = np.linspace(0.0, z_max, nz + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    occupancy_margin = 0.5 * (z_edges[1] - z_edges[0])

    occupancy = np.zeros((nx, ny, nz), dtype=bool)
    outside_hole = Xc * Xc + Yc * Yc >= hole_r * hole_r

    for k, zc in enumerate(z_centers):
        occupancy[:, :, k] = outside_hole & (zc <= top_surface + occupancy_margin)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_ids: dict[tuple[int, int, int], int] = {}

    def vertex_index(ix: int, iy: int, iz: int) -> int:
        key = (ix, iy, iz)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append([x_edges[ix], y_edges[iy], z_edges[iz]])
        return vertex_ids[key]

    def emit_face(corners: list[tuple[int, int, int]]) -> None:
        a, b, c, d = [vertex_index(*corner) for corner in corners]
        _add_quad(faces, a, b, c, d)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not occupancy[i, j, k]:
                    continue

                if i == 0 or not occupancy[i - 1, j, k]:
                    emit_face(
                        [
                            (i, j, k),
                            (i, j, k + 1),
                            (i, j + 1, k + 1),
                            (i, j + 1, k),
                        ]
                    )
                if i == nx - 1 or not occupancy[i + 1, j, k]:
                    emit_face(
                        [
                            (i + 1, j, k),
                            (i + 1, j + 1, k),
                            (i + 1, j + 1, k + 1),
                            (i + 1, j, k + 1),
                        ]
                    )
                if j == 0 or not occupancy[i, j - 1, k]:
                    emit_face(
                        [
                            (i, j, k),
                            (i + 1, j, k),
                            (i + 1, j, k + 1),
                            (i, j, k + 1),
                        ]
                    )
                if j == ny - 1 or not occupancy[i, j + 1, k]:
                    emit_face(
                        [
                            (i, j + 1, k),
                            (i, j + 1, k + 1),
                            (i + 1, j + 1, k + 1),
                            (i + 1, j + 1, k),
                        ]
                    )
                if k == 0 or not occupancy[i, j, k - 1]:
                    emit_face(
                        [
                            (i, j, k),
                            (i, j + 1, k),
                            (i + 1, j + 1, k),
                            (i + 1, j, k),
                        ]
                    )
                if k == nz - 1 or not occupancy[i, j, k + 1]:
                    emit_face(
                        [
                            (i, j, k + 1),
                            (i + 1, j, k + 1),
                            (i + 1, j + 1, k + 1),
                            (i, j + 1, k + 1),
                        ]
                    )

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        raise ValueError("Generated hole mesh is not watertight")

    return mesh


def make_smooth_visual_hole_block(
    L=0.08,
    W=0.08,
    H=0.03,
    hole_r=0.015,
    nx=160,
    ny=160,
    surface_fn=default_surface,
):
    x_edges = np.linspace(-L / 2, L / 2, nx + 1)
    y_edges = np.linspace(-W / 2, W / 2, ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    occupied = Xc * Xc + Yc * Yc >= hole_r * hole_r

    Xv, Yv = np.meshgrid(x_edges, y_edges, indexing="ij")
    top_heights = H + surface_fn(Xv, Yv)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_ids: dict[tuple[int, int, bool], int] = {}

    def vertex_index(i: int, j: int, top: bool) -> int:
        key = (i, j, top)
        if key not in vertex_ids:
            z = float(top_heights[i, j]) if top else 0.0
            vertex_ids[key] = len(vertices)
            vertices.append([float(x_edges[i]), float(y_edges[j]), z])
        return vertex_ids[key]

    def add_top_cell(i: int, j: int) -> None:
        a = vertex_index(i, j, True)
        b = vertex_index(i + 1, j, True)
        c = vertex_index(i + 1, j + 1, True)
        d = vertex_index(i, j + 1, True)
        _add_quad(faces, a, b, c, d)

    def add_bottom_cell(i: int, j: int) -> None:
        a = vertex_index(i, j, False)
        b = vertex_index(i, j + 1, False)
        c = vertex_index(i + 1, j + 1, False)
        d = vertex_index(i + 1, j, False)
        _add_quad(faces, a, b, c, d)

    def add_wall(i0: int, j0: int, i1: int, j1: int) -> None:
        a = vertex_index(i0, j0, False)
        b = vertex_index(i1, j1, False)
        c = vertex_index(i1, j1, True)
        d = vertex_index(i0, j0, True)
        _add_quad(faces, a, b, c, d)

    for i in range(nx):
        for j in range(ny):
            if not occupied[i, j]:
                continue

            add_top_cell(i, j)
            add_bottom_cell(i, j)

            if i == 0 or not occupied[i - 1, j]:
                add_wall(i, j, i, j + 1)
            if i == nx - 1 or not occupied[i + 1, j]:
                add_wall(i + 1, j + 1, i + 1, j)
            if j == 0 or not occupied[i, j - 1]:
                add_wall(i + 1, j, i, j)
            if j == ny - 1 or not occupied[i, j + 1]:
                add_wall(i, j + 1, i + 1, j + 1)

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        raise ValueError("Generated smooth visual hole mesh is not watertight")

    return mesh


def export_convex_parts(mesh, out_dir, prefix="part", max_hulls=32):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        parts = convex_decomposition(
            mesh,
            maxConvexHulls=max_hulls,
            resolution=400000,
            maxNumVerticesPerCH=64,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "vhacdx":
            raise ModuleNotFoundError(
                "Convex decomposition requires the optional 'vhacdx' package. "
                "Install it in the active environment to export convex parts."
            ) from exc
        raise

    for i, part in enumerate(parts):
        if isinstance(part, dict):
            part_mesh = trimesh.Trimesh(**part)
        else:
            part_mesh = part

        part_mesh.export(out_dir / f"{prefix}_{i}.stl")

    return parts


def _relative_posix_path(target_path: Path, start_dir: Path) -> str:
    return Path(os.path.relpath(target_path, start=start_dir)).as_posix()


def generate_parametric_hole_xml(
    part_name,
    output_path,
    visual_mesh_path,
    collision_mesh_dir,
    body_pos=(0.4, 0.0, 0.3),
    body_quat=(1.0, 0.0, 0.0, 0.0),
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collision_mesh_dir = Path(collision_mesh_dir)
    collision_meshes = sorted(collision_mesh_dir.glob("geometry_*.stl"))

    visual_rel = _relative_posix_path(Path(visual_mesh_path), output_path.parent)
    asset_lines = [f'        <mesh name="holesim" file="{visual_rel}"/>']
    geom_lines = ['            <geom mesh="holesim" class="visual" rgba="0.55 0.35 0.2 1"/>']

    for mesh_path in collision_meshes:
        mesh_name = mesh_path.stem
        mesh_rel = _relative_posix_path(mesh_path, output_path.parent)
        asset_lines.append(f'        <mesh name="{mesh_name}" file="{mesh_rel}"/>')
        geom_lines.append(f'            <geom mesh="{mesh_name}" class="collision" friction="0 0 0"/>')

    pos_str = " ".join(str(v) for v in body_pos)
    quat_str = " ".join(str(v) for v in body_quat)

    xml_text = "\n".join(
        [
            '<mujoco model="hole">',
            "    <asset>",
            *asset_lines,
            "    </asset>",
            "",
            "    <worldbody>",
            f'        <body name="task" pos="{pos_str}" quat="{quat_str}">',
            *geom_lines,
            "        </body>",
            "    </worldbody>",
            "</mujoco>",
            "",
        ]
    )

    output_path.write_text(xml_text)
    return output_path


def _prepare_output_locations(
    part_name: str,
    *,
    output_root: Path,
    models_dir: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    out_dir = output_root / part_name
    xml_path = models_dir / f"{part_name}.xml"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if xml_path.exists():
        xml_path.unlink()

    return out_dir, xml_path


def generate_part_assets(
    config: CadPartConfig,
    *,
    output_root: Path | str = REPO_ROOT / "generated_cad",
    models_dir: Path | str = REPO_ROOT / "models",
) -> GeneratedPartArtifacts:
    surface_model = config.validate()
    part_name = config.normalized_part_name()
    output_root = Path(output_root)
    models_dir = Path(models_dir)
    out_dir, xml_path = _prepare_output_locations(
        part_name,
        output_root=output_root,
        models_dir=models_dir,
    )

    surface_mesh = make_smooth_visual_hole_block(
        L=config.length,
        W=config.width,
        H=config.height,
        hole_r=config.hole_radius,
        nx=config.nx,
        ny=config.ny,
        surface_fn=surface_model.height,
    )

    visual_mesh_path = out_dir / "hole_block.stl"
    visual_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    surface_mesh.export(visual_mesh_path)
    metadata_path, surface_function_path = write_generation_metadata(
        out_dir,
        config=config,
        surface_model=surface_model,
    )
    convex_dir = out_dir / "hole_block_convex"
    export_convex_parts(surface_mesh, convex_dir, "geometry")
    generate_parametric_hole_xml(
        part_name=part_name,
        output_path=xml_path,
        visual_mesh_path=visual_mesh_path,
        collision_mesh_dir=convex_dir,
        body_pos=config.body_pos,
        body_quat=config.body_quat,
    )
    return GeneratedPartArtifacts(
        part_name=part_name,
        output_dir=out_dir,
        visual_mesh_path=visual_mesh_path,
        convex_dir=convex_dir,
        metadata_path=metadata_path,
        surface_function_path=surface_function_path,
        xml_path=xml_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CAD models with holes and export convex parts.")
    parser.add_argument("--part-name", type=str, default=DEFAULT_PART_NAME, help="Directory to save generated CAD files.")
    parser.add_argument("--block-length", type=float, default=DEFAULT_BLOCK_LENGTH, help="Block length in meters.")
    parser.add_argument("--block-width", type=float, default=DEFAULT_BLOCK_WIDTH, help="Block width in meters.")
    parser.add_argument("--block-height", type=float, default=DEFAULT_BLOCK_HEIGHT, help="Block height in meters.")
    parser.add_argument("--hole-radius", type=float, default=DEFAULT_HOLE_RADIUS, help="Centered circular through-hole radius in meters.")
    parser.add_argument("--nx", type=int, default=DEFAULT_NX, help="Number of x samples for the generated block.")
    parser.add_argument("--ny", type=int, default=DEFAULT_NY, help="Number of y samples for the generated block.")
    parser.add_argument(
        "--surface-family",
        type=str,
        choices=["default", "random_gaussian_two_peak"],
        default="default",
        help="Family of smooth surfaces to generate.",
    )
    parser.add_argument("--surface-amp", type=float, default=0.002, help="Overall amplitude of the top surface.")
    parser.add_argument("--surface-freq-x", type=float, default=80.0, help="X frequency for the default surface.")
    parser.add_argument("--surface-freq-y", type=float, default=60.0, help="Y frequency for the default surface.")
    parser.add_argument("--surface-seed", type=int, default=0, help="Seed used for randomized surface families.")
    parser.add_argument(
        "--gaussian-curvature",
        type=float,
        default=0.015,
        help="Gaussian spread for the two random peaks. Smaller values create sharper curvature.",
    )
    parser.add_argument(
        "--gaussian-peak-offset",
        type=float,
        default=0.02,
        help="Maximum absolute x/y offset of each random Gaussian peak from the center.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "generated_cad",
        help="Directory where generated CAD folders should be written.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory where the generated MuJoCo XML should be written.",
    )
    return parser.parse_args()


def main() -> GeneratedPartArtifacts:
    args = parse_args()
    config = cad_part_config_from_args(args)
    artifacts = generate_part_assets(
        config,
        output_root=args.output_root,
        models_dir=args.models_dir,
    )
    print(f"Saved CAD assets to {artifacts.output_dir}")
    print(f"Saved model XML to {artifacts.xml_path}")
    return artifacts


if __name__ == "__main__":
    main()
