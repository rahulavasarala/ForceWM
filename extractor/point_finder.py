from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml


DEFAULT_FR3_XML_PATH = Path(__file__).resolve().parents[1] / "models" / "fr3.xml"
DEFAULT_SCENE_XML_PATH = Path(__file__).resolve().parents[1] / "models" / "parametric_scene.xml"
DEFAULT_STATIONARY_CAMERA_NAME = "stationary_camera"
DEFAULT_CONTACT_GEOM_NAME = "ee_contact"
DEFAULT_NUM_RINGS = 3
DEFAULT_POINTS_PER_RING = 10
DEFAULT_CONTACT_COLOR_PATCH_RADIUS_PX = 2
DEFAULT_CONTACT_COLOR_CHANNEL_TOLERANCE = 95
DEFAULT_OCCLUSION_RADIUS_PX = 2.0
DEFAULT_MAX_REPROJECTION_ERROR_M = 0.02
MIN_CAMERA_DEPTH = 1e-6
# The rendered ee_contact_visual appears as a shaded light gray in the saved
# camera frames, so the color gate is centered on that observed BGR value.
DEFAULT_CONTACT_VISUAL_BGR = np.array([160, 160, 160], dtype=np.uint8)


@dataclass(frozen=True)
class CameraCalibration:
    fovy_degrees: float
    camera_position_world: np.ndarray
    camera_right_world: np.ndarray
    camera_down_world: np.ndarray
    camera_forward_world: np.ndarray


@dataclass(frozen=True)
class ContactCylinderSpec:
    radius_m: float
    half_height_m: float


@dataclass(frozen=True)
class BottomSurfaceRingConfig:
    radius_scale: float
    num_points: int


@dataclass(frozen=True)
class BottomSurfaceConfig:
    include_center: bool
    concentric_rings: tuple[BottomSurfaceRingConfig, ...]


@dataclass(frozen=True)
class RingPointConfig:
    height_fraction: float
    num_points: int


@dataclass(frozen=True)
class SimplePointConfig:
    bottom_surface: BottomSurfaceConfig
    middle_ring: RingPointConfig
    upper_ring: RingPointConfig
    female_surface_points: int = 256


@dataclass(frozen=True)
class PointSelectionDiagnostics:
    local_points: np.ndarray
    local_normals: np.ndarray
    world_points: np.ndarray
    world_normals: np.ndarray
    projected_pixels: np.ndarray
    rounded_pixels: np.ndarray
    camera_depths: np.ndarray
    facing_scores: np.ndarray
    finite_projection_mask: np.ndarray
    positive_depth_mask: np.ndarray
    facing_mask: np.ndarray
    in_bounds_mask: np.ndarray
    white_mask: np.ndarray
    depth_filter_applied: bool
    depth_observed_mask: np.ndarray
    observed_depth_m: np.ndarray
    depth_reprojected_world_points: np.ndarray
    reprojection_error_m: np.ndarray
    reprojection_error_mask: np.ndarray
    pre_occlusion_mask: np.ndarray
    occlusion_keep_mask: np.ndarray
    final_keep_mask: np.ndarray
    rejection_reasons: tuple[str, ...]
    num_rings: int
    points_per_ring: int
    frame_shape: tuple[int, int]


class SupportsAlignedEpisode(Protocol):
    positions: np.ndarray
    orientations: np.ndarray
    frames: np.ndarray


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def default_simple_point_config() -> SimplePointConfig:
    return SimplePointConfig(
        bottom_surface=BottomSurfaceConfig(
            include_center=True,
            concentric_rings=(
                BottomSurfaceRingConfig(radius_scale=0.5, num_points=6),
                BottomSurfaceRingConfig(radius_scale=1.0, num_points=8),
            ),
        ),
        middle_ring=RingPointConfig(height_fraction=0.5, num_points=8),
        upper_ring=RingPointConfig(height_fraction=1.0, num_points=8),
        female_surface_points=256,
    )


def _validate_simple_point_config(point_config: SimplePointConfig) -> SimplePointConfig:
    if not isinstance(point_config, SimplePointConfig):
        raise TypeError("point_config must be a SimplePointConfig instance.")

    bottom_surface = point_config.bottom_surface
    if not isinstance(bottom_surface.include_center, bool):
        raise ValueError("bottom_surface.include_center must be a boolean.")

    total_surface_points = 1 if bottom_surface.include_center else 0
    for ring_config in bottom_surface.concentric_rings:
        if not 0.0 <= float(ring_config.radius_scale) <= 1.0:
            raise ValueError("bottom_surface.concentric_rings[*].radius_scale must be in [0.0, 1.0].")
        if int(ring_config.num_points) <= 0:
            raise ValueError("bottom_surface.concentric_rings[*].num_points must be positive.")
        total_surface_points += int(ring_config.num_points)

    if total_surface_points <= 0:
        raise ValueError("bottom_surface must define at least one point.")

    for name, ring_config in (
        ("middle_ring", point_config.middle_ring),
        ("upper_ring", point_config.upper_ring),
    ):
        if not 0.0 <= float(ring_config.height_fraction) <= 1.0:
            raise ValueError(f"{name}.height_fraction must be in [0.0, 1.0].")
        if int(ring_config.num_points) <= 0:
            raise ValueError(f"{name}.num_points must be positive.")

    if int(point_config.female_surface_points) <= 0:
        raise ValueError("female_surface_points must be positive.")

    return point_config


def load_point_config(point_config_path: str | Path | None = None) -> SimplePointConfig:
    if point_config_path is None:
        return default_simple_point_config()

    point_config_path = Path(point_config_path).expanduser().resolve()
    if not point_config_path.exists():
        raise FileNotFoundError(f"Point-config file does not exist: {point_config_path}")

    with point_config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Point-config at {point_config_path} must contain a top-level mapping.")

    def require_mapping(parent: dict, field_name: str) -> dict:
        value = parent.get(field_name)
        if not isinstance(value, dict):
            raise ValueError(f"`{field_name}` must be a mapping in {point_config_path}.")
        return value

    def require_bool(parent: dict, field_name: str) -> bool:
        value = parent.get(field_name)
        if not isinstance(value, bool):
            raise ValueError(f"`{field_name}` must be a boolean in {point_config_path}.")
        return value

    def require_positive_int(parent: dict, field_name: str) -> int:
        if field_name not in parent:
            raise ValueError(f"`{field_name}` is required in {point_config_path}.")
        value = int(parent[field_name])
        if value <= 0:
            raise ValueError(f"`{field_name}` must be positive in {point_config_path}.")
        return value

    def require_fraction(parent: dict, field_name: str) -> float:
        if field_name not in parent:
            raise ValueError(f"`{field_name}` is required in {point_config_path}.")
        value = float(parent[field_name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"`{field_name}` must be in [0.0, 1.0] in {point_config_path}.")
        return value

    bottom_surface_raw = require_mapping(raw_config, "bottom_surface")
    ring_entries = bottom_surface_raw.get("concentric_rings")
    if not isinstance(ring_entries, list):
        raise ValueError(f"`bottom_surface.concentric_rings` must be a list in {point_config_path}.")

    bottom_surface = BottomSurfaceConfig(
        include_center=require_bool(bottom_surface_raw, "include_center"),
        concentric_rings=tuple(
            BottomSurfaceRingConfig(
                radius_scale=require_fraction(ring_raw, "radius_scale"),
                num_points=require_positive_int(ring_raw, "num_points"),
            )
            for ring_raw in ring_entries
            if isinstance(ring_raw, dict)
        ),
    )
    if len(bottom_surface.concentric_rings) != len(ring_entries):
        raise ValueError(f"Each entry in `bottom_surface.concentric_rings` must be a mapping in {point_config_path}.")

    middle_ring_raw = require_mapping(raw_config, "middle_ring")
    upper_ring_raw = require_mapping(raw_config, "upper_ring")
    point_config = SimplePointConfig(
        bottom_surface=bottom_surface,
        middle_ring=RingPointConfig(
            height_fraction=require_fraction(middle_ring_raw, "height_fraction"),
            num_points=require_positive_int(middle_ring_raw, "num_points"),
        ),
        upper_ring=RingPointConfig(
            height_fraction=require_fraction(upper_ring_raw, "height_fraction"),
            num_points=require_positive_int(upper_ring_raw, "num_points"),
        ),
        female_surface_points=require_positive_int(raw_config, "female_surface_points"),
    )
    return _validate_simple_point_config(point_config)


def parse_mujoco_float_vector(raw_value: str, expected_length: int, field_name: str) -> np.ndarray:
    values = np.fromstring(raw_value, sep=" ", dtype=np.float64)
    if len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} values for `{field_name}`, got {len(values)} from `{raw_value}`.")
    return values


def compute_camera_intrinsics(
    frame_height: int,
    frame_width: int,
    fovy_degrees: float,
) -> tuple[float, float, float, float]:
    fovy_radians = math.radians(float(fovy_degrees))
    fy = float(frame_height) / (2.0 * math.tan(fovy_radians / 2.0))
    fx = fy
    cx = (float(frame_width) - 1.0) / 2.0
    cy = (float(frame_height) - 1.0) / 2.0
    return fx, fy, cx, cy


def load_contact_cylinder_spec(
    model_xml_path: Path = DEFAULT_FR3_XML_PATH,
    geom_name: str = DEFAULT_CONTACT_GEOM_NAME,
) -> ContactCylinderSpec:
    if not model_xml_path.exists():
        raise FileNotFoundError(f"Contact geometry XML does not exist: {model_xml_path}")

    model_root = ET.parse(model_xml_path).getroot()

    for geom_element in model_root.iter("geom"):
        if geom_element.attrib.get("name") != geom_name:
            continue
        if geom_element.attrib.get("type") != "cylinder":
            raise ValueError(f"Geom `{geom_name}` in {model_xml_path} must have type `cylinder`.")
        if "size" not in geom_element.attrib:
            raise ValueError(f"Geom `{geom_name}` in {model_xml_path} is missing `size`.")
        size = parse_mujoco_float_vector(
            geom_element.attrib["size"],
            expected_length=2,
            field_name="geom size",
        ).astype(np.float32)
        return ContactCylinderSpec(radius_m=float(size[0]), half_height_m=float(size[1]))

    raise ValueError(f"Could not find geom `{geom_name}` in {model_xml_path}.")


def load_camera_calibration(
    camera_name: str = DEFAULT_STATIONARY_CAMERA_NAME,
    scene_xml_path: Path = DEFAULT_SCENE_XML_PATH,
) -> CameraCalibration:
    if not scene_xml_path.exists():
        raise FileNotFoundError(f"Camera scene XML does not exist: {scene_xml_path}")

    scene_root = ET.parse(scene_xml_path).getroot()
    for camera_element in scene_root.iter("camera"):
        if camera_element.attrib.get("name") != camera_name:
            continue

        if "pos" not in camera_element.attrib or "fovy" not in camera_element.attrib:
            raise ValueError(f"Camera `{camera_name}` in {scene_xml_path} is missing `pos` or `fovy`.")
        if "xyaxes" not in camera_element.attrib:
            raise ValueError(f"Camera `{camera_name}` in {scene_xml_path} is missing `xyaxes`.")

        camera_position_world = parse_mujoco_float_vector(
            camera_element.attrib["pos"],
            expected_length=3,
            field_name="camera pos",
        ).astype(np.float32)
        xyaxes = parse_mujoco_float_vector(
            camera_element.attrib["xyaxes"],
            expected_length=6,
            field_name="camera xyaxes",
        ).astype(np.float32)
        right_world = _normalize(xyaxes[:3].astype(np.float64)).astype(np.float32)
        up_world = _normalize(xyaxes[3:].astype(np.float64)).astype(np.float32)
        down_world = (-up_world).astype(np.float32)
        forward_world = (-_normalize(np.cross(right_world, up_world).astype(np.float64))).astype(np.float32)
        fovy_degrees = float(camera_element.attrib["fovy"])
        return CameraCalibration(
            fovy_degrees=fovy_degrees,
            camera_position_world=camera_position_world,
            camera_right_world=right_world,
            camera_down_world=down_world,
            camera_forward_world=forward_world,
        )

    raise ValueError(f"Could not find camera `{camera_name}` in {scene_xml_path}.")


def generate_contact_candidate_points(
    contact_spec: ContactCylinderSpec,
    num_rings: int = DEFAULT_NUM_RINGS,
    points_per_ring: int = DEFAULT_POINTS_PER_RING,
) -> tuple[np.ndarray, np.ndarray]:
    if num_rings <= 0:
        raise ValueError("num_rings must be positive.")
    if points_per_ring <= 0:
        raise ValueError("points_per_ring must be positive.")

    ring_heights = np.linspace(0.0, contact_spec.half_height_m, num_rings, dtype=np.float32)
    thetas = np.linspace(0.0, 2.0 * math.pi, points_per_ring, endpoint=False, dtype=np.float64)
    local_points = np.empty((num_rings * points_per_ring, 3), dtype=np.float32)
    local_normals = np.empty_like(local_points)

    point_index = 0
    for ring_height in ring_heights:
        for theta in thetas:
            cos_theta = float(math.cos(theta))
            sin_theta = float(math.sin(theta))
            local_points[point_index] = np.array(
                [contact_spec.radius_m * cos_theta, contact_spec.radius_m * sin_theta, -ring_height],
                dtype=np.float32,
            )
            local_normals[point_index] = np.array([cos_theta, sin_theta, 0.0], dtype=np.float32)
            point_index += 1

    return local_points, local_normals


def _sample_circle_points(radius_m: float, z_height_m: float, num_points: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive when sampling a circle.")

    thetas = np.linspace(0.0, 2.0 * math.pi, int(num_points), endpoint=False, dtype=np.float64)
    circle_points = np.empty((int(num_points), 3), dtype=np.float32)
    for point_index, theta in enumerate(thetas):
        circle_points[point_index] = np.array(
            [
                float(radius_m) * float(math.cos(theta)),
                float(radius_m) * float(math.sin(theta)),
                float(z_height_m),
            ],
            dtype=np.float32,
        )
    return circle_points


def generate_simple_point_template(
    contact_spec: ContactCylinderSpec,
    point_config: SimplePointConfig | None = None,
) -> np.ndarray:
    point_config = _validate_simple_point_config(
        default_simple_point_config() if point_config is None else point_config
    )

    local_points: list[np.ndarray] = []
    # The recorded eef_pos comes from the motion-force task control point,
    # which is placed on the contact face of the EE cylinder. Anchor the
    # synthetic template at that face so the bottom surface sits at z=0 and
    # the side rings span the full cylinder height away from the anchor.
    bottom_z = 0.0
    full_height = 2.0 * float(contact_spec.half_height_m)

    if point_config.bottom_surface.include_center:
        local_points.append(np.array([0.0, 0.0, bottom_z], dtype=np.float32))

    for ring_config in point_config.bottom_surface.concentric_rings:
        local_points.append(
            _sample_circle_points(
                radius_m=float(ring_config.radius_scale) * float(contact_spec.radius_m),
                z_height_m=bottom_z,
                num_points=int(ring_config.num_points),
            )
        )

    for ring_config in (point_config.middle_ring, point_config.upper_ring):
        ring_z = -float(ring_config.height_fraction) * full_height
        local_points.append(
            _sample_circle_points(
                radius_m=float(contact_spec.radius_m),
                z_height_m=ring_z,
                num_points=int(ring_config.num_points),
            )
        )

    if not local_points:
        raise ValueError("Synthetic point template must contain at least one point.")

    return np.concatenate(
        [
            point_block[None, :]
            if isinstance(point_block, np.ndarray) and point_block.ndim == 1
            else np.asarray(point_block, dtype=np.float32)
            for point_block in local_points
        ],
        axis=0,
    ).astype(np.float32)


def generate_points_simple(
    positions_world: np.ndarray,
    orientations_world: np.ndarray,
    contact_spec: ContactCylinderSpec,
    point_config: SimplePointConfig | None = None,
) -> np.ndarray:
    positions_world = np.asarray(positions_world, dtype=np.float32)
    orientations_world = np.asarray(orientations_world, dtype=np.float32)
    if positions_world.ndim != 2 or positions_world.shape[1] != 3:
        raise ValueError(f"`positions_world` must have shape (T, 3), got {positions_world.shape}.")
    if orientations_world.ndim != 3 or orientations_world.shape[1:] != (3, 3):
        raise ValueError(f"`orientations_world` must have shape (T, 3, 3), got {orientations_world.shape}.")
    if len(positions_world) != len(orientations_world):
        raise ValueError("positions_world and orientations_world must have matching lengths.")

    local_template = generate_simple_point_template(contact_spec, point_config=point_config)
    if not np.all(np.isfinite(positions_world)) or not np.all(np.isfinite(orientations_world)):
        raise ValueError("Synthetic point generation expects finite positions and orientations.")

    world_points = np.einsum("pj,tij->tpi", local_template.astype(np.float64), orientations_world.astype(np.float64))
    world_points = world_points + positions_world[:, None, :].astype(np.float64)
    return world_points.astype(np.float32)


def project_world_points_to_pixels(
    points_world: np.ndarray,
    camera_calibration: CameraCalibration,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    frame_height, frame_width = frame_shape
    fx, fy, cx, cy = compute_camera_intrinsics(
        frame_height=frame_height,
        frame_width=frame_width,
        fovy_degrees=camera_calibration.fovy_degrees,
    )

    rel_world = np.asarray(points_world, dtype=np.float32) - camera_calibration.camera_position_world[None, :]
    camera_x = rel_world @ camera_calibration.camera_right_world
    camera_y = rel_world @ camera_calibration.camera_down_world
    camera_z = rel_world @ camera_calibration.camera_forward_world

    projected_pixels = np.full((len(points_world), 2), np.nan, dtype=np.float32)
    valid_depth_mask = camera_z > MIN_CAMERA_DEPTH
    if np.any(valid_depth_mask):
        projected_pixels[valid_depth_mask, 0] = (
            fx * camera_x[valid_depth_mask] / camera_z[valid_depth_mask] + cx
        ).astype(np.float32)
        projected_pixels[valid_depth_mask, 1] = (
            fy * camera_y[valid_depth_mask] / camera_z[valid_depth_mask] + cy
        ).astype(np.float32)

    return projected_pixels, camera_z.astype(np.float32)


def patch_contains_contact_like_pixel(
    frame_bgr: np.ndarray,
    pixel_xy: np.ndarray,
    patch_radius_px: int = DEFAULT_CONTACT_COLOR_PATCH_RADIUS_PX,
    channel_tolerance: int = DEFAULT_CONTACT_COLOR_CHANNEL_TOLERANCE,
) -> bool:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        raise ValueError("Expected frame_bgr to have shape HxWx3.")

    frame_height, frame_width = frame_bgr.shape[:2]
    pixel_x = int(pixel_xy[0])
    pixel_y = int(pixel_xy[1])

    x0 = max(0, pixel_x - patch_radius_px)
    x1 = min(frame_width, pixel_x + patch_radius_px + 1)
    y0 = max(0, pixel_y - patch_radius_px)
    y1 = min(frame_height, pixel_y + patch_radius_px + 1)
    patch = frame_bgr[y0:y1, x0:x1, :3]
    if patch.size == 0:
        return False

    patch_int = patch.astype(np.int16)
    color_delta = np.abs(patch_int - DEFAULT_CONTACT_VISUAL_BGR.astype(np.int16)[None, None, :])
    color_match_mask = np.all(color_delta <= int(channel_tolerance), axis=2)
    return bool(np.any(color_match_mask))


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for depth-backed point selection. Install `opencv-python` in the active environment."
        ) from exc
    return cv2


def _read_depth_frame(frame_path: Path) -> np.ndarray:
    cv2 = _require_cv2()
    depth_frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if depth_frame is None:
        raise RuntimeError(f"Failed to read depth frame: {frame_path}")
    if depth_frame.ndim != 2 or depth_frame.dtype != np.uint16:
        raise ValueError(
            f"Depth frame `{frame_path}` must decode as HxW uint16, got shape={depth_frame.shape} dtype={depth_frame.dtype}."
        )
    return depth_frame


def _load_first_aligned_depth_frame(
    aligned_episode: SupportsAlignedEpisode,
    expected_frame_shape: tuple[int, int],
) -> np.ndarray | None:
    injected_depth_frames = getattr(aligned_episode, "depth_frames_mm", None)
    if injected_depth_frames is not None:
        depth_frames_array = np.asarray(injected_depth_frames)
        if depth_frames_array.ndim == 2:
            depth_frame = depth_frames_array
        elif depth_frames_array.ndim == 3 and len(depth_frames_array) > 0:
            depth_frame = depth_frames_array[0]
        else:
            raise ValueError("depth_frames_mm must have shape HxW or TxHxW when provided.")
        if tuple(depth_frame.shape) != tuple(expected_frame_shape):
            raise ValueError(
                f"Injected depth frame shape mismatch: expected {expected_frame_shape}, got {depth_frame.shape}."
            )
        return depth_frame.astype(np.uint16, copy=False)

    source_dir = getattr(aligned_episode, "source_dir", None)
    source_frame_indices = getattr(aligned_episode, "source_frame_indices", None)
    if source_dir is None or source_frame_indices is None:
        return None

    episode_dir = Path(source_dir)
    depth_frames_dir = episode_dir / "visual" / "depth" / "depth_frames"
    if not depth_frames_dir.exists() or not depth_frames_dir.is_dir():
        return None

    depth_frame_paths = sorted(
        path for path in depth_frames_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not depth_frame_paths:
        return None

    source_frame_indices_array = np.asarray(source_frame_indices).reshape(-1)
    if len(source_frame_indices_array) == 0:
        return None
    source_frame_index = int(source_frame_indices_array[0])
    if source_frame_index < 0 or source_frame_index >= len(depth_frame_paths):
        raise IndexError(
            f"Source frame index {source_frame_index} is out of range for {len(depth_frame_paths)} depth frames."
        )

    depth_frame = _read_depth_frame(depth_frame_paths[source_frame_index])
    if tuple(depth_frame.shape) != tuple(expected_frame_shape):
        raise ValueError(
            f"Depth frame shape mismatch for {depth_frame_paths[source_frame_index]}: "
            f"expected {expected_frame_shape}, got {depth_frame.shape}."
        )
    return depth_frame


def reconstruct_world_points_from_depth_pixels(
    pixel_xy: np.ndarray,
    depth_frame_mm: np.ndarray,
    camera_calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth_frame_mm.ndim != 2:
        raise ValueError(f"Expected depth_frame_mm to have shape HxW, got {depth_frame_mm.shape}.")

    pixel_xy = np.asarray(pixel_xy, dtype=np.int64)
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise ValueError(f"Expected pixel_xy to have shape (N, 2), got {pixel_xy.shape}.")

    frame_height, frame_width = depth_frame_mm.shape
    fx, fy, cx, cy = compute_camera_intrinsics(
        frame_height=frame_height,
        frame_width=frame_width,
        fovy_degrees=camera_calibration.fovy_degrees,
    )

    world_points = np.full((len(pixel_xy), 3), np.nan, dtype=np.float32)
    observed_depth_m = np.full(len(pixel_xy), np.nan, dtype=np.float32)
    valid_mask = (
        (pixel_xy[:, 0] >= 0)
        & (pixel_xy[:, 0] < frame_width)
        & (pixel_xy[:, 1] >= 0)
        & (pixel_xy[:, 1] < frame_height)
    )
    if not np.any(valid_mask):
        return world_points, observed_depth_m, valid_mask

    valid_indices = np.flatnonzero(valid_mask)
    valid_x = pixel_xy[valid_mask, 0]
    valid_y = pixel_xy[valid_mask, 1]
    depth_values_mm = depth_frame_mm[valid_y, valid_x]
    valid_depth_mask = depth_values_mm > 0
    if not np.any(valid_depth_mask):
        return world_points, observed_depth_m, np.zeros(len(pixel_xy), dtype=bool)

    reconstructed_indices = valid_indices[valid_depth_mask]
    depth_m = depth_values_mm[valid_depth_mask].astype(np.float32) / 1000.0
    pixel_x = valid_x[valid_depth_mask].astype(np.float32)
    pixel_y = valid_y[valid_depth_mask].astype(np.float32)
    observed_depth_m[reconstructed_indices] = depth_m

    camera_x = (pixel_x - cx) * depth_m / fx
    camera_y = (pixel_y - cy) * depth_m / fy
    world_points[reconstructed_indices] = (
        camera_calibration.camera_position_world[None, :]
        + camera_x[:, None] * camera_calibration.camera_right_world[None, :]
        + camera_y[:, None] * camera_calibration.camera_down_world[None, :]
        + depth_m[:, None] * camera_calibration.camera_forward_world[None, :]
    ).astype(np.float32)

    reconstructed_mask = np.zeros(len(pixel_xy), dtype=bool)
    reconstructed_mask[reconstructed_indices] = True
    return world_points, observed_depth_m, reconstructed_mask


def suppress_occluded_pixels(
    projected_pixels_xy: np.ndarray,
    camera_depths: np.ndarray,
    suppression_radius_px: float = DEFAULT_OCCLUSION_RADIUS_PX,
) -> np.ndarray:
    if len(projected_pixels_xy) == 0:
        return np.zeros(0, dtype=np.int64)

    radius_sq = float(suppression_radius_px) ** 2
    sorted_indices = np.argsort(camera_depths, kind="stable")
    kept_sorted: list[int] = []

    for candidate_index in sorted_indices.tolist():
        candidate_pixel = projected_pixels_xy[candidate_index]
        if not kept_sorted:
            kept_sorted.append(candidate_index)
            continue

        distances_sq = np.sum((projected_pixels_xy[np.asarray(kept_sorted)] - candidate_pixel) ** 2, axis=1)
        if np.all(distances_sq > radius_sq):
            kept_sorted.append(candidate_index)

    return np.sort(np.asarray(kept_sorted, dtype=np.int64))


def diagnose_point_selection(
    aligned_episode: SupportsAlignedEpisode,
    camera_calibration: CameraCalibration,
    contact_spec: ContactCylinderSpec,
    num_rings: int = DEFAULT_NUM_RINGS,
    points_per_ring: int = DEFAULT_POINTS_PER_RING,
) -> PointSelectionDiagnostics:
    if len(aligned_episode.frames) == 0:
        raise ValueError("Cannot diagnose point selection without any aligned frames.")

    first_frame = np.asarray(aligned_episode.frames[0])
    eef_position_world = np.asarray(aligned_episode.positions[0], dtype=np.float32).reshape(3)
    eef_orientation_world = np.asarray(aligned_episode.orientations[0], dtype=np.float32).reshape(3, 3)

    local_points, local_normals = generate_contact_candidate_points(
        contact_spec,
        num_rings=num_rings,
        points_per_ring=points_per_ring,
    )
    world_points = local_points @ eef_orientation_world.T + eef_position_world
    world_normals = local_normals @ eef_orientation_world.T

    projected_pixels, camera_depths = project_world_points_to_pixels(
        world_points,
        camera_calibration=camera_calibration,
        frame_shape=tuple(first_frame.shape[:2]),
    )
    rounded_pixels = np.full((len(projected_pixels), 2), -1, dtype=np.int64)
    finite_projection_mask = np.isfinite(projected_pixels).all(axis=1)
    rounded_pixels[finite_projection_mask] = np.rint(projected_pixels[finite_projection_mask]).astype(np.int64)

    frame_height, frame_width = first_frame.shape[:2]
    facing_scores = np.einsum(
        "ij,ij->i",
        world_normals,
        camera_calibration.camera_position_world[None, :] - world_points,
    )
    positive_depth_mask = camera_depths > MIN_CAMERA_DEPTH
    facing_mask = facing_scores > 0.0
    in_bounds_mask = (
        (rounded_pixels[:, 0] >= 0)
        & (rounded_pixels[:, 0] < frame_width)
        & (rounded_pixels[:, 1] >= 0)
        & (rounded_pixels[:, 1] < frame_height)
    )

    white_mask = np.zeros(len(projected_pixels), dtype=bool)
    pre_white_mask = finite_projection_mask & positive_depth_mask & facing_mask & in_bounds_mask
    for candidate_index in np.flatnonzero(pre_white_mask).tolist():
        white_mask[candidate_index] = patch_contains_contact_like_pixel(
            first_frame,
            rounded_pixels[candidate_index],
        )

    pre_depth_mask = pre_white_mask & white_mask
    depth_filter_applied = False
    depth_observed_mask = np.zeros(len(projected_pixels), dtype=bool)
    observed_depth_m = np.full(len(projected_pixels), np.nan, dtype=np.float32)
    depth_reprojected_world_points = np.full((len(projected_pixels), 3), np.nan, dtype=np.float32)
    reprojection_error_m = np.full(len(projected_pixels), np.nan, dtype=np.float32)
    reprojection_error_mask = np.ones(len(projected_pixels), dtype=bool)

    depth_frame_mm = _load_first_aligned_depth_frame(
        aligned_episode,
        expected_frame_shape=(frame_height, frame_width),
    )
    if depth_frame_mm is not None:
        depth_filter_applied = True
        depth_reprojected_world_points, observed_depth_m, depth_observed_mask = reconstruct_world_points_from_depth_pixels(
            rounded_pixels,
            depth_frame_mm=depth_frame_mm,
            camera_calibration=camera_calibration,
        )
        reprojection_candidate_mask = pre_depth_mask & depth_observed_mask
        reprojection_error_mask = np.zeros(len(projected_pixels), dtype=bool)
        if np.any(reprojection_candidate_mask):
            reprojection_error_m[reprojection_candidate_mask] = np.linalg.norm(
                depth_reprojected_world_points[reprojection_candidate_mask] - world_points[reprojection_candidate_mask],
                axis=1,
            ).astype(np.float32)
            reprojection_error_mask[reprojection_candidate_mask] = (
                reprojection_error_m[reprojection_candidate_mask] <= DEFAULT_MAX_REPROJECTION_ERROR_M
            )

    pre_occlusion_mask = pre_depth_mask & reprojection_error_mask
    occlusion_keep_mask = np.zeros(len(projected_pixels), dtype=bool)
    if np.any(pre_occlusion_mask):
        pre_occlusion_indices = np.flatnonzero(pre_occlusion_mask)
        kept_relative_indices = suppress_occluded_pixels(
            projected_pixels_xy=projected_pixels[pre_occlusion_indices],
            camera_depths=camera_depths[pre_occlusion_indices],
        )
        occlusion_keep_mask[pre_occlusion_indices[kept_relative_indices]] = True

    final_keep_mask = pre_occlusion_mask & occlusion_keep_mask
    rejection_reasons: list[str] = []
    for point_index in range(len(projected_pixels)):
        if final_keep_mask[point_index]:
            rejection_reasons.append("kept")
        elif not finite_projection_mask[point_index] or not positive_depth_mask[point_index]:
            rejection_reasons.append("behind_camera")
        elif not facing_mask[point_index]:
            rejection_reasons.append("back_facing")
        elif not in_bounds_mask[point_index]:
            rejection_reasons.append("out_of_frame")
        elif not white_mask[point_index]:
            rejection_reasons.append("not_contact_color")
        elif depth_filter_applied and not depth_observed_mask[point_index]:
            rejection_reasons.append("missing_depth")
        elif depth_filter_applied and not reprojection_error_mask[point_index]:
            rejection_reasons.append("large_reprojection_error")
        else:
            rejection_reasons.append("occluded")

    return PointSelectionDiagnostics(
        local_points=local_points,
        local_normals=local_normals,
        world_points=world_points,
        world_normals=world_normals,
        projected_pixels=projected_pixels,
        rounded_pixels=rounded_pixels,
        camera_depths=camera_depths,
        facing_scores=facing_scores.astype(np.float32),
        finite_projection_mask=finite_projection_mask,
        positive_depth_mask=positive_depth_mask,
        facing_mask=facing_mask,
        in_bounds_mask=in_bounds_mask,
        white_mask=white_mask,
        depth_filter_applied=depth_filter_applied,
        depth_observed_mask=depth_observed_mask,
        observed_depth_m=observed_depth_m,
        depth_reprojected_world_points=depth_reprojected_world_points,
        reprojection_error_m=reprojection_error_m,
        reprojection_error_mask=reprojection_error_mask,
        pre_occlusion_mask=pre_occlusion_mask,
        occlusion_keep_mask=occlusion_keep_mask,
        final_keep_mask=final_keep_mask,
        rejection_reasons=tuple(rejection_reasons),
        num_rings=num_rings,
        points_per_ring=points_per_ring,
        frame_shape=(frame_height, frame_width),
    )


def select_points_to_track(
    aligned_episode: SupportsAlignedEpisode,
    camera_calibration: CameraCalibration,
    contact_spec: ContactCylinderSpec,
) -> np.ndarray:
    if len(aligned_episode.frames) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    diagnostics = diagnose_point_selection(
        aligned_episode=aligned_episode,
        camera_calibration=camera_calibration,
        contact_spec=contact_spec,
    )
    if not np.any(diagnostics.final_keep_mask):
        return np.zeros((0, 2), dtype=np.int32)
    return diagnostics.rounded_pixels[diagnostics.final_keep_mask].astype(np.int32)
