from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_XML_PATH = REPO_ROOT / "models" / "parametric_scene.xml"
DEFAULT_CAMERA_NAME = "stationary_camera"
DEFAULT_CAMERA_KEY = "camera_01"
DEFAULT_POINT_SIZE = 3.0
DEFAULT_ELEVATION = 24.0
DEFAULT_AZIMUTH = -58.0
DEFAULT_STRIDE = 1


@dataclass(frozen=True)
class CameraModel:
    fovy_degrees: float
    position_world: np.ndarray
    right_world: np.ndarray
    down_world: np.ndarray
    forward_world: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dense 3D point cloud from the first saved depth frame of one raw episode "
            "and visualize the reconstructed scene."
        )
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
        type=str,
        help="Path to one raw episode_* directory.",
    )
    parser.add_argument(
        "--scene-xml",
        default=str(DEFAULT_SCENE_XML_PATH),
        type=str,
        help="Path to the MuJoCo scene XML containing the stationary camera.",
    )
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        type=str,
        help="Camera name to parse from the scene XML.",
    )
    parser.add_argument(
        "--camera-key",
        default=DEFAULT_CAMERA_KEY,
        type=str,
        help="RGB camera key used to locate the companion video frame.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("axial", "range"),
        default="axial",
        help="Interpret saved depth as camera-space z (`axial`) or Euclidean ray length (`range`).",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Render both axial and range-based reconstructions side by side.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Use every Nth pixel along each axis when building the dense point cloud.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=DEFAULT_POINT_SIZE,
        help="Scatter marker size for rendered points.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=DEFAULT_ELEVATION,
        help="Initial 3D elevation angle in degrees.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=DEFAULT_AZIMUTH,
        help="Initial 3D azimuth angle in degrees.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        type=str,
        help="Optional path to save the figure instead of only showing it.",
    )
    return parser.parse_args()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _parse_float_vector(raw_value: str, expected_length: int, field_name: str) -> np.ndarray:
    values = np.fromstring(raw_value, sep=" ", dtype=np.float64)
    if len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} values for `{field_name}`, got {len(values)} from `{raw_value}`."
        )
    return values


def load_camera_model(scene_xml_path: Path, camera_name: str) -> CameraModel:
    scene_xml_path = scene_xml_path.expanduser().resolve()
    if not scene_xml_path.exists():
        raise FileNotFoundError(f"Scene XML does not exist: {scene_xml_path}")

    scene_root = ET.parse(scene_xml_path).getroot()
    for camera_element in scene_root.iter("camera"):
        if camera_element.attrib.get("name") != camera_name:
            continue
        if "pos" not in camera_element.attrib or "xyaxes" not in camera_element.attrib or "fovy" not in camera_element.attrib:
            raise ValueError(f"Camera `{camera_name}` in {scene_xml_path} is missing `pos`, `xyaxes`, or `fovy`.")

        position_world = _parse_float_vector(
            camera_element.attrib["pos"],
            expected_length=3,
            field_name="camera pos",
        ).astype(np.float32)
        xyaxes = _parse_float_vector(
            camera_element.attrib["xyaxes"],
            expected_length=6,
            field_name="camera xyaxes",
        ).astype(np.float32)
        right_world = _normalize(xyaxes[:3].astype(np.float64)).astype(np.float32)
        up_world = _normalize(xyaxes[3:].astype(np.float64)).astype(np.float32)
        down_world = (-up_world).astype(np.float32)
        forward_world = (-_normalize(np.cross(right_world, up_world).astype(np.float64))).astype(np.float32)
        fovy_degrees = float(camera_element.attrib["fovy"])
        return CameraModel(
            fovy_degrees=fovy_degrees,
            position_world=position_world,
            right_world=right_world,
            down_world=down_world,
            forward_world=forward_world,
        )

    raise ValueError(f"Could not find camera `{camera_name}` in {scene_xml_path}.")


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


def discover_first_depth_frame_path(episode_dir: Path) -> Path:
    depth_frames_dir = episode_dir / "visual" / "depth" / "depth_frames"
    if not depth_frames_dir.exists():
        raise FileNotFoundError(f"Depth frame directory does not exist: {depth_frames_dir}")
    depth_paths = sorted(
        path for path in depth_frames_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not depth_paths:
        raise FileNotFoundError(f"No depth PNG frames found in {depth_frames_dir}")
    return depth_paths[0]


def read_depth_frame(depth_frame_path: Path) -> np.ndarray:
    import cv2

    depth_frame = cv2.imread(str(depth_frame_path), cv2.IMREAD_UNCHANGED)
    if depth_frame is None:
        raise RuntimeError(f"Failed to read depth frame: {depth_frame_path}")
    if depth_frame.ndim != 2 or depth_frame.dtype != np.uint16:
        raise ValueError(
            f"Depth frame `{depth_frame_path}` must decode as HxW uint16, got shape={depth_frame.shape} dtype={depth_frame.dtype}."
        )
    return depth_frame


def read_first_rgb_frame(episode_dir: Path, camera_key: str) -> np.ndarray | None:
    import cv2

    video_path = episode_dir / "visual" / f"{camera_key}.mp4"
    if not video_path.exists():
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open RGB video file: {video_path}")
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read the first RGB frame from: {video_path}")
    return np.asarray(frame)


def colorize_depth_frame(depth_frame_mm: np.ndarray) -> np.ndarray:
    import cv2

    valid_depth_mm = depth_frame_mm[depth_frame_mm > 0]
    if valid_depth_mm.size == 0:
        return np.zeros((*depth_frame_mm.shape, 3), dtype=np.uint8)

    lower = float(np.percentile(valid_depth_mm, 1.0))
    upper = float(np.percentile(valid_depth_mm, 99.0))
    if upper <= lower:
        upper = lower + 1.0

    clipped = np.clip(depth_frame_mm.astype(np.float32), lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    normalized[depth_frame_mm == 0] = 0.0
    grayscale = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    colorized = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    colorized[depth_frame_mm == 0] = (0, 0, 0)
    return colorized


def build_dense_point_cloud(
    depth_frame_mm: np.ndarray,
    camera_model: CameraModel,
    depth_mode: str = "axial",
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth_mode not in {"axial", "range"}:
        raise ValueError(f"Unsupported depth mode: {depth_mode}")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    frame_height, frame_width = depth_frame_mm.shape
    fx, fy, cx, cy = compute_camera_intrinsics(
        frame_height=frame_height,
        frame_width=frame_width,
        fovy_degrees=camera_model.fovy_degrees,
    )

    grid_y, grid_x = np.indices((frame_height, frame_width), dtype=np.int64)
    valid_mask = depth_frame_mm > 0
    if stride > 1:
        valid_mask &= (grid_x % stride == 0)
        valid_mask &= (grid_y % stride == 0)

    pixel_x = grid_x[valid_mask].astype(np.float32)
    pixel_y = grid_y[valid_mask].astype(np.float32)
    depth_m = depth_frame_mm[valid_mask].astype(np.float32) / 1000.0
    if depth_m.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    ray_x = (pixel_x - cx) / fx
    ray_y = (pixel_y - cy) / fy

    if depth_mode == "axial":
        camera_x = ray_x * depth_m
        camera_y = ray_y * depth_m
        camera_z = depth_m
    else:
        ray_norm = np.sqrt(ray_x ** 2 + ray_y ** 2 + 1.0)
        camera_x = depth_m * ray_x / ray_norm
        camera_y = depth_m * ray_y / ray_norm
        camera_z = depth_m / ray_norm

    world_points = (
        camera_model.position_world[None, :]
        + camera_x[:, None] * camera_model.right_world[None, :]
        + camera_y[:, None] * camera_model.down_world[None, :]
        + camera_z[:, None] * camera_model.forward_world[None, :]
    ).astype(np.float32)
    pixels_xy = np.column_stack((pixel_x.astype(np.int32), pixel_y.astype(np.int32)))
    return world_points, pixels_xy, depth_m


def sample_point_colors(
    pixels_xy: np.ndarray,
    rgb_frame_bgr: np.ndarray | None,
    depth_frame_mm: np.ndarray,
) -> np.ndarray:
    if len(pixels_xy) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if rgb_frame_bgr is not None:
        colors_rgb = rgb_frame_bgr[pixels_xy[:, 1], pixels_xy[:, 0], ::-1].astype(np.float32) / 255.0
        return colors_rgb

    depth_values_mm = depth_frame_mm[pixels_xy[:, 1], pixels_xy[:, 0]].astype(np.float32)
    valid_depth_values_mm = depth_values_mm[depth_values_mm > 0]
    if valid_depth_values_mm.size == 0:
        return np.full((len(pixels_xy), 3), 0.7, dtype=np.float32)

    lower = float(np.percentile(valid_depth_values_mm, 1.0))
    upper = float(np.percentile(valid_depth_values_mm, 99.0))
    if upper <= lower:
        upper = lower + 1.0
    normalized = np.clip((depth_values_mm - lower) / (upper - lower), 0.0, 1.0)

    import matplotlib.pyplot as plt

    colors_rgba = plt.get_cmap("turbo")(normalized)
    return np.asarray(colors_rgba[:, :3], dtype=np.float32)


def set_equal_axes(ax, points_xyz: np.ndarray, padding: float = 0.02) -> None:
    if len(points_xyz) == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        return

    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    if extent <= 0.0 or not np.isfinite(extent):
        extent = 0.1
    half_extent = extent / 2.0 + padding
    ax.set_xlim(float(center[0] - half_extent), float(center[0] + half_extent))
    ax.set_ylim(float(center[1] - half_extent), float(center[1] + half_extent))
    ax.set_zlim(float(center[2] - half_extent), float(center[2] + half_extent))
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _scatter_point_cloud(
    ax,
    world_points: np.ndarray,
    colors_rgb: np.ndarray,
    camera_model: CameraModel,
    title: str,
    point_size: float,
    elevation: float,
    azimuth: float,
) -> None:
    ax.view_init(elev=elevation, azim=azimuth)
    set_equal_axes(
        ax,
        np.vstack([world_points, camera_model.position_world.reshape(1, 3)]) if len(world_points) else camera_model.position_world.reshape(1, 3),
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    if len(world_points):
        ax.scatter(
            world_points[:, 0],
            world_points[:, 1],
            world_points[:, 2],
            c=colors_rgb,
            s=point_size,
            depthshade=False,
        )
    ax.scatter(
        [camera_model.position_world[0]],
        [camera_model.position_world[1]],
        [camera_model.position_world[2]],
        color="black",
        marker="*",
        s=120,
    )


def visualize_scene_point_cloud(
    episode_dir: Path,
    depth_frame_mm: np.ndarray,
    rgb_frame_bgr: np.ndarray | None,
    camera_model: CameraModel,
    depth_mode: str,
    compare_modes: bool,
    stride: int,
    point_size: float,
    elevation: float,
    azimuth: float,
    save_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    preview_image_rgb = (rgb_frame_bgr[..., ::-1] if rgb_frame_bgr is not None else colorize_depth_frame(depth_frame_mm)[..., ::-1])

    if compare_modes:
        figure = plt.figure("Point Cloud Generator", figsize=(18, 7))
        grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.15, 1.15))
        ax_preview = figure.add_subplot(grid[0, 0])
        ax_axial = figure.add_subplot(grid[0, 1], projection="3d")
        ax_range = figure.add_subplot(grid[0, 2], projection="3d")

        axial_points, axial_pixels, axial_depth_m = build_dense_point_cloud(
            depth_frame_mm,
            camera_model=camera_model,
            depth_mode="axial",
            stride=stride,
        )
        range_points, range_pixels, range_depth_m = build_dense_point_cloud(
            depth_frame_mm,
            camera_model=camera_model,
            depth_mode="range",
            stride=stride,
        )
        axial_colors = sample_point_colors(axial_pixels, rgb_frame_bgr, depth_frame_mm)
        range_colors = sample_point_colors(range_pixels, rgb_frame_bgr, depth_frame_mm)

        _scatter_point_cloud(
            ax_axial,
            axial_points,
            axial_colors,
            camera_model=camera_model,
            title=f"Axial depth reconstruction\n{len(axial_points)} points",
            point_size=point_size,
            elevation=elevation,
            azimuth=azimuth,
        )
        _scatter_point_cloud(
            ax_range,
            range_points,
            range_colors,
            camera_model=camera_model,
            title=f"Range depth reconstruction\n{len(range_points)} points",
            point_size=point_size,
            elevation=elevation,
            azimuth=azimuth,
        )
        point_count = len(axial_points)
        depth_values_m = axial_depth_m
    else:
        figure = plt.figure("Point Cloud Generator", figsize=(14, 7))
        grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.25))
        ax_preview = figure.add_subplot(grid[0, 0])
        ax_cloud = figure.add_subplot(grid[0, 1], projection="3d")

        world_points, pixels_xy, depth_values_m = build_dense_point_cloud(
            depth_frame_mm,
            camera_model=camera_model,
            depth_mode=depth_mode,
            stride=stride,
        )
        colors_rgb = sample_point_colors(pixels_xy, rgb_frame_bgr, depth_frame_mm)
        _scatter_point_cloud(
            ax_cloud,
            world_points,
            colors_rgb,
            camera_model=camera_model,
            title=f"{depth_mode.title()} depth reconstruction\n{len(world_points)} points",
            point_size=point_size,
            elevation=elevation,
            azimuth=azimuth,
        )
        point_count = len(world_points)

    ax_preview.imshow(preview_image_rgb)
    ax_preview.set_title(f"{episode_dir.name}\nFirst RGB/depth preview")
    ax_preview.set_xlabel("Pixel x")
    ax_preview.set_ylabel("Pixel y")

    valid_depth_m = depth_values_m[np.isfinite(depth_values_m)]
    min_depth_m = float(valid_depth_m.min()) if valid_depth_m.size else float("nan")
    max_depth_m = float(valid_depth_m.max()) if valid_depth_m.size else float("nan")
    figure.text(
        0.02,
        0.02,
        f"Points: {point_count}  |  Depth range: {min_depth_m:.4f}m to {max_depth_m:.4f}m  |  Stride: {stride}",
        fontsize=10,
    )
    figure.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()


def main() -> int:
    args = _parse_args()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    scene_xml_path = Path(args.scene_xml).expanduser().resolve()
    save_path = None if args.save_path is None else Path(args.save_path).expanduser().resolve()

    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")

    depth_frame_path = discover_first_depth_frame_path(episode_dir)
    depth_frame_mm = read_depth_frame(depth_frame_path)
    rgb_frame_bgr = read_first_rgb_frame(episode_dir, camera_key=str(args.camera_key))
    camera_model = load_camera_model(scene_xml_path, camera_name=str(args.camera_name))

    print(f"Episode: {episode_dir}")
    print(f"First depth frame: {depth_frame_path}")
    print(f"Depth frame shape: {depth_frame_mm.shape}")
    print(f"Depth mode: {'axial + range comparison' if args.compare_modes else args.depth_mode}")

    valid_depth_mm = depth_frame_mm[depth_frame_mm > 0]
    if valid_depth_mm.size:
        print(
            f"Valid depth range: {int(valid_depth_mm.min())}mm to {int(valid_depth_mm.max())}mm "
            f"({float(valid_depth_mm.min()) / 1000.0:.4f}m to {float(valid_depth_mm.max()) / 1000.0:.4f}m)"
        )
    else:
        print("No valid depth pixels were found in the first depth frame.")

    visualize_scene_point_cloud(
        episode_dir=episode_dir,
        depth_frame_mm=depth_frame_mm,
        rgb_frame_bgr=rgb_frame_bgr,
        camera_model=camera_model,
        depth_mode=str(args.depth_mode),
        compare_modes=bool(args.compare_modes),
        stride=int(args.stride),
        point_size=float(args.point_size),
        elevation=float(args.elevation),
        azimuth=float(args.azimuth),
        save_path=save_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
