from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from extractor.extract_to_parquet import (
    align_episode_to_lowdim,
    discover_depth_frame_paths,
    load_episode_data,
    read_depth_frame,
)
from extractor.point_finder import (
    DEFAULT_CONTACT_COLOR_CHANNEL_TOLERANCE,
    DEFAULT_CONTACT_COLOR_PATCH_RADIUS_PX,
    DEFAULT_MAX_REPROJECTION_ERROR_M,
    DEFAULT_CONTACT_VISUAL_BGR,
    DEFAULT_CONTACT_GEOM_NAME,
    DEFAULT_FR3_XML_PATH,
    DEFAULT_NUM_RINGS,
    DEFAULT_POINTS_PER_RING,
    DEFAULT_SCENE_XML_PATH,
    DEFAULT_STATIONARY_CAMERA_NAME,
    PointSelectionDiagnostics,
    compute_camera_intrinsics,
    diagnose_point_selection,
    load_camera_calibration,
    load_contact_cylinder_spec,
)


REASON_COLORS = {
    "kept": "#1b9e77",
    "behind_camera": "#7f7f7f",
    "back_facing": "#d95f02",
    "out_of_frame": "#d73027",
    "not_contact_color": "#4575b4",
    "missing_depth": "#313695",
    "large_reprojection_error": "#f46d43",
    "occluded": "#8c510a",
}

REASON_MARKERS = {
    "kept": "o",
    "behind_camera": "x",
    "back_facing": "^",
    "out_of_frame": "s",
    "not_contact_color": "D",
    "missing_depth": "v",
    "large_reprojection_error": "X",
    "occluded": "P",
}


def reconstruct_depth_points(
    diagnostics: PointSelectionDiagnostics,
    depth_frame_mm: np.ndarray,
    camera_calibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth_frame_mm.shape != diagnostics.frame_shape:
        raise ValueError(
            f"Depth frame shape {depth_frame_mm.shape} does not match diagnostics frame shape {diagnostics.frame_shape}."
        )

    keep_indices = np.flatnonzero(diagnostics.final_keep_mask)
    world_points = np.full((len(keep_indices), 3), np.nan, dtype=np.float32)
    observed_depth_m = np.full(len(keep_indices), np.nan, dtype=np.float32)
    point_errors_m = np.full(len(keep_indices), np.nan, dtype=np.float32)
    if len(keep_indices) == 0:
        return world_points, observed_depth_m, point_errors_m

    frame_height, frame_width = diagnostics.frame_shape
    fx, fy, cx, cy = compute_camera_intrinsics(
        frame_height=frame_height,
        frame_width=frame_width,
        fovy_degrees=camera_calibration.fovy_degrees,
    )

    rounded_pixels = diagnostics.rounded_pixels[keep_indices]
    valid_depth_indices: list[int] = []
    valid_pixels_x: list[float] = []
    valid_pixels_y: list[float] = []
    valid_depth_m: list[float] = []

    for relative_index, (pixel_x, pixel_y) in enumerate(rounded_pixels.tolist()):
        if not (0 <= pixel_x < frame_width and 0 <= pixel_y < frame_height):
            continue
        depth_mm = int(depth_frame_mm[pixel_y, pixel_x])
        if depth_mm <= 0:
            continue
        valid_depth_indices.append(relative_index)
        valid_pixels_x.append(float(pixel_x))
        valid_pixels_y.append(float(pixel_y))
        valid_depth_m.append(float(depth_mm) / 1000.0)

    if not valid_depth_indices:
        return world_points, observed_depth_m, point_errors_m

    valid_depth_indices_array = np.asarray(valid_depth_indices, dtype=np.int64)
    valid_pixels_x_array = np.asarray(valid_pixels_x, dtype=np.float32)
    valid_pixels_y_array = np.asarray(valid_pixels_y, dtype=np.float32)
    valid_depth_m_array = np.asarray(valid_depth_m, dtype=np.float32)
    observed_depth_m[valid_depth_indices_array] = valid_depth_m_array

    camera_x = (valid_pixels_x_array - cx) * valid_depth_m_array / fx
    camera_y = (valid_pixels_y_array - cy) * valid_depth_m_array / fy
    world_points[valid_depth_indices_array] = (
        camera_calibration.camera_position_world[None, :]
        + camera_x[:, None] * camera_calibration.camera_right_world[None, :]
        + camera_y[:, None] * camera_calibration.camera_down_world[None, :]
        + valid_depth_m_array[:, None] * camera_calibration.camera_forward_world[None, :]
    ).astype(np.float32)

    expected_world_points = diagnostics.world_points[keep_indices[valid_depth_indices_array]]
    point_errors_m[valid_depth_indices_array] = np.linalg.norm(
        world_points[valid_depth_indices_array] - expected_world_points,
        axis=1,
    ).astype(np.float32)
    return world_points, observed_depth_m, point_errors_m


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the end-effector point-selection diagnostics for the first aligned frame "
            "of one raw episode directory."
        )
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
        type=str,
        help="Path to one raw episode_* directory containing lowdim and visual data.",
    )
    parser.add_argument(
        "--scene-xml",
        default=str(DEFAULT_SCENE_XML_PATH),
        type=str,
        help="Path to the MuJoCo scene XML containing the stationary camera.",
    )
    parser.add_argument(
        "--fr3-xml",
        default=str(DEFAULT_FR3_XML_PATH),
        type=str,
        help="Path to the FR3 XML containing the ee_contact cylinder.",
    )
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_STATIONARY_CAMERA_NAME,
        type=str,
        help="Camera name to parse from the scene XML.",
    )
    parser.add_argument(
        "--contact-geom-name",
        default=DEFAULT_CONTACT_GEOM_NAME,
        type=str,
        help="Contact geom name to parse from the FR3 XML.",
    )
    parser.add_argument(
        "--num-rings",
        default=DEFAULT_NUM_RINGS,
        type=int,
        help="Number of axial rings used for the EE wireframe.",
    )
    parser.add_argument(
        "--points-per-ring",
        default=DEFAULT_POINTS_PER_RING,
        type=int,
        help="Number of azimuth samples used per ring.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        type=str,
        help="Optional path to save the debug figure instead of only showing it.",
    )
    return parser.parse_args()


def _wireframe_edges(num_rings: int, points_per_ring: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for ring_index in range(num_rings):
        ring_offset = ring_index * points_per_ring
        for point_index in range(points_per_ring):
            current_index = ring_offset + point_index
            next_index = ring_offset + ((point_index + 1) % points_per_ring)
            edges.append((current_index, next_index))

    for ring_index in range(num_rings - 1):
        lower_offset = ring_index * points_per_ring
        upper_offset = (ring_index + 1) * points_per_ring
        for point_index in range(points_per_ring):
            edges.append((lower_offset + point_index, upper_offset + point_index))
    return edges


def _set_equal_axes(ax, points_xyz: np.ndarray, padding: float = 0.03) -> None:
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


def _print_diagnostics_table(
    diagnostics: PointSelectionDiagnostics,
    depth_reprojected_world_points: np.ndarray | None = None,
    observed_depth_m: np.ndarray | None = None,
    point_errors_m: np.ndarray | None = None,
) -> None:
    keep_indices = np.flatnonzero(diagnostics.final_keep_mask)
    kept_lookup = {int(point_index): kept_offset for kept_offset, point_index in enumerate(keep_indices.tolist())}

    print("\nPoint selection diagnostics")
    print(
        " index | reason         | cam_z(m) | depth(m) | err(m)  | facing   | pixel(x,y)  | world(x,y,z)"
    )
    print("-" * 122)
    for point_index, reason in enumerate(diagnostics.rejection_reasons):
        pixel_x = int(diagnostics.rounded_pixels[point_index, 0])
        pixel_y = int(diagnostics.rounded_pixels[point_index, 1])
        world_point = diagnostics.world_points[point_index]
        camera_depth = float(diagnostics.camera_depths[point_index])
        facing = float(diagnostics.facing_scores[point_index])
        observed_depth = float("nan")
        error_m = float("nan")
        if point_index in kept_lookup:
            kept_offset = kept_lookup[point_index]
            if observed_depth_m is not None:
                observed_depth = float(observed_depth_m[kept_offset])
            if point_errors_m is not None:
                error_m = float(point_errors_m[kept_offset])
        print(
            f"{point_index:6d} | "
            f"{reason:14s} | "
            f"{camera_depth:8.4f} | "
            f"{observed_depth:8.4f} | "
            f"{error_m:7.4f} | "
            f"{facing:8.4f} | "
            f"({pixel_x:4d},{pixel_y:4d}) | "
            f"({world_point[0]: .4f},{world_point[1]: .4f},{world_point[2]: .4f})"
        )

    counts = Counter(diagnostics.rejection_reasons)
    print("\nCounts by reason")
    for reason in [
        "kept",
        "behind_camera",
        "back_facing",
        "out_of_frame",
        "not_contact_color",
        "missing_depth",
        "large_reprojection_error",
        "occluded",
    ]:
        print(f"  {reason:14s}: {counts.get(reason, 0)}")
    if point_errors_m is not None and np.isfinite(point_errors_m).any():
        print(
            f"\nDepth reprojection error over kept points: "
            f"mean={float(np.nanmean(point_errors_m)):.4f}m  max={float(np.nanmax(point_errors_m)):.4f}m"
        )


def visualize_diagnostics(
    episode_dir: Path,
    diagnostics: PointSelectionDiagnostics,
    frame_bgr: np.ndarray,
    depth_reprojected_world_points: np.ndarray,
    observed_depth_m: np.ndarray,
    point_errors_m: np.ndarray,
    eef_position_world: np.ndarray,
    camera_position_world: np.ndarray,
    save_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure("Point Finder Diagnostics", figsize=(16, 9))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.15), height_ratios=(1.0, 0.35))
    ax_3d = figure.add_subplot(grid[0, 0], projection="3d")
    ax_image = figure.add_subplot(grid[0, 1])
    ax_text = figure.add_subplot(grid[1, :])

    edges = _wireframe_edges(diagnostics.num_rings, diagnostics.points_per_ring)
    for start_index, end_index in edges:
        segment = diagnostics.world_points[[start_index, end_index]]
        ax_3d.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#c9c9c9", linewidth=1.0, alpha=0.9)

    for reason, color in REASON_COLORS.items():
        reason_indices = np.flatnonzero(np.asarray(diagnostics.rejection_reasons) == reason)
        if len(reason_indices) == 0:
            continue
        points_xyz = diagnostics.world_points[reason_indices]
        ax_3d.scatter(
            points_xyz[:, 0],
            points_xyz[:, 1],
            points_xyz[:, 2],
            color=color,
            s=80 if reason == "kept" else 52,
            marker=REASON_MARKERS[reason],
            label=f"{reason} ({len(reason_indices)})",
            depthshade=False,
        )
        for idx, point_xyz in zip(reason_indices.tolist(), points_xyz):
            ax_3d.text(point_xyz[0], point_xyz[1], point_xyz[2], str(idx), fontsize=8, color=color)

    keep_indices = np.flatnonzero(diagnostics.final_keep_mask)
    valid_reprojection_mask = np.isfinite(depth_reprojected_world_points).all(axis=1)
    if np.any(valid_reprojection_mask):
        reprojected_points_xyz = depth_reprojected_world_points[valid_reprojection_mask]
        ax_3d.scatter(
            reprojected_points_xyz[:, 0],
            reprojected_points_xyz[:, 1],
            reprojected_points_xyz[:, 2],
            color="#e41a1c",
            s=72,
            marker="X",
            label=f"depth reprojection ({len(reprojected_points_xyz)})",
            depthshade=False,
        )
        for kept_offset in np.flatnonzero(valid_reprojection_mask).tolist():
            point_index = int(keep_indices[kept_offset])
            expected_point = diagnostics.world_points[point_index]
            reprojected_point = depth_reprojected_world_points[kept_offset]
            ax_3d.plot(
                [expected_point[0], reprojected_point[0]],
                [expected_point[1], reprojected_point[1]],
                [expected_point[2], reprojected_point[2]],
                color="#e41a1c",
                linewidth=1.2,
                alpha=0.85,
            )
            ax_3d.text(
                reprojected_point[0],
                reprojected_point[1],
                reprojected_point[2],
                f"{point_index}*",
                fontsize=8,
                color="#e41a1c",
            )

    ax_3d.scatter(
        [eef_position_world[0]],
        [eef_position_world[1]],
        [eef_position_world[2]],
        color="black",
        marker="x",
        s=100,
        label="ee bottom center",
    )
    ax_3d.scatter(
        [camera_position_world[0]],
        [camera_position_world[1]],
        [camera_position_world[2]],
        color="black",
        marker="*",
        s=140,
        label="camera",
    )

    all_points_for_bounds = np.vstack(
        [
            diagnostics.world_points,
            eef_position_world.reshape(1, 3),
            camera_position_world.reshape(1, 3),
        ]
    )
    if np.isfinite(depth_reprojected_world_points).any():
        all_points_for_bounds = np.vstack(
            [
                all_points_for_bounds,
                depth_reprojected_world_points[np.isfinite(depth_reprojected_world_points).all(axis=1)],
            ]
        )
    _set_equal_axes(ax_3d, all_points_for_bounds)
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("3D EE wireframe candidates")
    ax_3d.legend(loc="upper left", fontsize=8)

    ax_image.imshow(frame_bgr[..., ::-1])
    ax_image.set_title("First aligned frame with projected candidates")
    ax_image.set_xlabel("Pixel x")
    ax_image.set_ylabel("Pixel y")

    frame_height, frame_width = frame_bgr.shape[:2]
    for point_index, reason in enumerate(diagnostics.rejection_reasons):
        if not diagnostics.finite_projection_mask[point_index]:
            continue
        pixel_x = float(diagnostics.rounded_pixels[point_index, 0])
        pixel_y = float(diagnostics.rounded_pixels[point_index, 1])
        color = REASON_COLORS[reason]
        marker = REASON_MARKERS[reason]
        in_image = 0 <= pixel_x < frame_width and 0 <= pixel_y < frame_height
        ax_image.scatter(
            [pixel_x],
            [pixel_y],
            s=90 if reason == "kept" else 60,
            c=[color],
            marker=marker,
            edgecolors="black" if marker != "x" else None,
            linewidths=0.8 if marker != "x" else 1.2,
        )
        text_dx = 3.0 if in_image else 0.0
        text_dy = -3.0 if in_image else 0.0
        ax_image.text(pixel_x + text_dx, pixel_y + text_dy, str(point_index), color=color, fontsize=8, weight="bold")

    ax_image.set_xlim(0, frame_width - 1)
    ax_image.set_ylim(frame_height - 1, 0)

    counts = Counter(diagnostics.rejection_reasons)
    kept_indices = np.flatnonzero(diagnostics.final_keep_mask).tolist()
    summary_lines = [
        f"Episode: {episode_dir.name}",
        f"Frame shape: {frame_width} x {frame_height}",
        f"Wireframe layout: {diagnostics.num_rings} rings x {diagnostics.points_per_ring} points = {len(diagnostics.local_points)} candidates",
        (
            "Contact color gate: "
            f"BGR={DEFAULT_CONTACT_VISUAL_BGR.tolist()}, "
            f"radius={DEFAULT_CONTACT_COLOR_PATCH_RADIUS_PX}px, "
            f"tolerance={DEFAULT_CONTACT_COLOR_CHANNEL_TOLERANCE}"
        ),
        f"Max reprojection error: {DEFAULT_MAX_REPROJECTION_ERROR_M:.3f} m",
        f"Kept indices: {kept_indices if kept_indices else 'none'}",
        "",
        "Counts by reason:",
    ]
    for reason in [
        "kept",
        "behind_camera",
        "back_facing",
        "out_of_frame",
        "not_contact_color",
        "missing_depth",
        "large_reprojection_error",
        "occluded",
    ]:
        summary_lines.append(f"  {reason}: {counts.get(reason, 0)}")
    valid_reprojection_mask = np.isfinite(point_errors_m)
    if np.any(valid_reprojection_mask):
        summary_lines.extend(
            [
                "",
                "Depth reprojection over kept points:",
                f"  mean error: {float(np.nanmean(point_errors_m)):.4f} m",
                f"  max error:  {float(np.nanmax(point_errors_m)):.4f} m",
            ]
        )
    summary_lines.extend(
        [
            "",
            "Red X points in 3D are depth-reprojected positions from the kept pixels.",
            "Red line segments connect expected EE points to their depth reprojections.",
            "See terminal output for the full per-point table.",
        ]
    )

    ax_text.axis("off")
    ax_text.text(
        0.01,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    figure.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"\nSaved debug figure to {save_path}")
    plt.show()


def main() -> int:
    args = _parse_args()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    scene_xml = Path(args.scene_xml).expanduser().resolve()
    fr3_xml = Path(args.fr3_xml).expanduser().resolve()
    save_path = None if args.save_path is None else Path(args.save_path).expanduser().resolve()

    episode = load_episode_data(episode_dir)
    aligned_episode = align_episode_to_lowdim(episode)
    if aligned_episode is None:
        raise RuntimeError(f"Could not align episode to lowdim data for {episode_dir}.")

    camera_calibration = load_camera_calibration(
        camera_name=args.camera_name,
        scene_xml_path=scene_xml,
    )
    contact_spec = load_contact_cylinder_spec(
        model_xml_path=fr3_xml,
        geom_name=args.contact_geom_name,
    )
    depth_frame_paths = discover_depth_frame_paths(episode_dir)
    if depth_frame_paths is None:
        raise RuntimeError(f"No depth frames were found under {episode_dir}.")
    diagnostics = diagnose_point_selection(
        aligned_episode=aligned_episode,
        camera_calibration=camera_calibration,
        contact_spec=contact_spec,
        num_rings=int(args.num_rings),
        points_per_ring=int(args.points_per_ring),
    )
    first_source_frame_index = int(aligned_episode.source_frame_indices[0])
    if first_source_frame_index >= len(depth_frame_paths):
        raise IndexError(
            f"First aligned frame index {first_source_frame_index} exceeds the available depth frames ({len(depth_frame_paths)})."
        )
    depth_frame_mm = read_depth_frame(depth_frame_paths[first_source_frame_index])
    depth_reprojected_world_points, observed_depth_m, point_errors_m = reconstruct_depth_points(
        diagnostics=diagnostics,
        depth_frame_mm=depth_frame_mm,
        camera_calibration=camera_calibration,
    )

    _print_diagnostics_table(
        diagnostics,
        depth_reprojected_world_points=depth_reprojected_world_points,
        observed_depth_m=observed_depth_m,
        point_errors_m=point_errors_m,
    )
    visualize_diagnostics(
        episode_dir=episode_dir,
        diagnostics=diagnostics,
        frame_bgr=np.asarray(aligned_episode.frames[0]),
        depth_reprojected_world_points=depth_reprojected_world_points,
        observed_depth_m=observed_depth_m,
        point_errors_m=point_errors_m,
        eef_position_world=np.asarray(aligned_episode.positions[0], dtype=np.float32).reshape(3),
        camera_position_world=camera_calibration.camera_position_world,
        save_path=save_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
