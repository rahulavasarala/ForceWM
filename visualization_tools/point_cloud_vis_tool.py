from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_POINT_CLOUD_PATH = Path(
    "/Users/rahulavasarala/Desktop/ForceWM/data_storage/depth_collection_v1_extracted/"
    "point_clouds/episode_0001/chunk_0001.npy"
)
DEFAULT_WINDOW_TITLE = "ForceWM Point Cloud Viewer"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a single point-cloud .npy file with shape (T, N, 3)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_POINT_CLOUD_PATH),
        help="Path to one point-cloud .npy file.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Frame index to open first.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="Scatter marker size for rendered points.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=24.0,
        help="Initial 3D elevation angle in degrees.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=-58.0,
        help="Initial 3D azimuth angle in degrees.",
    )
    return parser.parse_args()


def load_point_cloud_sequence(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a single .npy file, got directory: {path}")
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got: {path}")

    point_clouds = np.load(path)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 3:
        raise ValueError(
            f"Point-cloud file `{path}` must have shape (T, N, 3), got {point_clouds.shape}."
        )
    if len(point_clouds) == 0:
        raise ValueError(f"Point-cloud file `{path}` does not contain any frames.")
    return np.asarray(point_clouds, dtype=np.float32)


def compute_global_bounds(point_clouds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_points = point_clouds.reshape(-1, 3)
    valid_points = flat_points[np.isfinite(flat_points).all(axis=1)]

    if len(valid_points) == 0:
        mins = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        maxs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return mins, maxs

    mins = valid_points.min(axis=0)
    maxs = valid_points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    if not np.isfinite(extent) or extent <= 0.0:
        extent = 1.0
    half_extent = extent / 2.0
    return (center - half_extent).astype(np.float32), (center + half_extent).astype(np.float32)


def set_equal_axes(ax, mins: np.ndarray, maxs: np.ndarray) -> None:
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_box_aspect((1.0, 1.0, 1.0))


def visualize_point_cloud_file(
    path: Path,
    start_index: int = 0,
    point_size: float = 18.0,
    elevation: float = 24.0,
    azimuth: float = -58.0,
) -> None:
    import matplotlib.pyplot as plt

    point_clouds = load_point_cloud_sequence(path)
    global_mins, global_maxs = compute_global_bounds(point_clouds)
    start_index = int(np.clip(start_index, 0, len(point_clouds) - 1))
    state = {"index": start_index}

    figure = plt.figure(DEFAULT_WINDOW_TITLE, figsize=(9, 8))
    ax = figure.add_subplot(111, projection="3d")
    ax.view_init(elev=elevation, azim=azimuth)

    def _render_current_frame() -> None:
        point_cloud = point_clouds[state["index"]]
        valid_mask = np.isfinite(point_cloud).all(axis=1)
        valid_points = point_cloud[valid_mask]
        valid_indices = np.flatnonzero(valid_mask)

        ax.clear()
        ax.view_init(elev=ax.elev, azim=ax.azim)
        set_equal_axes(ax, global_mins, global_maxs)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        if len(valid_points):
            ax.scatter(
                valid_points[:, 0],
                valid_points[:, 1],
                valid_points[:, 2],
                c=valid_indices,
                cmap="turbo",
                s=point_size,
                depthshade=True,
                vmin=0,
                vmax=max(1, point_cloud.shape[0] - 1),
            )

        valid_count = int(valid_mask.sum())
        ax.set_title(
            "\n".join(
                [
                    f"{path.name}",
                    f"Frame {state['index'] + 1}/{len(point_clouds)}  |  Valid points: {valid_count}/{len(point_cloud)}",
                ]
            ),
            fontsize=11,
        )

        figure.texts.clear()
        figure.text(
            0.02,
            0.02,
            "Controls: space/right next, backspace/left previous, home first, end last, q/esc quit",
            fontsize=10,
        )
        figure.canvas.draw_idle()

    def _on_key_press(event) -> None:
        key = event.key
        if key in {"q", "escape"}:
            plt.close(figure)
            return
        if key in {"right", "space"} and state["index"] < len(point_clouds) - 1:
            state["index"] += 1
        elif key in {"left", "backspace"} and state["index"] > 0:
            state["index"] -= 1
        elif key == "home":
            state["index"] = 0
        elif key == "end":
            state["index"] = len(point_clouds) - 1
        else:
            return
        _render_current_frame()

    figure.canvas.mpl_connect("key_press_event", _on_key_press)
    _render_current_frame()
    plt.show()


def main() -> None:
    args = _parse_args()
    visualize_point_cloud_file(
        path=Path(args.path).expanduser().resolve(),
        start_index=args.start_index,
        point_size=float(args.point_size),
        elevation=float(args.elevation),
        azimuth=float(args.azimuth),
    )


if __name__ == "__main__":
    main()
