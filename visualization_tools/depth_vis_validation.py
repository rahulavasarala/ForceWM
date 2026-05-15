from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DEPTH_PATH = Path(
    "/Users/rahulavasarala/Desktop/ForceWM/data_storage/depth_collection_v1/"
    "episode_000001/visual/depth/depth_frames"
)
WINDOW_NAME = "ForceWM Depth Viewer"
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW_COLOR = (0, 0, 0)


@dataclass(frozen=True)
class DepthFrame:
    path: Path
    sequence_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved uint16 depth PNG frames and step through them with the keyboard."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DEPTH_PATH),
        help=(
            "Path to a depth_frames directory, an episode directory, or a dataset/repository "
            "root that contains depth frames."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Frame index to open first after discovery.",
    )
    parser.add_argument(
        "--min-depth-mm",
        type=int,
        default=None,
        help="Optional lower bound for visualization scaling in millimeters.",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=int,
        default=None,
        help="Optional upper bound for visualization scaling in millimeters.",
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=1.0,
        help="Lower percentile for automatic display scaling when explicit min/max are not given.",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=99.0,
        help="Upper percentile for automatic display scaling when explicit min/max are not given.",
    )
    return parser.parse_args()


def _sorted_pngs(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def _discover_depth_frames(input_path: Path) -> list[DepthFrame]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            raise ValueError(f"Expected a depth PNG file, got: {input_path}")
        return [DepthFrame(path=input_path, sequence_name=input_path.parent.name)]

    candidate_directories: list[Path] = []
    direct_pngs = _sorted_pngs(input_path)
    if direct_pngs:
        candidate_directories.append(input_path)
    else:
        candidate_directories.extend(
            sorted(
                path
                for path in input_path.rglob("depth_frames")
                if path.is_dir() and _sorted_pngs(path)
            )
        )

    if not candidate_directories:
        raise FileNotFoundError(
            f"No depth PNG frames were found under {input_path}. "
            "Point the tool at a `depth_frames` directory or a repo/dataset root that contains one."
        )

    discovered_frames: list[DepthFrame] = []
    for directory in candidate_directories:
        sequence_name = str(directory.relative_to(input_path)) if directory != input_path else directory.name
        for frame_path in _sorted_pngs(directory):
            discovered_frames.append(DepthFrame(path=frame_path, sequence_name=sequence_name))

    return discovered_frames


def _read_depth_frame(frame_path: Path) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Failed to read depth frame: {frame_path}")
    if frame.ndim != 2 or frame.dtype != np.uint16:
        raise ValueError(
            f"Depth frame `{frame_path}` must decode as HxW uint16, "
            f"got shape={frame.shape} dtype={frame.dtype}."
        )
    return frame


def _resolve_visualization_range(
    depth_frame_mm: np.ndarray,
    min_depth_mm: int | None,
    max_depth_mm: int | None,
    percentile_min: float,
    percentile_max: float,
) -> tuple[float, float]:
    valid_depth = depth_frame_mm[depth_frame_mm > 0]
    if valid_depth.size == 0:
        return 0.0, 1.0

    if min_depth_mm is not None and max_depth_mm is not None:
        lower = float(min_depth_mm)
        upper = float(max_depth_mm)
    else:
        lower = float(np.percentile(valid_depth, percentile_min))
        upper = float(np.percentile(valid_depth, percentile_max))
        if min_depth_mm is not None:
            lower = float(min_depth_mm)
        if max_depth_mm is not None:
            upper = float(max_depth_mm)

    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _colorize_depth_frame(
    depth_frame_mm: np.ndarray,
    min_depth_mm: int | None,
    max_depth_mm: int | None,
    percentile_min: float,
    percentile_max: float,
) -> tuple[np.ndarray, float, float]:
    lower, upper = _resolve_visualization_range(
        depth_frame_mm,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
    )

    clipped = np.clip(depth_frame_mm.astype(np.float32), lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    normalized[depth_frame_mm == 0] = 0.0
    grayscale = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    colorized = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    colorized[depth_frame_mm == 0] = (0, 0, 0)
    return colorized, lower, upper


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.55,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_SHADOW_COLOR,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _make_display_image(
    frame: DepthFrame,
    depth_frame_mm: np.ndarray,
    frame_index: int,
    frame_count: int,
    min_depth_mm: int | None,
    max_depth_mm: int | None,
    percentile_min: float,
    percentile_max: float,
) -> np.ndarray:
    display_image, lower, upper = _colorize_depth_frame(
        depth_frame_mm,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
    )
    valid_depth = depth_frame_mm[depth_frame_mm > 0]
    valid_min = int(valid_depth.min()) if valid_depth.size else 0
    valid_max = int(valid_depth.max()) if valid_depth.size else 0

    _draw_text(display_image, f"Frame {frame_index + 1}/{frame_count}", (12, 24))
    _draw_text(display_image, f"Sequence: {frame.sequence_name}", (12, 48))
    _draw_text(display_image, f"File: {frame.path.name}", (12, 72))
    _draw_text(
        display_image,
        f"Valid depth range: {valid_min}mm to {valid_max}mm",
        (12, 96),
    )
    _draw_text(
        display_image,
        f"Display scale: {lower:.1f}mm to {upper:.1f}mm",
        (12, 120),
    )
    _draw_text(
        display_image,
        "Controls: space/-> next, backspace/<- previous, q or esc quit",
        (12, 144),
        font_scale=0.5,
    )

    return display_image


def visualize_depth_frames(
    path: Path,
    start_index: int = 0,
    min_depth_mm: int | None = None,
    max_depth_mm: int | None = None,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> None:
    frames = _discover_depth_frames(path)
    if not frames:
        raise RuntimeError(f"No depth frames discovered under {path}.")

    current_index = int(np.clip(start_index, 0, len(frames) - 1))
    print(f"Loaded {len(frames)} depth frames from `{Path(path).expanduser().resolve()}`")
    print("Controls: space/right-arrow = next, backspace/left-arrow = previous, q/esc = quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 900)

    while True:
        frame = frames[current_index]
        depth_frame_mm = _read_depth_frame(frame.path)
        display_image = _make_display_image(
            frame,
            depth_frame_mm,
            frame_index=current_index,
            frame_count=len(frames),
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            percentile_min=percentile_min,
            percentile_max=percentile_max,
        )
        cv2.imshow(WINDOW_NAME, display_image)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (32, 2555904):
            current_index = min(current_index + 1, len(frames) - 1)
            continue
        if key in (8, 2424832):
            current_index = max(current_index - 1, 0)
            continue

    cv2.destroyAllWindows()


def main() -> int:
    args = _parse_args()
    visualize_depth_frames(
        path=Path(args.path),
        start_index=args.start_index,
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
