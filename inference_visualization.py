from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from training.inference import load_prediction_artifact


SCENE_POINT_COLOR = "saddlebrown"
PREDICTED_DEPTH_COLOR = "#127475"
TARGET_DEPTH_COLOR = "#ff7f11"
PREDICTED_EE_COLOR = "#004e64"
TARGET_EE_COLOR = "#c44536"
VECTOR_COMPONENT_COLORS = ("#1f77b4", "#2ca02c", "#d62728")


def load_visualization_artifact(path: str | Path) -> dict[str, Any]:
    return load_prediction_artifact(path)


def prepare_episode_records(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = artifact.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Prediction artifact does not contain any episode records.")
    return episodes


def _resolve_start_episode(episodes: list[dict[str, Any]], requested_episode: int | None) -> int:
    if requested_episode is None:
        return 0
    if requested_episode < 0 or requested_episode >= len(episodes):
        raise IndexError(f"Episode index {requested_episode} is out of bounds for {len(episodes)} episodes.")
    return int(requested_episode)


def _plot_xyz_series(axis, x_values: np.ndarray, target_values: np.ndarray, predicted_values: np.ndarray, title: str) -> None:
    axis.clear()
    labels = ("x", "y", "z")
    for component_index, color in enumerate(VECTOR_COMPONENT_COLORS):
        axis.plot(
            x_values,
            target_values[:, component_index],
            linestyle="--",
            linewidth=1.8,
            color=color,
            alpha=0.95,
            label=f"target {labels[component_index]}",
        )
        axis.plot(
            x_values,
            predicted_values[:, component_index],
            linestyle="-",
            linewidth=1.6,
            color=color,
            alpha=0.75,
            label=f"pred {labels[component_index]}",
        )
    axis.set_title(title)
    axis.set_xlabel("episode step")
    axis.grid(True, alpha=0.25)


def _plot_force_dimension(axis, x_values: np.ndarray, target_values: np.ndarray, predicted_values: np.ndarray) -> None:
    axis.clear()
    axis.step(x_values, target_values, where="mid", linestyle="--", linewidth=2.0, color=TARGET_EE_COLOR, label="target")
    axis.step(
        x_values,
        predicted_values,
        where="mid",
        linestyle="-",
        linewidth=1.8,
        color=PREDICTED_EE_COLOR,
        label="predicted",
    )
    axis.set_title("force_dimension")
    axis.set_xlabel("episode step")
    axis.set_yticks(sorted({int(value) for value in np.concatenate([target_values, predicted_values])}))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)


def _plot_ee_traces(axis, artifact: dict[str, Any], episode: dict[str, Any]) -> None:
    axis.clear()
    scene_points = np.asarray(episode["scene_points"], dtype=np.float32)
    predicted_points = np.asarray(episode["predicted_depth_points"], dtype=np.float32)
    target_points = np.asarray(episode["target_depth_points"], dtype=np.float32)
    depth_mask = np.asarray(episode["depth_mask"]).astype(bool, copy=False)
    dataset_indices = np.asarray(episode["dataset_indices"], dtype=np.int64)
    episode_index = int(episode["episode_index"])

    if len(scene_points):
        axis.scatter(
            scene_points[:, 0],
            scene_points[:, 1],
            scene_points[:, 2],
            s=12,
            alpha=0.55,
            color=SCENE_POINT_COLOR,
            label="scene points",
        )

    ee_valid_mask = depth_mask[:, 0]
    if np.any(ee_valid_mask):
        predicted_trace = predicted_points[ee_valid_mask, 0, :]
        actual_trace = target_points[ee_valid_mask, 0, :]

        axis.plot(
            actual_trace[:, 0],
            actual_trace[:, 1],
            actual_trace[:, 2],
            linestyle="--",
            linewidth=2.8,
            alpha=0.95,
            color=TARGET_EE_COLOR,
            label="actual EE trace",
        )
        axis.plot(
            predicted_trace[:, 0],
            predicted_trace[:, 1],
            predicted_trace[:, 2],
            linestyle="-",
            linewidth=2.8,
            alpha=0.95,
            color=PREDICTED_EE_COLOR,
            label="predicted EE trace",
        )

        axis.scatter(
            actual_trace[0:1, 0],
            actual_trace[0:1, 1],
            actual_trace[0:1, 2],
            s=34,
            color=TARGET_EE_COLOR,
            alpha=0.9,
            label="actual EE start",
        )
        axis.scatter(
            actual_trace[-1:, 0],
            actual_trace[-1:, 1],
            actual_trace[-1:, 2],
            s=40,
            color=TARGET_DEPTH_COLOR,
            alpha=0.9,
            label="actual EE end",
        )
        axis.scatter(
            predicted_trace[0:1, 0],
            predicted_trace[0:1, 1],
            predicted_trace[0:1, 2],
            s=36,
            color=PREDICTED_EE_COLOR,
            alpha=0.9,
            label="pred EE start",
        )
        axis.scatter(
            predicted_trace[-1:, 0],
            predicted_trace[-1:, 1],
            predicted_trace[-1:, 2],
            s=42,
            color=PREDICTED_DEPTH_COLOR,
            alpha=0.9,
            label="pred EE end",
        )

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_box_aspect((1, 1, 1))
    axis.set_title(
        "Predicted Vs Actual EE Trace\n"
        f"split={artifact['metadata']['split']} | episode={episode_index} | rows={len(dataset_indices)} | "
        f"indices={dataset_indices[0]}-{dataset_indices[-1]}"
    )
    axis.legend(loc="upper right", fontsize=8)


def _draw_episode(
    figure,
    axes: dict[str, Any],
    artifact: dict[str, Any],
    episodes: list[dict[str, Any]],
    state: dict[str, int],
) -> None:
    episode = episodes[state["episode_cursor"]]
    x_values = np.arange(len(episode["dataset_indices"]), dtype=np.int64)

    _plot_ee_traces(axes["depth"], artifact, episode)
    _plot_force_dimension(
        axes["force_dimension"],
        x_values,
        np.asarray(episode["target_force_dimension"], dtype=np.int64),
        np.asarray(episode["predicted_force_dimension"], dtype=np.int64),
    )
    _plot_xyz_series(
        axes["motion_axis"],
        x_values,
        np.asarray(episode["target_motion_or_force_axis"], dtype=np.float32),
        np.asarray(episode["predicted_motion_or_force_axis"], dtype=np.float32),
        "motion_or_force_axis",
    )
    _plot_xyz_series(
        axes["sensed_force"],
        x_values,
        np.asarray(episode["target_sensed_force"], dtype=np.float32),
        np.asarray(episode["predicted_sensed_force"], dtype=np.float32),
        "sensed_force",
    )
    _plot_xyz_series(
        axes["sensed_moment"],
        x_values,
        np.asarray(episode["target_sensed_moment"], dtype=np.float32),
        np.asarray(episode["predicted_sensed_moment"], dtype=np.float32),
        "sensed_moment",
    )
    axes["motion_axis"].legend(loc="upper right", fontsize=7)
    axes["sensed_force"].legend(loc="upper right", fontsize=7)
    axes["sensed_moment"].legend(loc="upper right", fontsize=7)
    figure.suptitle(
        "CRWM One-Step Inference Visualization\n"
        "Left/right: next/previous episode | q/esc: quit",
        fontsize=13,
    )
    figure.canvas.draw_idle()


def visualize_prediction_artifact(path: str | Path, *, episode_index: int | None = None) -> None:
    artifact = load_visualization_artifact(path)
    episodes = prepare_episode_records(artifact)
    start_episode = _resolve_start_episode(episodes, episode_index)

    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid_spec = figure.add_gridspec(4, 2, width_ratios=[2.25, 1.75])
    axes = {
        "depth": figure.add_subplot(grid_spec[:, 0], projection="3d"),
        "force_dimension": figure.add_subplot(grid_spec[0, 1]),
        "motion_axis": figure.add_subplot(grid_spec[1, 1]),
        "sensed_force": figure.add_subplot(grid_spec[2, 1]),
        "sensed_moment": figure.add_subplot(grid_spec[3, 1]),
    }
    state = {"episode_cursor": int(start_episode)}

    def redraw() -> None:
        _draw_episode(figure, axes, artifact, episodes, state)

    def on_key(event) -> None:
        if event.key in ("right", " "):
            state["episode_cursor"] = (state["episode_cursor"] + 1) % len(episodes)
            redraw()
        elif event.key in ("left", "backspace"):
            state["episode_cursor"] = (state["episode_cursor"] - 1) % len(episodes)
            redraw()
        elif event.key in ("q", "escape"):
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CRWM one-step inference exports.")
    parser.add_argument("--artifact", required=True, type=str, help="Path to the exported `.npy` prediction artifact.")
    parser.add_argument(
        "--episode",
        default=None,
        type=int,
        help="Optional episode cursor to open first.",
    )
    args = parser.parse_args()
    visualize_prediction_artifact(args.artifact, episode_index=args.episode)


if __name__ == "__main__":
    main()
