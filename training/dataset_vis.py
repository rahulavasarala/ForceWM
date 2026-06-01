from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dataset import MultiModalDataset, POINT_CLOUD_MASK_SUFFIX

FIXED_AXIS_LIMIT = 0.1
AXIS_PANEL_LIMIT = 1.15
FORCE_VECTOR_SCALE_M_PER_N = 0.01
MAX_FORCE_VECTOR_LENGTH = 0.12
MAX_ACTION_VECTOR_LENGTH = 0.16

FORCE_DIMENSION_KEY = "force_dimension"
MOTION_OR_FORCE_AXIS_KEYS = ("motion_or_force_axis", "force_or_motion_axis")
SENSED_FORCE_KEY = "sensed_force"
SENSED_MOMENT_KEY = "sensed_moment"
ACTION_DELTA_POS_KEY = "action_delta_pos"
SCENE_POINT_COLOR = "saddlebrown"


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _resolve_dataset_index(dataset: MultiModalDataset, requested_index: int | None) -> int:
    if requested_index is not None:
        dataset_index = int(requested_index)
        if dataset_index < 0 or dataset_index >= len(dataset):
            raise IndexError(f"Index {dataset_index} is out of bounds for dataset with {len(dataset)} rows")
        return dataset_index

    rng = np.random.default_rng()
    return int(rng.integers(0, len(dataset)))


def _select_point_cloud_key(dataset: MultiModalDataset) -> str:
    if not dataset.point_cloud_keys:
        raise ValueError("The dataset does not expose any point-cloud observation keys.")
    return dataset.point_cloud_keys[0]


def _select_scene_point_cloud_key(dataset: MultiModalDataset) -> str | None:
    if not dataset.static_point_cloud_keys:
        return None
    return dataset.static_point_cloud_keys[0]


def _split_depth_and_ee_points(valid_points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    if len(valid_points) == 0:
        return None, np.empty((0, 3), dtype=np.float32)

    ee_point = np.asarray(valid_points[0], dtype=np.float32)
    depth_points = np.asarray(valid_points[1:], dtype=np.float32)
    return ee_point, depth_points


def _format_vector(vector: np.ndarray | None) -> str:
    if vector is None:
        return "n/a"
    return np.array2string(
        np.asarray(vector, dtype=np.float32).reshape(3),
        precision=4,
        suppress_small=True,
        floatmode="fixed",
    )


def _normalize_axis(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / norm


def _orthonormal_complement(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _normalize_axis(axis)
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    basis_1 = reference - float(np.dot(reference, axis)) * axis
    basis_1 = _normalize_axis(basis_1)
    basis_2 = _normalize_axis(np.cross(axis, basis_1))
    return basis_1, basis_2


def _axis_visualization(force_dimension: int, axis: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    world_axes = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]

    if force_dimension <= 0:
        return world_axes, []
    if force_dimension >= 3:
        return [], world_axes

    axis = _normalize_axis(axis)
    orthogonal_1, orthogonal_2 = _orthonormal_complement(axis)
    if force_dimension == 1:
        return [orthogonal_1, orthogonal_2], [axis]
    return [axis], [orthogonal_1, orthogonal_2]


def _extract_modal_series(
    modal_dict: dict,
    key_candidates: str | tuple[str, ...],
) -> np.ndarray | None:
    candidate_names = (key_candidates,) if isinstance(key_candidates, str) else tuple(key_candidates)

    for key_name in candidate_names:
        if key_name not in modal_dict:
            continue

        value = np.asarray(_to_numpy(modal_dict[key_name]))
        if value.ndim == 0:
            return value.reshape(1)
        if value.shape[0] == 0:
            return None
        return value

    return None


def _extract_latest_modal_value(modal_dict: dict, key_candidates: str | tuple[str, ...]) -> np.ndarray | None:
    value = _extract_modal_series(modal_dict, key_candidates)
    if value is None:
        return None
    return np.asarray(value[-1])


def _extract_latest_scalar(modal_dict: dict, key_candidates: str | tuple[str, ...]) -> int | None:
    value = _extract_latest_modal_value(modal_dict, key_candidates)
    if value is None:
        return None

    flat_value = np.asarray(value).reshape(-1)
    if flat_value.size != 1:
        raise ValueError(f"Expected scalar value for {key_candidates}, got shape {np.asarray(value).shape}.")
    return int(flat_value[0])


def _extract_latest_vector(
    modal_dict: dict,
    key_candidates: str | tuple[str, ...],
    *,
    length: int = 3,
) -> np.ndarray | None:
    value = _extract_latest_modal_value(modal_dict, key_candidates)
    if value is None:
        return None

    flat_value = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat_value.size != length:
        raise ValueError(
            f"Expected {length} values for {key_candidates}, got shape {np.asarray(value).shape}."
        )
    return flat_value


def _extract_vector_sequence(
    modal_dict: dict,
    key_candidates: str | tuple[str, ...],
    *,
    length: int = 3,
) -> np.ndarray | None:
    values = _extract_modal_series(modal_dict, key_candidates)
    if values is None:
        return None

    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != length:
        raise ValueError(
            f"Expected shape (T, {length}) for {key_candidates}, got {values.shape}."
        )
    return values


def _extract_scalar_sequence(modal_dict: dict, key_candidates: str | tuple[str, ...]) -> np.ndarray | None:
    values = _extract_modal_series(modal_dict, key_candidates)
    if values is None:
        return None

    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError(f"Expected shape (T,) for {key_candidates}, got {values.shape}.")
    return values.astype(np.int64, copy=False)


def _scaled_force_vector(force_vector: np.ndarray | None) -> np.ndarray | None:
    if force_vector is None:
        return None

    scaled_force = np.asarray(force_vector, dtype=np.float32) * np.float32(FORCE_VECTOR_SCALE_M_PER_N)
    norm = float(np.linalg.norm(scaled_force))
    if norm <= 1e-9:
        return scaled_force
    if norm > MAX_FORCE_VECTOR_LENGTH:
        scaled_force = scaled_force / norm * np.float32(MAX_FORCE_VECTOR_LENGTH)
    return scaled_force.astype(np.float32, copy=False)


def _clamp_vector_length(vector: np.ndarray | None, max_length: float) -> np.ndarray | None:
    if vector is None:
        return None

    clipped = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(clipped))
    if norm <= 1e-9 or norm <= max_length:
        return clipped
    return (clipped / norm * np.float32(max_length)).astype(np.float32, copy=False)


def _load_point_cloud_sample(
    dataset: MultiModalDataset,
    dataset_index: int,
) -> tuple[dict, str, np.ndarray, np.ndarray, int, int, int, np.ndarray]:
    point_cloud_key = _select_point_cloud_key(dataset)
    sample = dataset[dataset_index]
    point_clouds = _to_numpy(sample["obs_dict"][point_cloud_key]).astype(np.float32, copy=False)
    point_cloud_mask = _to_numpy(
        sample["obs_dict"][f"{point_cloud_key}{POINT_CLOUD_MASK_SUFFIX}"]
    ).astype(bool, copy=False)

    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 3:
        raise ValueError(
            f"Expected `{point_cloud_key}` to have shape (T, P, 3), got {point_clouds.shape}."
        )
    if point_cloud_mask.shape != point_clouds.shape[:2]:
        raise ValueError(
            f"Expected `{point_cloud_key}{POINT_CLOUD_MASK_SUFFIX}` to have shape {point_clouds.shape[:2]}, "
            f"got {point_cloud_mask.shape}."
        )

    episode_index, episode_start, episode_end = dataset.get_episode_bounds(dataset_index)
    point_counts = point_cloud_mask.sum(axis=1).astype(np.int64)

    return (
        sample,
        point_cloud_key,
        point_clouds,
        point_cloud_mask,
        episode_index,
        episode_start,
        episode_end,
        point_counts,
    )


def _load_prediction_point_cloud_sample(
    sample: dict,
    point_cloud_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_dict = sample["prediction"]
    if point_cloud_key not in prediction_dict:
        raise KeyError(f"Prediction dictionary is missing point-cloud key `{point_cloud_key}`.")

    point_clouds = _to_numpy(prediction_dict[point_cloud_key]).astype(np.float32, copy=False)
    point_cloud_mask = _to_numpy(
        prediction_dict[f"{point_cloud_key}{POINT_CLOUD_MASK_SUFFIX}"]
    ).astype(bool, copy=False)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 3:
        raise ValueError(
            f"Expected prediction `{point_cloud_key}` to have shape (T, P, 3), got {point_clouds.shape}."
        )
    if point_cloud_mask.shape != point_clouds.shape[:2]:
        raise ValueError(
            f"Expected prediction `{point_cloud_key}{POINT_CLOUD_MASK_SUFFIX}` to have shape {point_clouds.shape[:2]}, "
            f"got {point_cloud_mask.shape}."
        )

    point_counts = point_cloud_mask.sum(axis=1).astype(np.int64)
    return point_clouds, point_cloud_mask, point_counts


def _draw_vector(
    axis,
    *,
    origin: np.ndarray,
    vector: np.ndarray | None,
    color: str,
    label: str,
    linewidth: float = 2.4,
    alpha: float = 1.0,
) -> None:
    if vector is None:
        return

    vector = np.asarray(vector, dtype=np.float32).reshape(3)
    if float(np.linalg.norm(vector)) <= 1e-9:
        return

    axis.quiver(
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(vector[0]),
        float(vector[1]),
        float(vector[2]),
        color=color,
        linewidth=linewidth,
        arrow_length_ratio=0.18,
        label=label,
        alpha=alpha,
    )


def _draw_force_motion_axis_panel(
    axis,
    *,
    force_dimension: int | None,
    motion_or_force_axis: np.ndarray | None,
    sensed_force: np.ndarray | None,
    action_delta_pos: np.ndarray | None,
    predicted_force_dimensions: np.ndarray | None,
    predicted_axes: np.ndarray | None,
    predicted_sensed_forces: np.ndarray | None,
    predicted_sensed_moments: np.ndarray | None,
) -> None:
    axis.clear()

    world_axes = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    for axis_index, basis_axis in enumerate(world_axes):
        axis.plot(
            [-basis_axis[0], basis_axis[0]],
            [-basis_axis[1], basis_axis[1]],
            [-basis_axis[2], basis_axis[2]],
            linestyle="--",
            linewidth=1.2,
            color="0.7",
        )
        axis.text(
            1.08 * basis_axis[0],
            1.08 * basis_axis[1],
            1.08 * basis_axis[2],
            f"e{axis_index + 1}",
            color="0.45",
        )

    if force_dimension is not None and motion_or_force_axis is not None:
        motion_axes, force_axes = _axis_visualization(force_dimension, motion_or_force_axis)
    else:
        motion_axes, force_axes = [], []

    for basis_axis in motion_axes:
        axis.quiver(
            0.0,
            0.0,
            0.0,
            basis_axis[0],
            basis_axis[1],
            basis_axis[2],
            length=0.95,
            normalize=True,
            color="royalblue",
            linewidth=2.8,
        )

    for basis_axis in force_axes:
        axis.quiver(
            0.0,
            0.0,
            0.0,
            basis_axis[0],
            basis_axis[1],
            basis_axis[2],
            length=0.95,
            normalize=True,
            color="crimson",
            linewidth=2.8,
        )

    if predicted_force_dimensions is not None and predicted_axes is not None:
        for timestep, (pred_force_dimension, pred_axis) in enumerate(
            zip(predicted_force_dimensions, predicted_axes, strict=True)
        ):
            pred_motion_axes, pred_force_axes = _axis_visualization(int(pred_force_dimension), pred_axis)
            alpha = min(0.35 + 0.18 * timestep, 0.9)

            for basis_axis in pred_motion_axes:
                axis.quiver(
                    0.0,
                    0.0,
                    0.0,
                    basis_axis[0],
                    basis_axis[1],
                    basis_axis[2],
                    length=0.72,
                    normalize=True,
                    color="deepskyblue",
                    linewidth=1.6,
                    alpha=alpha,
                )
            for basis_axis in pred_force_axes:
                axis.quiver(
                    0.0,
                    0.0,
                    0.0,
                    basis_axis[0],
                    basis_axis[1],
                    basis_axis[2],
                    length=0.72,
                    normalize=True,
                    color="lightcoral",
                    linewidth=1.6,
                    alpha=alpha,
                )

    axis.set_xlim(-AXIS_PANEL_LIMIT, AXIS_PANEL_LIMIT)
    axis.set_ylim(-AXIS_PANEL_LIMIT, AXIS_PANEL_LIMIT)
    axis.set_zlim(-AXIS_PANEL_LIMIT, AXIS_PANEL_LIMIT)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.set_title("Force / Motion Axes", fontsize=11)
    axis.legend(
        handles=[
            Line2D([0], [0], color="0.7", linestyle="--", linewidth=1.4, label="World basis"),
            Line2D([0], [0], color="royalblue", linewidth=2.8, label="Motion axis"),
            Line2D([0], [0], color="crimson", linewidth=2.8, label="Force axis"),
            Line2D([0], [0], color="deepskyblue", linewidth=1.8, label="Pred motion axis"),
            Line2D([0], [0], color="lightcoral", linewidth=1.8, label="Pred force axis"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=True,
    )

    info_lines = [
        f"fdim: {force_dimension if force_dimension is not None else 'n/a'}",
        f"axis: {_format_vector(motion_or_force_axis)}",
        f"sensed_force: {_format_vector(sensed_force)}",
        f"action_dpos: {_format_vector(action_delta_pos)}",
        f"force scale: {FORCE_VECTOR_SCALE_M_PER_N:.3f} m/N",
    ]
    if predicted_force_dimensions is not None:
        for timestep, pred_force_dimension in enumerate(predicted_force_dimensions):
            pred_axis = None if predicted_axes is None else predicted_axes[timestep]
            pred_force = None if predicted_sensed_forces is None else predicted_sensed_forces[timestep]
            pred_moment = None if predicted_sensed_moments is None else predicted_sensed_moments[timestep]
            info_lines.append(f"pred[{timestep}] fdim: {int(pred_force_dimension)}")
            info_lines.append(f"pred[{timestep}] axis: {_format_vector(pred_axis)}")
            info_lines.append(f"pred[{timestep}] F: {_format_vector(pred_force)}")
            info_lines.append(f"pred[{timestep}] M: {_format_vector(pred_moment)}")
    axis.text2D(
        0.02,
        0.02,
        "\n".join(info_lines),
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        family="monospace",
        fontsize=7.2,
    )


def _draw_point_cloud_sample(
    point_cloud_axis,
    side_axis,
    dataset: MultiModalDataset,
    dataset_index: int,
    *,
    show_history: bool,
) -> None:
    (
        sample,
        point_cloud_key,
        point_clouds,
        point_cloud_mask,
        episode_index,
        episode_start,
        episode_end,
        point_counts,
    ) = _load_point_cloud_sample(dataset, dataset_index)
    obs_dict = sample["obs_dict"]
    prediction_dict = sample["prediction"]
    prediction_point_clouds, prediction_point_cloud_mask, prediction_point_counts = _load_prediction_point_cloud_sample(
        sample,
        point_cloud_key,
    )
    scene_point_cloud_key = _select_scene_point_cloud_key(dataset)
    force_dimension = _extract_latest_scalar(obs_dict, FORCE_DIMENSION_KEY)
    motion_or_force_axis = _extract_latest_vector(obs_dict, MOTION_OR_FORCE_AXIS_KEYS)
    sensed_force = _extract_latest_vector(obs_dict, SENSED_FORCE_KEY)
    action_delta_pos = _extract_latest_vector(obs_dict, ACTION_DELTA_POS_KEY)
    predicted_force_dimensions = _extract_scalar_sequence(prediction_dict, FORCE_DIMENSION_KEY)
    predicted_axes = _extract_vector_sequence(prediction_dict, MOTION_OR_FORCE_AXIS_KEYS)
    predicted_sensed_forces = _extract_vector_sequence(prediction_dict, SENSED_FORCE_KEY)
    predicted_sensed_moments = _extract_vector_sequence(prediction_dict, SENSED_MOMENT_KEY)

    timesteps_to_draw = list(range(point_clouds.shape[0])) if show_history else [point_clouds.shape[0] - 1]
    mode_name = "history" if show_history else "latest-only"

    print(
        f"Visualizing dataset index {dataset_index} | "
        f"episode {episode_index} ({episode_start}-{episode_end}) | "
        f"mode={mode_name} | obs steps={point_clouds.shape[0]} | obs point counts={point_counts.tolist()} | "
        f"pred steps={prediction_point_clouds.shape[0]} | pred point counts={prediction_point_counts.tolist()}",
        flush=True,
    )

    point_cloud_axis.clear()
    obs_color_map = plt.get_cmap("viridis", point_clouds.shape[0])
    pred_color_map = plt.get_cmap("plasma", prediction_point_clouds.shape[0])
    latest_valid_points = point_clouds[-1][point_cloud_mask[-1]]
    latest_ee_point, _ = _split_depth_and_ee_points(latest_valid_points)

    if scene_point_cloud_key is not None and scene_point_cloud_key in obs_dict:
        scene_point_clouds = _to_numpy(obs_dict[scene_point_cloud_key]).astype(np.float32, copy=False)
        scene_point_cloud_mask = _to_numpy(
            obs_dict[f"{scene_point_cloud_key}{POINT_CLOUD_MASK_SUFFIX}"]
        ).astype(bool, copy=False)
        if scene_point_clouds.ndim != 3 or scene_point_clouds.shape[-1] != 3:
            raise ValueError(
                f"Expected `{scene_point_cloud_key}` to have shape (T, P, 3), got {scene_point_clouds.shape}."
            )
        if scene_point_cloud_mask.shape != scene_point_clouds.shape[:2]:
            raise ValueError(
                f"Expected `{scene_point_cloud_key}{POINT_CLOUD_MASK_SUFFIX}` to have shape "
                f"{scene_point_clouds.shape[:2]}, got {scene_point_cloud_mask.shape}."
            )

        scene_valid_points = scene_point_clouds[0][scene_point_cloud_mask[0]]
        if len(scene_valid_points) != 0:
            point_cloud_axis.scatter(
                scene_valid_points[:, 0],
                scene_valid_points[:, 1],
                scene_valid_points[:, 2],
                s=16,
                alpha=0.9,
                color=SCENE_POINT_COLOR,
                label=f"scene points ({len(scene_valid_points)} pts)",
            )

    ee_label_drawn = False
    for timestep in timesteps_to_draw:
        valid_points = point_clouds[timestep][point_cloud_mask[timestep]]
        if len(valid_points) == 0:
            continue
        ee_point, depth_points = _split_depth_and_ee_points(valid_points)

        if len(depth_points) != 0:
            point_cloud_axis.scatter(
                depth_points[:, 0],
                depth_points[:, 1],
                depth_points[:, 2],
                s=18,
                alpha=0.85,
                color=obs_color_map(timestep),
                label=f"obs t={timestep} depth ({len(depth_points)} pts)",
            )

        if ee_point is not None:
            point_cloud_axis.scatter(
                [ee_point[0]],
                [ee_point[1]],
                [ee_point[2]],
                s=85,
                alpha=1.0 if timestep == timesteps_to_draw[-1] else 0.55,
                color="red",
                edgecolors="black",
                linewidths=0.8,
                label="Obs end effector" if not ee_label_drawn else None,
            )
            ee_label_drawn = True

    pred_ee_label_drawn = False
    pred_force_label_drawn = False
    for timestep in range(prediction_point_clouds.shape[0]):
        valid_points = prediction_point_clouds[timestep][prediction_point_cloud_mask[timestep]]
        if len(valid_points) == 0:
            continue
        ee_point, depth_points = _split_depth_and_ee_points(valid_points)
        pred_color = pred_color_map(timestep)

        if len(depth_points) != 0:
            point_cloud_axis.scatter(
                depth_points[:, 0],
                depth_points[:, 1],
                depth_points[:, 2],
                s=14,
                alpha=0.65,
                marker="^",
                color=pred_color,
                label=f"pred t+{timestep + 1} depth ({len(depth_points)} pts)",
            )

        if ee_point is not None:
            point_cloud_axis.scatter(
                [ee_point[0]],
                [ee_point[1]],
                [ee_point[2]],
                s=70,
                alpha=min(0.45 + 0.15 * timestep, 0.9),
                marker="s",
                color=pred_color,
                edgecolors="black",
                linewidths=0.7,
                label="Pred end effector" if not pred_ee_label_drawn else None,
            )
            pred_ee_label_drawn = True

            pred_force_vector = None
            if predicted_sensed_forces is not None and timestep < len(predicted_sensed_forces):
                pred_force_vector = _scaled_force_vector(predicted_sensed_forces[timestep])
            _draw_vector(
                point_cloud_axis,
                origin=ee_point,
                vector=pred_force_vector,
                color="goldenrod",
                label="Pred sensed force" if not pred_force_label_drawn else None,
                linewidth=1.8,
                alpha=min(0.35 + 0.18 * timestep, 0.85),
            )
            pred_force_label_drawn = pred_force_label_drawn or pred_force_vector is not None

    if latest_ee_point is not None:
        _draw_vector(
            point_cloud_axis,
            origin=latest_ee_point,
            vector=_scaled_force_vector(sensed_force),
            color="darkorange",
            label="Sensed force",
        )
        _draw_vector(
            point_cloud_axis,
            origin=latest_ee_point,
            vector=_clamp_vector_length(action_delta_pos, MAX_ACTION_VECTOR_LENGTH),
            color="limegreen",
            label="Action delta pos",
        )

    point_cloud_axis.set_xlim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    point_cloud_axis.set_ylim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    point_cloud_axis.set_zlim(-FIXED_AXIS_LIMIT, FIXED_AXIS_LIMIT)
    point_cloud_axis.set_box_aspect((1, 1, 1))
    point_cloud_axis.set_xlabel("X")
    point_cloud_axis.set_ylabel("Y")
    point_cloud_axis.set_zlabel("Z")
    point_cloud_axis.legend(loc="upper right", fontsize=8)
    point_cloud_axis.set_title(
        "Depth Point Clouds And Future Prediction Targets\n"
        f"Key: {point_cloud_key} | Obs mode: {mode_name} | Index: {dataset_index} | "
        f"Episode: {episode_index} ({episode_start}-{episode_end})"
    )
    point_cloud_axis.text2D(
        0.02,
        0.02,
        "Space/right: next | left/backspace: previous | h: toggle obs history | q/esc: quit",
        transform=point_cloud_axis.transAxes,
    )
    _draw_force_motion_axis_panel(
        side_axis,
        force_dimension=force_dimension,
        motion_or_force_axis=motion_or_force_axis,
        sensed_force=sensed_force,
        action_delta_pos=action_delta_pos,
        predicted_force_dimensions=predicted_force_dimensions,
        predicted_axes=predicted_axes,
        predicted_sensed_forces=predicted_sensed_forces,
        predicted_sensed_moments=predicted_sensed_moments,
    )


def visualize_dataset_browser(dataset: MultiModalDataset, start_index: int, *, show_history: bool) -> None:
    figure = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid_spec = figure.add_gridspec(1, 2, width_ratios=[4.5, 1.7])
    point_cloud_axis = figure.add_subplot(grid_spec[0, 0], projection="3d")
    side_axis = figure.add_subplot(grid_spec[0, 1], projection="3d")
    state = {"dataset_index": int(start_index), "show_history": bool(show_history)}

    def redraw() -> None:
        _draw_point_cloud_sample(
            point_cloud_axis,
            side_axis,
            dataset,
            state["dataset_index"],
            show_history=bool(state["show_history"]),
        )
        figure.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in (" ", "right"):
            state["dataset_index"] = (state["dataset_index"] + 1) % len(dataset)
            redraw()
        elif event.key in ("left", "backspace"):
            state["dataset_index"] = (state["dataset_index"] - 1) % len(dataset)
            redraw()
        elif event.key == "h":
            state["show_history"] = not bool(state["show_history"])
            redraw()
        elif event.key in ("q", "escape"):
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point-cloud observation stack from the dataset.")
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        required=True,
        type=str,
        help="Directory that contains the extracted dataset artifacts.",
    )
    parser.add_argument(
        "--universal-contract",
        dest="universal_contract",
        required=True,
        type=str,
        help="Path to the universal contract file.",
    )
    parser.add_argument(
        "--index",
        dest="dataset_index",
        default=None,
        type=int,
        help="Optional dataset index to visualize. If omitted, a random sample is chosen.",
    )
    parser.add_argument(
        "--show-history",
        dest="show_history",
        action="store_true",
        help="Show the full observation history instead of only the most recent depth points.",
    )

    args = parser.parse_args()
    dataset = MultiModalDataset(
        args.dataset_path,
        universal_contract=args.universal_contract,
    )
    dataset_index = _resolve_dataset_index(dataset, args.dataset_index)
    visualize_dataset_browser(dataset, dataset_index, show_history=bool(args.show_history))
