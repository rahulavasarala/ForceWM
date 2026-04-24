import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset import MultiModalDataset


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _quat_to_direction(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float32)
    if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternions with shape (T, 4), got {quaternions.shape}."
        )

    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    quaternions = quaternions / norms

    x = quaternions[:, 0]
    y = quaternions[:, 1]
    z = quaternions[:, 2]
    w = quaternions[:, 3]

    # Rotate the unit x-axis by each quaternion so orientation can be shown as a 3D arrow.
    direction = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        axis=-1,
    )
    return direction


def _episode_bounds(episode_ends):
    starts = np.concatenate(([0], episode_ends[:-1] + 1))
    return list(zip(starts.tolist(), episode_ends.tolist()))


def _build_samples_by_episode(test_dataset, num_samples, rng):
    samples_by_episode = []
    for episode_index, (start, end) in enumerate(_episode_bounds(test_dataset.episode_ends)):
        frame_indices = np.arange(start, end + 1)
        sample_count = min(num_samples, len(frame_indices))
        chosen_indices = rng.choice(frame_indices, size=sample_count, replace=False)
        chosen_indices = np.sort(chosen_indices)

        episode_samples = []
        for dataset_index in chosen_indices:
            sample = test_dataset[int(dataset_index)]
            episode_samples.append(
                {
                    "dataset_index": int(dataset_index),
                    "episode_index": episode_index,
                    "episode_start": int(start),
                    "episode_end": int(end),
                    "obs_pos": _to_numpy(sample["obs"]["eef_pos"]),
                    "obs_ori": _to_numpy(sample["obs"]["eef_ori"]),
                    "action_pos": _to_numpy(sample["actions"]["eef_pos"]),
                    "action_ori": _to_numpy(sample["actions"]["eef_ori"]),
                }
            )

        samples_by_episode.append(episode_samples)

    return samples_by_episode


def _compute_plot_limits(sample):
    points = np.concatenate([sample["obs_pos"], sample["action_pos"]], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = np.max(maxs - mins)
    half_range = max(extent / 2.0, 0.05) * 1.2
    return center, half_range


def _draw_sample(ax, sample, sample_index, total_samples):
    obs_pos = sample["obs_pos"]
    action_pos = sample["action_pos"]
    obs_dir = _quat_to_direction(sample["obs_ori"])
    action_dir = _quat_to_direction(sample["action_ori"])

    ax.clear()
    ax.plot(
        obs_pos[:, 0],
        obs_pos[:, 1],
        obs_pos[:, 2],
        color="tab:blue",
        marker="o",
        linewidth=2,
        label="Observation positions",
    )
    ax.plot(
        action_pos[:, 0],
        action_pos[:, 1],
        action_pos[:, 2],
        color="tab:orange",
        marker="^",
        linewidth=2,
        linestyle="--",
        label="Action positions",
    )

    ax.quiver(
        obs_pos[:, 0],
        obs_pos[:, 1],
        obs_pos[:, 2],
        obs_dir[:, 0],
        obs_dir[:, 1],
        obs_dir[:, 2],
        length=0.03,
        color="tab:blue",
        normalize=True,
    )
    ax.quiver(
        action_pos[:, 0],
        action_pos[:, 1],
        action_pos[:, 2],
        action_dir[:, 0],
        action_dir[:, 1],
        action_dir[:, 2],
        length=0.03,
        color="tab:orange",
        normalize=True,
    )

    center, half_range = _compute_plot_limits(sample)
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    ax.set_title(
        "Dataset Visualization\n"
        f"Episode {sample['episode_index']} | Frame {sample['dataset_index']} "
        f"({sample['episode_start']}-{sample['episode_end']}) | "
        f"Sample {sample_index + 1}/{total_samples}"
    )
    ax.text2D(
        0.02,
        0.02,
        "Press space for next sample, left/right to navigate, q to quit.",
        transform=ax.transAxes,
    )


def visualize_dataset(test_dataset, num_samples):
    # test_dataset is of the form -------
    # test_dataset = MultiModalDataset(dataset_path, universal_contract= contract_path)

    # sample random frames from each episode and then visualize the points, orientations, and
    # actions (orientations and positions) in matplotlib as a 3D graph.
    if num_samples <= 0:
        raise ValueError("`num_samples` must be a positive integer.")

    episode_ends = test_dataset.episode_ends

    if len(episode_ends) == 0:
        raise ValueError("The dataset has no episodes to visualize.")

    rng = np.random.default_rng()
    samples_by_episode = _build_samples_by_episode(test_dataset, num_samples, rng)
    non_empty_samples = [episode_samples for episode_samples in samples_by_episode if episode_samples]
    if not non_empty_samples:
        raise ValueError("Unable to collect any samples from the dataset.")

    ordered_samples = [sample for episode_samples in non_empty_samples for sample in episode_samples]

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    state = {"index": 0}

    def redraw():
        sample = ordered_samples[state["index"]]
        _draw_sample(axis, sample, state["index"], len(ordered_samples))
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key in (" ", "right"):
            state["index"] = (state["index"] + 1) % len(ordered_samples)
            redraw()
        elif event.key == "left":
            state["index"] = (state["index"] - 1) % len(ordered_samples)
            redraw()
        elif event.key in ("q", "escape"):
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Argument Parser for the dataset")

    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        required=True,
        type=str,
        help="Directory that contains the named recording buffers.",
    )
   
    parser.add_argument(
        "--universal-contract",
        dest="universal_contract",
        required=True,
        type=str,
        help="Path to the universal contract file.",
    )

    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        default=1,
        type=int,
        help="Number of random anchor frames to sample per episode.",
    )

    args = parser.parse_args()
    dataset = MultiModalDataset(
        args.dataset_path,
        universal_contract=args.universal_contract,
    )
    dataset.action_type = "absolute"
    visualize_dataset(dataset, args.num_samples)

