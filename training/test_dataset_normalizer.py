from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None


PYARROW_AVAILABLE = pa is not None and pq is not None

if PYARROW_AVAILABLE:
    from training.dataset import MultiModalDataset
    from training.normalizer import DatasetNormalizer, build_normalizer
else:
    MultiModalDataset = None
    DatasetNormalizer = None
    build_normalizer = None


def _fixed_size_list_column(array: np.ndarray, value_type) -> pa.Array:
    array = np.asarray(array)
    list_size = int(np.prod(array.shape[1:], dtype=np.int64))
    flattened = np.ascontiguousarray(array.reshape(len(array), list_size))
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flattened.reshape(-1), type=value_type),
        list_size,
    )


def _write_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    dataset_path.mkdir(parents=True, exist_ok=True)

    angles = np.array([0.0, 0.15, 0.3, 0.45, 0.6], dtype=np.float32)
    eef_ori = R.from_euler("z", angles).as_matrix().astype(np.float32).reshape(-1, 9)

    columns = {
        "eef_pos": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [2.0, 1.0, 0.5],
                [3.0, 1.5, 1.0],
                [4.0, 2.0, 1.5],
            ],
            dtype=np.float32,
        ),
        "eef_ori": eef_ori,
        "action_delta_pos": np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.2, 0.0],
                [0.2, 0.3, 0.1],
                [0.3, 0.4, 0.1],
                [0.4, 0.5, 0.2],
            ],
            dtype=np.float32,
        ),
        "action_delta_rotvec": np.array(
            [
                [0.0, 0.0, -0.2],
                [0.0, 0.0, -0.1],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.2],
            ],
            dtype=np.float32,
        ),
        "action_force_magnitude": np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32),
        "sensed_force": np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0],
                [5.0, 4.0, 3.0],
            ],
            dtype=np.float32,
        ),
        "force_dimension": np.array([0, 1, 2, 1, 0], dtype=np.int64),
        "motion_or_force_axis": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    table = pa.table(
        {
            "eef_pos": _fixed_size_list_column(columns["eef_pos"], pa.float32()),
            "eef_ori": _fixed_size_list_column(columns["eef_ori"], pa.float32()),
            "action_delta_pos": _fixed_size_list_column(columns["action_delta_pos"], pa.float32()),
            "action_delta_rotvec": _fixed_size_list_column(columns["action_delta_rotvec"], pa.float32()),
            "action_force_magnitude": pa.array(columns["action_force_magnitude"], type=pa.float32()),
            "sensed_force": _fixed_size_list_column(columns["sensed_force"], pa.float32()),
            "force_dimension": pa.array(columns["force_dimension"], type=pa.int64()),
            "motion_or_force_axis": _fixed_size_list_column(columns["motion_or_force_axis"], pa.float32()),
        }
    )
    pq.write_table(table, dataset_path / "dummy.parquet")
    np.savez(
        dataset_path / "metadata.npz",
        episode_ends=np.array([len(columns["eef_pos"]) - 1], dtype=np.int64),
        chunk_size=np.array(2, dtype=np.int64),
    )
    return columns


def _write_multiepisode_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    dataset_path.mkdir(parents=True, exist_ok=True)

    num_rows = 6
    columns = {
        "action_delta_pos": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "action_delta_rotvec": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.2],
                [0.0, 0.1, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.3, 0.0],
            ],
            dtype=np.float32,
        ),
        "action_force_magnitude": np.linspace(1.0, 2.0, num_rows, dtype=np.float32),
        "force_dimension": np.array([0, 1, 2, 1, 2, 3], dtype=np.int64),
        "motion_or_force_axis": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "sensed_force": np.array(
            [
                [1.0, 0.0, 0.0],
                [1.5, 0.5, 0.0],
                [2.0, 1.0, 0.0],
                [2.5, 1.0, 0.5],
                [3.0, 1.5, 0.5],
                [3.5, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "sensed_moment": np.array(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
                [0.5, 0.6, 0.7],
            ],
            dtype=np.float32,
        ),
    }
    table = pa.table(
        {
            "action_delta_pos": _fixed_size_list_column(columns["action_delta_pos"], pa.float32()),
            "action_delta_rotvec": _fixed_size_list_column(columns["action_delta_rotvec"], pa.float32()),
            "action_force_magnitude": pa.array(columns["action_force_magnitude"], type=pa.float32()),
            "force_dimension": pa.array(columns["force_dimension"], type=pa.int64()),
            "motion_or_force_axis": _fixed_size_list_column(columns["motion_or_force_axis"], pa.float32()),
            "sensed_force": _fixed_size_list_column(columns["sensed_force"], pa.float32()),
            "sensed_moment": _fixed_size_list_column(columns["sensed_moment"], pa.float32()),
        }
    )
    pq.write_table(table, dataset_path / "dummy.parquet")
    np.savez(
        dataset_path / "metadata.npz",
        episode_ends=np.array([2, 5], dtype=np.int64),
        chunk_size=np.array(2, dtype=np.int64),
    )
    return columns


def _write_point_cloud_chunks(dataset_path: Path, episode_lengths: list[int], *, num_points: int = 4) -> np.ndarray:
    point_cloud_root = dataset_path / "point_clouds"
    point_cloud_root.mkdir(parents=True, exist_ok=True)
    all_episodes: list[np.ndarray] = []
    for episode_idx, episode_length in enumerate(episode_lengths):
        frames = np.zeros((episode_length, num_points, 3), dtype=np.float32)
        base = float(episode_idx * 10)
        for frame_idx in range(episode_length):
            frames[frame_idx, :, 0] = base + frame_idx
            frames[frame_idx, :, 1] = np.arange(num_points, dtype=np.float32)
            frames[frame_idx, :, 2] = episode_idx
        all_episodes.append(frames)

        episode_dir = point_cloud_root / f"episode_{episode_idx + 1:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        chunk_index = 0
        for frame_start in range(0, episode_length, 2):
            chunk = frames[frame_start : frame_start + 2]
            np.save(episode_dir / f"chunk_{chunk_index + 1:04d}.npy", chunk)
            chunk_index += 1

    return np.stack(all_episodes, axis=0)


def _write_contract(
    contract_path: Path,
    *,
    loader_entries: list[tuple[str, dict[str, object]]],
    include_depth_key: bool = False,
    scene_points_path: str | None = None,
) -> None:
    robot_cfg: dict[str, object] = {
        "data_loader": {
            "keys": [{key_name: key_cfg} for key_name, key_cfg in loader_entries],
            "prediction": {
                "window": 2,
                "dss": 1,
            },
        }
    }

    if include_depth_key or scene_points_path is not None:
        visual_keys: list[dict[str, object]] = []
        if include_depth_key:
            visual_keys.append(
                {
                    "camera_01_depth": {
                        "type": "depth",
                    }
                }
            )
        if scene_points_path is not None:
            visual_keys.append(
                {
                    "scene_points": {
                        "type": "scene_points",
                        "path": scene_points_path,
                    }
                }
            )
        robot_cfg["data_sources"] = {
            "visual": {
                "keys": visual_keys
            }
        }

    with contract_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"robot": robot_cfg}, handle, sort_keys=False)


@unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
class DatasetNormalizerTests(unittest.TestCase):
    def test_build_normalizer_includes_only_normalized_keys_and_default_representation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("eef_ori", {"obs_window": 2, "obs_dss": 1, "normalize": True, "normalization_representation": "matrix"}),
                    ("action_delta_rotvec", {"obs_window": 2, "obs_dss": 1, "normalize": True, "normalization_representation": "rotvec"}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1, "normalize": True}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            output_path = build_normalizer(dataset_path, contract_path)
            normalizer = DatasetNormalizer.load(output_path)

        self.assertEqual(set(normalizer.key_stats.keys()), {"eef_ori", "action_delta_rotvec", "sensed_force"})
        self.assertEqual(normalizer.require_key("eef_ori")["representation"], "matrix")
        self.assertEqual(normalizer.require_key("action_delta_rotvec")["representation"], "rotvec")
        self.assertEqual(normalizer.require_key("sensed_force")["representation"], "standard")
        self.assertEqual(normalizer.require_key("eef_ori")["feature_shape"], (9,))

    def test_dataset_rejects_point_cloud_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("camera_01_depth", {"obs_window": 2, "obs_dss": 1, "normalize": True}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                ],
                include_depth_key=True,
            )

            with self.assertRaisesRegex(ValueError, "point-cloud key"):
                build_normalizer(dataset_path, contract_path)

            with self.assertRaisesRegex(ValueError, "point-cloud key"):
                MultiModalDataset(dataset_path, contract_path)

    def test_loader_returns_normalized_lowdim_observations_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            columns = _write_dataset(dataset_path)

            normalized_contract = Path(tmp_dir) / "normalized.yaml"
            raw_contract = Path(tmp_dir) / "raw.yaml"
            _write_contract(
                normalized_contract,
                loader_entries=[
                    ("action_delta_rotvec", {"obs_window": 2, "obs_dss": 1, "normalize": True, "normalization_representation": "rotvec"}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1, "normalize": True}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
            )
            _write_contract(
                raw_contract,
                loader_entries=[
                    ("action_delta_rotvec", {"obs_window": 2, "obs_dss": 1}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            build_normalizer(dataset_path, normalized_contract)
            normalized_dataset = MultiModalDataset(dataset_path, normalized_contract)
            raw_dataset = MultiModalDataset(dataset_path, raw_contract)

            normalized_sample = normalized_dataset[2]
            raw_sample = raw_dataset[2]

        raw_obs = columns["sensed_force"][1:3]
        expected = normalized_dataset.normalizer.normalize_key("sensed_force", raw_obs)
        np.testing.assert_allclose(raw_sample["obs_dict"]["sensed_force"].numpy(), raw_obs)
        np.testing.assert_allclose(normalized_sample["obs_dict"]["sensed_force"].numpy(), expected, atol=1e-6)
        self.assertFalse(
            np.allclose(
                normalized_sample["obs_dict"]["sensed_force"].numpy(),
                raw_sample["obs_dict"]["sensed_force"].numpy(),
            )
        )

    def test_prediction_lowdim_values_are_normalized_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            columns = _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1, "normalize": True}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            build_normalizer(dataset_path, contract_path)
            dataset = MultiModalDataset(dataset_path, contract_path)
            sample = dataset[1]

        expected_prediction = dataset.normalizer.normalize_key("sensed_force", columns["sensed_force"][2:4])
        np.testing.assert_allclose(sample["prediction"]["sensed_force"].numpy(), expected_prediction, atol=1e-6)

    def test_action_delta_rotvec_round_trips_for_numpy_and_torch_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            columns = _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("action_delta_rotvec", {"obs_window": 2, "obs_dss": 1, "normalize": True, "normalization_representation": "rotvec"}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            normalizer_path = build_normalizer(dataset_path, contract_path)
            normalizer = DatasetNormalizer.load(normalizer_path)

        rotvec_sequence = columns["action_delta_rotvec"][1:4]
        normalized_sequence = normalizer.normalize_key("action_delta_rotvec", rotvec_sequence)
        restored_sequence = normalizer.denormalize_key("action_delta_rotvec", normalized_sequence)
        np.testing.assert_allclose(restored_sequence, rotvec_sequence, atol=1e-6)

        batched_rotvecs = torch.from_numpy(np.stack([rotvec_sequence, rotvec_sequence + 0.05], axis=0))
        normalized_batched = normalizer.normalize_key("action_delta_rotvec", batched_rotvecs)
        restored_batched = normalizer.denormalize_key("action_delta_rotvec", normalized_batched)
        self.assertIsInstance(normalized_batched, torch.Tensor)
        self.assertEqual(normalized_batched.dtype, torch.float32)
        torch.testing.assert_close(restored_batched, batched_rotvecs.to(dtype=torch.float32), atol=1e-6, rtol=0.0)

    def test_matrix_orientation_round_trip_uses_flattened_trailing_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            columns = _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("eef_ori", {"obs_window": 2, "obs_dss": 1, "normalize": True, "normalization_representation": "matrix"}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            normalizer_path = build_normalizer(dataset_path, contract_path)
            normalizer = DatasetNormalizer.load(normalizer_path)

        orientations = columns["eef_ori"][1:4]
        normalized = normalizer.normalize_key("eef_ori", orientations)
        restored = normalizer.denormalize_key("eef_ori", normalized)
        self.assertEqual(normalizer.require_key("eef_ori")["feature_shape"], (9,))
        np.testing.assert_allclose(restored, orientations, atol=1e-6)

    def test_normalize_and_denormalize_sample_api_handles_nested_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            columns = _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1, "normalize": True}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            normalizer_path = build_normalizer(dataset_path, contract_path)
            normalizer = DatasetNormalizer.load(normalizer_path)

        sample = {
            "obs_dict": {
                "sensed_force": columns["sensed_force"][0:2],
                "force_dimension": columns["force_dimension"][0:2],
            },
            "prediction": {
                "sensed_force": columns["sensed_force"][2:4],
            },
            "metadata": {"episode_id": 0},
        }
        normalized_sample = normalizer.normalize_sample(sample)
        restored_sample = normalizer.denormalize_sample(normalized_sample)

        np.testing.assert_allclose(restored_sample["obs_dict"]["sensed_force"], sample["obs_dict"]["sensed_force"], atol=1e-6)
        np.testing.assert_allclose(
            restored_sample["prediction"]["sensed_force"],
            sample["prediction"]["sensed_force"],
            atol=1e-6,
        )
        np.testing.assert_array_equal(restored_sample["obs_dict"]["force_dimension"], sample["obs_dict"]["force_dimension"])
        self.assertEqual(restored_sample["metadata"], sample["metadata"])

    def test_dataset_requires_normalizer_when_normalization_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1, "normalize": True}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            with self.assertRaisesRegex(FileNotFoundError, "Missing normalizer file"):
                MultiModalDataset(dataset_path, contract_path)

    def test_dataset_requires_stats_for_every_normalized_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("action_delta_rotvec", {"obs_window": 2, "obs_dss": 1, "normalize": True, "normalization_representation": "rotvec"}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1, "normalize": True}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
            )

            normalizer_path = build_normalizer(dataset_path, contract_path)
            normalizer = DatasetNormalizer.load(normalizer_path)
            del normalizer.key_stats["action_delta_rotvec"]
            normalizer.save(normalizer_path)

            with self.assertRaisesRegex(KeyError, "action_delta_rotvec"):
                MultiModalDataset(dataset_path, contract_path)

    def test_scene_points_load_per_episode_and_stay_out_of_prediction_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            scene_points = np.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                    [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            scene_points_path = Path(tmp_dir) / "scene_points.npy"
            np.save(scene_points_path, scene_points)

            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("camera_01_depth", {"obs_window": 2, "obs_dss": 1}),
                    ("scene_points", {"obs_window": 1, "obs_dss": 1}),
                    ("motion_or_force_axis", {"obs_window": 2, "obs_dss": 1}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                    ("action_delta_pos", {"obs_window": 2, "obs_dss": 1}),
                    ("action_delta_rotvec", {"obs_window": 2, "obs_dss": 1}),
                    ("action_force_magnitude", {"obs_window": 2, "obs_dss": 1}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                    ("sensed_moment", {"obs_window": 2, "obs_dss": 1}),
                ],
                include_depth_key=True,
                scene_points_path=str(scene_points_path),
            )

            dataset = MultiModalDataset(dataset_path, contract_path)
            sample_episode_0 = dataset[1]
            sample_episode_1 = dataset[4]

        np.testing.assert_allclose(sample_episode_0["obs_dict"]["scene_points"].numpy()[0], scene_points[0], atol=1e-6)
        np.testing.assert_allclose(sample_episode_1["obs_dict"]["scene_points"].numpy()[0], scene_points[1], atol=1e-6)
        np.testing.assert_array_equal(
            sample_episode_0["obs_dict"]["scene_points_mask"].numpy(),
            np.ones((1, scene_points.shape[1]), dtype=bool),
        )
        self.assertNotIn("scene_points", sample_episode_0["prediction"])
        self.assertEqual(dataset.static_point_cloud_keys, ["scene_points"])

    def test_scene_points_reject_mismatched_episode_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            scene_points_path = Path(tmp_dir) / "scene_points.npy"
            np.save(scene_points_path, np.zeros((1, 3, 3), dtype=np.float32))

            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("camera_01_depth", {"obs_window": 2, "obs_dss": 1}),
                    ("scene_points", {"obs_window": 1, "obs_dss": 1}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                    ("sensed_moment", {"obs_window": 2, "obs_dss": 1}),
                    ("motion_or_force_axis", {"obs_window": 2, "obs_dss": 1}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
                include_depth_key=True,
                scene_points_path=str(scene_points_path),
            )

            with self.assertRaisesRegex(ValueError, "metadata declares 2 episodes"):
                MultiModalDataset(dataset_path, contract_path)

    def test_scene_points_reject_wrong_trailing_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            scene_points_path = Path(tmp_dir) / "scene_points.npy"
            np.save(scene_points_path, np.zeros((2, 3, 4), dtype=np.float32))

            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("camera_01_depth", {"obs_window": 2, "obs_dss": 1}),
                    ("scene_points", {"obs_window": 1, "obs_dss": 1}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                    ("sensed_moment", {"obs_window": 2, "obs_dss": 1}),
                    ("motion_or_force_axis", {"obs_window": 2, "obs_dss": 1}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
                include_depth_key=True,
                scene_points_path=str(scene_points_path),
            )

            with self.assertRaisesRegex(ValueError, "xyz dimension 3"):
                MultiModalDataset(dataset_path, contract_path)

    def test_scene_points_require_single_step_loader_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            scene_points_path = Path(tmp_dir) / "scene_points.npy"
            np.save(scene_points_path, np.zeros((2, 3, 3), dtype=np.float32))

            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_contract(
                contract_path,
                loader_entries=[
                    ("camera_01_depth", {"obs_window": 2, "obs_dss": 1}),
                    ("scene_points", {"obs_window": 2, "obs_dss": 1}),
                    ("sensed_force", {"obs_window": 2, "obs_dss": 1}),
                    ("sensed_moment", {"obs_window": 2, "obs_dss": 1}),
                    ("motion_or_force_axis", {"obs_window": 2, "obs_dss": 1}),
                    ("force_dimension", {"obs_window": 2, "obs_dss": 1}),
                ],
                include_depth_key=True,
                scene_points_path=str(scene_points_path),
            )

            with self.assertRaisesRegex(ValueError, "obs_window` must be 1"):
                MultiModalDataset(dataset_path, contract_path)


if __name__ == "__main__":
    unittest.main()
