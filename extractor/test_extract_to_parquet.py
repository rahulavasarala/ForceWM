from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from extractor import extract_to_parquet as mod
from extractor import point_finder


def _action_label_config() -> mod.ActionLabelConfig:
    return mod.ActionLabelConfig(
        current_position_key="eef_pos",
        current_orientation_key="eef_ori",
        desired_position_key="desired_eef_pos",
        desired_orientation_key="desired_eef_ori",
        desired_force_magnitude_key="desired_force_magnitude",
        frame="female_part",
        orientation_encoding="rotvec",
        target_resample="hold_last",
    )


class ExtractToParquetTests(unittest.TestCase):
    def test_parse_action_label_config_requires_contract_block(self) -> None:
        with self.assertRaises(KeyError):
            mod.parse_action_label_config({"robot": {}})

    def test_load_lowdim_episode_requires_mapped_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            np.savez(
                episode_dir / "lowdim.npz",
                timestamp_s=np.array([0.0, 1.0], dtype=np.float64),
                eef_pos=np.zeros((2, 3), dtype=np.float32),
                eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
            )

            with self.assertRaises(KeyError):
                mod.load_lowdim_episode(episode_dir, _action_label_config())

    def test_load_lowdim_episode_requires_strictly_increasing_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            np.savez(
                episode_dir / "lowdim.npz",
                timestamp_s=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                eef_pos=np.zeros((3, 3), dtype=np.float32),
                eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                desired_eef_pos=np.zeros((3, 3), dtype=np.float32),
                desired_eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                desired_force_magnitude=np.ones(3, dtype=np.float32),
            )

            with self.assertRaises(ValueError):
                mod.load_lowdim_episode(episode_dir, _action_label_config())

    def test_load_lowdim_episode_rejects_negative_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            np.savez(
                episode_dir / "lowdim.npz",
                timestamp_s=np.array([0.0, 1.0], dtype=np.float64),
                eef_pos=np.zeros((2, 3), dtype=np.float32),
                eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
                desired_eef_pos=np.zeros((2, 3), dtype=np.float32),
                desired_eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
                desired_force_magnitude=np.array([1.0, -1.0], dtype=np.float32),
            )

            with self.assertRaises(ValueError):
                mod.load_lowdim_episode(episode_dir, _action_label_config())

    def test_load_lowdim_episode_preserves_passthrough_lowdim_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            np.savez(
                episode_dir / "lowdim.npz",
                timestamp_s=np.array([0.0, 1.0], dtype=np.float64),
                eef_pos=np.zeros((2, 3), dtype=np.float32),
                eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
                desired_eef_pos=np.zeros((2, 3), dtype=np.float32),
                desired_eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
                desired_force_magnitude=np.array([1.0, 2.0], dtype=np.float32),
                force_dimension=np.array([1, 0], dtype=np.int32),
                motion_or_force_axis=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            )

            episode = mod.load_lowdim_episode(episode_dir, _action_label_config())

        self.assertEqual(list(episode.passthrough_lowdim.keys()), ["force_dimension", "motion_or_force_axis"])
        np.testing.assert_array_equal(episode.passthrough_lowdim["force_dimension"], np.array([1, 0], dtype=np.int64))
        np.testing.assert_allclose(
            episode.passthrough_lowdim["motion_or_force_axis"],
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )

    def test_compute_action_labels_returns_relative_pose_and_absolute_force(self) -> None:
        Rotation = mod._require_scipy_rotation()
        desired_orientation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix().astype(np.float32)

        delta_pos, delta_rotvec, force = mod.compute_action_labels(
            current_positions_world=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            current_orientations_world=np.eye(3, dtype=np.float32)[None],
            desired_positions_world=np.array([[2.0, 4.0, 7.0]], dtype=np.float32),
            desired_orientations_world=desired_orientation[None],
            desired_force_magnitudes=np.array([3.5], dtype=np.float32),
            female_part_position_world=np.array([0.25, -0.5, 1.0], dtype=np.float32),
        )

        np.testing.assert_allclose(delta_pos, np.array([[1.0, 2.0, 4.0]], dtype=np.float32))
        np.testing.assert_allclose(delta_rotvec, np.array([[0.0, 0.0, math.pi / 2.0]], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(force, np.array([3.5], dtype=np.float32))

    def test_build_prune_keep_mask_reuses_edge_trim_and_stationary_pruning(self) -> None:
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        keep_mask = mod.build_prune_keep_mask(
            timestamps=timestamps,
            positions=positions,
            trim_start=1,
            trim_end=0,
            vel_thresh=1e-6,
            stationary_window=2,
        )

        self.assertEqual(keep_mask.tolist(), [False, True, True, False, False])

    def test_write_parquet_includes_action_columns(self) -> None:
        pa, pq = mod._require_pyarrow()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            episode = mod.ProcessedEpisode(
                source_dir=output_dir / "episode_000001",
                source_name="episode_000001",
                timestamps=np.array([0.0, 1.0], dtype=np.float64),
                positions=np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
                orientations=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
                action_delta_positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                action_delta_rotvecs=np.array([[0.0, 0.0, 0.1], [0.0, 0.2, 0.0]], dtype=np.float32),
                action_force_magnitudes=np.array([2.0, 3.0], dtype=np.float32),
                passthrough_lowdim={
                    "force_dimension": np.array([1, 0], dtype=np.int64),
                    "motion_or_force_axis": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                },
            )

            total_rows = mod.write_parquet(output_dir, [episode])
            table = pq.read_table(output_dir / mod.DEFAULT_PARQUET_NAME)

        self.assertEqual(total_rows, 2)
        self.assertEqual(
            table.schema.names,
            [
                "eef_pos",
                "eef_ori",
                "action_delta_pos",
                "action_delta_rotvec",
                "action_force_magnitude",
                "force_dimension",
                "motion_or_force_axis",
            ],
        )
        self.assertEqual(table.column("action_force_magnitude").type, pa.float32())
        self.assertEqual(table.column("force_dimension").type, pa.int64())

    def test_process_episode_builds_fixed_size_point_clouds(self) -> None:
        point_config = point_finder.default_simple_point_config()
        contact_spec = point_finder.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            np.savez(
                episode_dir / "lowdim.npz",
                timestamp_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
                eef_pos=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
                eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                desired_eef_pos=np.array([[0.0, 0.1, 0.0], [0.1, 0.1, 0.0], [0.2, 0.1, 0.0]], dtype=np.float32),
                desired_eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                desired_force_magnitude=np.array([1.0, 1.5, 2.0], dtype=np.float32),
                force_dimension=np.array([0, 1, 0], dtype=np.int32),
                sensed_force=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], dtype=np.float32),
            )

            result = mod.process_episode(
                episode_dir=episode_dir,
                trim_start=0,
                trim_end=0,
                vel_thresh=-1.0,
                stationary_window=1,
                action_label_config=_action_label_config(),
                female_part_position_world=np.zeros(3, dtype=np.float32),
                contact_spec=contact_spec,
                point_config=point_config,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.point_clouds.shape[0], 3)
        self.assertEqual(result.point_clouds.shape[2], 3)
        self.assertEqual(result.point_clouds.shape[1], 32)
        np.testing.assert_allclose(result.point_clouds[:, 0, :], result.processed_episode.positions, atol=1e-6)
        np.testing.assert_array_equal(
            result.processed_episode.passthrough_lowdim["force_dimension"],
            np.array([0, 1, 0], dtype=np.int64),
        )
        np.testing.assert_allclose(
            result.processed_episode.passthrough_lowdim["sensed_force"],
            np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], dtype=np.float32),
        )

    def test_extract_dataset_writes_lowdim_only_outputs(self) -> None:
        contract_text = """
robot:
  action_labels:
    current_position_key: "eef_pos"
    current_orientation_key: "eef_ori"
    desired_position_key: "desired_eef_pos"
    desired_orientation_key: "desired_eef_ori"
    desired_force_magnitude_key: "desired_force_magnitude"
    frame: "female_part"
    orientation_encoding: "rotvec"
    target_resample: "hold_last"
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            input_dir = temp_root / "input"
            episode_dir = input_dir / "episode_000001"
            episode_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                episode_dir / "lowdim.npz",
                timestamp_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
                eef_pos=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
                eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                desired_eef_pos=np.array([[0.0, 0.1, 0.0], [0.1, 0.1, 0.0], [0.2, 0.1, 0.0]], dtype=np.float32),
                desired_eef_ori=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                desired_force_magnitude=np.array([1.0, 1.5, 2.0], dtype=np.float32),
                force_dimension=np.array([1, 0, 1], dtype=np.int32),
                motion_or_force_axis=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                sensed_force=np.array([[4.0, 5.0, 6.0], [4.5, 5.5, 6.5], [5.0, 6.0, 7.0]], dtype=np.float32),
            )

            contract_path = temp_root / "contract.yaml"
            contract_path.write_text(contract_text, encoding="utf-8")
            output_dir = temp_root / "output"

            with mock.patch.object(
                mod,
                "load_female_part_position_world",
                return_value=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ), mock.patch.object(
                mod,
                "load_contact_cylinder_spec",
                return_value=point_finder.ContactCylinderSpec(radius_m=0.1, half_height_m=0.2),
            ):
                mod.extract_dataset(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    universal_contract_path=contract_path,
                    point_config_path=None,
                    chunk_size=2,
                    trim_start=0,
                    trim_end=0,
                    vel_thresh=-1.0,
                    stationary_window=1,
                )

            self.assertTrue((output_dir / "dataset.parquet").exists())
            self.assertTrue((output_dir / "metadata.npz").exists())
            self.assertFalse((output_dir / "videos").exists())
            chunk_paths = sorted((output_dir / "point_clouds" / "episode_0001").glob("chunk_*.npy"))
            self.assertEqual([path.name for path in chunk_paths], ["chunk_0001.npy", "chunk_0002.npy"])

            first_chunk = np.load(chunk_paths[0])
            second_chunk = np.load(chunk_paths[1])
            self.assertEqual(first_chunk.shape, (2, 32, 3))
            self.assertEqual(second_chunk.shape, (1, 32, 3))

            with np.load(output_dir / "metadata.npz") as metadata:
                np.testing.assert_array_equal(metadata["episode_ends"], np.array([2], dtype=np.int64))
                self.assertEqual(int(np.asarray(metadata["chunk_size"]).item()), 2)

            _, pq = mod._require_pyarrow()
            table = pq.read_table(output_dir / "dataset.parquet")
            self.assertEqual(table.num_rows, 3)
            self.assertEqual(
                table.schema.names,
                [
                    "eef_pos",
                    "eef_ori",
                    "action_delta_pos",
                    "action_delta_rotvec",
                    "action_force_magnitude",
                    "force_dimension",
                    "motion_or_force_axis",
                    "sensed_force",
                ],
            )


if __name__ == "__main__":
    unittest.main()
