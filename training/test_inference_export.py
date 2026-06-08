from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
except ModuleNotFoundError:
    matplotlib = None

try:
    from inference_visualization import (
        build_synthetic_ee_centers,
        extract_ee_point,
        load_visualization_dataset,
        prepare_episode_visualization,
        resolve_synthetic_distance,
        resolve_stdev,
    )
except ModuleNotFoundError:
    build_synthetic_ee_centers = None
    extract_ee_point = None
    load_visualization_dataset = None
    prepare_episode_visualization = None
    resolve_synthetic_distance = None
    resolve_stdev = None
from training.crwm_model import CRWMModel
from training.inference import export_predictions, load_prediction_artifact
from training.test_crwm_stack import (
    PYARROW_AVAILABLE,
    _make_train_config,
    _write_crwm_contract,
    _write_multiepisode_dataset,
    _write_point_cloud_chunks,
    _write_scene_points,
)
from training.train import train


def _build_obs_batch(batch_size: int = 2) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "obs_dict": {
            "camera_01_depth": torch.randn(batch_size, 2, 4, 3),
            "camera_01_depth_mask": torch.ones(batch_size, 2, 4, dtype=torch.bool),
            "scene_points": torch.randn(batch_size, 1, 4, 3),
            "scene_points_mask": torch.ones(batch_size, 1, 4, dtype=torch.bool),
            "motion_or_force_axis": torch.randn(batch_size, 2, 3),
            "force_dimension": torch.tensor([[0, 1], [2, 3]], dtype=torch.long)[:batch_size],
            "action_delta_pos": torch.randn(batch_size, 2, 3),
            "action_delta_rotvec": torch.randn(batch_size, 2, 3),
            "action_force_magnitude": torch.randn(batch_size, 2),
            "sensed_force": torch.randn(batch_size, 2, 3),
            "sensed_moment": torch.randn(batch_size, 2, 3),
        },
        "prediction": {
            "camera_01_depth": torch.randn(batch_size, 1, 4, 3),
            "camera_01_depth_mask": torch.ones(batch_size, 1, 4, dtype=torch.bool),
            "motion_or_force_axis": torch.randn(batch_size, 1, 3),
            "force_dimension": torch.tensor([[1], [2]], dtype=torch.long)[:batch_size],
            "sensed_force": torch.randn(batch_size, 1, 3),
            "sensed_moment": torch.randn(batch_size, 1, 3),
        },
    }


class CRWMSamplerTests(unittest.TestCase):
    def test_sample_one_step_is_deterministic_and_decodes_direct_latent_delta(self) -> None:
        model = CRWMModel(
            depth_key="camera_01_depth",
            scene_points_key="scene_points",
            num_depth_points=4,
            depth_encoder_config={"type": "dummy", "hidden_dim": 16, "point_feature_dim": 12, "global_latent_dim": 8, "num_blocks": 1},
            contact_encoder_config={"hidden_dim": 16, "output_dim": 6, "num_force_dimensions": 4, "force_embedding_dim": 4},
            action_encoder_config={"hidden_dim": 16, "output_dim": 5},
            flow_config={"model_dim": 24, "num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0},
            decoder_config={"depth_hidden_dim": 16, "contact_hidden_dim": 16},
            max_history_steps=4,
        )
        batch = _build_obs_batch(batch_size=2)

        class _ConstantDeltaPredictor(torch.nn.Module):
            def __init__(self, latent_dim: int) -> None:
                super().__init__()
                self.latent_dim = int(latent_dim)

            def forward(self, condition_tokens: torch.Tensor) -> torch.Tensor:
                batch_size = int(condition_tokens.shape[0])
                return torch.ones(
                    batch_size,
                    self.latent_dim,
                    dtype=condition_tokens.dtype,
                    device=condition_tokens.device,
                )

        model.flow_model = _ConstantDeltaPredictor(model.latent_dim)
        context = model._build_inference_context(batch["obs_dict"])
        expected_predicted_delta = torch.ones(2, model.latent_dim)

        outputs_a = model.sample_one_step(
            batch["obs_dict"],
        )
        outputs_b = model.sample_one_step(
            batch["obs_dict"],
        )

        torch.testing.assert_close(outputs_a["predicted_delta"], expected_predicted_delta, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(outputs_a["predicted_delta"], outputs_b["predicted_delta"], atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            outputs_a["predicted_depth_latent"],
            context["last_observed_depth"] + expected_predicted_delta[:, : model.depth_latent_dim],
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            outputs_a["predicted_contact_latent"],
            context["last_observed_contact"] + expected_predicted_delta[:, model.depth_latent_dim :],
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(tuple(outputs_a["predicted_depth_points"].shape), (2, 4, 3))
        self.assertEqual(tuple(outputs_a["predicted_force_dimension_logits"].shape), (2, 4))
        self.assertEqual(tuple(outputs_a["predicted_force_dimension"].shape), (2,))


@unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
class InferenceExportTests(unittest.TestCase):
    def test_export_predictions_supports_train_val_all_and_denormalizes_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(
                contract_path,
                prediction_window=1,
                normalize_force=True,
            )
            output_dir = Path(tmp_dir) / "run"
            config = _make_train_config(
                dataset_path=dataset_path,
                contract_path=contract_path,
                output_dir=output_dir,
                normalize_force=True,
            )
            train(config)
            expected_val_forces = np.array(
                [
                    [3.0, 1.5, 0.5],
                    [3.5, 2.0, 1.0],
                    [3.5, 2.0, 1.0],
                ],
                dtype=np.float32,
            )
            expected_val_moments = np.array(
                [
                    [0.4, 0.5, 0.6],
                    [0.5, 0.6, 0.7],
                    [0.5, 0.6, 0.7],
                ],
                dtype=np.float32,
            )

            artifact_train = export_predictions(config, split="train", artifact_path=output_dir / "train_preds.npy")
            artifact_val = export_predictions(config, split="val", artifact_path=output_dir / "val_preds.npy")
            artifact_all = export_predictions(config, split="all", artifact_path=output_dir / "all_preds.npy")
            self.assertTrue((output_dir / "train_preds.npy").exists())
            self.assertTrue((output_dir / "val_preds.npy").exists())
            self.assertTrue((output_dir / "all_preds.npy").exists())
            self.assertEqual([episode["episode_index"] for episode in artifact_train["episodes"]], [0])
            self.assertEqual([episode["episode_index"] for episode in artifact_val["episodes"]], [1])
            self.assertEqual([episode["episode_index"] for episode in artifact_all["episodes"]], [0, 1])
            np.testing.assert_array_equal(
                artifact_train["episodes"][0]["dataset_indices"],
                np.array([0, 1, 2], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                artifact_val["episodes"][0]["dataset_indices"],
                np.array([3, 4, 5], dtype=np.int64),
            )
            np.testing.assert_array_equal(artifact_all["metadata"]["selected_indices"], np.arange(6, dtype=np.int64))
            self.assertEqual(artifact_all["artifact_version"], 2)
            self.assertEqual(artifact_all["metadata"]["predictor_type"], "direct_latent_delta")
            self.assertNotIn("seed", artifact_all["metadata"])
            self.assertNotIn("sampling_steps", artifact_all["metadata"])
            self.assertNotIn("solver", artifact_all["metadata"])
            np.testing.assert_allclose(
                artifact_val["episodes"][0]["target_sensed_force"],
                expected_val_forces,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                artifact_val["episodes"][0]["target_sensed_moment"],
                expected_val_moments,
                atol=1e-6,
            )
            self.assertEqual(tuple(artifact_val["episodes"][0]["predicted_depth_points"].shape), (3, 4, 3))
            self.assertEqual(tuple(artifact_val["episodes"][0]["depth_mask"].shape), (3, 4))
            self.assertTrue(str(artifact_val["metadata"]["checkpoint_path"]).endswith("best.pt"))

    def test_extract_ee_point_uses_first_point(self) -> None:
        if extract_ee_point is None:
            raise unittest.SkipTest("inference visualizer is unavailable")
        points = np.array(
            [
                [1.0, 2.0, 3.0],
                [9.0, 9.0, 9.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(extract_ee_point(points), points[0])

    def test_load_visualization_dataset_rejects_non_one_step_predictions(self) -> None:
        if load_visualization_dataset is None:
            raise unittest.SkipTest("inference visualizer is unavailable")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=2, normalize_force=False)

            with self.assertRaisesRegex(ValueError, "one-step prediction targets"):
                load_visualization_dataset(dataset_path, contract_path)

    def test_prepare_episode_visualization_uses_scene_points_and_one_step_targets(self) -> None:
        if (
            load_visualization_dataset is None
            or prepare_episode_visualization is None
            or build_synthetic_ee_centers is None
        ):
            raise unittest.SkipTest("inference visualizer is unavailable")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            scene_points = _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1, normalize_force=False)

            dataset = load_visualization_dataset(dataset_path, contract_path)
            payload = prepare_episode_visualization(dataset, 0, synthetic_distance=0.0)
            centers = build_synthetic_ee_centers(dataset, 0)

        expected_ground_truth_path = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        expected_centers = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(payload.scene_points, scene_points[0], atol=1e-6)
        np.testing.assert_allclose(payload.ground_truth_ee_path, expected_ground_truth_path, atol=1e-6)
        np.testing.assert_allclose(centers, expected_centers, atol=1e-6)
        np.testing.assert_allclose(payload.synthetic_ee_centers, expected_centers, atol=1e-6)
        np.testing.assert_allclose(payload.synthetic_ee_path, expected_ground_truth_path, atol=1e-6)

    def test_prepare_episode_visualization_generates_deterministic_fixed_distance_path(self) -> None:
        if (
            load_visualization_dataset is None
            or prepare_episode_visualization is None
            or resolve_synthetic_distance is None
            or resolve_stdev is None
        ):
            raise unittest.SkipTest("inference visualizer is unavailable")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1, normalize_force=False)

            dataset = load_visualization_dataset(dataset_path, contract_path)
            payload_a = prepare_episode_visualization(dataset, 1, synthetic_distance=0.25)
            payload_b = prepare_episode_visualization(dataset, 1, synthetic_distance=0.25)
            payload_zero_distance = prepare_episode_visualization(dataset, 1, synthetic_distance=0.0)
            payload_smaller_distance = prepare_episode_visualization(dataset, 1, synthetic_distance=0.1)
            payload_jitter_a = prepare_episode_visualization(dataset, 1, synthetic_distance=0.25, stdev=0.05)
            payload_jitter_b = prepare_episode_visualization(dataset, 1, synthetic_distance=0.25, stdev=0.05)

        self.assertAlmostEqual(resolve_synthetic_distance(0.25), 0.25, places=6)
        self.assertAlmostEqual(resolve_stdev(0.05), 0.05, places=6)
        np.testing.assert_allclose(payload_a.synthetic_ee_path, payload_b.synthetic_ee_path, atol=1e-6)
        np.testing.assert_allclose(payload_jitter_a.synthetic_ee_path, payload_jitter_b.synthetic_ee_path, atol=1e-6)
        np.testing.assert_allclose(payload_a.synthetic_ee_path[0], payload_a.ground_truth_ee_path[0], atol=1e-6)
        np.testing.assert_allclose(payload_zero_distance.synthetic_ee_path[1:], payload_zero_distance.synthetic_ee_centers, atol=1e-6)
        offsets = payload_a.synthetic_ee_path[1:] - payload_a.synthetic_ee_centers
        smaller_offsets = payload_smaller_distance.synthetic_ee_path[1:] - payload_smaller_distance.synthetic_ee_centers
        jitter_offsets = payload_jitter_a.synthetic_ee_path[1:] - payload_jitter_a.synthetic_ee_centers
        np.testing.assert_allclose(
            np.linalg.norm(offsets, axis=1),
            np.full(len(offsets), resolve_synthetic_distance(0.25), dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(smaller_offsets, axis=1),
            np.full(len(smaller_offsets), resolve_synthetic_distance(0.1), dtype=np.float32),
            atol=1e-6,
        )
        self.assertGreater(
            float(np.max(np.abs(np.linalg.norm(jitter_offsets, axis=1) - resolve_synthetic_distance(0.25)))),
            1e-4,
        )
        self.assertFalse(np.allclose(payload_jitter_a.synthetic_ee_path, payload_a.synthetic_ee_path, atol=1e-6))

    def test_resolve_synthetic_distance_rejects_negative_values(self) -> None:
        if resolve_synthetic_distance is None:
            raise unittest.SkipTest("inference visualizer is unavailable")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_synthetic_distance(-1.0)

    def test_resolve_stdev_rejects_negative_values_and_requires_synthetic_distance(self) -> None:
        if prepare_episode_visualization is None or resolve_stdev is None or load_visualization_dataset is None:
            raise unittest.SkipTest("inference visualizer is unavailable")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_stdev(-1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1, normalize_force=False)

            dataset = load_visualization_dataset(dataset_path, contract_path)
            with self.assertRaisesRegex(ValueError, "requires `--synthetic-distance`"):
                prepare_episode_visualization(dataset, 0, stdev=0.1)


if __name__ == "__main__":
    unittest.main()
