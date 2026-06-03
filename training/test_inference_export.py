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
    from inference_visualization import load_visualization_artifact, prepare_episode_records
except ModuleNotFoundError:
    load_visualization_artifact = None
    prepare_episode_records = None
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
    def test_sample_one_step_uses_seeded_gaussian_rollout_and_decodes_outputs(self) -> None:
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

        class _ConstantVelocityFlow(torch.nn.Module):
            def forward(
                self,
                x_t: torch.Tensor,
                timesteps: torch.Tensor,
                condition_tokens: torch.Tensor,
            ) -> torch.Tensor:
                _ = timesteps, condition_tokens
                return torch.ones_like(x_t)

        model.flow_model = _ConstantVelocityFlow()

        generator_a = torch.Generator().manual_seed(11)
        outputs_a = model.sample_one_step(
            batch["obs_dict"],
            generator=generator_a,
            sampling_steps=32,
            solver="heun",
        )
        generator_b = torch.Generator().manual_seed(11)
        outputs_b = model.sample_one_step(
            batch["obs_dict"],
            generator=generator_b,
            sampling_steps=32,
            solver="heun",
        )
        generator_c = torch.Generator().manual_seed(12)
        outputs_c = model.sample_one_step(
            batch["obs_dict"],
            generator=generator_c,
            sampling_steps=32,
            solver="heun",
        )

        expected_initial = torch.randn((2, model.latent_dim), generator=torch.Generator().manual_seed(11))
        torch.testing.assert_close(outputs_a["predicted_state"], expected_initial + 1.0, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(outputs_a["predicted_state"], outputs_b["predicted_state"], atol=1e-6, rtol=0.0)
        self.assertFalse(torch.allclose(outputs_a["predicted_state"], outputs_c["predicted_state"]))
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

    def test_visualizer_loader_reads_exported_artifact(self) -> None:
        if load_visualization_artifact is None or prepare_episode_records is None:
            raise unittest.SkipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1, normalize_force=False)
            output_dir = Path(tmp_dir) / "run"
            config = _make_train_config(
                dataset_path=dataset_path,
                contract_path=contract_path,
                output_dir=output_dir,
            )
            train(config)
            artifact_path = output_dir / "visualizer_preds.npy"
            export_predictions(config, split="all", artifact_path=artifact_path)

            inference_artifact = load_prediction_artifact(artifact_path)
            visualization_artifact = load_visualization_artifact(artifact_path)
            episodes = prepare_episode_records(visualization_artifact)

        self.assertEqual(inference_artifact["artifact_version"], 1)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(tuple(episodes[0]["predicted_depth_points"].shape), (3, 4, 3))
        self.assertEqual(tuple(episodes[1]["target_motion_or_force_axis"].shape), (3, 3))


if __name__ == "__main__":
    unittest.main()
