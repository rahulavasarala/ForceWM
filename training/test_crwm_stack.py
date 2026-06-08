from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
import yaml
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

from training.crwm_model import CRWMModel, DummyDepthEncoder
from training.train import ModuleEMA, _build_model_size_report, train


PYARROW_AVAILABLE = pa is not None and pq is not None


def _fixed_size_list_column(array: np.ndarray, value_type) -> "pa.Array":
    if not PYARROW_AVAILABLE:
        raise unittest.SkipTest("pyarrow not installed")
    array = np.asarray(array)
    list_size = int(np.prod(array.shape[1:], dtype=np.int64))
    flattened = np.ascontiguousarray(array.reshape(len(array), list_size))
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flattened.reshape(-1), type=value_type),
        list_size,
    )


def _write_multiepisode_dataset(dataset_path: Path) -> None:
    if not PYARROW_AVAILABLE:
        raise unittest.SkipTest("pyarrow not installed")
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


def _write_point_cloud_chunks(dataset_path: Path, episode_lengths: list[int], *, num_points: int = 4) -> None:
    point_cloud_root = dataset_path / "point_clouds"
    point_cloud_root.mkdir(parents=True, exist_ok=True)
    for episode_idx, episode_length in enumerate(episode_lengths):
        frames = np.zeros((episode_length, num_points, 3), dtype=np.float32)
        base = float(episode_idx * 10)
        for frame_idx in range(episode_length):
            frames[frame_idx, :, 0] = base + frame_idx
            frames[frame_idx, :, 1] = np.arange(num_points, dtype=np.float32)
            frames[frame_idx, :, 2] = episode_idx

        episode_dir = point_cloud_root / f"episode_{episode_idx + 1:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        chunk_index = 0
        for frame_start in range(0, episode_length, 2):
            chunk = frames[frame_start : frame_start + 2]
            np.save(episode_dir / f"chunk_{chunk_index + 1:04d}.npy", chunk)
            chunk_index += 1


def _write_contract(
    contract_path: Path,
    *,
    normalize_force: bool,
    include_scene_points: bool = True,
) -> None:
    loader_keys = [
        {"camera_01_depth": {"obs_window": 2, "obs_dss": 1}},
        {"motion_or_force_axis": {"obs_window": 2, "obs_dss": 1}},
        {"force_dimension": {"obs_window": 2, "obs_dss": 1}},
        {"action_delta_pos": {"obs_window": 2, "obs_dss": 1}},
        {"action_delta_rotvec": {"obs_window": 2, "obs_dss": 1}},
        {"action_force_magnitude": {"obs_window": 2, "obs_dss": 1}},
        {"sensed_force": {"obs_window": 2, "obs_dss": 1, "normalize": normalize_force}},
        {"sensed_moment": {"obs_window": 2, "obs_dss": 1, "normalize": normalize_force}},
    ]
    if include_scene_points:
        loader_keys.insert(1, {"scene_points": {"obs_window": 1, "obs_dss": 1}})

    contract = {
        "robot": {
            "data_sources": {
                "visual": {
                    "keys": [
                        {"camera_01_depth": {"type": "depth"}},
                    ]
                }
            },
            "data_loader": {
                "keys": loader_keys,
                "prediction": {
                    "window": 1,
                    "dss": 1,
                },
            },
        }
    }
    with contract_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(contract, handle, sort_keys=False)


def _write_scene_points(path: Path) -> np.ndarray:
    scene_points = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 1.5, 0.0]],
            [[5.0, 0.0, 0.0], [5.0, 0.5, 0.0], [5.0, 1.0, 0.0], [5.0, 1.5, 0.0]],
        ],
        dtype=np.float32,
    )
    np.save(path, scene_points)
    return scene_points


def _write_crwm_contract(
    contract_path: Path,
    *,
    prediction_window: int = 1,
    normalize_force: bool = False,
    include_scene_points: bool = True,
) -> None:
    _write_contract(
        contract_path,
        normalize_force=normalize_force,
        include_scene_points=include_scene_points,
    )
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    contract["robot"]["data_loader"]["prediction"]["window"] = int(prediction_window)
    with contract_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(contract, handle, sort_keys=False)


def _make_train_config(
    *,
    dataset_path: Path,
    contract_path: Path,
    output_dir: Path,
    normalize_force: bool = False,
) -> dict[str, object]:
    return {
        "dataset_path": str(dataset_path),
        "universal_contract": str(contract_path),
        "output_dir": str(output_dir),
        "device": "cpu",
        "seed": 7,
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "val_fraction": 0.5,
        "val_every_epochs": 1,
        "log_every": 0,
        "depth_encoder_trainable_epochs": 0,
        "contact_encoder_trainable_epochs": 0,
        "wandb": {
            "enabled": False,
            "project": "forcewm-test",
            "entity": None,
            "run_name": None,
        },
        "depth_encoder_ema": {"decay": 0.9, "update_after_step": 0, "update_every": 1},
        "contact_encoder_ema": {"decay": 0.9, "update_after_step": 0, "update_every": 1},
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        "scheduler": {"warmup_steps": 0, "min_lr_scale": 0.5},
        "model": {
            "max_history_steps": 4,
            "depth_encoder": {
                "type": "dummy",
                "hidden_dim": 16,
                "point_feature_dim": 12,
                "global_latent_dim": 8,
                "num_blocks": 1,
            },
            "contact_encoder": {
                "hidden_dim": 16,
                "output_dim": 6,
                "num_force_dimensions": 4,
                "force_embedding_dim": 4,
            },
            "action_encoder": {
                "hidden_dim": 16,
                "output_dim": 5,
            },
            "flow": {
                "model_dim": 24,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
            },
            "decoder": {
                "depth_hidden_dim": 16,
                "contact_hidden_dim": 16,
            },
            "loss_weights": {
                "flow": 1.0,
                "depth_recon": 0.5,
                "contact_recon": 0.25,
            },
        },
    }


class DummyDepthEncoderTests(unittest.TestCase):
    def test_dummy_depth_encoder_ignores_invalid_points(self) -> None:
        encoder = DummyDepthEncoder(hidden_dim=16, point_feature_dim=12, global_latent_dim=8, num_blocks=1)
        points = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-77.0, -77.0, -77.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, True, False], [True, True, False]], dtype=torch.bool)

        outputs = encoder(points, mask)

        self.assertEqual(tuple(outputs.global_latent.shape), (2, 8))
        self.assertEqual(tuple(outputs.point_features.shape), (2, 3, 12))
        self.assertEqual(tuple(outputs.valid_mask.shape), (2, 3))
        torch.testing.assert_close(outputs.global_latent[0], outputs.global_latent[1], atol=1e-6, rtol=0.0)


class CRWMModelTests(unittest.TestCase):
    def test_crwm_forward_returns_expected_shapes(self) -> None:
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
        batch = {
            "obs_dict": {
                "camera_01_depth": torch.randn(2, 2, 4, 3),
                "camera_01_depth_mask": torch.ones(2, 2, 4, dtype=torch.bool),
                "scene_points": torch.randn(2, 1, 4, 3),
                "scene_points_mask": torch.ones(2, 1, 4, dtype=torch.bool),
                "motion_or_force_axis": torch.randn(2, 2, 3),
                "force_dimension": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
                "action_delta_pos": torch.randn(2, 2, 3),
                "action_delta_rotvec": torch.randn(2, 2, 3),
                "action_force_magnitude": torch.randn(2, 2),
                "sensed_force": torch.randn(2, 2, 3),
                "sensed_moment": torch.randn(2, 2, 3),
            },
            "prediction": {
                "camera_01_depth": torch.randn(2, 1, 4, 3),
                "camera_01_depth_mask": torch.ones(2, 1, 4, dtype=torch.bool),
                "motion_or_force_axis": torch.randn(2, 1, 3),
                "force_dimension": torch.tensor([[1], [2]], dtype=torch.long),
                "sensed_force": torch.randn(2, 1, 3),
                "sensed_moment": torch.randn(2, 1, 3),
            },
        }

        outputs = model(batch)

        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(tuple(outputs["predicted_delta"].shape), (2, 14))
        self.assertEqual(tuple(outputs["latent_target"].shape), (2, 14))
        self.assertEqual(tuple(outputs["predicted_depth_points"].shape), (2, 4, 3))
        self.assertEqual(tuple(outputs["predicted_force_dimension_logits"].shape), (2, 4))
        self.assertTrue(torch.isfinite(outputs["latent_delta_loss"]))
        self.assertTrue(torch.isfinite(outputs["ee_position_mse"]))
        self.assertEqual(model.depth_decoder.decoder[0].in_features, model.depth_latent_dim)
        self.assertEqual(tuple(model.modality_embeddings.shape), (4, model.model_dim))
        self.assertIn("contact_force_dimension_ce", outputs)
        self.assertIn("contact_motion_axis_mse", outputs)
        self.assertIn("contact_sensed_force_mse", outputs)
        self.assertIn("contact_sensed_moment_mse", outputs)
        self.assertTrue(
            torch.allclose(
                outputs["contact_recon_loss"],
                outputs["contact_force_dimension_ce"]
                + outputs["contact_motion_axis_mse"]
                + outputs["contact_sensed_force_mse"]
                + outputs["contact_sensed_moment_mse"],
            )
        )

    def test_crwm_requires_scene_points_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires `scene_points_key`"):
            CRWMModel(
                depth_key="camera_01_depth",
                scene_points_key=None,
                num_depth_points=4,
                depth_encoder_config={"type": "dummy", "hidden_dim": 16, "point_feature_dim": 12, "global_latent_dim": 8, "num_blocks": 1},
                contact_encoder_config={"hidden_dim": 16, "output_dim": 6, "num_force_dimensions": 4, "force_embedding_dim": 4},
                action_encoder_config={"hidden_dim": 16, "output_dim": 5},
                flow_config={"model_dim": 24, "num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0},
                decoder_config={"depth_hidden_dim": 16, "contact_hidden_dim": 16},
                max_history_steps=4,
            )

    def test_crwm_forward_uses_scene_for_conditioning_not_decoder_input(self) -> None:
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
        batch = {
            "obs_dict": {
                "camera_01_depth": torch.randn(2, 2, 4, 3),
                "camera_01_depth_mask": torch.ones(2, 2, 4, dtype=torch.bool),
                "scene_points": torch.randn(2, 1, 4, 3),
                "scene_points_mask": torch.ones(2, 1, 4, dtype=torch.bool),
                "motion_or_force_axis": torch.randn(2, 2, 3),
                "force_dimension": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
                "action_delta_pos": torch.randn(2, 2, 3),
                "action_delta_rotvec": torch.randn(2, 2, 3),
                "action_force_magnitude": torch.randn(2, 2),
                "sensed_force": torch.randn(2, 2, 3),
                "sensed_moment": torch.randn(2, 2, 3),
            },
            "prediction": {
                "camera_01_depth": torch.randn(2, 1, 4, 3),
                "camera_01_depth_mask": torch.ones(2, 1, 4, dtype=torch.bool),
                "motion_or_force_axis": torch.randn(2, 1, 3),
                "force_dimension": torch.tensor([[1], [2]], dtype=torch.long),
                "sensed_force": torch.randn(2, 1, 3),
                "sensed_moment": torch.randn(2, 1, 3),
            },
        }

        class _RecordingFlowModel(torch.nn.Module):
            def __init__(self, latent_dim: int) -> None:
                super().__init__()
                self.latent_dim = int(latent_dim)
                self.last_condition_tokens: torch.Tensor | None = None

            def forward(self, condition_tokens: torch.Tensor) -> torch.Tensor:
                self.last_condition_tokens = condition_tokens.detach().clone()
                return torch.zeros(
                    condition_tokens.shape[0],
                    self.latent_dim,
                    dtype=condition_tokens.dtype,
                    device=condition_tokens.device,
                )

        class _RecordingDepthDecoder(torch.nn.Module):
            def __init__(self, num_points: int) -> None:
                super().__init__()
                self.num_points = int(num_points)
                self.last_input: torch.Tensor | None = None

            def forward(self, latent: torch.Tensor) -> torch.Tensor:
                self.last_input = latent.detach().clone()
                return torch.zeros(latent.shape[0], self.num_points, 3, dtype=latent.dtype, device=latent.device)

        recording_flow_model = _RecordingFlowModel(model.latent_dim)
        recording_depth_decoder = _RecordingDepthDecoder(model.num_depth_points)
        model.flow_model = recording_flow_model
        model.depth_decoder = recording_depth_decoder

        outputs = model(batch)

        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(tuple(outputs["predicted_depth_points"].shape), (2, 4, 3))
        assert recording_flow_model.last_condition_tokens is not None
        assert recording_depth_decoder.last_input is not None
        self.assertEqual(tuple(recording_flow_model.last_condition_tokens.shape), (2, 7, model.model_dim))
        self.assertEqual(tuple(recording_depth_decoder.last_input.shape), (2, model.depth_latent_dim))

    def test_ee_position_mse_uses_only_first_depth_point_and_mask(self) -> None:
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
        predicted_points = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [99.0, 99.0, 99.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[7.0, 8.0, 9.0], [55.0, 55.0, 55.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        target_points = torch.zeros_like(predicted_points)
        target_mask = torch.tensor(
            [
                [True, True, True, True],
                [False, True, True, True],
            ],
            dtype=torch.bool,
        )

        ee_position_mse = model._ee_position_mse(predicted_points, target_points, target_mask)

        self.assertAlmostEqual(float(ee_position_mse.item()), (1.0**2 + 2.0**2 + 3.0**2) / 3.0, places=6)

    def test_model_size_report_counts_scene_conditioning_components(self) -> None:
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

        report = _build_model_size_report(model)

        conditioning_components = report["components"]["conditioning_stack"]["components"]
        self.assertIn("scene_token_projection", conditioning_components)
        self.assertEqual(model.depth_decoder.decoder[0].in_features, model.depth_latent_dim)


class TrainerSmokeTests(unittest.TestCase):
    def test_module_ema_updates_toward_source_weights(self) -> None:
        source = torch.nn.Linear(4, 4, bias=False)
        source.weight.data.fill_(2.0)
        ema = ModuleEMA(source, config=type("Cfg", (), {"decay": 0.5, "update_after_step": 0, "update_every": 1})())

        source.weight.data.fill_(4.0)
        ema.maybe_update(source, step=1)

        self.assertTrue(torch.all(ema.module.weight > 2.0))
        self.assertTrue(torch.all(ema.module.weight < 4.0))

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_rejects_prediction_windows_other_than_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            scene_points_path = dataset_path / "scene_points.npy"
            _write_scene_points(scene_points_path)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=2)

            config = {
                "dataset_path": str(dataset_path),
                "universal_contract": str(contract_path),
                "output_dir": str(Path(tmp_dir) / "run"),
                "device": "cpu",
                "epochs": 1,
            }

            with self.assertRaisesRegex(ValueError, "prediction.window == 1"):
                train(config)

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_requires_scene_points_loader_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(
                contract_path,
                prediction_window=1,
                include_scene_points=False,
            )

            config = {
                "dataset_path": str(dataset_path),
                "universal_contract": str(contract_path),
                "output_dir": str(Path(tmp_dir) / "run"),
                "device": "cpu",
                "epochs": 1,
            }

            with self.assertRaisesRegex(ValueError, "expects exactly one `scene_points` key"):
                train(config)

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_builds_normalizer_and_writes_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            scene_points_path = dataset_path / "scene_points.npy"
            _write_scene_points(scene_points_path)
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

            metrics = train(config)
            latest_checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")
            self.assertTrue((dataset_path / "normalizer.npy").exists())
            self.assertTrue((output_dir / "latest.pt").exists())
            self.assertTrue((output_dir / "best.pt").exists())
            self.assertTrue((output_dir / "config_snapshot.yaml").exists())
            self.assertTrue((output_dir / "contract_snapshot.yaml").exists())
            self.assertIn("train_loss", metrics)
            self.assertIn("val_loss", metrics)
            self.assertIn("depth_ema_state_dict", latest_checkpoint)
            self.assertIn("contact_ema_state_dict", latest_checkpoint)
            self.assertIn("model_state_dict", latest_checkpoint)

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_prints_startup_report_before_epoch_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1)
            output_dir = Path(tmp_dir) / "run"
            config = _make_train_config(
                dataset_path=dataset_path,
                contract_path=contract_path,
                output_dir=output_dir,
            )

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                metrics = train(config)

        output = stdout_buffer.getvalue()
        emitted_lines = [line for line in output.splitlines() if line.strip()]
        self.assertGreaterEqual(len(emitted_lines), 2)
        self.assertEqual(emitted_lines[0], "Startup Preflight:")
        self.assertIn("  stage: startup_dummy_pass", output)
        self.assertIn("  device: cpu", output)
        self.assertIn("  depth_encoder_trainable: False", output)
        self.assertIn("  contact_encoder_trainable: False", output)
        self.assertIn(
            "  batch_source: split=train indices=[0, 1] configured_batch_size=2 effective_batch_size=2",
            output,
        )
        self.assertIn("Inputs / obs_dict:", output)
        self.assertIn("Prediction Targets:", output)
        self.assertIn("  - camera_01_depth: shape=[2, 2, 4, 3]", output)
        self.assertIn("  - action_force_magnitude: shape=[2, 2]", output)
        self.assertIn("Model Components:", output)
        self.assertIn("  - full_model: total_params=", output)
        self.assertIn("  - depth_encoder: total_params=", output)
        self.assertIn("  - conditioning_stack: total_params=", output)
        self.assertIn("Outputs:", output)
        self.assertIn("  - predicted_delta: shape=[2, 14]", output)
        self.assertIn("Loss Targets:", output)
        self.assertIn("  latent_delta_loss:", output)
        self.assertIn("  depth_recon_loss:", output)
        self.assertIn("  contact_recon_loss:", output)
        self.assertIn("Normalization:", output)
        self.assertIn("  - none", output)
        self.assertNotIn('"stage"', output)
        self.assertNotIn("preview", output)
        self.assertNotIn("values", output)
        self.assertTrue(emitted_lines[-1].startswith("epoch=1 "))
        self.assertIn("val_ran=1", emitted_lines[-1])
        self.assertIn("contact_encoder_trainable=0", emitted_lines[-1])
        self.assertIn("train_ee_mse=", emitted_lines[-1])
        self.assertIn("train_loss", metrics)

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_startup_report_includes_normalizer_stats_for_normalized_keys(self) -> None:
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

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                train(config)

        output = stdout_buffer.getvalue()
        self.assertIn("Normalization:", output)
        self.assertIn(
            "  - sensed_force: mean=[2.2500, 1.0000, 0.3333] std=[0.8539, 0.6455, 0.3727]",
            output,
        )
        self.assertIn(
            "  - sensed_moment: mean=[0.2500, 0.3500, 0.4500] std=[0.1708, 0.1708, 0.1708]",
            output,
        )
        self.assertNotIn("motion_or_force_axis: mean=", output)

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_freezes_contact_encoder_after_configured_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1)
            output_dir = Path(tmp_dir) / "run"
            config = _make_train_config(
                dataset_path=dataset_path,
                contract_path=contract_path,
                output_dir=output_dir,
            )
            config["epochs"] = 2
            config["depth_encoder_trainable_epochs"] = 1
            config["contact_encoder_trainable_epochs"] = 1

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                train(config)

        epoch_lines = [
            line for line in stdout_buffer.getvalue().splitlines() if line.startswith("epoch=")
        ]
        self.assertEqual(len(epoch_lines), 2)
        self.assertIn("epoch=1 depth_encoder_trainable=1 contact_encoder_trainable=1", epoch_lines[0])
        self.assertIn("epoch=2 depth_encoder_trainable=0 contact_encoder_trainable=0", epoch_lines[1])

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_skips_validation_until_configured_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1)
            output_dir = Path(tmp_dir) / "run"
            config = _make_train_config(
                dataset_path=dataset_path,
                contract_path=contract_path,
                output_dir=output_dir,
            )
            config["epochs"] = 2
            config["val_every_epochs"] = 2

            epoch_metrics: list[dict[str, float]] = []

            def _collect_epoch(epoch: int, metrics: dict[str, float], output_dir: Path) -> None:
                _ = epoch, output_dir
                epoch_metrics.append(dict(metrics))

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                metrics = train(config, on_epoch_end=_collect_epoch)
            best_checkpoint_exists = (output_dir / "best.pt").exists()

        output = stdout_buffer.getvalue()
        epoch_lines = [line for line in output.splitlines() if line.startswith("epoch=")]
        self.assertEqual(len(epoch_lines), 2)
        self.assertIn("epoch=1", epoch_lines[0])
        self.assertIn("val_ran=0", epoch_lines[0])
        self.assertIn("val_loss=nan", epoch_lines[0])
        self.assertIn("epoch=2", epoch_lines[1])
        self.assertIn("val_ran=1", epoch_lines[1])
        self.assertEqual(len(epoch_metrics), 2)
        self.assertEqual(epoch_metrics[0]["val_ran"], 0.0)
        self.assertNotIn("val_loss", epoch_metrics[0])
        self.assertEqual(epoch_metrics[1]["val_ran"], 1.0)
        self.assertIn("val_loss", epoch_metrics[1])
        self.assertEqual(metrics["val_ran"], 1.0)
        self.assertIn("val_loss", metrics)
        self.assertTrue(best_checkpoint_exists)

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_logs_wandb_metrics_when_enabled(self) -> None:
        class _FakeWandb:
            def __init__(self) -> None:
                self.init_kwargs: dict[str, object] | None = None
                self.log_calls: list[tuple[dict[str, float], int | None]] = []
                self.finish_calls = 0

            def init(self, **kwargs):
                self.init_kwargs = kwargs
                return object()

            def log(self, metrics, step=None):
                self.log_calls.append((dict(metrics), step))

            def finish(self):
                self.finish_calls += 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1)
            output_dir = Path(tmp_dir) / "run"
            config = _make_train_config(
                dataset_path=dataset_path,
                contract_path=contract_path,
                output_dir=output_dir,
            )
            config["wandb"] = {
                "enabled": True,
                "project": "forcewm-test",
                "entity": "forcewm",
                "run_name": "unit-test-run",
            }

            fake_wandb = _FakeWandb()
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                train(config)

        self.assertIsNotNone(fake_wandb.init_kwargs)
        assert fake_wandb.init_kwargs is not None
        self.assertEqual(fake_wandb.init_kwargs["project"], "forcewm-test")
        self.assertEqual(fake_wandb.init_kwargs["entity"], "forcewm")
        self.assertEqual(fake_wandb.init_kwargs["name"], "unit-test-run")
        self.assertEqual(fake_wandb.finish_calls, 1)
        step_logs = [metrics for metrics, step in fake_wandb.log_calls if step is not None and "train_step/loss" in metrics]
        self.assertGreaterEqual(len(step_logs), 1)
        self.assertIn("train_step/contact_force_dimension_ce", step_logs[0])
        self.assertIn("train_step/contact_motion_axis_mse", step_logs[0])
        self.assertIn("train_step/contact_sensed_force_mse", step_logs[0])
        self.assertIn("train_step/contact_sensed_moment_mse", step_logs[0])
        self.assertIn("trainer/global_step", step_logs[0])
        self.assertIn("trainer/epoch", step_logs[0])
        self.assertIn("trainer/lr", step_logs[0])
        epoch_logs = [metrics for metrics, _ in fake_wandb.log_calls if "train_epoch/loss" in metrics]
        self.assertEqual(len(epoch_logs), 1)
        self.assertIn("val/loss", epoch_logs[0])
        self.assertIn("trainer/val_ran", epoch_logs[0])

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow not installed")
    def test_train_replays_captured_startup_noise_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"
            _write_multiepisode_dataset(dataset_path)
            _write_point_cloud_chunks(dataset_path, [3, 3], num_points=4)
            _write_scene_points(dataset_path / "scene_points.npy")
            contract_path = Path(tmp_dir) / "contract.yaml"
            _write_crwm_contract(contract_path, prediction_window=1)
            output_dir = Path(tmp_dir) / "run"
            config = _make_train_config(
                dataset_path=dataset_path,
                contract_path=contract_path,
                output_dir=output_dir,
            )

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            def _failing_build_depth_encoder(config=None):
                print("startup noise from depth encoder")
                raise RuntimeError("synthetic startup failure")

            with mock.patch("training.crwm_model.build_depth_encoder", side_effect=_failing_build_depth_encoder):
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    with self.assertRaisesRegex(RuntimeError, "synthetic startup failure"):
                        train(config)

        self.assertEqual(stdout_buffer.getvalue(), "")
        self.assertIn("startup noise from depth encoder", stderr_buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
