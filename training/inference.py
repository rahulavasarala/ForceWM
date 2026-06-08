from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from tqdm import tqdm

if __package__:
    from .crwm_model import CRWMModel
    from .train import (
        ModuleEMA,
        _enforce_prediction_window_is_one,
        _infer_num_depth_points,
        _load_yaml,
        _move_to_device,
        _resolve_depth_and_scene_keys,
        _resolve_ema_config,
        _resolve_split_indices,
        _select_device,
        ensure_normalizer,
    )
else:
    from crwm_model import CRWMModel
    from train import (
        ModuleEMA,
        _enforce_prediction_window_is_one,
        _infer_num_depth_points,
        _load_yaml,
        _move_to_device,
        _resolve_depth_and_scene_keys,
        _resolve_ema_config,
        _resolve_split_indices,
        _select_device,
        ensure_normalizer,
    )


DEFAULT_EXPORT_SPLIT = "val"
DEFAULT_ARTIFACT_NAME_TEMPLATE = "one_step_predictions_{split}.npy"
ARTIFACT_VERSION = 2
PREDICTOR_TYPE = "direct_latent_delta"
SUPPORTED_SPLITS = frozenset({"train", "val", "all"})


@dataclass
class InferenceRuntime:
    config: dict[str, Any]
    contract: dict[str, Any]
    dataset: Any
    model: CRWMModel
    depth_ema: ModuleEMA
    device: torch.device
    checkpoint_path: Path
    checkpoint_metadata: dict[str, Any]
    output_dir: Path
    dataset_path: Path
    contract_path: Path


def _resolve_export_split_indices(
    dataset: Any,
    *,
    split: str,
    val_fraction: float,
) -> list[int]:
    split_name = str(split).strip().lower()
    if split_name not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split `{split}`. Expected one of {sorted(SUPPORTED_SPLITS)}.")

    train_indices, val_indices = _resolve_split_indices(dataset, float(val_fraction))
    if split_name == "train":
        return train_indices
    if split_name == "val":
        return val_indices
    return list(range(len(dataset)))


def _resolve_axis_key(mapping: Mapping[str, torch.Tensor], candidates: tuple[str, ...]) -> str:
    for key_name in candidates:
        if key_name in mapping:
            return key_name
    raise KeyError(f"Missing motion/force axis key. Tried {list(candidates)}.")


def _stack_row_dicts(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    return {
        key: np.stack([np.asarray(row[key]) for row in rows], axis=0)
        for key in rows[0]
    }


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_checkpoint_for_inference(
    checkpoint_path: str | Path,
    *,
    model: CRWMModel,
    depth_ema: ModuleEMA,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    depth_ema.load_state_dict(checkpoint["depth_ema_state_dict"])
    return checkpoint


def _build_model_from_config(
    config: Mapping[str, Any],
    *,
    dataset: Any,
    device: torch.device,
) -> tuple[CRWMModel, ModuleEMA]:
    depth_key, scene_points_key = _resolve_depth_and_scene_keys(dataset)
    model_cfg = dict(config.get("model", {}))
    depth_encoder_cfg = dict(model_cfg.get("depth_encoder", {}))
    depth_encoder_type = str(depth_encoder_cfg.get("type", "dummy")).lower()
    if depth_encoder_type not in {"dummy", "ptv3"}:
        raise ValueError(f"`model.depth_encoder.type` must be `dummy` or `ptv3`, got `{depth_encoder_type}`.")
    if depth_encoder_type == "ptv3" and device.type != "cuda":
        raise ValueError(
            "Concerto/PTv3 inference requires a CUDA device. "
            f"Resolved device `{device}` is unsupported for `model.depth_encoder.type: ptv3`."
        )

    num_depth_points = int(model_cfg.get("num_depth_points", _infer_num_depth_points(dataset, depth_key)))
    model = CRWMModel(
        depth_key=depth_key,
        scene_points_key=scene_points_key,
        num_depth_points=num_depth_points,
        depth_encoder_config=depth_encoder_cfg,
        contact_encoder_config=dict(model_cfg.get("contact_encoder", {})),
        action_encoder_config=dict(model_cfg.get("action_encoder", {})),
        flow_config=dict(model_cfg.get("flow", {})),
        decoder_config=dict(model_cfg.get("decoder", {})),
        loss_weights=dict(model_cfg.get("loss_weights", {})),
        max_history_steps=int(model_cfg.get("max_history_steps", 16)),
        action_delta_pos_key=str(model_cfg.get("action_delta_pos_key", "action_delta_pos")),
        action_delta_rotvec_key=str(model_cfg.get("action_delta_rotvec_key", "action_delta_rotvec")),
        action_force_magnitude_key=str(model_cfg.get("action_force_magnitude_key", "action_force_magnitude")),
        force_dimension_key=str(model_cfg.get("force_dimension_key", "force_dimension")),
        motion_or_force_axis_key=str(model_cfg.get("motion_or_force_axis_key", "motion_or_force_axis")),
        sensed_force_key=str(model_cfg.get("sensed_force_key", "sensed_force")),
        sensed_moment_key=str(model_cfg.get("sensed_moment_key", "sensed_moment")),
    ).to(device)
    depth_ema = ModuleEMA(model.depth_encoder, _resolve_ema_config(config.get("depth_encoder_ema")))
    return model, depth_ema


def load_inference_runtime(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    device_name: str | None = None,
) -> InferenceRuntime:
    config_payload = copy.deepcopy(dict(config))
    dataset_path = Path(config_payload["dataset_path"]).expanduser().resolve()
    contract_path = Path(config_payload["universal_contract"]).expanduser().resolve()
    output_dir = Path(config_payload.get("output_dir", "training_runs/crwm")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = _load_yaml(contract_path)
    _enforce_prediction_window_is_one(contract)
    ensure_normalizer(dataset_path, contract_path)

    if __package__:
        from .dataset import MultiModalDataset
    else:
        from dataset import MultiModalDataset

    dataset = MultiModalDataset(
        dataset_path,
        universal_contract=contract_path,
        pointcloud_cache_size=int(config_payload.get("pointcloud_cache_size", 4)),
    )
    device = _select_device(device_name or config_payload.get("device"))
    model, depth_ema = _build_model_from_config(config_payload, dataset=dataset, device=device)

    resolved_checkpoint_path = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else (output_dir / "best.pt").resolve()
    )
    checkpoint = _load_checkpoint_for_inference(
        resolved_checkpoint_path,
        model=model,
        depth_ema=depth_ema,
        device=device,
    )
    model.eval()
    depth_ema.module.eval()
    return InferenceRuntime(
        config=config_payload,
        contract=contract,
        dataset=dataset,
        model=model,
        depth_ema=depth_ema,
        device=device,
        checkpoint_path=resolved_checkpoint_path,
        checkpoint_metadata={
            "epoch": int(checkpoint.get("epoch", 0)),
            "global_step": int(checkpoint.get("global_step", 0)),
            "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        },
        output_dir=output_dir,
        dataset_path=dataset_path,
        contract_path=contract_path,
    )


def export_predictions(
    config: Mapping[str, Any],
    *,
    split: str = DEFAULT_EXPORT_SPLIT,
    checkpoint_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    runtime = load_inference_runtime(
        config,
        checkpoint_path=checkpoint_path,
        device_name=device_name,
    )
    dataset = runtime.dataset
    selected_indices = _resolve_export_split_indices(
        dataset,
        split=split,
        val_fraction=float(runtime.config.get("val_fraction", 0.2)),
    )
    if not selected_indices:
        raise ValueError(f"No dataset rows were selected for split `{split}`.")

    resolved_batch_size = int(batch_size or runtime.config.get("batch_size", 8))
    if resolved_batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {resolved_batch_size}.")

    split_name = str(split).strip().lower()
    resolved_artifact_path = (
        Path(artifact_path).expanduser().resolve()
        if artifact_path is not None
        else (runtime.output_dir / DEFAULT_ARTIFACT_NAME_TEMPLATE.format(split=split_name)).resolve()
    )
    resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    episodes: list[dict[str, Any]] = []
    active_episode_index: int | None = None
    active_rows: list[dict[str, Any]] = []
    axis_key_name: str | None = None

    def _flush_episode() -> None:
        nonlocal active_episode_index, active_rows
        if active_episode_index is None:
            return
        scene_points = np.asarray(
            dataset.static_point_cloud_data[runtime.model.scene_points_key][active_episode_index],
            dtype=np.float32,
        )
        stacked = _stack_row_dicts(active_rows)
        episodes.append(
            {
                "episode_index": int(active_episode_index),
                "dataset_indices": stacked.pop("dataset_index").astype(np.int64, copy=False),
                "scene_points": scene_points,
                **stacked,
            }
        )
        active_episode_index = None
        active_rows = []

    num_batches = (len(selected_indices) + resolved_batch_size - 1) // resolved_batch_size
    batch_iterator = tqdm(
        range(0, len(selected_indices), resolved_batch_size),
        total=num_batches,
        desc=f"inference[{split_name}]",
        disable=not bool(show_progress),
    )

    with torch.inference_mode():
        for batch_start in batch_iterator:
            batch_indices = selected_indices[batch_start : batch_start + resolved_batch_size]
            raw_batch = dataset.collate_fn([dataset[int(dataset_index)] for dataset_index in batch_indices])
            batch = _move_to_device(raw_batch, runtime.device)
            predictions = runtime.model.sample_one_step(
                batch["obs_dict"],
                ema_depth_encoder=runtime.depth_ema.module,
            )

            prediction_dict = raw_batch["prediction"]
            axis_key_name = axis_key_name or _resolve_axis_key(
                prediction_dict,
                runtime.model.motion_or_force_axis_candidates,
            )
            predicted_contact = {
                runtime.model.force_dimension_key: predictions["predicted_force_dimension"],
                axis_key_name: predictions["predicted_motion_or_force_axis"],
                runtime.model.sensed_force_key: predictions["predicted_sensed_force"],
                runtime.model.sensed_moment_key: predictions["predicted_sensed_moment"],
            }
            target_contact = {
                runtime.model.force_dimension_key: prediction_dict[runtime.model.force_dimension_key][:, 0],
                axis_key_name: prediction_dict[axis_key_name][:, 0, :],
                runtime.model.sensed_force_key: prediction_dict[runtime.model.sensed_force_key][:, 0, :],
                runtime.model.sensed_moment_key: prediction_dict[runtime.model.sensed_moment_key][:, 0, :],
            }
            if dataset.normalizer is not None:
                predicted_contact = dataset.normalizer.denormalize_modal_dict(predicted_contact)
                target_contact = dataset.normalizer.denormalize_modal_dict(target_contact)

            for row_index, dataset_index in enumerate(batch_indices):
                episode_index = int(dataset.get_episode(int(dataset_index)))
                if active_episode_index is None:
                    active_episode_index = episode_index
                if episode_index != active_episode_index:
                    _flush_episode()
                    active_episode_index = episode_index

                active_rows.append(
                    {
                        "dataset_index": np.asarray(int(dataset_index), dtype=np.int64),
                        "predicted_depth_points": _to_numpy(predictions["predicted_depth_points"][row_index]).astype(
                            np.float32,
                            copy=False,
                        ),
                        "target_depth_points": _to_numpy(prediction_dict[runtime.model.depth_key][row_index, 0]).astype(
                            np.float32,
                            copy=False,
                        ),
                        "depth_mask": _to_numpy(
                            prediction_dict[f"{runtime.model.depth_key}_mask"][row_index, 0]
                        ).astype(bool, copy=False),
                        "predicted_force_dimension": _to_numpy(
                            predicted_contact[runtime.model.force_dimension_key][row_index]
                        ).astype(np.int64, copy=False),
                        "predicted_force_dimension_logits": _to_numpy(
                            predictions["predicted_force_dimension_logits"][row_index]
                        ).astype(np.float32, copy=False),
                        "target_force_dimension": _to_numpy(
                            target_contact[runtime.model.force_dimension_key][row_index]
                        ).astype(np.int64, copy=False),
                        "predicted_motion_or_force_axis": _to_numpy(predicted_contact[axis_key_name][row_index]).astype(
                            np.float32,
                            copy=False,
                        ),
                        "target_motion_or_force_axis": _to_numpy(target_contact[axis_key_name][row_index]).astype(
                            np.float32,
                            copy=False,
                        ),
                        "predicted_sensed_force": _to_numpy(
                            predicted_contact[runtime.model.sensed_force_key][row_index]
                        ).astype(np.float32, copy=False),
                        "target_sensed_force": _to_numpy(
                            target_contact[runtime.model.sensed_force_key][row_index]
                        ).astype(np.float32, copy=False),
                        "predicted_sensed_moment": _to_numpy(
                            predicted_contact[runtime.model.sensed_moment_key][row_index]
                        ).astype(np.float32, copy=False),
                        "target_sensed_moment": _to_numpy(
                            target_contact[runtime.model.sensed_moment_key][row_index]
                        ).astype(np.float32, copy=False),
                    }
                )

    _flush_episode()
    if axis_key_name is None:
        raise RuntimeError("Failed to resolve the motion/force axis key from the prediction targets.")

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "metadata": {
            "split": split_name,
            "predictor_type": PREDICTOR_TYPE,
            "batch_size": int(resolved_batch_size),
            "dataset_path": str(runtime.dataset_path),
            "contract_path": str(runtime.contract_path),
            "output_dir": str(runtime.output_dir),
            "checkpoint_path": str(runtime.checkpoint_path),
            "checkpoint_epoch": int(runtime.checkpoint_metadata["epoch"]),
            "checkpoint_global_step": int(runtime.checkpoint_metadata["global_step"]),
            "checkpoint_best_val_loss": float(runtime.checkpoint_metadata["best_val_loss"]),
            "selected_indices": np.asarray(selected_indices, dtype=np.int64),
        },
        "keys": {
            "depth_key": runtime.model.depth_key,
            "scene_points_key": runtime.model.scene_points_key,
            "force_dimension_key": runtime.model.force_dimension_key,
            "motion_or_force_axis_key": axis_key_name,
            "sensed_force_key": runtime.model.sensed_force_key,
            "sensed_moment_key": runtime.model.sensed_moment_key,
        },
        "episodes": episodes,
    }
    np.save(resolved_artifact_path, artifact, allow_pickle=True)
    artifact["artifact_path"] = str(resolved_artifact_path)
    return artifact


def load_prediction_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    loaded = np.load(artifact_path, allow_pickle=True)
    if not isinstance(loaded, np.ndarray) or loaded.shape != ():
        raise ValueError(f"Prediction artifact `{artifact_path}` must be a scalar object array.")
    artifact = loaded.item()
    if not isinstance(artifact, dict):
        raise ValueError(f"Prediction artifact `{artifact_path}` must contain a dictionary payload.")
    if "episodes" not in artifact or "metadata" not in artifact or "keys" not in artifact:
        raise ValueError(f"Prediction artifact `{artifact_path}` is missing required top-level fields.")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic one-step CRWM predictions for a dataset split.")
    parser.add_argument("--config", required=True, type=str, help="Path to the CRWM training config YAML.")
    parser.add_argument(
        "--split",
        default=DEFAULT_EXPORT_SPLIT,
        choices=sorted(SUPPORTED_SPLITS),
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=str,
        help="Optional checkpoint override. Defaults to `<output_dir>/best.pt` from the config.",
    )
    parser.add_argument(
        "--artifact-path",
        default=None,
        type=str,
        help="Optional output `.npy` path. Defaults to `<output_dir>/one_step_predictions_<split>.npy`.",
    )
    args = parser.parse_args()
    artifact = export_predictions(
        _load_yaml(args.config),
        split=args.split,
        checkpoint_path=args.checkpoint_path,
        artifact_path=args.artifact_path,
        show_progress=True,
    )
    print(yaml.safe_dump({"artifact_path": artifact["artifact_path"]}, sort_keys=False).strip())


if __name__ == "__main__":
    main()
