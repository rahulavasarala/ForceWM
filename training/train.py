from __future__ import annotations

import argparse
import contextlib
import copy
import io
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

if __package__:
    from .crwm_model import CRWMModel
else:
    from crwm_model import CRWMModel

if TYPE_CHECKING:
    if __package__:
        from .dataset import MultiModalDataset
    else:
        from dataset import MultiModalDataset


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file `{config_path}` must parse to a dictionary.")
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(inner_value, device) for key, inner_value in value.items()}
    return value


def _contract_uses_normalization(contract: Mapping[str, Any]) -> bool:
    key_entries = contract.get("robot", {}).get("data_loader", {}).get("keys", [])
    if not isinstance(key_entries, list):
        return False
    for entry in key_entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        _, key_cfg = next(iter(entry.items()))
        if isinstance(key_cfg, dict) and bool(key_cfg.get("normalize", False)):
            return True
    return False


def ensure_normalizer(dataset_path: str | Path, contract_path: str | Path) -> Path | None:
    dataset_path = Path(dataset_path).expanduser().resolve()
    contract = _load_yaml(contract_path)
    if not _contract_uses_normalization(contract):
        return None

    normalizer_path = dataset_path / "normalizer.npy"
    if normalizer_path.exists():
        return normalizer_path
    if __package__:
        from .normalizer import build_normalizer as _build_normalizer
    else:
        from normalizer import build_normalizer as _build_normalizer
    return _build_normalizer(dataset_path, contract_path)


def _enforce_prediction_window_is_one(contract: Mapping[str, Any]) -> None:
    prediction_cfg = contract.get("robot", {}).get("data_loader", {}).get("prediction", {})
    if not isinstance(prediction_cfg, dict):
        raise ValueError("`robot.data_loader.prediction` must be a dictionary.")
    window = int(prediction_cfg.get("window", 0))
    if window != 1:
        raise ValueError(
            f"CRWM v1 requires `robot.data_loader.prediction.window == 1`, but received `{window}`."
        )


def _resolve_split_indices(dataset: MultiModalDataset, val_fraction: float) -> tuple[list[int], list[int]]:
    num_episodes = len(dataset.episode_ends)
    all_indices = list(range(len(dataset)))
    if num_episodes <= 1:
        split_index = max(1, min(len(all_indices) - 1, int(round(len(all_indices) * (1.0 - val_fraction)))))
        return all_indices[:split_index], all_indices[split_index:]

    episode_starts = [0] + [int(value) + 1 for value in dataset.episode_ends[:-1]]
    episode_ranges = [
        list(range(int(start), int(end) + 1))
        for start, end in zip(episode_starts, dataset.episode_ends.tolist(), strict=True)
    ]
    num_val_episodes = max(1, min(num_episodes - 1, int(round(num_episodes * val_fraction))))
    train_ranges = episode_ranges[: num_episodes - num_val_episodes]
    val_ranges = episode_ranges[num_episodes - num_val_episodes :]
    train_indices = [idx for episode_indices in train_ranges for idx in episode_indices]
    val_indices = [idx for episode_indices in val_ranges for idx in episode_indices]
    return train_indices, val_indices


def _resolve_depth_and_scene_keys(dataset: MultiModalDataset) -> tuple[str, str]:
    if len(dataset.point_cloud_keys) != 1:
        raise ValueError(
            "CRWM v1 expects exactly one dynamic depth key in `robot.data_loader.keys`, "
            f"but found {dataset.point_cloud_keys}."
        )
    if len(dataset.static_point_cloud_keys) != 1:
        raise ValueError(
            "CRWM v1 expects exactly one `scene_points` key in `robot.data_loader.keys`, "
            f"but found {dataset.static_point_cloud_keys}."
        )
    scene_points_key = dataset.static_point_cloud_keys[0]
    return dataset.point_cloud_keys[0], scene_points_key


def _infer_num_depth_points(dataset: MultiModalDataset, depth_key: str) -> int:
    sample = dataset[0]
    return int(sample["prediction"][depth_key].shape[1])


def _build_dataloaders(
    dataset: MultiModalDataset,
    *,
    batch_size: int,
    num_workers: int,
    train_indices: list[int],
    val_indices: list[int],
) -> tuple[DataLoader, DataLoader]:
    if not train_indices or not val_indices:
        raise ValueError(
            f"Unable to create a non-empty train/validation split. train={len(train_indices)} val={len(val_indices)}"
        )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return train_loader, val_loader


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_scale: float = 0.1,
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_scale = float(min_lr_scale)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = 0.0 if total_steps == warmup_steps else (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@dataclass
class EMAConfig:
    decay: float = 0.999
    update_after_step: int = 0
    update_every: int = 1


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = ""
    entity: str | None = None
    run_name: str | None = None


class ModuleEMA:
    def __init__(self, module: nn.Module, config: EMAConfig) -> None:
        self.module = copy.deepcopy(module).eval()
        _set_module_trainable(self.module, False)
        self.config = config

    def maybe_update(self, source_module: nn.Module, step: int) -> None:
        if step < self.config.update_after_step:
            return
        if self.config.update_every <= 0:
            return
        if step % self.config.update_every != 0:
            return
        decay = float(self.config.decay)
        with torch.no_grad():
            ema_state = dict(self.module.named_parameters())
            source_state = dict(source_module.named_parameters())
            for name, parameter in source_state.items():
                ema_state[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
            ema_buffers = dict(self.module.named_buffers())
            source_buffers = dict(source_module.named_buffers())
            for name, buffer in source_buffers.items():
                ema_buffers[name].copy_(buffer.detach())

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state_dict)


def _resolve_ema_config(config: Mapping[str, Any] | None = None) -> EMAConfig:
    config = dict(config or {})
    return EMAConfig(
        decay=float(config.get("decay", 0.999)),
        update_after_step=int(config.get("update_after_step", 0)),
        update_every=int(config.get("update_every", 1)),
    )


class WandbLogger:
    def __init__(
        self,
        *,
        config: WandbConfig,
        run_config: Mapping[str, Any],
        output_dir: Path,
    ) -> None:
        self.enabled = bool(config.enabled)
        self._wandb: Any | None = None
        if not self.enabled:
            return

        project = str(config.project).strip()
        if not project:
            raise ValueError("`wandb.project` is required when `wandb.enabled` is true.")

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging is enabled but the `wandb` package is not installed in this environment."
            ) from exc

        run_name = str(config.run_name).strip() if config.run_name is not None else output_dir.name
        entity = str(config.entity).strip() if config.entity is not None else ""
        self._wandb = wandb
        wandb.init(
            project=project,
            entity=entity or None,
            name=run_name or output_dir.name,
            config=copy.deepcopy(dict(run_config)),
            dir=str(output_dir),
        )

    def log(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(dict(metrics), step=step)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()


def _resolve_wandb_config(config: Mapping[str, Any] | None) -> WandbConfig:
    config = dict(config or {})
    entity = config.get("entity")
    run_name = config.get("run_name")
    return WandbConfig(
        enabled=bool(config.get("enabled", False)),
        project=str(config.get("project", "")),
        entity=None if entity in {None, ""} else str(entity),
        run_name=None if run_name in {None, ""} else str(run_name),
    )


def _aggregate_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    return {
        key: float(sum(row[key] for row in metric_rows) / len(metric_rows))
        for key in keys
    }


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    cpu_tensor = detached.to(device="cpu")
    numel = int(cpu_tensor.numel())
    summary: dict[str, Any] = {
        "shape": [int(dim) for dim in cpu_tensor.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": numel,
    }

    if numel == 1:
        summary["value"] = cpu_tensor.reshape(-1)[0].item()
        return summary

    if numel <= 32:
        summary["values"] = cpu_tensor.tolist()
        return summary

    flattened = cpu_tensor.reshape(-1)
    summary["preview"] = flattened[:8].tolist()
    stats_tensor = flattened.to(dtype=torch.float32)
    summary["stats"] = {
        "min": float(stats_tensor.min().item()),
        "max": float(stats_tensor.max().item()),
        "mean": float(stats_tensor.mean().item()),
    }
    return summary


def _mapping_tensor_summary(mapping: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        str(key): _tensor_summary(value)
        for key, value in mapping.items()
    }


def _build_normalization_report(dataset: MultiModalDataset) -> dict[str, Any]:
    normalized_keys = [str(key_name) for key_name in dataset.normalized_lowdim_keys]
    normalizer_path = dataset.dataset_path / "normalizer.npy"
    report: dict[str, Any] = {
        "normalizer_path": str(normalizer_path),
        "keys": {},
    }
    if not normalized_keys:
        return report
    if dataset.normalizer is None:
        raise RuntimeError(
            "Normalization is enabled for at least one loader key, but no dataset normalizer is loaded."
        )

    key_reports: dict[str, Any] = {}
    for key_name in normalized_keys:
        key_stats = dataset.normalizer.require_key(key_name)
        key_reports[key_name] = {
            "representation": str(key_stats["representation"]),
            "feature_shape": [int(dim) for dim in tuple(key_stats["feature_shape"])],
            "count": int(key_stats["count"]),
            "mean": _tensor_summary(torch.from_numpy(np.asarray(key_stats["mean"]).copy())),
            "std": _tensor_summary(torch.from_numpy(np.asarray(key_stats["std"]).copy())),
        }
    report["keys"] = key_reports
    return report


def _format_shape(shape: list[int]) -> str:
    return "[" + ", ".join(str(int(dim)) for dim in shape) + "]"


def _format_float_list(values: list[Any]) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def _format_mapping_shapes(title: str, mapping: Mapping[str, Any]) -> list[str]:
    lines = [f"{title}:"]
    for key_name, summary in mapping.items():
        shape = summary.get("shape", [])
        lines.append(f"  - {key_name}: shape={_format_shape(shape)}")
    return lines


def _format_loss_target_shapes(title: str, mapping: Mapping[str, Any]) -> list[str]:
    lines = [f"{title}:"]
    for loss_name, loss_mapping in mapping.items():
        lines.append(f"  {loss_name}:")
        for key_name, summary in loss_mapping.items():
            shape = summary.get("shape", [])
            lines.append(f"    - {key_name}: shape={_format_shape(shape)}")
    return lines


def _format_parameter_summary(name: str, summary: Mapping[str, Any]) -> str:
    total_params = int(summary["total_params"])
    trainable_params = int(summary["trainable_params"])
    return f"  - {name}: total_params={total_params} trainable_params={trainable_params}"


def _format_normalization_preflight(report: Mapping[str, Any]) -> list[str]:
    lines = ["Normalization:"]
    key_reports = report.get("keys", {})
    if not key_reports:
        lines.append("  - none")
        return lines

    for key_name, key_report in key_reports.items():
        mean_values = list(key_report["mean"].get("values", []))
        std_values = list(key_report["std"].get("values", []))
        lines.append(
            f"  - {key_name}: mean={_format_float_list(mean_values)} std={_format_float_list(std_values)}"
        )
    return lines


def _format_startup_preflight_report(report: Mapping[str, Any]) -> str:
    batch_source = report["batch_source"]
    model_report = report["model"]
    component_reports = model_report["components"]
    lines = [
        "Startup Preflight:",
        f"  stage: {report['stage']}",
        f"  device: {report['device']}",
        f"  depth_encoder_trainable: {report['depth_encoder_trainable']}",
        f"  contact_encoder_trainable: {report['contact_encoder_trainable']}",
        (
            "  batch_source: "
            f"split={batch_source['split']} "
            f"indices={batch_source['indices']} "
            f"configured_batch_size={batch_source['configured_batch_size']} "
            f"effective_batch_size={batch_source['effective_batch_size']}"
        ),
    ]
    lines.extend(_format_mapping_shapes("Inputs / obs_dict", report["inputs"]["obs_dict"]))
    lines.extend(_format_mapping_shapes("Prediction Targets", report["inputs"]["prediction"]))
    lines.append("Model Components:")
    lines.append(_format_parameter_summary("full_model", model_report["full"]))
    lines.append(_format_parameter_summary("depth_encoder", component_reports["depth_encoder"]))
    lines.append(_format_parameter_summary("contact_encoder", component_reports["contact_encoder"]))
    lines.append(_format_parameter_summary("action_encoder", component_reports["action_encoder"]))
    lines.append(
        _format_parameter_summary(
            "conditioning_stack",
            component_reports["conditioning_stack"]["total"],
        )
    )
    lines.append(_format_parameter_summary("flow_model", component_reports["flow_model"]))
    lines.append(_format_parameter_summary("contact_decoder", component_reports["contact_decoder"]))
    lines.append(_format_parameter_summary("depth_decoder", component_reports["depth_decoder"]))
    lines.extend(_format_mapping_shapes("Outputs", report["outputs"]))
    lines.extend(_format_loss_target_shapes("Loss Targets", report["loss_targets"]))
    lines.extend(_format_normalization_preflight(report["normalization"]))
    return "\n".join(lines)


def _parameter_summary_from_parameters(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    total_params = int(sum(parameter.numel() for parameter in parameters))
    trainable_parameters = [parameter for parameter in parameters if parameter.requires_grad]
    trainable_params = int(sum(parameter.numel() for parameter in trainable_parameters))
    parameter_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in parameters))
    trainable_parameter_bytes = int(
        sum(parameter.numel() * parameter.element_size() for parameter in trainable_parameters)
    )
    mib_denominator = float(1024 * 1024)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "parameter_bytes": parameter_bytes,
        "parameter_mib": float(parameter_bytes / mib_denominator),
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "trainable_parameter_mib": float(trainable_parameter_bytes / mib_denominator),
    }


def _module_parameter_summary(module: nn.Module) -> dict[str, Any]:
    return _parameter_summary_from_parameters(list(module.parameters()))


def _combine_parameter_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_params = int(sum(summary["total_params"] for summary in summaries))
    trainable_params = int(sum(summary["trainable_params"] for summary in summaries))
    parameter_bytes = int(sum(summary["parameter_bytes"] for summary in summaries))
    trainable_parameter_bytes = int(sum(summary["trainable_parameter_bytes"] for summary in summaries))
    mib_denominator = float(1024 * 1024)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "parameter_bytes": parameter_bytes,
        "parameter_mib": float(parameter_bytes / mib_denominator),
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "trainable_parameter_mib": float(trainable_parameter_bytes / mib_denominator),
    }


def _build_model_size_report(model: CRWMModel) -> dict[str, Any]:
    conditioning_components = {
        "action_token_projection": _module_parameter_summary(model.action_token_projection),
        "depth_token_projection": _module_parameter_summary(model.depth_token_projection),
        "contact_token_projection": _module_parameter_summary(model.contact_token_projection),
        "scene_token_projection": _module_parameter_summary(model.scene_token_projection),
        "modality_embeddings": _parameter_summary_from_parameters([model.modality_embeddings]),
        "temporal_embeddings": _module_parameter_summary(model.temporal_embeddings),
    }
    conditioning_total = _combine_parameter_summaries(list(conditioning_components.values()))

    component_reports = {
        "depth_encoder": _module_parameter_summary(model.depth_encoder),
        "contact_encoder": _module_parameter_summary(model.contact_encoder),
        "action_encoder": _module_parameter_summary(model.action_encoder),
        "conditioning_stack": {
            "total": conditioning_total,
            "components": conditioning_components,
        },
        "flow_model": _module_parameter_summary(model.flow_model),
        "contact_decoder": _module_parameter_summary(model.contact_decoder),
        "depth_decoder": _module_parameter_summary(model.depth_decoder),
    }
    full_model = _module_parameter_summary(model)
    component_summaries = [
        component_reports["depth_encoder"],
        component_reports["contact_encoder"],
        component_reports["action_encoder"],
        conditioning_total,
        component_reports["flow_model"],
        component_reports["contact_decoder"],
        component_reports["depth_decoder"],
    ]
    component_totals = _combine_parameter_summaries(component_summaries)
    return {
        "full": full_model,
        "components": component_reports,
        "parameter_accounting": {
            "component_total_params": component_totals["total_params"],
            "component_trainable_params": component_totals["trainable_params"],
            "matches_full_model": (
                component_totals["total_params"] == full_model["total_params"]
                and component_totals["trainable_params"] == full_model["trainable_params"]
            ),
        },
    }


def _build_startup_batch(
    dataset: MultiModalDataset,
    *,
    train_indices: list[int],
    batch_size: int,
) -> tuple[dict[str, Any], list[int]]:
    if not train_indices:
        raise ValueError("Cannot build a startup batch because the training split is empty.")
    effective_batch_size = min(max(1, int(batch_size)), len(train_indices))
    selected_indices = [int(index) for index in train_indices[:effective_batch_size]]
    batch = dataset.collate_fn([dataset[index] for index in selected_indices])
    return batch, selected_indices


def _first_present_tensor(mapping: Mapping[str, torch.Tensor], candidates: tuple[str, ...], source_name: str) -> torch.Tensor:
    for key_name in candidates:
        if key_name in mapping:
            return mapping[key_name]
    raise KeyError(f"Missing required key for {source_name}. Tried {list(candidates)}.")


def _build_startup_loss_target_report(
    model: CRWMModel,
    batch: Mapping[str, Mapping[str, torch.Tensor]],
    outputs: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    prediction_dict = batch["prediction"]
    depth_mask_key = f"{model.depth_key}_mask"
    return {
        "latent_delta_loss": {
            "predicted_delta": _tensor_summary(outputs["predicted_delta"]),
            "latent_target": _tensor_summary(outputs["latent_target"]),
        },
        "depth_recon_loss": {
            "predicted_depth_points": _tensor_summary(outputs["predicted_depth_points"]),
            "future_depth_target": _tensor_summary(prediction_dict[model.depth_key][:, 0, :, :]),
            "future_depth_target_mask": _tensor_summary(prediction_dict[depth_mask_key][:, 0, :]),
        },
        "contact_recon_loss": {
            "predicted_force_dimension_logits": _tensor_summary(outputs["predicted_force_dimension_logits"]),
            "target_force_dimension": _tensor_summary(prediction_dict[model.force_dimension_key][:, 0]),
            "predicted_motion_or_force_axis": _tensor_summary(outputs["predicted_motion_or_force_axis"]),
            "target_motion_or_force_axis": _tensor_summary(
                _first_present_tensor(
                    prediction_dict,
                    model.motion_or_force_axis_candidates,
                    source_name="startup prediction motion/force axis",
                )[:, 0, :]
            ),
            "predicted_sensed_force": _tensor_summary(outputs["predicted_sensed_force"]),
            "target_sensed_force": _tensor_summary(prediction_dict[model.sensed_force_key][:, 0, :]),
            "predicted_sensed_moment": _tensor_summary(outputs["predicted_sensed_moment"]),
            "target_sensed_moment": _tensor_summary(prediction_dict[model.sensed_moment_key][:, 0, :]),
        },
    }


def _build_startup_report(
    *,
    model: CRWMModel,
    dataset: MultiModalDataset,
    train_indices: list[int],
    batch_size: int,
    depth_ema: ModuleEMA,
    contact_ema: ModuleEMA,
    device: torch.device,
    seed: int,
    depth_encoder_trainable: bool,
    contact_encoder_trainable: bool,
) -> dict[str, Any]:
    raw_batch, selected_indices = _build_startup_batch(
        dataset,
        train_indices=train_indices,
        batch_size=batch_size,
    )
    batch = _move_to_device(raw_batch, device)

    previous_model_training = bool(model.training)
    previous_depth_ema_training = bool(depth_ema.module.training)
    previous_contact_ema_training = bool(contact_ema.module.training)
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        try:
            model.eval()
            depth_ema.module.eval()
            contact_ema.module.eval()
            with torch.no_grad():
                outputs = model(
                    batch,
                    ema_depth_encoder=depth_ema.module,
                    ema_contact_encoder=contact_ema.module,
                )
        finally:
            model.train(previous_model_training)
            depth_ema.module.train(previous_depth_ema_training)
            contact_ema.module.train(previous_contact_ema_training)

    return {
        "stage": "startup_dummy_pass",
        "device": str(device),
        "depth_encoder_trainable": bool(depth_encoder_trainable),
        "contact_encoder_trainable": bool(contact_encoder_trainable),
        "batch_source": {
            "split": "train",
            "indices": selected_indices,
            "configured_batch_size": int(batch_size),
            "effective_batch_size": len(selected_indices),
        },
        "inputs": {
            "obs_dict": _mapping_tensor_summary(batch["obs_dict"]),
            "prediction": _mapping_tensor_summary(batch["prediction"]),
        },
        "normalization": _build_normalization_report(dataset),
        "model": _build_model_size_report(model),
        "outputs": _mapping_tensor_summary(outputs),
        "loss_targets": _build_startup_loss_target_report(model, batch, outputs),
    }


def _replay_captured_startup_streams(stdout_buffer: io.StringIO, stderr_buffer: io.StringIO) -> None:
    captured_stdout = stdout_buffer.getvalue()
    captured_stderr = stderr_buffer.getvalue()
    if captured_stdout:
        sys.stderr.write(captured_stdout)
    if captured_stderr:
        sys.stderr.write(captured_stderr)
    if captured_stdout or captured_stderr:
        sys.stderr.flush()


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _run_epoch(
    *,
    model: CRWMModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    depth_ema: ModuleEMA,
    contact_ema: ModuleEMA,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    gradient_clip_norm: float | None,
    global_step: int,
    training: bool,
    log_every: int,
    on_train_step: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[dict[str, float], int]:
    metric_rows: list[dict[str, float]] = []
    if training:
        model.train()
        if not any(parameter.requires_grad for parameter in model.depth_encoder.parameters()):
            model.depth_encoder.eval()
        if not any(parameter.requires_grad for parameter in model.contact_encoder.parameters()):
            model.contact_encoder.eval()
        depth_ema.module.eval()
        contact_ema.module.eval()
    else:
        model.eval()

    autocast_enabled = use_amp and device.type == "cuda"

    for step_index, raw_batch in enumerate(data_loader, start=1):
        batch = _move_to_device(raw_batch, device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                outputs = model(
                    batch,
                    ema_depth_encoder=depth_ema.module,
                    ema_contact_encoder=contact_ema.module,
                )
                loss = outputs["loss"]

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if autocast_enabled and scaler is not None:
                scaler.scale(loss).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            global_step += 1
            if scheduler is not None:
                scheduler.step()
            depth_ema.maybe_update(model.depth_encoder, global_step)
            contact_ema.maybe_update(model.contact_encoder, global_step)

        metric_row = {
            "loss": float(outputs["loss"].detach().cpu().item()),
            "latent_delta_loss": float(outputs["latent_delta_loss"].detach().cpu().item()),
            "depth_recon_loss": float(outputs["depth_recon_loss"].detach().cpu().item()),
            "contact_recon_loss": float(outputs["contact_recon_loss"].detach().cpu().item()),
            "contact_force_dimension_ce": float(outputs["contact_force_dimension_ce"].detach().cpu().item()),
            "contact_motion_axis_mse": float(outputs["contact_motion_axis_mse"].detach().cpu().item()),
            "contact_sensed_force_mse": float(outputs["contact_sensed_force_mse"].detach().cpu().item()),
            "contact_sensed_moment_mse": float(outputs["contact_sensed_moment_mse"].detach().cpu().item()),
            "force_dimension_accuracy": float(outputs["force_dimension_accuracy"].detach().cpu().item()),
            "ee_position_mse": float(outputs["ee_position_mse"].detach().cpu().item()),
        }
        metric_rows.append(metric_row)
        if training and on_train_step is not None:
            on_train_step(global_step, metric_row)

        if log_every > 0 and step_index % log_every == 0:
            phase = "train" if training else "val"
            print(
                f"[{phase}] step={step_index} "
                f"loss={metric_row['loss']:.4f} "
                f"delta={metric_row['latent_delta_loss']:.4f} "
                f"depth={metric_row['depth_recon_loss']:.4f} "
                f"contact={metric_row['contact_recon_loss']:.4f} "
                f"contact_ce={metric_row['contact_force_dimension_ce']:.4f} "
                f"contact_axis={metric_row['contact_motion_axis_mse']:.4f} "
                f"contact_force={metric_row['contact_sensed_force_mse']:.4f} "
                f"contact_moment={metric_row['contact_sensed_moment_mse']:.4f} "
                f"force_acc={metric_row['force_dimension_accuracy']:.4f} "
                f"ee_mse={metric_row['ee_position_mse']:.4f}"
            )

    return _aggregate_metrics(metric_rows), global_step


def _save_checkpoint(
    path: Path,
    *,
    model: CRWMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    depth_ema: ModuleEMA,
    contact_ema: ModuleEMA,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_val_loss": float(best_val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "depth_ema_state_dict": depth_ema.state_dict(),
            "contact_ema_state_dict": contact_ema.state_dict(),
            "config": dict(config),
        },
        path,
    )


def _load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: CRWMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    depth_ema: ModuleEMA,
    contact_ema: ModuleEMA,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    depth_ema.load_state_dict(checkpoint["depth_ema_state_dict"])
    contact_ema.load_state_dict(checkpoint["contact_ema_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
    )


def train(
    config: Mapping[str, Any],
    *,
    on_epoch_end: Callable[[int, dict[str, float], Path], None] | None = None,
) -> dict[str, float]:
    dataset_path = Path(config["dataset_path"]).expanduser().resolve()
    contract_path = Path(config["universal_contract"]).expanduser().resolve()
    output_dir = Path(config.get("output_dir", "training_runs/crwm")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = _load_yaml(contract_path)
    _enforce_prediction_window_is_one(contract)
    ensure_normalizer(dataset_path, contract_path)

    if __package__:
        from .dataset import MultiModalDataset
    else:
        from dataset import MultiModalDataset

    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    device = _select_device(config.get("device"))

    dataset = MultiModalDataset(
        dataset_path,
        universal_contract=contract_path,
        pointcloud_cache_size=int(config.get("pointcloud_cache_size", 4)),
    )
    val_fraction = float(config.get("val_fraction", 0.2))
    train_indices, val_indices = _resolve_split_indices(dataset, val_fraction)
    depth_key, scene_points_key = _resolve_depth_and_scene_keys(dataset)
    model_cfg = dict(config.get("model", {}))
    depth_encoder_cfg = dict(model_cfg.get("depth_encoder", {}))
    depth_encoder_type = str(depth_encoder_cfg.get("type", "dummy")).lower()
    if depth_encoder_type not in {"dummy", "ptv3"}:
        raise ValueError(f"`model.depth_encoder.type` must be `dummy` or `ptv3`, got `{depth_encoder_type}`.")
    if depth_encoder_type == "ptv3" and device.type != "cuda":
        raise ValueError(
            "Concerto/PTv3 training requires a CUDA device. "
            f"Resolved device `{device}` is unsupported for `model.depth_encoder.type: ptv3`."
        )

    num_depth_points = int(model_cfg.get("num_depth_points", _infer_num_depth_points(dataset, depth_key)))
    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    train_loader, val_loader = _build_dataloaders(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    epochs = int(config.get("epochs", 50))
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    gradient_clip_norm = config.get("gradient_clip_norm")
    gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
    log_every = int(config.get("log_every", 0))
    val_every_epochs = int(config.get("val_every_epochs", 1))
    if val_every_epochs <= 0:
        raise ValueError(f"`val_every_epochs` must be positive, got `{val_every_epochs}`.")
    depth_encoder_trainable_epochs = int(config.get("depth_encoder_trainable_epochs", 10))
    contact_encoder_trainable_epochs = int(config.get("contact_encoder_trainable_epochs", 10))
    wandb_config = _resolve_wandb_config(config.get("wandb"))

    startup_stdout = io.StringIO()
    startup_stderr = io.StringIO()
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    startup_report: dict[str, Any]
    try:
        with contextlib.redirect_stdout(startup_stdout), contextlib.redirect_stderr(startup_stderr):
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

            optimizer_cfg = dict(config.get("optimizer", {}))
            optimizer = AdamW(
                model.parameters(),
                lr=float(optimizer_cfg.get("lr", 1e-4)),
                weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
            )
            scheduler_cfg = dict(config.get("scheduler", {}))
            scheduler = _build_scheduler(
                optimizer,
                total_steps=max(1, epochs * len(train_loader)),
                warmup_steps=int(scheduler_cfg.get("warmup_steps", 0)),
                min_lr_scale=float(scheduler_cfg.get("min_lr_scale", 0.1)),
            )

            depth_ema = ModuleEMA(model.depth_encoder, _resolve_ema_config(config.get("depth_encoder_ema")))
            contact_ema = ModuleEMA(model.contact_encoder, _resolve_ema_config(config.get("contact_encoder_ema")))

            resume_from = config.get("resume_from")
            if resume_from:
                start_epoch, global_step, best_val_loss = _load_checkpoint(
                    resume_from,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    depth_ema=depth_ema,
                    contact_ema=contact_ema,
                    device=device,
                )

            initial_epoch = start_epoch + 1
            initial_depth_encoder_trainable = initial_epoch <= depth_encoder_trainable_epochs
            initial_contact_encoder_trainable = initial_epoch <= contact_encoder_trainable_epochs
            _set_module_trainable(model.depth_encoder, initial_depth_encoder_trainable)
            _set_module_trainable(model.contact_encoder, initial_contact_encoder_trainable)
            if not initial_depth_encoder_trainable:
                model.depth_encoder.eval()
            if not initial_contact_encoder_trainable:
                model.contact_encoder.eval()

            startup_report = _build_startup_report(
                model=model,
                dataset=dataset,
                train_indices=train_indices,
                batch_size=batch_size,
                depth_ema=depth_ema,
                contact_ema=contact_ema,
                device=device,
                seed=seed,
                depth_encoder_trainable=initial_depth_encoder_trainable,
                contact_encoder_trainable=initial_contact_encoder_trainable,
            )
    except Exception:
        _replay_captured_startup_streams(startup_stdout, startup_stderr)
        raise

    print(_format_startup_preflight_report(startup_report))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    config_snapshot_path = output_dir / "config_snapshot.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    contract_snapshot_path = output_dir / "contract_snapshot.yaml"
    with contract_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(contract, handle, sort_keys=False)

    latest_checkpoint_path = output_dir / "latest.pt"
    best_checkpoint_path = output_dir / "best.pt"
    final_metrics: dict[str, float] = {}
    wandb_logger = WandbLogger(config=wandb_config, run_config=config, output_dir=output_dir)
    try:
        for epoch_index in range(start_epoch, epochs):
            current_epoch = epoch_index + 1
            depth_encoder_trainable = current_epoch <= depth_encoder_trainable_epochs
            contact_encoder_trainable = current_epoch <= contact_encoder_trainable_epochs
            val_ran = current_epoch % val_every_epochs == 0
            _set_module_trainable(model.depth_encoder, depth_encoder_trainable)
            _set_module_trainable(model.contact_encoder, contact_encoder_trainable)
            if not depth_encoder_trainable:
                model.depth_encoder.eval()
            if not contact_encoder_trainable:
                model.contact_encoder.eval()

            def _log_train_step(step: int, metrics: dict[str, float]) -> None:
                wandb_logger.log(
                    {
                        "train_step/loss": metrics["loss"],
                        "train_step/latent_delta_loss": metrics["latent_delta_loss"],
                        "train_step/depth_recon_loss": metrics["depth_recon_loss"],
                        "train_step/contact_recon_loss": metrics["contact_recon_loss"],
                        "train_step/contact_force_dimension_ce": metrics["contact_force_dimension_ce"],
                        "train_step/contact_motion_axis_mse": metrics["contact_motion_axis_mse"],
                        "train_step/contact_sensed_force_mse": metrics["contact_sensed_force_mse"],
                        "train_step/contact_sensed_moment_mse": metrics["contact_sensed_moment_mse"],
                        "train_step/force_dimension_accuracy": metrics["force_dimension_accuracy"],
                        "train_step/ee_position_mse": metrics["ee_position_mse"],
                        "trainer/epoch": float(current_epoch),
                        "trainer/global_step": float(step),
                        "trainer/lr": _current_learning_rate(optimizer),
                        "trainer/depth_encoder_trainable": float(int(depth_encoder_trainable)),
                        "trainer/contact_encoder_trainable": float(int(contact_encoder_trainable)),
                    },
                    step=step,
                )

            train_metrics, global_step = _run_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                depth_ema=depth_ema,
                contact_ema=contact_ema,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
                gradient_clip_norm=gradient_clip_norm,
                global_step=global_step,
                training=True,
                log_every=log_every,
                on_train_step=_log_train_step,
            )

            val_metrics: dict[str, float] = {}
            if val_ran:
                val_metrics, global_step = _run_epoch(
                    model=model,
                    data_loader=val_loader,
                    optimizer=None,
                    scheduler=None,
                    depth_ema=depth_ema,
                    contact_ema=contact_ema,
                    device=device,
                    use_amp=False,
                    scaler=None,
                    gradient_clip_norm=None,
                    global_step=global_step,
                    training=False,
                    log_every=0,
                )

            final_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            final_metrics["val_ran"] = float(int(val_ran))
            final_metrics["global_step"] = float(global_step)
            final_metrics["lr"] = _current_learning_rate(optimizer)
            if val_ran:
                final_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})

            wandb_epoch_metrics: dict[str, float] = {
                f"train_epoch/{key}": value for key, value in train_metrics.items()
            }
            wandb_epoch_metrics.update(
                {
                    "trainer/epoch": float(current_epoch),
                    "trainer/global_step": float(global_step),
                    "trainer/lr": _current_learning_rate(optimizer),
                    "trainer/val_ran": float(int(val_ran)),
                }
            )
            if val_ran:
                wandb_epoch_metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
            wandb_logger.log(wandb_epoch_metrics, step=global_step)

            val_loss = val_metrics.get("loss", float("nan"))
            val_force_acc = val_metrics.get("force_dimension_accuracy", float("nan"))
            train_ee_position_mse = train_metrics.get("ee_position_mse", float("nan"))
            val_ee_position_mse = val_metrics.get("ee_position_mse", float("nan"))
            print(
                f"epoch={current_epoch} "
                f"depth_encoder_trainable={int(depth_encoder_trainable)} "
                f"contact_encoder_trainable={int(contact_encoder_trainable)} "
                f"val_ran={int(val_ran)} "
                f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
                f"val_loss={val_loss:.4f} "
                f"train_ee_mse={train_ee_position_mse:.4f} "
                f"val_ee_mse={val_ee_position_mse:.4f} "
                f"train_force_acc={train_metrics.get('force_dimension_accuracy', float('nan')):.4f} "
                f"val_force_acc={val_force_acc:.4f}"
            )

            _save_checkpoint(
                latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                depth_ema=depth_ema,
                contact_ema=contact_ema,
                epoch=current_epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                config=config,
            )

            if val_ran:
                current_val_loss = float(val_metrics.get("loss", float("inf")))
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    _save_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        depth_ema=depth_ema,
                        contact_ema=contact_ema,
                        epoch=current_epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        config=config,
                    )
            if on_epoch_end is not None:
                on_epoch_end(current_epoch, final_metrics, output_dir)
    finally:
        wandb_logger.finish()

    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the contact-rich world model.")
    parser.add_argument("--config", required=True, type=str, help="Path to the CRWM training config YAML.")
    args = parser.parse_args()
    train(_load_yaml(args.config))


if __name__ == "__main__":
    main()
