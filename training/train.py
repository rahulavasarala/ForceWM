from __future__ import annotations

import argparse
import copy
import math
import random
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


def _resolve_depth_and_scene_keys(dataset: MultiModalDataset) -> tuple[str, str | None]:
    if len(dataset.point_cloud_keys) != 1:
        raise ValueError(
            "CRWM v1 expects exactly one dynamic depth key in `robot.data_loader.keys`, "
            f"but found {dataset.point_cloud_keys}."
        )
    if len(dataset.static_point_cloud_keys) > 1:
        raise ValueError(
            "CRWM v1 expects at most one `scene_points` key in `robot.data_loader.keys`, "
            f"but found {dataset.static_point_cloud_keys}."
        )
    scene_points_key = dataset.static_point_cloud_keys[0] if dataset.static_point_cloud_keys else None
    return dataset.point_cloud_keys[0], scene_points_key


def _infer_num_depth_points(dataset: MultiModalDataset, depth_key: str) -> int:
    sample = dataset[0]
    return int(sample["prediction"][depth_key].shape[1])


def _build_dataloaders(
    dataset: MultiModalDataset,
    *,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
) -> tuple[DataLoader, DataLoader]:
    train_indices, val_indices = _resolve_split_indices(dataset, val_fraction)
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


def _aggregate_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    return {
        key: float(sum(row[key] for row in metric_rows) / len(metric_rows))
        for key in keys
    }


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
) -> tuple[dict[str, float], int]:
    metric_rows: list[dict[str, float]] = []
    if training:
        model.train()
        if not any(parameter.requires_grad for parameter in model.depth_encoder.parameters()):
            model.depth_encoder.eval()
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
            "flow_loss": float(outputs["flow_loss"].detach().cpu().item()),
            "depth_recon_loss": float(outputs["depth_recon_loss"].detach().cpu().item()),
            "contact_recon_loss": float(outputs["contact_recon_loss"].detach().cpu().item()),
            "force_dimension_accuracy": float(outputs["force_dimension_accuracy"].detach().cpu().item()),
        }
        metric_rows.append(metric_row)

        if log_every > 0 and step_index % log_every == 0:
            phase = "train" if training else "val"
            print(
                f"[{phase}] step={step_index} "
                f"loss={metric_row['loss']:.4f} "
                f"flow={metric_row['flow_loss']:.4f} "
                f"depth={metric_row['depth_recon_loss']:.4f} "
                f"contact={metric_row['contact_recon_loss']:.4f} "
                f"force_acc={metric_row['force_dimension_accuracy']:.4f}"
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

    train_loader, val_loader = _build_dataloaders(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        num_workers=int(config.get("num_workers", 0)),
        val_fraction=float(config.get("val_fraction", 0.2)),
    )

    optimizer_cfg = dict(config.get("optimizer", {}))
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 50))
    scheduler_cfg = dict(config.get("scheduler", {}))
    scheduler = _build_scheduler(
        optimizer,
        total_steps=max(1, epochs * len(train_loader)),
        warmup_steps=int(scheduler_cfg.get("warmup_steps", 0)),
        min_lr_scale=float(scheduler_cfg.get("min_lr_scale", 0.1)),
    )

    depth_ema = ModuleEMA(model.depth_encoder, _resolve_ema_config(config.get("depth_encoder_ema")))
    contact_ema = ModuleEMA(model.contact_encoder, _resolve_ema_config(config.get("contact_encoder_ema")))

    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
    gradient_clip_norm = config.get("gradient_clip_norm")
    gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
    log_every = int(config.get("log_every", 0))
    depth_encoder_trainable_epochs = int(config.get("depth_encoder_trainable_epochs", 10))

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
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

    config_snapshot_path = output_dir / "config_snapshot.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)

    latest_checkpoint_path = output_dir / "latest.pt"
    best_checkpoint_path = output_dir / "best.pt"
    final_metrics: dict[str, float] = {}

    for epoch_index in range(start_epoch, epochs):
        current_epoch = epoch_index + 1
        depth_encoder_trainable = current_epoch <= depth_encoder_trainable_epochs
        _set_module_trainable(model.depth_encoder, depth_encoder_trainable)
        if not depth_encoder_trainable:
            model.depth_encoder.eval()

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
        )
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
        final_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        print(
            f"epoch={current_epoch} "
            f"depth_encoder_trainable={int(depth_encoder_trainable)} "
            f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.4f} "
            f"train_force_acc={train_metrics.get('force_dimension_accuracy', float('nan')):.4f} "
            f"val_force_acc={val_metrics.get('force_dimension_accuracy', float('nan')):.4f}"
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

    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the contact-rich world model.")
    parser.add_argument("--config", required=True, type=str, help="Path to the CRWM training config YAML.")
    args = parser.parse_args()
    train(_load_yaml(args.config))


if __name__ == "__main__":
    main()
