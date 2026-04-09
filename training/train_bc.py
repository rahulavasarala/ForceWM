from __future__ import annotations

import argparse
import math
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "PyTorch is required for BC training. Install the `pytorch` package in the runtime environment."
    ) from exc

try:
    from training.bc_policy import (
        DatasetSpec,
        build_policy_from_config,
        infer_dataset_spec,
        normalize_observation_dict,
        normalize_tensor,
    )
    from training.dataloader import (
        ForceWMParquetDataset,
        build_relative_action_targets,
        compute_action_indices,
        compute_observation_indices,
        fetch_lowdim_sequence,
        rotation_matrix_to_rot6d,
    )
except ModuleNotFoundError:
    from bc_policy import (
        DatasetSpec,
        build_policy_from_config,
        infer_dataset_spec,
        normalize_observation_dict,
        normalize_tensor,
    )
    from dataloader import (
        ForceWMParquetDataset,
        build_relative_action_targets,
        compute_action_indices,
        compute_observation_indices,
        fetch_lowdim_sequence,
        rotation_matrix_to_rot6d,
    )


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None

    def update(self, values: torch.Tensor | np.ndarray) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float64)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.reshape(-1, tensor.shape[-1])

        if self.sum is None:
            self.sum = tensor.sum(dim=0)
            self.sum_sq = (tensor * tensor).sum(dim=0)
        else:
            self.sum += tensor.sum(dim=0)
            self.sum_sq += (tensor * tensor).sum(dim=0)
        self.count += int(tensor.shape[0])

    def finalize(self) -> dict[str, torch.Tensor]:
        if self.count <= 0 or self.sum is None or self.sum_sq is None:
            raise ValueError("RunningStats has no values to finalize.")
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count) - mean.square()
        variance = torch.clamp(variance, min=1e-6)
        std = torch.sqrt(variance)
        return {"mean": mean.float().cpu(), "std": std.float().cpu()}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return payload


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_override: str | None = None) -> torch.device:
    if device_override is not None:
        requested = device_override.lower()
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_compute_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported compute dtype `{dtype_name}`. Expected one of: {sorted(dtype_map.keys())}."
        )
    return dtype_map[normalized]


def resolve_autocast_settings(
    device: torch.device,
    optimization_config: dict[str, Any],
) -> tuple[bool, torch.dtype, str]:
    requested_dtype_name = str(optimization_config.get("compute_dtype", "float32"))
    requested_dtype = resolve_compute_dtype(requested_dtype_name)
    amp_requested = bool(optimization_config.get("use_amp", False))

    if not amp_requested or requested_dtype == torch.float32:
        return False, torch.float32, "float32"

    probe = torch.ones(2, device=device, dtype=torch.float32)
    try:
        with torch.autocast(device_type=device.type, dtype=requested_dtype, enabled=True):
            _ = probe * probe
    except Exception as exc:
        warnings.warn(
            f"Requested compute_dtype={requested_dtype_name} is not supported on device `{device}`. "
            f"Falling back to float32. Original error: {exc}",
            stacklevel=2,
        )
        return False, torch.float32, "float32"

    return True, requested_dtype, requested_dtype_name.lower()


def build_episode_split_indices(
    dataset: ForceWMParquetDataset,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    episode_ids = sorted(dataset.episode_ranges.keys())
    if len(episode_ids) < 2 or val_fraction <= 0:
        warnings.warn(
            "Dataset has fewer than two episodes or val_fraction <= 0; training will run without validation.",
            stacklevel=2,
        )
        return list(range(len(dataset))), []

    rng = np.random.default_rng(seed)
    shuffled_episode_ids = episode_ids.copy()
    rng.shuffle(shuffled_episode_ids)

    num_val = max(1, int(round(len(shuffled_episode_ids) * val_fraction)))
    num_val = min(num_val, len(shuffled_episode_ids) - 1)
    val_episode_ids = set(shuffled_episode_ids[:num_val])
    train_episode_ids = set(shuffled_episode_ids[num_val:])

    train_indices = [
        index for index, row in enumerate(dataset.rows) if int(row["episode_id"]) in train_episode_ids
    ]
    val_indices = [index for index, row in enumerate(dataset.rows) if int(row["episode_id"]) in val_episode_ids]
    return train_indices, val_indices


def compute_normalization_stats(
    dataset: ForceWMParquetDataset,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    observation_stats = {key_name: RunningStats() for key_name in dataset_spec.lowdim_keys}
    action_stats = {"robot_pos": RunningStats(), "robot_ori": RunningStats()}
    orientation_source_key = dataset.action_specs["robot_ori"].source_key

    for row in dataset.rows:
        episode_id = int(row["episode_id"])
        center_episode_frame = int(row["episode_frame_index"])
        episode_range = dataset.episode_ranges[episode_id]

        for key_name, key_spec in dataset.lowdim_specs.items():
            frame_indices = compute_observation_indices(
                center_episode_frame,
                key_spec.obs_window,
                key_spec.obs_dss,
                episode_range.last_episode_frame,
                episode_range.first_episode_frame,
            )
            sequence = fetch_lowdim_sequence(
                dataset.rows,
                dataset.episode_frame_lookup,
                episode_id,
                frame_indices,
                key_name,
            )
            if key_name == orientation_source_key:
                sequence = rotation_matrix_to_rot6d(sequence)
            observation_stats[key_name].update(sequence)

        action_frame_indices = compute_action_indices(
            center_episode_frame,
            dataset.action_window,
            dataset.action_dss,
            episode_range.last_episode_frame,
            episode_range.first_episode_frame,
        )
        position_sequence = fetch_lowdim_sequence(
            dataset.rows,
            dataset.episode_frame_lookup,
            episode_id,
            action_frame_indices,
            dataset.action_specs["robot_pos"].source_key,
        )
        orientation_sequence = fetch_lowdim_sequence(
            dataset.rows,
            dataset.episode_frame_lookup,
            episode_id,
            action_frame_indices,
            dataset.action_specs["robot_ori"].source_key,
        )
        relative_positions, relative_rot6d = build_relative_action_targets(
            position_sequence,
            orientation_sequence,
        )
        action_stats["robot_pos"].update(relative_positions)
        action_stats["robot_ori"].update(relative_rot6d)

    return {
        "observation": {
            key_name: running_stats.finalize() for key_name, running_stats in observation_stats.items()
        },
        "action": {
            key_name: running_stats.finalize() for key_name, running_stats in action_stats.items()
        },
        "metadata": {
            "num_samples": len(dataset),
            "computed_from": "full_dataset",
            "contract_path": str(dataset.contract.get("_resolved_contract_path", "")),
            "parquet_path": str(dataset.parquet_path),
        },
    }


def create_dataloader(
    dataset: ForceWMParquetDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    subset = Subset(dataset, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        generator=generator,
    )


def move_nested_tensors_to_device(payload: Any, device: torch.device) -> Any:
    if isinstance(payload, torch.Tensor):
        return payload.to(device, non_blocking=device.type == "cuda")
    if isinstance(payload, dict):
        return {key: move_nested_tensors_to_device(value, device) for key, value in payload.items()}
    return payload


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler `{scheduler_name}`. Expected `none` or `cosine`.")

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        if epochs <= warmup_epochs:
            return 1.0
        progress = float(epoch - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_bc_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_loss = F.mse_loss(predictions["robot_pos"], targets["robot_pos"])
    ori_loss = F.mse_loss(predictions["robot_ori"], targets["robot_ori"])
    pos_weight = float(loss_config.get("pos_loss_weight", 1.0))
    ori_weight = float(loss_config.get("ori_loss_weight", 1.0))
    total_loss = pos_weight * pos_loss + ori_weight * ori_loss
    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "pos_loss": float(pos_loss.detach().cpu()),
        "ori_loss": float(ori_loss.detach().cpu()),
    }
    return total_loss, metrics


def prepare_batch(
    batch: dict[str, Any],
    device: torch.device,
    normalization_stats: dict[str, Any],
    lowdim_keys: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    observation_batch = move_nested_tensors_to_device(batch["observation"], device)
    action_batch = move_nested_tensors_to_device(batch["action"], device)
    observation = normalize_observation_dict(
        observation_batch,
        normalization_stats,
        lowdim_keys,
    )
    action = {
        key_name: normalize_tensor(value, normalization_stats["action"][key_name])
        for key_name, value in action_batch.items()
    }
    return observation, action


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalization_stats: dict[str, Any],
    lowdim_keys: tuple[str, ...],
    loss_config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
    scaler: torch.cuda.amp.GradScaler,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    log_every_steps: int,
    epoch_index: int,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    if train and optimizer is None:
        raise ValueError("Optimizer must be provided when train=True.")

    accumulated_loss = 0.0
    accumulated_pos_loss = 0.0
    accumulated_ori_loss = 0.0
    num_batches = 0

    for step_index, batch in enumerate(dataloader, start=1):
        observation, action_targets = prepare_batch(batch, device, normalization_stats, lowdim_keys)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            predictions = model(observation)
            loss, metrics = compute_bc_loss(predictions, action_targets, loss_config)

        if train:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        accumulated_loss += metrics["loss"]
        accumulated_pos_loss += metrics["pos_loss"]
        accumulated_ori_loss += metrics["ori_loss"]
        num_batches += 1

        if log_every_steps > 0 and step_index % log_every_steps == 0:
            phase = "train" if train else "val"
            print(
                f"[{phase}] epoch={epoch_index + 1} step={step_index}/{len(dataloader)} "
                f"loss={metrics['loss']:.6f} pos={metrics['pos_loss']:.6f} ori={metrics['ori_loss']:.6f}"
            )

    if num_batches == 0:
        return {"loss": float("nan"), "pos_loss": float("nan"), "ori_loss": float("nan")}

    return {
        "loss": accumulated_loss / num_batches,
        "pos_loss": accumulated_pos_loss / num_batches,
        "ori_loss": accumulated_ori_loss / num_batches,
    }


def contract_summary(dataset: ForceWMParquetDataset, dataset_spec: DatasetSpec) -> dict[str, Any]:
    return {
        "visual_key": dataset_spec.image_key,
        "lowdim_keys": list(dataset_spec.lowdim_keys),
        "action_keys": sorted(dataset.action_specs.keys()),
        "action_window": dataset_spec.action_window,
        "image_shape": list(dataset_spec.image_shape),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    config: dict[str, Any],
    normalization_stats: dict[str, Any],
    dataset_spec: DatasetSpec,
    contract_summary_payload: dict[str, Any],
    best_val_loss: float | None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "config": config,
        "normalization_stats": normalization_stats,
        "dataset_spec": dataset_spec.to_dict(),
        "contract_summary": contract_summary_payload,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def prune_epoch_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    epoch_checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    for stale_checkpoint in epoch_checkpoints[:-keep_last]:
        stale_checkpoint.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple BC transformer policy for ForceWM.")
    parser.add_argument(
        "--config",
        default="training/configs/bc_default.yaml",
        help="Path to the training config YAML.",
    )
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (cpu, cuda, mps). Defaults to automatic selection.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run validation only from a checkpoint without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolved_config = deepcopy(config)

    if args.output_dir is not None:
        resolved_config.setdefault("data", {})["output_dir"] = args.output_dir

    if args.eval_only and args.resume is None:
        raise ValueError("`--eval-only` requires `--resume <checkpoint.pt>`.")

    data_config = resolved_config["data"]
    model_config = resolved_config["model"]
    optimization_config = resolved_config["optimization"]
    checkpoint_config = resolved_config["checkpoint"]
    logging_config = resolved_config["logging"]

    seed = int(data_config["seed"])
    set_random_seed(seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    parquet_path = Path(data_config["parquet_path"]).expanduser().resolve()
    contract_path = (
        Path(data_config["contract_path"]).expanduser().resolve()
        if data_config.get("contract_path")
        else None
    )
    output_dir = Path(data_config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "normalization_stats.pt"
    resolved_config_path = output_dir / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    dataset = ForceWMParquetDataset(
        parquet_path,
        contract_path=contract_path,
        max_open_video_chunks=int(data_config.get("max_open_video_chunks", 8)),
    )
    dataset.contract["_resolved_contract_path"] = str(contract_path) if contract_path is not None else ""
    dataset_spec = infer_dataset_spec(dataset)
    print(f"Loaded dataset with {len(dataset)} samples from {parquet_path}")

    normalization_stats = compute_normalization_stats(dataset, dataset_spec)
    torch.save(normalization_stats, stats_path)
    print(f"Saved normalization stats to {stats_path}")

    train_indices, val_indices = build_episode_split_indices(
        dataset,
        float(data_config.get("val_fraction", 0.0)),
        seed,
    )
    print(
        f"Episode split -> train_samples={len(train_indices)} val_samples={len(val_indices)} "
        f"episodes={len(dataset.episode_ranges)}"
    )

    pin_memory = bool(data_config.get("pin_memory", False)) and device.type == "cuda"
    train_loader = create_dataloader(
        dataset,
        train_indices,
        batch_size=int(data_config["batch_size"]),
        shuffle=not args.eval_only,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=pin_memory,
        seed=seed,
    )
    val_loader = (
        create_dataloader(
            dataset,
            val_indices,
            batch_size=int(data_config["batch_size"]),
            shuffle=False,
            num_workers=int(data_config.get("num_workers", 0)),
            pin_memory=pin_memory,
            seed=seed,
        )
        if val_indices
        else None
    )

    model = build_policy_from_config(model_config, dataset_spec).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(optimization_config["lr"]),
        weight_decay=float(optimization_config.get("weight_decay", 0.0)),
        betas=tuple(float(value) for value in optimization_config.get("betas", [0.9, 0.999])),
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=str(optimization_config.get("scheduler", "none")),
        epochs=int(optimization_config["epochs"]),
        warmup_epochs=int(optimization_config.get("warmup_epochs", 0)),
    )

    autocast_enabled, autocast_dtype, effective_dtype_name = resolve_autocast_settings(
        device,
        optimization_config,
    )
    print(
        f"Autocast: {'enabled' if autocast_enabled else 'disabled'} "
        f"dtype={effective_dtype_name}"
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and device.type == "cuda" and autocast_dtype == torch.float16,
    )

    start_epoch = 0
    best_val_loss: float | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        normalization_stats = checkpoint.get("normalization_stats", normalization_stats)
        if not args.eval_only:
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = checkpoint.get("best_val_loss")
        print(f"Resumed from checkpoint {args.resume} at epoch {start_epoch}")

    if args.eval_only:
        evaluation_loader = val_loader if val_loader is not None else train_loader
        if val_loader is None:
            warnings.warn("No validation split available; evaluating on the training split instead.", stacklevel=2)
        metrics = run_epoch(
            model=model,
            dataloader=evaluation_loader,
            device=device,
            normalization_stats=normalization_stats,
            lowdim_keys=dataset_spec.lowdim_keys,
            loss_config=resolved_config["loss"],
            optimizer=None,
            grad_clip_norm=None,
            scaler=scaler,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            log_every_steps=int(logging_config.get("log_every_steps", 0)),
            epoch_index=start_epoch,
            train=False,
        )
        print(
            f"[eval] loss={metrics['loss']:.6f} pos={metrics['pos_loss']:.6f} ori={metrics['ori_loss']:.6f}"
        )
        dataset.close()
        return

    num_epochs = int(optimization_config["epochs"])
    save_every_epochs = int(checkpoint_config.get("save_every_epochs", 0))
    keep_last = int(checkpoint_config.get("keep_last", 0))
    contract_summary_payload = contract_summary(dataset, dataset_spec)

    for epoch_index in range(start_epoch, num_epochs):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            normalization_stats=normalization_stats,
            lowdim_keys=dataset_spec.lowdim_keys,
            loss_config=resolved_config["loss"],
            optimizer=optimizer,
            grad_clip_norm=float(optimization_config.get("grad_clip_norm", 0.0)),
            scaler=scaler,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            log_every_steps=int(logging_config.get("log_every_steps", 0)),
            epoch_index=epoch_index,
            train=True,
        )

        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    normalization_stats=normalization_stats,
                    lowdim_keys=dataset_spec.lowdim_keys,
                    loss_config=resolved_config["loss"],
                    optimizer=None,
                    grad_clip_norm=None,
                    scaler=scaler,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                    log_every_steps=int(logging_config.get("log_every_steps", 0)),
                    epoch_index=epoch_index,
                    train=False,
                )

        if scheduler is not None:
            scheduler.step()

        if val_metrics is None:
            print(
                f"[epoch {epoch_index + 1}] train_loss={train_metrics['loss']:.6f} "
                f"train_pos={train_metrics['pos_loss']:.6f} train_ori={train_metrics['ori_loss']:.6f}"
            )
        else:
            print(
                f"[epoch {epoch_index + 1}] train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} train_pos={train_metrics['pos_loss']:.6f} "
                f"val_pos={val_metrics['pos_loss']:.6f} train_ori={train_metrics['ori_loss']:.6f} "
                f"val_ori={val_metrics['ori_loss']:.6f}"
            )

        if val_metrics is not None and bool(checkpoint_config.get("save_best", True)):
            current_val_loss = float(val_metrics["loss"])
            if best_val_loss is None or current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_checkpoint_path = checkpoint_dir / "best.pt"
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch_index,
                    config=resolved_config,
                    normalization_stats=normalization_stats,
                    dataset_spec=dataset_spec,
                    contract_summary_payload=contract_summary_payload,
                    best_val_loss=best_val_loss,
                )

        latest_checkpoint_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch_index,
            config=resolved_config,
            normalization_stats=normalization_stats,
            dataset_spec=dataset_spec,
            contract_summary_payload=contract_summary_payload,
            best_val_loss=best_val_loss,
        )

        if save_every_epochs > 0 and (epoch_index + 1) % save_every_epochs == 0:
            epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch_index + 1:04d}.pt"
            save_checkpoint(
                epoch_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index,
                config=resolved_config,
                normalization_stats=normalization_stats,
                dataset_spec=dataset_spec,
                contract_summary_payload=contract_summary_payload,
                best_val_loss=best_val_loss,
            )
            prune_epoch_checkpoints(checkpoint_dir, keep_last)

    dataset.close()


if __name__ == "__main__":
    main()
