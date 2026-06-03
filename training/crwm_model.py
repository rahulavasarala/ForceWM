from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


POINT_CLOUD_MASK_SUFFIX = "_mask"
_CONCERTO_MODEL_CHANNELS: dict[str, tuple[int, ...]] = {
    "concerto_tiny": (16, 32, 64, 128, 256),
    "concerto_small": (32, 64, 128, 256, 512),
    "concerto_base": (48, 96, 192, 384, 512),
    "concerto_large": (64, 128, 256, 512, 768),
}


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    return mask.to(dtype=torch.bool)


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _as_bool_mask(mask)
    weighted = features * mask.unsqueeze(-1).to(dtype=features.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=features.dtype)
    return weighted.sum(dim=1) / denom


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps.to(dtype=torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _infer_scalar_feature(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(-1)
    if value.ndim == 3 and value.shape[-1] == 1:
        return value
    raise ValueError(f"Expected scalar feature with shape (B, T) or (B, T, 1), got {tuple(value.shape)}.")


def _first_present(mapping: Mapping[str, torch.Tensor], candidates: tuple[str, ...], source_name: str) -> torch.Tensor:
    for key_name in candidates:
        if key_name in mapping:
            return mapping[key_name]
    raise KeyError(f"Missing required key for {source_name}. Tried {list(candidates)}.")


def _randn_like_shape(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # MPS currently expects CPU-backed RNG state, so sample there then move.
    if device.type == "mps":
        return torch.randn(shape, dtype=dtype, generator=generator).to(device)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


@dataclass
class DepthEncoderOutput:
    global_latent: torch.Tensor
    point_features: torch.Tensor
    valid_mask: torch.Tensor


class DepthEncoderBase(nn.Module):
    global_latent_dim: int
    point_feature_dim: int

    def forward(self, points: torch.Tensor, valid_mask: torch.Tensor) -> DepthEncoderOutput:
        raise NotImplementedError


class MaskedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return residual + x


class DummyDepthEncoder(DepthEncoderBase):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        point_feature_dim: int = 128,
        global_latent_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.global_latent_dim = int(global_latent_dim)
        self.point_feature_dim = int(point_feature_dim)

        self.input_projection = nn.Linear(3, hidden_dim)
        self.blocks = nn.ModuleList(
            [MaskedResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(max(1, int(num_blocks)))]
        )
        self.point_projection = nn.Linear(hidden_dim, self.point_feature_dim)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(self.point_feature_dim),
            nn.Linear(self.point_feature_dim, self.global_latent_dim),
        )

    def forward(self, points: torch.Tensor, valid_mask: torch.Tensor) -> DepthEncoderOutput:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Depth encoder expects points with shape (B, P, 3), got {tuple(points.shape)}.")
        if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(points.shape[:2]):
            raise ValueError(
                f"Depth encoder mask must have shape {tuple(points.shape[:2])}, got {tuple(valid_mask.shape)}."
            )

        mask = _as_bool_mask(valid_mask)
        x = self.input_projection(points.to(dtype=torch.float32))
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, mask)

        point_features = self.point_projection(x) * mask.unsqueeze(-1).to(dtype=x.dtype)
        global_latent = self.global_projection(_masked_mean(point_features, mask))
        return DepthEncoderOutput(
            global_latent=global_latent,
            point_features=point_features,
            valid_mask=mask,
        )


def _resolve_concerto_restored_feature_dim(
    *,
    model_name: str,
    checkpoint_path: str | None,
    restored_feature_dim: int | None,
) -> int:
    if restored_feature_dim is not None:
        return int(restored_feature_dim)

    model_key = str(model_name).lower()
    channels = _CONCERTO_MODEL_CHANNELS.get(model_key)
    if channels is not None and len(channels) >= 3:
        return int(channels[-1] + channels[-2] + channels[-3])

    if checkpoint_path:
        checkpoint_label = Path(checkpoint_path).name
        raise ValueError(
            "Unable to infer Concerto output feature width from checkpoint "
            f"`{checkpoint_label}`. Set `model.depth_encoder.restored_feature_dim` explicitly."
        )

    raise ValueError(
        "Unable to infer Concerto output feature width. Set "
        "`model.depth_encoder.restored_feature_dim` explicitly."
    )


class PointTransformerV3Adapter(DepthEncoderBase):
    def __init__(
        self,
        *,
        model_name: str = "concerto_small",
        repo_id: str = "Pointcept/Concerto",
        checkpoint_path: str | None = None,
        download_root: str | None = None,
        restored_feature_dim: int | None = None,
        point_feature_dim: int | None = None,
        global_latent_dim: int | None = None,
        patch_size: int = 256,
        enc_patch_size: list[int] | tuple[int, ...] | int | None = None,
        enable_flash: bool = False,
        grid_size: float = 0.02,
        **_: Any,
    ) -> None:
        super().__init__()
        try:
            import concerto
        except ImportError as exc:
            raise ImportError(
                "PointTransformerV3Adapter requires the standalone Concerto package. "
                "Install it with `pip install --no-build-isolation git+https://github.com/Pointcept/Concerto.git`."
            ) from exc

        if enc_patch_size is None:
            resolved_enc_patch_size = [int(patch_size) for _ in range(5)]
        elif isinstance(enc_patch_size, (list, tuple)):
            resolved_enc_patch_size = [int(value) for value in enc_patch_size]
        else:
            resolved_enc_patch_size = [int(enc_patch_size) for _ in range(5)]

        self.restored_feature_dim = _resolve_concerto_restored_feature_dim(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            restored_feature_dim=restored_feature_dim,
        )
        self.point_feature_dim = int(point_feature_dim or self.restored_feature_dim)
        self.global_latent_dim = int(global_latent_dim or self.point_feature_dim)
        self.model_name = str(model_name)
        self.repo_id = str(repo_id)
        self.checkpoint_path = checkpoint_path
        self.enable_flash = bool(enable_flash)
        self.grid_size = float(grid_size)
        self._custom_config = {
            "enc_patch_size": resolved_enc_patch_size,
            "enable_flash": self.enable_flash,
        }
        self._transform_config = [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=self.grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        self._transform: Any | None = None
        self.backbone = self._load_backbone(
            concerto=concerto,
            model_name=self.model_name,
            repo_id=self.repo_id,
            checkpoint_path=self.checkpoint_path,
            download_root=download_root,
            custom_config=self._custom_config,
        )
        self.point_projection = (
            nn.Identity()
            if self.point_feature_dim == self.restored_feature_dim
            else nn.Linear(self.restored_feature_dim, self.point_feature_dim)
        )
        self.global_projection = (
            nn.Identity()
            if self.global_latent_dim == self.point_feature_dim
            else nn.Linear(self.point_feature_dim, self.global_latent_dim)
        )

    @staticmethod
    def _load_backbone(
        *,
        concerto: Any,
        model_name: str,
        repo_id: str,
        checkpoint_path: str | None,
        download_root: str | None,
        custom_config: Mapping[str, Any],
    ) -> nn.Module:
        load_target = checkpoint_path or model_name
        load_kwargs: dict[str, Any] = {"custom_config": dict(custom_config)}
        if checkpoint_path is None:
            load_kwargs["repo_id"] = repo_id
        if download_root is not None:
            load_kwargs["download_root"] = download_root

        try:
            return concerto.model.load(load_target, **load_kwargs)
        except TypeError:
            return concerto.load(load_target, **load_kwargs)

    @staticmethod
    def _move_point_dict_to_device(point: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in point.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _restore_multiscale_features(encoded_point: Any) -> Any:
        point = encoded_point
        for _ in range(2):
            if "pooling_parent" not in point.keys():
                break
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point

    def _get_transform(self) -> Any:
        if self._transform is None:
            import concerto

            self._transform = concerto.transform.Compose(self._transform_config)
        return self._transform

    def _encode_single_point_cloud(
        self,
        points: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        valid_points = points[valid_mask]
        num_points = int(points.shape[0])
        num_valid_points = int(valid_points.shape[0])
        if num_valid_points == 0:
            return torch.zeros(
                (num_points, self.restored_feature_dim),
                dtype=points.dtype,
                device=device,
            )

        coord = valid_points.detach().to(device="cpu", dtype=torch.float32).numpy()
        zeros = np.zeros_like(coord, dtype=np.float32)
        point = {"coord": coord, "color": zeros, "normal": zeros}
        transformed = self._get_transform()(point)
        transformed = self._move_point_dict_to_device(transformed, device)

        backbone_trainable = any(parameter.requires_grad for parameter in self.backbone.parameters())
        grad_context = contextlib.nullcontext() if backbone_trainable else torch.no_grad()
        with grad_context:
            encoded = self.backbone(transformed)

        restored = self._restore_multiscale_features(encoded)
        restored_feat = restored.feat[restored.inverse] if "inverse" in restored.keys() else restored.feat
        if restored_feat.ndim != 2:
            raise ValueError(
                f"Concerto restored features must have shape (N, C), got {tuple(restored_feat.shape)}."
            )
        if int(restored_feat.shape[0]) != num_valid_points:
            raise ValueError(
                "Concerto restored feature count does not match the number of valid input points. "
                f"Expected {num_valid_points}, got {int(restored_feat.shape[0])}."
            )
        if int(restored_feat.shape[1]) != self.restored_feature_dim:
            raise ValueError(
                "Concerto restored feature width does not match the configured adapter width. "
                f"Expected {self.restored_feature_dim}, got {int(restored_feat.shape[1])}."
            )

        padded_features = torch.zeros(
            (num_points, self.restored_feature_dim),
            dtype=restored_feat.dtype,
            device=device,
        )
        padded_features[valid_mask] = restored_feat
        return padded_features

    def forward(self, points: torch.Tensor, valid_mask: torch.Tensor) -> DepthEncoderOutput:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Depth encoder expects points with shape (B, P, 3), got {tuple(points.shape)}.")
        if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(points.shape[:2]):
            raise ValueError(
                f"Depth encoder mask must have shape {tuple(points.shape[:2])}, got {tuple(valid_mask.shape)}."
            )

        mask = _as_bool_mask(valid_mask)
        encoded_rows = [
            self._encode_single_point_cloud(sample_points, sample_mask, device=points.device)
            for sample_points, sample_mask in zip(points, mask, strict=True)
        ]
        restored_point_features = torch.stack(encoded_rows, dim=0)
        point_features = self.point_projection(restored_point_features)
        point_features = point_features * mask.unsqueeze(-1).to(dtype=point_features.dtype)
        global_latent = self.global_projection(_masked_mean(point_features, mask))
        return DepthEncoderOutput(
            global_latent=global_latent,
            point_features=point_features,
            valid_mask=mask,
        )


def build_depth_encoder(config: Mapping[str, Any] | None = None) -> DepthEncoderBase:
    config = dict(config or {})
    encoder_type = str(config.get("type", "dummy")).lower()
    if encoder_type == "dummy":
        return DummyDepthEncoder(
            hidden_dim=int(config.get("hidden_dim", 128)),
            point_feature_dim=int(config.get("point_feature_dim", config.get("hidden_dim", 128))),
            global_latent_dim=int(config.get("global_latent_dim", 128)),
            num_blocks=int(config.get("num_blocks", 3)),
            dropout=float(config.get("dropout", 0.0)),
        )
    if encoder_type == "ptv3":
        return PointTransformerV3Adapter(**config)
    raise ValueError(f"Unsupported depth encoder type `{encoder_type}`. Expected `dummy` or `ptv3`.")


class ContactPatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_force_dimensions: int = 4,
        force_embedding_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_force_dimensions = int(num_force_dimensions)
        self.force_embedding = nn.Embedding(self.num_force_dimensions, int(force_embedding_dim))
        input_dim = int(force_embedding_dim) + 9
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(
        self,
        force_dimension: torch.Tensor,
        motion_or_force_axis: torch.Tensor,
        sensed_force: torch.Tensor,
        sensed_moment: torch.Tensor,
    ) -> torch.Tensor:
        if force_dimension.ndim != 2:
            raise ValueError(
                f"Force-dimension tensor must have shape (B, T), got {tuple(force_dimension.shape)}."
            )

        embedded_dimension = self.force_embedding(
            force_dimension.to(dtype=torch.long).clamp(min=0, max=self.num_force_dimensions - 1)
        )
        numeric = torch.cat(
            [
                motion_or_force_axis.to(dtype=torch.float32),
                sensed_force.to(dtype=torch.float32),
                sensed_moment.to(dtype=torch.float32),
            ],
            dim=-1,
        )
        return self.mlp(torch.cat([embedded_dimension, numeric], dim=-1))


class ActionEncoder(nn.Module):
    def __init__(self, *, hidden_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(
        self,
        action_delta_pos: torch.Tensor,
        action_delta_rotvec: torch.Tensor,
        action_force_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        magnitude = _infer_scalar_feature(action_force_magnitude).to(dtype=torch.float32)
        features = torch.cat(
            [
                action_delta_pos.to(dtype=torch.float32),
                action_delta_rotvec.to(dtype=torch.float32),
                magnitude,
            ],
            dim=-1,
        )
        return self.mlp(features)


class TimestepEmbedder(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or model_dim * 4)
        self.model_dim = int(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.model_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_timestep_embedding(timesteps, self.model_dim))


class DiTBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        mlp_hidden_dim = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, model_dim),
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 6),
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(time_embedding).chunk(6, dim=-1)
        attn_input = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_output
        mlp_input = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class FinalLayer(nn.Module):
    def __init__(self, model_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 2),
        )
        self.projection = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(time_embedding).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1.0 + scale) + shift
        return self.projection(x)


class LatentFlowTransformer(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(latent_dim, model_dim)
        self.time_embedder = TimestepEmbedder(model_dim=model_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.final_layer = FinalLayer(model_dim=model_dim, output_dim=latent_dim)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
        state_token = self.input_projection(x_t).unsqueeze(1)
        sequence = torch.cat([state_token, condition_tokens], dim=1)
        time_embedding = self.time_embedder(timesteps)
        for block in self.blocks:
            sequence = block(sequence, time_embedding)
        return self.final_layer(sequence[:, 0, :], time_embedding)


class ContactPatchDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_force_dimensions: int = 4,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.force_dimension_head = nn.Linear(hidden_dim, num_force_dimensions)
        self.motion_axis_head = nn.Linear(hidden_dim, 3)
        self.sensed_force_head = nn.Linear(hidden_dim, 3)
        self.sensed_moment_head = nn.Linear(hidden_dim, 3)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(latent)
        return {
            "force_dimension_logits": self.force_dimension_head(hidden),
            "motion_or_force_axis": self.motion_axis_head(hidden),
            "sensed_force": self.sensed_force_head(hidden),
            "sensed_moment": self.sensed_moment_head(hidden),
        }


class DepthReconstructionDecoder(nn.Module):
    def __init__(self, *, input_dim: int, num_points: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_points * 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latent)
        return decoded.reshape(latent.shape[0], self.num_points, 3)


class CRWMModel(nn.Module):
    def __init__(
        self,
        *,
        depth_key: str,
        scene_points_key: str | None,
        num_depth_points: int,
        depth_encoder_config: Mapping[str, Any] | None = None,
        contact_encoder_config: Mapping[str, Any] | None = None,
        action_encoder_config: Mapping[str, Any] | None = None,
        flow_config: Mapping[str, Any] | None = None,
        decoder_config: Mapping[str, Any] | None = None,
        loss_weights: Mapping[str, float] | None = None,
        max_history_steps: int = 16,
        action_delta_pos_key: str = "action_delta_pos",
        action_delta_rotvec_key: str = "action_delta_rotvec",
        action_force_magnitude_key: str = "action_force_magnitude",
        force_dimension_key: str = "force_dimension",
        motion_or_force_axis_key: str = "motion_or_force_axis",
        sensed_force_key: str = "sensed_force",
        sensed_moment_key: str = "sensed_moment",
    ) -> None:
        super().__init__()
        self.depth_key = str(depth_key)
        if scene_points_key is None or not str(scene_points_key).strip():
            raise ValueError(
                "CRWMModel requires `scene_points_key` because scene conditioning is mandatory for CRWM."
            )
        self.scene_points_key = str(scene_points_key)
        self.action_delta_pos_key = str(action_delta_pos_key)
        self.action_delta_rotvec_key = str(action_delta_rotvec_key)
        self.action_force_magnitude_key = str(action_force_magnitude_key)
        self.force_dimension_key = str(force_dimension_key)
        self.motion_or_force_axis_candidates = (
            str(motion_or_force_axis_key),
            "force_or_motion_axis",
        )
        self.sensed_force_key = str(sensed_force_key)
        self.sensed_moment_key = str(sensed_moment_key)

        self.depth_encoder = build_depth_encoder(depth_encoder_config)
        contact_encoder_config = dict(contact_encoder_config or {})
        action_encoder_config = dict(action_encoder_config or {})
        flow_config = dict(flow_config or {})
        decoder_config = dict(decoder_config or {})

        self.contact_encoder = ContactPatchEncoder(
            num_force_dimensions=int(contact_encoder_config.get("num_force_dimensions", 4)),
            force_embedding_dim=int(contact_encoder_config.get("force_embedding_dim", 16)),
            hidden_dim=int(contact_encoder_config.get("hidden_dim", 128)),
            output_dim=int(contact_encoder_config.get("output_dim", 128)),
        )
        self.action_encoder = ActionEncoder(
            hidden_dim=int(action_encoder_config.get("hidden_dim", 128)),
            output_dim=int(action_encoder_config.get("output_dim", 128)),
        )

        self.depth_latent_dim = int(self.depth_encoder.global_latent_dim)
        self.contact_latent_dim = int(self.contact_encoder.output_dim)
        self.action_latent_dim = int(self.action_encoder.output_dim)
        self.latent_dim = self.depth_latent_dim + self.contact_latent_dim
        self.model_dim = int(flow_config.get("model_dim", 256))
        self.max_history_steps = int(max_history_steps)
        self.num_depth_points = int(num_depth_points)

        self.action_token_projection = nn.Linear(self.action_latent_dim, self.model_dim)
        self.depth_token_projection = nn.Linear(self.depth_latent_dim, self.model_dim)
        self.contact_token_projection = nn.Linear(self.contact_latent_dim, self.model_dim)
        self.scene_token_projection = nn.Linear(self.depth_latent_dim, self.model_dim)
        self.modality_embeddings = nn.Parameter(torch.zeros(4, self.model_dim))
        self.temporal_embeddings = nn.Embedding(self.max_history_steps, self.model_dim)

        self.flow_model = LatentFlowTransformer(
            latent_dim=self.latent_dim,
            model_dim=self.model_dim,
            num_layers=int(flow_config.get("num_layers", 6)),
            num_heads=int(flow_config.get("num_heads", 8)),
            mlp_ratio=float(flow_config.get("mlp_ratio", 4.0)),
            dropout=float(flow_config.get("dropout", 0.0)),
        )

        self.contact_decoder = ContactPatchDecoder(
            input_dim=self.contact_latent_dim,
            hidden_dim=int(decoder_config.get("contact_hidden_dim", 128)),
            num_force_dimensions=int(contact_encoder_config.get("num_force_dimensions", 4)),
        )
        self.depth_decoder = DepthReconstructionDecoder(
            input_dim=self.depth_latent_dim,
            num_points=self.num_depth_points,
            hidden_dim=int(decoder_config.get("depth_hidden_dim", 256)),
        )

        weights = dict(loss_weights or {})
        self.loss_weights = {
            "flow": float(weights.get("flow", 1.0)),
            "depth_recon": float(weights.get("depth_recon", 1.0)),
            "contact_recon": float(weights.get("contact_recon", 1.0)),
        }

    def _build_inference_context(
        self,
        obs_dict: Mapping[str, torch.Tensor],
        *,
        scene_depth_encoder: DepthEncoderBase | None = None,
    ) -> dict[str, torch.Tensor]:
        observed_depth = obs_dict[self.depth_key]
        observed_depth_mask = obs_dict[f"{self.depth_key}{POINT_CLOUD_MASK_SUFFIX}"]
        scene_points = obs_dict[self.scene_points_key]
        scene_points_mask = obs_dict[f"{self.scene_points_key}{POINT_CLOUD_MASK_SUFFIX}"]

        observed_depth_latents = self._encode_depth_sequence(self.depth_encoder, observed_depth, observed_depth_mask)
        observed_contact_latents = self._encode_contact_sequence(self.contact_encoder, obs_dict)
        action_latents = self._encode_action_sequence(obs_dict)

        resolved_scene_encoder = scene_depth_encoder or self.depth_encoder
        with torch.no_grad():
            scene_depth_latents = self._encode_depth_sequence(
                resolved_scene_encoder,
                scene_points,
                scene_points_mask,
            )

        scene_latent = scene_depth_latents.global_latent[:, 0, :]
        condition_tokens = self._build_condition_tokens(
            action_latents=action_latents,
            depth_latents=observed_depth_latents.global_latent,
            contact_latents=observed_contact_latents,
            scene_latent=scene_latent,
        )
        return {
            "condition_tokens": condition_tokens,
            "last_observed_depth": observed_depth_latents.global_latent[:, -1, :],
            "last_observed_contact": observed_contact_latents[:, -1, :],
        }

    def _encode_depth_sequence(
        self,
        encoder: DepthEncoderBase,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> DepthEncoderOutput:
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError(f"Depth sequence must have shape (B, T, P, 3), got {tuple(points.shape)}.")
        if mask.ndim != 3 or tuple(mask.shape) != tuple(points.shape[:3]):
            raise ValueError(f"Depth mask must have shape {tuple(points.shape[:3])}, got {tuple(mask.shape)}.")

        batch_size, time_steps, num_points, _ = points.shape
        encoded = encoder(
            points.reshape(batch_size * time_steps, num_points, 3),
            mask.reshape(batch_size * time_steps, num_points),
        )
        return DepthEncoderOutput(
            global_latent=encoded.global_latent.reshape(batch_size, time_steps, -1),
            point_features=encoded.point_features.reshape(batch_size, time_steps, num_points, -1),
            valid_mask=encoded.valid_mask.reshape(batch_size, time_steps, num_points),
        )

    def _encode_contact_sequence(
        self,
        encoder: ContactPatchEncoder,
        modal_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        force_dimension = modal_dict[self.force_dimension_key]
        motion_or_force_axis = _first_present(
            modal_dict,
            self.motion_or_force_axis_candidates,
            source_name="motion/force axis",
        )
        sensed_force = modal_dict[self.sensed_force_key]
        sensed_moment = modal_dict[self.sensed_moment_key]
        return encoder(force_dimension, motion_or_force_axis, sensed_force, sensed_moment)

    def _encode_action_sequence(self, modal_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.action_encoder(
            modal_dict[self.action_delta_pos_key],
            modal_dict[self.action_delta_rotvec_key],
            modal_dict[self.action_force_magnitude_key],
        )

    def _build_condition_tokens(
        self,
        action_latents: torch.Tensor,
        depth_latents: torch.Tensor,
        contact_latents: torch.Tensor,
        scene_latent: torch.Tensor,
    ) -> torch.Tensor:
        if action_latents.shape[:2] != depth_latents.shape[:2] or action_latents.shape[:2] != contact_latents.shape[:2]:
            raise ValueError(
                "Action, depth, and contact histories must share the same batch/time dimensions. "
                f"Got action={tuple(action_latents.shape)}, depth={tuple(depth_latents.shape)}, "
                f"contact={tuple(contact_latents.shape)}."
            )

        batch_size, time_steps, _ = action_latents.shape
        if scene_latent.ndim != 2 or scene_latent.shape[0] != batch_size or scene_latent.shape[1] != self.depth_latent_dim:
            raise ValueError(
                "Scene latent must have shape "
                f"(B, {self.depth_latent_dim}), got {tuple(scene_latent.shape)}."
            )
        if time_steps > self.max_history_steps:
            raise ValueError(
                f"History length {time_steps} exceeds `max_history_steps={self.max_history_steps}` configured "
                "for CRWMModel."
            )

        temporal_indices = torch.arange(time_steps, device=action_latents.device)
        temporal_embeddings = self.temporal_embeddings(temporal_indices).unsqueeze(0).unsqueeze(2)
        history_modality_embeddings = self.modality_embeddings[:3].view(1, 1, 3, self.model_dim)

        tokens = torch.stack(
            [
                self.action_token_projection(action_latents),
                self.depth_token_projection(depth_latents),
                self.contact_token_projection(contact_latents),
            ],
            dim=2,
        )
        tokens = tokens + temporal_embeddings + history_modality_embeddings
        history_tokens = tokens.reshape(batch_size, time_steps * 3, self.model_dim)

        scene_token = self.scene_token_projection(scene_latent).unsqueeze(1)
        scene_token = scene_token + self.modality_embeddings[3].view(1, 1, self.model_dim)
        return torch.cat([history_tokens, scene_token], dim=1)

    def _split_predicted_latent(self, predicted_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return predicted_velocity.split([self.depth_latent_dim, self.contact_latent_dim], dim=-1)

    def _decode_predicted_state(
        self,
        predicted_state: torch.Tensor,
        *,
        last_observed_depth: torch.Tensor,
        last_observed_contact: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted_depth_delta, predicted_contact_delta = self._split_predicted_latent(predicted_state)
        predicted_depth_latent = last_observed_depth + predicted_depth_delta
        predicted_contact_latent = last_observed_contact + predicted_contact_delta
        predicted_contact = self.contact_decoder(predicted_contact_latent)
        predicted_force_dimension = predicted_contact["force_dimension_logits"].argmax(dim=-1)
        return {
            "predicted_state": predicted_state,
            "predicted_depth_latent": predicted_depth_latent,
            "predicted_contact_latent": predicted_contact_latent,
            "predicted_depth_points": self.depth_decoder(predicted_depth_latent),
            "predicted_force_dimension": predicted_force_dimension,
            "predicted_force_dimension_logits": predicted_contact["force_dimension_logits"],
            "predicted_motion_or_force_axis": predicted_contact["motion_or_force_axis"],
            "predicted_sensed_force": predicted_contact["sensed_force"],
            "predicted_sensed_moment": predicted_contact["sensed_moment"],
        }

    def _integrate_flow_ode(
        self,
        *,
        condition_tokens: torch.Tensor,
        initial_state: torch.Tensor,
        sampling_steps: int,
        solver: str,
    ) -> torch.Tensor:
        if sampling_steps <= 0:
            raise ValueError(f"`sampling_steps` must be positive, got {sampling_steps}.")

        solver_name = str(solver).strip().lower()
        if solver_name not in {"euler", "heun"}:
            raise ValueError(f"Unsupported solver `{solver}`. Expected `euler` or `heun`.")

        state = initial_state
        step_size = 1.0 / float(sampling_steps)
        batch_size = int(initial_state.shape[0])
        for step_index in range(sampling_steps):
            current_t = step_index * step_size
            timestep = torch.full(
                (batch_size,),
                float(current_t),
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self.flow_model(state, timestep, condition_tokens)
            if solver_name == "euler":
                state = state + step_size * velocity
                continue

            predicted_state = state + step_size * velocity
            next_t = min(1.0, current_t + step_size)
            next_timestep = torch.full(
                (batch_size,),
                float(next_t),
                device=state.device,
                dtype=state.dtype,
            )
            next_velocity = self.flow_model(predicted_state, next_timestep, condition_tokens)
            state = state + 0.5 * step_size * (velocity + next_velocity)

        return state

    def sample_one_step(
        self,
        obs_dict: Mapping[str, torch.Tensor],
        *,
        ema_depth_encoder: DepthEncoderBase | None = None,
        generator: torch.Generator | None = None,
        sampling_steps: int = 32,
        solver: str = "heun",
    ) -> dict[str, torch.Tensor]:
        context = self._build_inference_context(
            obs_dict,
            scene_depth_encoder=ema_depth_encoder,
        )
        last_observed_depth = context["last_observed_depth"]
        initial_state = _randn_like_shape(
            (int(last_observed_depth.shape[0]), self.latent_dim),
            device=last_observed_depth.device,
            dtype=last_observed_depth.dtype,
            generator=generator,
        )
        predicted_state = self._integrate_flow_ode(
            condition_tokens=context["condition_tokens"],
            initial_state=initial_state,
            sampling_steps=int(sampling_steps),
            solver=solver,
        )
        return self._decode_predicted_state(
            predicted_state,
            last_observed_depth=context["last_observed_depth"],
            last_observed_contact=context["last_observed_contact"],
        )

    def _depth_reconstruction_loss(
        self,
        predicted_points: torch.Tensor,
        target_points: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(predicted_points.shape) != tuple(target_points.shape):
            raise ValueError(
                f"Predicted depth points must match target shape. "
                f"Got predicted={tuple(predicted_points.shape)} target={tuple(target_points.shape)}."
            )
        squared_error = (predicted_points - target_points.to(dtype=torch.float32)).pow(2).sum(dim=-1)
        mask = _as_bool_mask(target_mask)
        return squared_error.masked_select(mask).mean()

    def _contact_reconstruction_loss(
        self,
        predictions: Mapping[str, torch.Tensor],
        prediction_dict: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        force_dimension_target = prediction_dict[self.force_dimension_key][:, 0].to(dtype=torch.long)
        motion_or_force_axis_target = _first_present(
            prediction_dict,
            self.motion_or_force_axis_candidates,
            source_name="prediction motion/force axis",
        )[:, 0, :].to(dtype=torch.float32)
        sensed_force_target = prediction_dict[self.sensed_force_key][:, 0, :].to(dtype=torch.float32)
        sensed_moment_target = prediction_dict[self.sensed_moment_key][:, 0, :].to(dtype=torch.float32)

        force_dimension_loss = F.cross_entropy(predictions["force_dimension_logits"], force_dimension_target)
        motion_axis_loss = F.mse_loss(predictions["motion_or_force_axis"], motion_or_force_axis_target)
        sensed_force_loss = F.mse_loss(predictions["sensed_force"], sensed_force_target)
        sensed_moment_loss = F.mse_loss(predictions["sensed_moment"], sensed_moment_target)
        total_loss = force_dimension_loss + motion_axis_loss + sensed_force_loss + sensed_moment_loss

        predicted_labels = predictions["force_dimension_logits"].argmax(dim=-1)
        accuracy = (predicted_labels == force_dimension_target).to(dtype=torch.float32).mean()
        return (
            {
                "contact_recon_loss": total_loss,
                "contact_force_dimension_ce": force_dimension_loss,
                "contact_motion_axis_mse": motion_axis_loss,
                "contact_sensed_force_mse": sensed_force_loss,
                "contact_sensed_moment_mse": sensed_moment_loss,
            },
            accuracy,
        )

    def forward(
        self,
        batch: Mapping[str, Mapping[str, torch.Tensor]],
        *,
        ema_depth_encoder: DepthEncoderBase | None = None,
        ema_contact_encoder: ContactPatchEncoder | None = None,
    ) -> dict[str, torch.Tensor]:
        obs_dict = batch["obs_dict"]
        prediction_dict = batch["prediction"]

        ema_depth_encoder = ema_depth_encoder or self.depth_encoder
        ema_contact_encoder = ema_contact_encoder or self.contact_encoder

        future_depth = prediction_dict[self.depth_key]
        future_depth_mask = prediction_dict[f"{self.depth_key}{POINT_CLOUD_MASK_SUFFIX}"]

        with torch.no_grad():
            target_depth_latents = self._encode_depth_sequence(ema_depth_encoder, future_depth, future_depth_mask)
        context = self._build_inference_context(obs_dict, scene_depth_encoder=ema_depth_encoder)

        with torch.no_grad():
            target_contact_latents = self._encode_contact_sequence(ema_contact_encoder, prediction_dict)
        condition_tokens = context["condition_tokens"]
        last_observed_depth = context["last_observed_depth"]
        last_observed_contact = context["last_observed_contact"]
        future_depth_latent = target_depth_latents.global_latent[:, 0, :]
        future_contact_latent = target_contact_latents[:, 0, :]

        p1_target = future_depth_latent - last_observed_depth
        p2_target = future_contact_latent - last_observed_contact
        latent_target = torch.cat([p1_target, p2_target], dim=-1)

        x0 = torch.randn_like(latent_target)
        timesteps = torch.rand(latent_target.shape[0], device=latent_target.device, dtype=latent_target.dtype)
        x_t = (1.0 - timesteps.unsqueeze(-1)) * x0 + timesteps.unsqueeze(-1) * latent_target
        target_velocity = latent_target - x0

        predicted_velocity = self.flow_model(x_t, timesteps, condition_tokens)
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)

        decoded_prediction = self._decode_predicted_state(
            predicted_velocity,
            last_observed_depth=last_observed_depth,
            last_observed_contact=last_observed_contact,
        )
        predicted_contact = {
            "force_dimension_logits": decoded_prediction["predicted_force_dimension_logits"],
            "motion_or_force_axis": decoded_prediction["predicted_motion_or_force_axis"],
            "sensed_force": decoded_prediction["predicted_sensed_force"],
            "sensed_moment": decoded_prediction["predicted_sensed_moment"],
        }
        contact_losses, force_dimension_accuracy = self._contact_reconstruction_loss(
            predictions=predicted_contact,
            prediction_dict=prediction_dict,
        )
        contact_recon_loss = contact_losses["contact_recon_loss"]

        predicted_depth_points = decoded_prediction["predicted_depth_points"]
        future_depth_target = future_depth[:, 0, :, :]
        future_depth_target_mask = future_depth_mask[:, 0, :]
        depth_recon_loss = self._depth_reconstruction_loss(
            predicted_points=predicted_depth_points,
            target_points=future_depth_target,
            target_mask=future_depth_target_mask,
        )

        total_loss = (
            self.loss_weights["flow"] * flow_loss
            + self.loss_weights["depth_recon"] * depth_recon_loss
            + self.loss_weights["contact_recon"] * contact_recon_loss
        )

        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "depth_recon_loss": depth_recon_loss,
            "contact_recon_loss": contact_recon_loss,
            "contact_force_dimension_ce": contact_losses["contact_force_dimension_ce"],
            "contact_motion_axis_mse": contact_losses["contact_motion_axis_mse"],
            "contact_sensed_force_mse": contact_losses["contact_sensed_force_mse"],
            "contact_sensed_moment_mse": contact_losses["contact_sensed_moment_mse"],
            "force_dimension_accuracy": force_dimension_accuracy,
            "predicted_velocity": predicted_velocity,
            "latent_target": latent_target,
            "predicted_depth_points": predicted_depth_points,
            "predicted_contact_latent": decoded_prediction["predicted_contact_latent"],
            "predicted_depth_latent": decoded_prediction["predicted_depth_latent"],
            "predicted_force_dimension_logits": predicted_contact["force_dimension_logits"],
            "predicted_motion_or_force_axis": predicted_contact["motion_or_force_axis"],
            "predicted_sensed_force": predicted_contact["sensed_force"],
            "predicted_sensed_moment": predicted_contact["sensed_moment"],
        }
