from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "PyTorch is required for the BC policy. Install the `pytorch` package in the runtime environment."
    ) from exc

try:
    import timm
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "timm is required for the BC policy backbone. Install the `timm` package in the runtime environment."
    ) from exc

try:
    from training.dataloader import ForceWMParquetDataset
except ModuleNotFoundError:
    from dataloader import ForceWMParquetDataset


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class DatasetSpec:
    image_key: str
    image_observation_horizon: int
    image_shape: tuple[int, int, int]
    lowdim_keys: tuple[str, ...]
    lowdim_observation_horizon: int
    lowdim_obs_dss: int
    lowdim_feature_dims: dict[str, int]
    action_window: int
    action_dims: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lowdim_keys"] = list(self.lowdim_keys)
        payload["image_shape"] = list(self.image_shape)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSpec":
        return cls(
            image_key=str(payload["image_key"]),
            image_observation_horizon=int(payload["image_observation_horizon"]),
            image_shape=tuple(int(value) for value in payload["image_shape"]),
            lowdim_keys=tuple(str(value) for value in payload["lowdim_keys"]),
            lowdim_observation_horizon=int(payload["lowdim_observation_horizon"]),
            lowdim_obs_dss=int(payload["lowdim_obs_dss"]),
            lowdim_feature_dims={
                str(key): int(value) for key, value in dict(payload["lowdim_feature_dims"]).items()
            },
            action_window=int(payload["action_window"]),
            action_dims={str(key): int(value) for key, value in dict(payload["action_dims"]).items()},
        )


def infer_dataset_spec(dataset: ForceWMParquetDataset) -> DatasetSpec:
    lowdim_schedule = {
        (int(spec.obs_window), int(spec.obs_dss)) for spec in dataset.lowdim_specs.values()
    }
    if len(lowdim_schedule) != 1:
        raise ValueError(
            "This first BC model uses one lowdim token per timestep and requires all lowdim keys "
            "to share the same observation schedule."
        )

    sample = dataset[0]
    lowdim_keys = tuple(dataset.lowdim_specs.keys())
    lowdim_feature_dims = {
        key_name: int(np.prod(sample["observation"][key_name].shape[1:], dtype=np.int64))
        for key_name in lowdim_keys
    }

    lowdim_obs_window, lowdim_obs_dss = next(iter(lowdim_schedule))
    image_key = dataset.visual_key_name
    image_shape = tuple(int(value) for value in sample["observation"][image_key].shape[1:])
    action_dims = {
        key_name: int(sample["action"][key_name].shape[-1])
        for key_name in ("robot_pos", "robot_ori")
    }

    return DatasetSpec(
        image_key=image_key,
        image_observation_horizon=int(sample["observation"][image_key].shape[0]),
        image_shape=image_shape,
        lowdim_keys=lowdim_keys,
        lowdim_observation_horizon=lowdim_obs_window,
        lowdim_obs_dss=lowdim_obs_dss,
        lowdim_feature_dims=lowdim_feature_dims,
        action_window=int(sample["action"]["robot_pos"].shape[0]),
        action_dims=action_dims,
    )


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _broadcast_stats(stats: dict[str, torch.Tensor | np.ndarray], tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(stats["mean"], device=tensor.device, dtype=tensor.dtype)
    std = torch.as_tensor(stats["std"], device=tensor.device, dtype=tensor.dtype)
    view_shape = [1] * tensor.ndim
    view_shape[-1] = mean.shape[0]
    return mean.view(*view_shape), std.view(*view_shape)


def normalize_tensor(tensor: torch.Tensor, stats: dict[str, torch.Tensor | np.ndarray]) -> torch.Tensor:
    mean, std = _broadcast_stats(stats, tensor)
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor, stats: dict[str, torch.Tensor | np.ndarray]) -> torch.Tensor:
    mean, std = _broadcast_stats(stats, tensor)
    return tensor * std + mean


def normalize_images_clip(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"Expected images with shape [B, T, C, H, W]. Got {tuple(images.shape)}.")

    images = images.float()
    if images.max() > 1.0:
        images = images / 255.0

    mean = images.new_tensor(CLIP_IMAGE_MEAN).view(1, 1, 3, 1, 1)
    std = images.new_tensor(CLIP_IMAGE_STD).view(1, 1, 3, 1, 1)
    return (images - mean) / std


def normalize_observation_dict(
    observation: dict[str, torch.Tensor],
    normalization_stats: dict[str, Any],
    lowdim_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    observation_stats = normalization_stats.get("observation", {})
    for key_name, value in observation.items():
        if key_name in lowdim_keys:
            normalized[key_name] = normalize_tensor(value, observation_stats[key_name])
        else:
            normalized[key_name] = value
    return normalized


def denormalize_action_dict(
    action: dict[str, torch.Tensor],
    normalization_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    action_stats = normalization_stats.get("action", {})
    return {
        key_name: denormalize_tensor(value, action_stats[key_name])
        for key_name, value in action.items()
    }


def _build_2d_sincos_position_encoding(height: int, width: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4 for 2D sine-cos positional encoding. Got {dim}.")

    device = torch.device("cpu")
    half_dim = dim // 4
    omega = torch.arange(half_dim, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / max(half_dim - 1, 1)))

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid_y = grid_y.reshape(-1, 1)
    grid_x = grid_x.reshape(-1, 1)

    out_y = grid_y * omega.reshape(1, -1)
    out_x = grid_x * omega.reshape(1, -1)

    pos_embedding = torch.cat(
        (out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()),
        dim=1,
    )
    return pos_embedding


class BCTransformerPolicy(nn.Module):
    def __init__(self, model_config: dict[str, Any], dataset_spec: DatasetSpec) -> None:
        super().__init__()
        self.dataset_spec = dataset_spec
        self.model_config = model_config

        self.d_model = int(model_config["d_model"])
        self.dropout = float(model_config.get("dropout", 0.0))
        self.backbone_name = str(model_config["backbone_name"])

        self.image_backbone = timm.create_model(
            self.backbone_name,
            pretrained=bool(model_config.get("backbone_pretrained", True)),
            features_only=True,
            out_indices=(4,),
        )
        if bool(model_config.get("freeze_backbone", False)):
            for parameter in self.image_backbone.parameters():
                parameter.requires_grad = False

        channels = int(self.image_backbone.feature_info.channels()[-1])
        self.image_projection = nn.Linear(channels, self.d_model) if channels != self.d_model else nn.Identity()

        with torch.no_grad():
            dummy_image = torch.zeros(1, *self.dataset_spec.image_shape, dtype=torch.float32)
            image_features = self.image_backbone(dummy_image)[0]
        self.image_feature_grid = (int(image_features.shape[-2]), int(image_features.shape[-1]))
        spatial_encoding = _build_2d_sincos_position_encoding(
            self.image_feature_grid[0],
            self.image_feature_grid[1],
            self.d_model,
        )
        self.register_buffer("image_spatial_position_encoding", spatial_encoding, persistent=False)

        self.image_frame_embedding = nn.Embedding(self.dataset_spec.image_observation_horizon, self.d_model)
        self.image_modality_embedding = nn.Parameter(torch.zeros(1, 1, self.d_model))

        lowdim_concat_dim = int(sum(self.dataset_spec.lowdim_feature_dims.values()))
        self.lowdim_token_encoder = MLP(
            input_dim=lowdim_concat_dim,
            hidden_dim=int(model_config["lowdim_mlp_hidden_dim"]),
            output_dim=self.d_model,
        )
        self.lowdim_timestep_embedding = nn.Embedding(
            self.dataset_spec.lowdim_observation_horizon,
            self.d_model,
        )
        self.lowdim_modality_embedding = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.action_query_tokens = nn.Parameter(
            torch.randn(self.dataset_spec.action_window, self.d_model) * 0.02
        )
        self.action_modality_embedding = nn.Parameter(torch.zeros(1, 1, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(model_config["num_heads"]),
            dim_feedforward=int(self.d_model * float(model_config["mlp_ratio"])),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(model_config["num_layers"]),
            norm=nn.LayerNorm(self.d_model),
        )

        action_head_hidden_dim = int(model_config["action_head_hidden_dim"])
        self.action_pos_head = MLP(self.d_model, action_head_hidden_dim, self.dataset_spec.action_dims["robot_pos"])
        self.action_ori_head = MLP(self.d_model, action_head_hidden_dim, self.dataset_spec.action_dims["robot_ori"])

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        images = normalize_images_clip(images)
        batch_size, num_frames, channels, height, width = images.shape
        if (channels, height, width) != self.dataset_spec.image_shape:
            raise ValueError(
                "Image tensor shape does not match the dataset spec. "
                f"Expected {self.dataset_spec.image_shape}, got {(channels, height, width)}."
            )

        features = self.image_backbone(images.reshape(batch_size * num_frames, channels, height, width))[0]
        feat_height, feat_width = int(features.shape[-2]), int(features.shape[-1])
        if (feat_height, feat_width) != self.image_feature_grid:
            raise ValueError(
                "Backbone feature grid changed unexpectedly. "
                f"Expected {self.image_feature_grid}, got {(feat_height, feat_width)}."
            )

        features = features.flatten(2).transpose(1, 2)
        features = self.image_projection(features)
        features = features.reshape(batch_size, num_frames, feat_height * feat_width, self.d_model)

        frame_positions = torch.arange(num_frames, device=features.device)
        frame_encoding = self.image_frame_embedding(frame_positions).view(1, num_frames, 1, self.d_model)
        spatial_encoding = self.image_spatial_position_encoding.to(features.device, dtype=features.dtype).view(
            1, 1, feat_height * feat_width, self.d_model
        )
        tokens = features + frame_encoding + spatial_encoding + self.image_modality_embedding.to(features.dtype)
        return tokens.reshape(batch_size, num_frames * feat_height * feat_width, self.d_model)

    def encode_lowdim_tokens(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_parts: list[torch.Tensor] = []
        batch_size = None
        for key_name in self.dataset_spec.lowdim_keys:
            if key_name not in observation:
                raise KeyError(f"Missing lowdim observation key `{key_name}`.")
            value = observation[key_name]
            if value.ndim < 3:
                raise ValueError(
                    f"Lowdim observation `{key_name}` must have shape [B, T, ...]. Got {tuple(value.shape)}."
                )
            if batch_size is None:
                batch_size = int(value.shape[0])
            if int(value.shape[1]) != self.dataset_spec.lowdim_observation_horizon:
                raise ValueError(
                    f"Lowdim observation `{key_name}` must have horizon {self.dataset_spec.lowdim_observation_horizon}. "
                    f"Got {int(value.shape[1])}."
                )
            encoded_parts.append(value.reshape(value.shape[0], value.shape[1], -1))

        lowdim_concat = torch.cat(encoded_parts, dim=-1)
        tokens = self.lowdim_token_encoder(lowdim_concat)
        timestep_ids = torch.arange(self.dataset_spec.lowdim_observation_horizon, device=tokens.device)
        timestep_encoding = self.lowdim_timestep_embedding(timestep_ids).view(
            1,
            self.dataset_spec.lowdim_observation_horizon,
            self.d_model,
        )
        return tokens + timestep_encoding + self.lowdim_modality_embedding.to(tokens.dtype)

    def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = observation[self.dataset_spec.image_key]
        image_tokens = self.encode_image_tokens(images)
        lowdim_tokens = self.encode_lowdim_tokens(observation)

        batch_size = image_tokens.shape[0]
        action_queries = self.action_query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        action_queries = action_queries + self.action_modality_embedding.to(action_queries.dtype)

        transformer_inputs = torch.cat((image_tokens, lowdim_tokens, action_queries), dim=1)
        transformer_outputs = self.transformer(transformer_inputs)
        action_token_outputs = transformer_outputs[:, -self.dataset_spec.action_window :, :]

        return {
            "robot_pos": self.action_pos_head(action_token_outputs),
            "robot_ori": self.action_ori_head(action_token_outputs),
        }

    def predict_actions(
        self,
        observation: dict[str, torch.Tensor],
        normalization_stats: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        model_device = next(self.parameters()).device
        batched_observation: dict[str, torch.Tensor] = {}
        was_unbatched = False
        for key_name, value in observation.items():
            tensor_value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            tensor_value = tensor_value.to(model_device)
            if key_name == self.dataset_spec.image_key and tensor_value.ndim == 4:
                batched_observation[key_name] = tensor_value.unsqueeze(0)
                was_unbatched = True
            elif key_name in self.dataset_spec.lowdim_keys and tensor_value.ndim == 2:
                batched_observation[key_name] = tensor_value.unsqueeze(0)
                was_unbatched = True
            else:
                batched_observation[key_name] = tensor_value

        normalized_observation = normalize_observation_dict(
            batched_observation,
            normalization_stats,
            self.dataset_spec.lowdim_keys,
        )
        predictions = self(normalized_observation)
        predictions = denormalize_action_dict(predictions, normalization_stats)

        if was_unbatched:
            return {key_name: value.squeeze(0) for key_name, value in predictions.items()}
        return predictions


def build_policy_from_config(model_config: dict[str, Any], dataset_spec: DatasetSpec) -> BCTransformerPolicy:
    return BCTransformerPolicy(model_config=model_config, dataset_spec=dataset_spec)


def load_policy_from_checkpoint(
    checkpoint_path,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> tuple[BCTransformerPolicy, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" not in checkpoint or "dataset_spec" not in checkpoint:
        raise ValueError("Checkpoint is missing `config` or `dataset_spec`.")

    config = checkpoint["config"]
    dataset_spec = DatasetSpec.from_dict(checkpoint["dataset_spec"])
    policy = build_policy_from_config(config["model"], dataset_spec)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    if eval_mode:
        policy.eval()

    normalization_stats = checkpoint.get("normalization_stats")
    if normalization_stats is None:
        raise ValueError("Checkpoint is missing `normalization_stats`.")
    return policy, normalization_stats, config
