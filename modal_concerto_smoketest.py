from __future__ import annotations

import json
from pathlib import Path

import modal
import numpy as np


APP_NAME = "concerto-encoder-smoketest"
GPU_TYPE = "L4"
PYTHON_VERSION = "3.10"
CUDA_BASE_IMAGE = "12.4.1-devel-ubuntu22.04"
TORCH_VERSION = "2.5.0"
TORCHVISION_VERSION = "0.20.0"
CUDA_TAG = "124"
CACHE_ROOT = "/root/.cache"


def _load_npz(npz_path: str) -> dict[str, np.ndarray]:
    path = Path(npz_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    with np.load(path) as data:
        if "coord" not in data:
            raise KeyError(f"{path} must contain a 'coord' array")
        point = {"coord": data["coord"]}
        if "color" in data:
            point["color"] = data["color"]
        if "normal" in data:
            point["normal"] = data["normal"]
    return point


def _sanitize_point_dict(point: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    coord = np.asarray(point["coord"], dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"'coord' must have shape (N, 3), got {coord.shape}")

    color_in = point.get("color")
    if color_in is None:
        color = np.zeros_like(coord, dtype=np.float32)
    else:
        color = np.asarray(color_in, dtype=np.float32)
        if color.ndim != 2 or color.shape[1] != 3:
            raise ValueError(f"'color' must have shape (N, 3), got {color.shape}")
        if color.shape[0] != coord.shape[0]:
            raise ValueError("'color' must have the same number of rows as 'coord'")
        if color.size and float(color.max()) <= 1.0:
            color = color * 255.0

    normal_in = point.get("normal")
    if normal_in is None:
        normal = np.zeros_like(coord, dtype=np.float32)
    else:
        normal = np.asarray(normal_in, dtype=np.float32)
        if normal.ndim != 2 or normal.shape[1] != 3:
            raise ValueError(f"'normal' must have shape (N, 3), got {normal.shape}")
        if normal.shape[0] != coord.shape[0]:
            raise ValueError("'normal' must have the same number of rows as 'coord'")

    return {"coord": coord, "color": color, "normal": normal}


def _make_synthetic_point_cloud(num_points: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    coord = rng.normal(size=(num_points, 3)).astype(np.float32)
    coord /= np.linalg.norm(coord, axis=1, keepdims=True).clip(min=1e-6)
    coord *= rng.uniform(0.3, 1.0, size=(num_points, 1)).astype(np.float32)
    normal = coord / np.linalg.norm(coord, axis=1, keepdims=True).clip(min=1e-6)
    color = rng.integers(0, 255, size=(num_points, 3), dtype=np.uint8).astype(
        np.float32
    )
    return {"coord": coord, "color": color, "normal": normal}


def _resolve_input_point(
    point_data: dict[str, np.ndarray] | None,
    num_points: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], str]:
    if point_data is None:
        return (
            _make_synthetic_point_cloud(num_points=num_points, seed=seed),
            f"synthetic(seed={seed})",
        )
    return _sanitize_point_dict(point_data), "npz"


image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_BASE_IMAGE}",
        add_python=PYTHON_VERSION,
    )
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "python -m pip install --upgrade pip 'setuptools<81' wheel",
        (
            "python -m pip install "
            f"torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} "
            f"--index-url https://download.pytorch.org/whl/cu{CUDA_TAG}"
        ),
        f"python -m pip install spconv-cu{CUDA_TAG}",
        (
            "python -m pip install torch-scatter "
            f"-f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu{CUDA_TAG}.html"
        ),
        (
            "python -m pip install "
            "numpy==1.26.4 pyyaml addict einops scipy termcolor timm "
            "huggingface_hub natsort"
        ),
        (
            "python -m pip install --no-build-isolation "
            "git+https://github.com/Pointcept/Concerto.git"
        ),
    )
    .env(
        {
            "HF_HOME": CACHE_ROOT,
            "HF_HUB_CACHE": f"{CACHE_ROOT}/huggingface",
        }
    )
)

app = modal.App(APP_NAME, image=image)
cache_volume = modal.Volume.from_name("concerto-smoketest-cache", create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    timeout=60 * 30,
    volumes={CACHE_ROOT: cache_volume},
)
def run_encoder(
    point_data: dict[str, np.ndarray] | None = None,
    model_name: str = "concerto_small",
    num_points: int = 4096,
    patch_size: int = 256,
    seed: int = 7,
) -> dict[str, object]:
    import torch

    import concerto

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal function")

    device = torch.device("cuda")
    raw_point, input_source = _resolve_input_point(
        point_data=point_data,
        num_points=num_points,
        seed=seed,
    )

    transform = concerto.transform.default()
    point = transform(raw_point)

    for key, value in list(point.items()):
        if isinstance(value, torch.Tensor):
            point[key] = value.to(device, non_blocking=True)

    custom_config = {
        "enc_patch_size": [patch_size for _ in range(5)],
        "enable_flash": False,
    }

    try:
        model = concerto.model.load(
            model_name,
            repo_id="Pointcept/Concerto",
            custom_config=custom_config,
        )
    except TypeError:
        model = concerto.load(
            model_name,
            repo_id="Pointcept/Concerto",
            custom_config=custom_config,
        )

    model = model.to(device).eval()

    with torch.inference_mode():
        encoded = model(point)

        # Mirror the official Concerto README logic to restore point features.
        for _ in range(2):
            if "pooling_parent" not in encoded.keys():
                break
            parent = encoded.pop("pooling_parent")
            inverse = encoded.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, encoded.feat[inverse]], dim=-1)
            encoded = parent

        while "pooling_parent" in encoded.keys():
            parent = encoded.pop("pooling_parent")
            inverse = encoded.pop("pooling_inverse")
            parent.feat = encoded.feat[inverse]
            encoded = parent

        restored_feat = (
            encoded.feat[encoded.inverse] if "inverse" in encoded.keys() else encoded.feat
        )

    result = {
        "model_name": model_name,
        "input_source": input_source,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "input_points": int(raw_point["coord"].shape[0]),
        "encoded_points": int(encoded.feat.shape[0]),
        "encoded_feature_shape": list(encoded.feat.shape),
        "restored_feature_shape": list(restored_feat.shape),
        "feature_dtype": str(restored_feat.dtype),
        "patch_size": patch_size,
        "flash_attention_enabled": False,
    }

    # Persist Hugging Face / model cache writes so future runs can reuse them.
    cache_volume.commit()
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    npz_path: str = "",
    model_name: str = "concerto_small",
    num_points: int = 4096,
    patch_size: int = 256,
    seed: int = 7,
):
    point_data = _load_npz(npz_path) if npz_path else None
    result = run_encoder.remote(
        point_data=point_data,
        model_name=model_name,
        num_points=num_points,
        patch_size=patch_size,
        seed=seed,
    )
    print(json.dumps(result, indent=2))
