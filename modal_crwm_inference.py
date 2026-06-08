from __future__ import annotations

import copy
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import modal
import yaml


APP_NAME = "forcewm-crwm-inference"
GPU_TYPE = "H100"
PYTHON_VERSION = "3.10"
CUDA_BASE_IMAGE = "12.4.1-devel-ubuntu22.04"
TORCH_VERSION = "2.5.0"
TORCHVISION_VERSION = "0.20.0"
CUDA_TAG = "124"
PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_DATASET_MOUNT = "/mnt/datasets"
REMOTE_OUTPUT_MOUNT = "/mnt/outputs"
REMOTE_CACHE_MOUNT = "/root/.cache"
REMOTE_CONTRACT_PATH = "/tmp/forcewm_universal_contract.yaml"
DATASET_VOLUME_NAME = os.environ.get("FORCEWM_MODAL_DATASET_VOLUME", "forcewm-datasets")
OUTPUT_VOLUME_NAME = os.environ.get("FORCEWM_MODAL_OUTPUT_VOLUME", "forcewm-training-runs")
CACHE_VOLUME_NAME = os.environ.get("FORCEWM_MODAL_CACHE_VOLUME", "concerto-smoketest-cache")
DEFAULT_CHECKPOINT_NAME = "best.pt"
DEFAULT_ARTIFACT_TEMPLATE = "one_step_predictions_{split}.npy"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file `{yaml_path}` must parse to a dictionary.")
    return payload


def _normalize_subdir(value: str, *, default: str) -> str:
    cleaned = str(value or "").strip().strip("/")
    return cleaned or default


def _container_path(mount_root: str, subdir: str) -> str:
    return str(PurePosixPath(mount_root) / subdir)


def _resolve_artifact_name(artifact_name: str | None, split: str) -> str:
    cleaned = str(artifact_name or "").strip()
    if cleaned:
        return cleaned if cleaned.endswith(".npy") else f"{cleaned}.npy"
    return DEFAULT_ARTIFACT_TEMPLATE.format(split=str(split).strip().lower())


def _to_output_volume_path(remote_output_path: str) -> str:
    output_prefix = f"{REMOTE_OUTPUT_MOUNT}/"
    if remote_output_path.startswith(output_prefix):
        return remote_output_path[len(output_prefix) :]
    path = PurePosixPath(remote_output_path)
    parts = path.parts
    if len(parts) >= 5 and parts[1] == "__modal" and parts[2] == "volumes":
        return str(PurePosixPath(*parts[4:]))
    return str(path.as_posix()).lstrip("/")


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
            "numpy==1.26.4 pyyaml pyarrow scipy timm tqdm wandb "
            "huggingface_hub natsort addict einops termcolor"
        ),
        (
            "python -m pip install --no-build-isolation "
            "git+https://github.com/Pointcept/Concerto.git"
        ),
    )
    .env(
        {
            "HF_HOME": REMOTE_CACHE_MOUNT,
            "HF_HUB_CACHE": f"{REMOTE_CACHE_MOUNT}/huggingface",
        }
    )
    .add_local_dir(
        PROJECT_ROOT / "training",
        remote_path="/root/training",
        ignore=["__pycache__", "*.pyc"],
    )
)

app = modal.App(APP_NAME, image=image)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _download_output_artifact(remote_output_path: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    volume_path = _to_output_volume_path(remote_output_path)
    data = b"".join(output_volume.read_file(volume_path))
    local_path.write_bytes(data)
    return local_path


@app.function(
    gpu=GPU_TYPE,
    timeout=60 * 60 * 24,
    volumes={
        REMOTE_DATASET_MOUNT: dataset_volume,
        REMOTE_OUTPUT_MOUNT: output_volume,
        REMOTE_CACHE_MOUNT: cache_volume,
    },
)
def run_inference(
    config: dict[str, Any],
    contract_payload: dict[str, Any],
    *,
    split: str,
    artifact_name: str | None = None,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    from training.inference import export_predictions

    contract_path = Path(REMOTE_CONTRACT_PATH)
    contract_path.write_text(yaml.safe_dump(contract_payload, sort_keys=False), encoding="utf-8")

    remote_config = copy.deepcopy(config)
    remote_config["universal_contract"] = str(contract_path)
    remote_config["device"] = "cuda"

    remote_output_dir = PurePosixPath(str(remote_config["output_dir"]))
    resolved_artifact_name = _resolve_artifact_name(artifact_name, split)
    artifact_path = remote_output_dir / resolved_artifact_name
    checkpoint_path = remote_output_dir / str(checkpoint_name).strip()

    artifact = export_predictions(
        remote_config,
        split=split,
        checkpoint_path=checkpoint_path,
        artifact_path=artifact_path,
        device_name="cuda",
        show_progress=True,
    )
    summary = {
        "artifact_path": str(artifact_path),
        "split": artifact["metadata"]["split"],
        "episodes": len(artifact["episodes"]),
        "rows": int(len(artifact["metadata"]["selected_indices"])),
        "checkpoint_path": str(checkpoint_path),
    }
    output_volume.commit()
    cache_volume.commit()
    print(json.dumps(summary, sort_keys=True))
    return summary


@app.local_entrypoint()
def main(
    config: str,
    split: str = "val",
    run_subdir: str = "",
    dataset_subdir: str = "",
    artifact_name: str = "",
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    download_artifact: bool = False,
    local_artifact_path: str = "",
):
    config_path = Path(config).expanduser().resolve()
    config_payload = _load_yaml(config_path)
    contract_payload = _load_yaml(config_payload["universal_contract"])

    dataset_local_path = Path(config_payload["dataset_path"]).expanduser().resolve()
    dataset_subdir = _normalize_subdir(dataset_subdir, default=dataset_local_path.name)
    default_output_name = Path(str(config_payload.get("output_dir", "training_runs/crwm"))).name
    run_subdir = _normalize_subdir(run_subdir, default=default_output_name)

    dataset_container_path = _container_path(REMOTE_DATASET_MOUNT, dataset_subdir)
    output_container_path = _container_path(REMOTE_OUTPUT_MOUNT, run_subdir)
    remote_config = copy.deepcopy(config_payload)
    remote_config["dataset_path"] = dataset_container_path
    remote_config["output_dir"] = output_container_path
    remote_config["device"] = "cuda"

    resolved_artifact_name = _resolve_artifact_name(artifact_name, split)
    function_call = run_inference.spawn(
        remote_config,
        contract_payload,
        split=split,
        artifact_name=resolved_artifact_name,
        checkpoint_name=checkpoint_name,
    )
    result = function_call.get()

    if download_artifact:
        if local_artifact_path:
            local_path = Path(local_artifact_path).expanduser().resolve()
        else:
            local_path = Path.cwd() / resolved_artifact_name
        downloaded_path = _download_output_artifact(str(result["artifact_path"]), local_path)
        result["downloaded_to"] = str(downloaded_path)

    print(json.dumps(result, indent=2, sort_keys=True))
