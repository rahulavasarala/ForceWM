from __future__ import annotations

import copy
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import modal
import yaml


APP_NAME = "forcewm-crwm-train"
GPU_TYPE = "L4"
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


def _volume_path(subdir: str) -> str:
    return str(PurePosixPath("/") / subdir)


def _rewrite_contract_for_remote(
    contract: dict[str, Any],
    *,
    dataset_local_path: Path,
    dataset_container_path: str,
    dataset_volume_path: str,
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    rewritten = copy.deepcopy(contract)
    extra_uploads: list[tuple[Path, str]] = []

    visual_cfg = rewritten.get("robot", {}).get("data_sources", {}).get("visual", {})
    key_entries = visual_cfg.get("keys", [])
    if not isinstance(key_entries, list):
        return rewritten, extra_uploads

    dataset_container_root = PurePosixPath(dataset_container_path)
    dataset_volume_root = PurePosixPath(dataset_volume_path)
    for entry in key_entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        _, key_cfg = next(iter(entry.items()))
        if not isinstance(key_cfg, dict):
            continue
        if str(key_cfg.get("type", "")).lower() != "scene_points":
            continue

        scene_path_value = key_cfg.get("path")
        if not isinstance(scene_path_value, str) or not scene_path_value.strip():
            raise ValueError("Scene-points entries must include a non-empty `path`.")
        local_scene_path = Path(scene_path_value).expanduser().resolve()
        try:
            relative_scene_path = local_scene_path.relative_to(dataset_local_path)
            remote_container_path = dataset_container_root / relative_scene_path.as_posix()
        except ValueError:
            remote_relative_path = PurePosixPath("_scene_points") / local_scene_path.name
            remote_container_path = dataset_container_root / remote_relative_path
            extra_uploads.append((local_scene_path, str(dataset_volume_root / remote_relative_path)))

        key_cfg["path"] = str(remote_container_path)

    return rewritten, extra_uploads


def _resolve_remote_resume_path(resume_from: str | None, output_container_path: str) -> str | None:
    if resume_from is None:
        return None
    cleaned = str(resume_from).strip()
    if not cleaned:
        return None
    if cleaned.startswith(REMOTE_OUTPUT_MOUNT):
        return cleaned
    return str(PurePosixPath(output_container_path) / Path(cleaned).name)


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
            "numpy==1.26.4 pyyaml pyarrow scipy timm tqdm "
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


@app.function(
    gpu=GPU_TYPE,
    timeout=60 * 60 * 24,
    volumes={
        REMOTE_DATASET_MOUNT: dataset_volume,
        REMOTE_OUTPUT_MOUNT: output_volume,
        REMOTE_CACHE_MOUNT: cache_volume,
    },
)
def run_training(config: dict[str, Any], contract_payload: dict[str, Any]) -> dict[str, float]:
    import sys

    sys.path.insert(0, "/root")
    from training.train import train

    contract_path = Path(REMOTE_CONTRACT_PATH)
    contract_path.write_text(yaml.safe_dump(contract_payload, sort_keys=False), encoding="utf-8")

    remote_config = copy.deepcopy(config)
    remote_config["universal_contract"] = str(contract_path)
    remote_config["device"] = "cuda"

    def _commit_epoch(epoch: int, metrics: dict[str, float], output_dir: Path) -> None:
        summary = {"epoch": int(epoch), "output_dir": str(output_dir), **metrics}
        print(json.dumps(summary, sort_keys=True))
        output_volume.commit()
        cache_volume.commit()

    metrics = train(remote_config, on_epoch_end=_commit_epoch)
    output_volume.commit()
    cache_volume.commit()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


@app.local_entrypoint()
def main(
    config: str,
    sync_dataset: bool = False,
    force_sync: bool = False,
    dataset_subdir: str = "",
    output_subdir: str = "",
    resume_from: str = "",
):
    config_path = Path(config).expanduser().resolve()
    config_payload = _load_yaml(config_path)
    contract_payload = _load_yaml(config_payload["universal_contract"])

    dataset_local_path = Path(config_payload["dataset_path"]).expanduser().resolve()
    dataset_subdir = _normalize_subdir(dataset_subdir, default=dataset_local_path.name)
    default_output_name = Path(str(config_payload.get("output_dir", "training_runs/crwm"))).name
    output_subdir = _normalize_subdir(output_subdir, default=default_output_name)

    dataset_container_path = _container_path(REMOTE_DATASET_MOUNT, dataset_subdir)
    dataset_volume_path = _volume_path(dataset_subdir)
    output_container_path = _container_path(REMOTE_OUTPUT_MOUNT, output_subdir)

    rewritten_contract, extra_uploads = _rewrite_contract_for_remote(
        contract_payload,
        dataset_local_path=dataset_local_path,
        dataset_container_path=dataset_container_path,
        dataset_volume_path=dataset_volume_path,
    )

    if sync_dataset:
        with dataset_volume.batch_upload(force=force_sync) as batch:
            batch.put_directory(str(dataset_local_path), dataset_volume_path)
            for local_path, remote_path in extra_uploads:
                batch.put_file(str(local_path), remote_path)

    remote_config = copy.deepcopy(config_payload)
    remote_config["dataset_path"] = dataset_container_path
    remote_config["output_dir"] = output_container_path
    remote_config["device"] = "cuda"

    resolved_resume_path = _resolve_remote_resume_path(
        resume_from if resume_from else remote_config.get("resume_from"),
        output_container_path,
    )
    if resolved_resume_path is None:
        remote_config.pop("resume_from", None)
    else:
        remote_config["resume_from"] = resolved_resume_path

    result = run_training.remote(remote_config, rewritten_contract)
    print(json.dumps(result, indent=2, sort_keys=True))
