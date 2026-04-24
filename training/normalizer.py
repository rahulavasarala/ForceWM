from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


OPENAI_CLIP_IMAGE_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
OPENAI_CLIP_IMAGE_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
STD_EPS = 1e-6


def _resolve_parquet_path(dataset_path: Path) -> Path:
    dummy_path = dataset_path / "dummy.parquet"
    if dummy_path.exists():
        return dummy_path

    dataset_parquet_path = dataset_path / "dataset.parquet"
    if dataset_parquet_path.exists():
        return dataset_parquet_path

    raise FileNotFoundError(
        f"Could not find parquet data in {dataset_path}. Expected `dummy.parquet` or `dataset.parquet`."
    )


def _to_numeric_array(column) -> np.ndarray:
    values = column.combine_chunks().to_numpy(zero_copy_only=False)
    if values.dtype == object and len(values) and isinstance(values[0], (list, tuple, np.ndarray)):
        values = np.stack(values, axis=0)

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return array


def _convert_orientation_to_quat(column_array: np.ndarray) -> np.ndarray:
    matrices = np.asarray(column_array, dtype=np.float32).reshape(-1, 3, 3)
    return R.from_matrix(matrices).as_quat().astype(np.float32)


def build_normalizer(dataset_path):
    dataset_path = Path(dataset_path).expanduser().resolve()
    parquet_path = _resolve_parquet_path(dataset_path)
    parquet_table = pq.read_table(parquet_path)

    lowdim_stats: dict[str, dict[str, np.ndarray]] = {}
    for column_name in parquet_table.column_names:
        column_array = _to_numeric_array(parquet_table[column_name])
        if column_name == "eef_ori":
            column_array = _convert_orientation_to_quat(column_array)
        column_std = column_array.std(axis=0)
        column_std = np.maximum(column_std, STD_EPS)

        lowdim_stats[column_name] = {
            "mean": column_array.mean(axis=0).astype(np.float32),
            "std": column_std.astype(np.float32),
        }

    normalizer = {
        "lowdim": lowdim_stats,
        "images": {
            "mean": OPENAI_CLIP_IMAGE_MEAN.copy(),
            "std": OPENAI_CLIP_IMAGE_STD.copy(),
        },
    }

    output_path = dataset_path / "normalizer.npy"
    np.save(output_path, normalizer, allow_pickle=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build normalization statistics for a dataset.")
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        required=True,
        type=str,
        help="Path to the extracted dataset directory.",
    )
    args = parser.parse_args()

    output_path = build_normalizer(args.dataset_path)
    print(f"Wrote normalizer to {output_path}")
