from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from policies.surface_models import (
    load_surface_model_from_generation_metadata,
    surface_model_from_generation_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "generated_cad" / "default_part" / "generation_metadata.json"
)


class SurfaceModelsMetadataTests(unittest.TestCase):
    def test_default_generation_metadata_reproduces_surface_height(self) -> None:
        metadata = json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
        surface = surface_model_from_generation_metadata(metadata)

        sample_x = np.array([-0.03, 0.0, 0.021], dtype=float)
        sample_y = np.array([-0.02, 0.01, 0.035], dtype=float)
        surface_cfg = metadata["surface"]
        block_height = metadata["block_dimensions"]["height"]
        expected_height = block_height + surface_cfg["base_height"] + surface_cfg["amp"] * np.sin(
            surface_cfg["freq_x"] * sample_x
        ) * np.cos(surface_cfg["freq_y"] * sample_y)

        np.testing.assert_allclose(
            surface.height(sample_x, sample_y),
            expected_height,
            atol=1e-12,
        )

    def test_gaussian_generation_metadata_uses_explicit_peak_data(self) -> None:
        metadata = {
            "block_dimensions": {"length": 0.1, "width": 0.08, "height": 0.03},
            "surface": {
                "family": "random_gaussian_two_peak",
                "base_height": 0.001,
                "amp": 0.006,
                "freq_x": 0.0,
                "freq_y": 0.0,
                "seed": 999,
                "gaussian_curvature": 0.02,
                "gaussian_peak_offset": 0.04,
                "origin_x": 0.0,
                "origin_y": 0.0,
                "gaussian_centers_local": [[0.01, -0.01], [-0.015, 0.02]],
                "gaussian_peak_amps": [0.0025, 0.0040],
                "gaussian_sigma": 0.018,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "generation_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            surface = load_surface_model_from_generation_metadata(metadata_path)

        sample_x = np.array([0.0, 0.012, -0.01], dtype=float)
        sample_y = np.array([0.0, -0.006, 0.018], dtype=float)
        sigma_sq = metadata["surface"]["gaussian_sigma"] ** 2
        expected_height = np.full_like(
            sample_x,
            metadata["block_dimensions"]["height"] + metadata["surface"]["base_height"],
            dtype=float,
        )
        for center, peak_amp in zip(
            metadata["surface"]["gaussian_centers_local"],
            metadata["surface"]["gaussian_peak_amps"],
            strict=True,
        ):
            dx = sample_x - center[0]
            dy = sample_y - center[1]
            expected_height += peak_amp * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_sq))

        np.testing.assert_allclose(surface.height(sample_x, sample_y), expected_height, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
