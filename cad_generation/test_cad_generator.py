from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cad_generation.cad_generator import (
    CadPartConfig,
    default_surface_config,
    generate_part_assets,
    make_default_cad_part_config,
)


def _has_vhacdx() -> bool:
    try:
        import vhacdx  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


HAS_VHACDX = _has_vhacdx()


def _base_config(part_name: str = "test_part") -> CadPartConfig:
    return replace(
        make_default_cad_part_config(part_name),
        length=0.08,
        width=0.07,
        height=0.03,
        hole_radius=0.012,
        nx=18,
        ny=16,
        surface=replace(default_surface_config(), amp=0.0015, freq_x=30.0, freq_y=24.0),
    )


@unittest.skipUnless(HAS_VHACDX, "vhacdx is required for CAD generation tests.")
class CadGeneratorTests(unittest.TestCase):
    def test_generate_part_assets_writes_expected_artifacts(self) -> None:
        config = _base_config("artifact_part")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = generate_part_assets(
                config,
                output_root=root / "generated_cad",
                models_dir=root / "models",
            )

            self.assertTrue(artifacts.visual_mesh_path.exists())
            self.assertTrue(artifacts.metadata_path.exists())
            self.assertTrue(artifacts.surface_function_path.exists())
            self.assertTrue(artifacts.xml_path.exists())
            convex_files = sorted(artifacts.convex_dir.glob("geometry_*.stl"))
            self.assertGreater(len(convex_files), 0)

            metadata = json.loads(artifacts.metadata_path.read_text())
            self.assertEqual(metadata["part_name"], "artifact_part")
            self.assertAlmostEqual(metadata["block_dimensions"]["length"], config.length)
            self.assertAlmostEqual(metadata["hole_dimensions"]["radius"], config.hole_radius)

    def test_generate_part_assets_removes_stale_convex_files_on_overwrite(self) -> None:
        config = _base_config("overwrite_part")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = generate_part_assets(
                config,
                output_root=root / "generated_cad",
                models_dir=root / "models",
            )
            stale_file = artifacts.convex_dir / "stale_geometry.stl"
            stale_file.write_text("stale")
            self.assertTrue(stale_file.exists())

            artifacts = generate_part_assets(
                config,
                output_root=root / "generated_cad",
                models_dir=root / "models",
            )

            self.assertFalse(stale_file.exists())
            self.assertGreater(len(sorted(artifacts.convex_dir.glob("geometry_*.stl"))), 0)

    def test_validate_geometry_rejects_hole_radius_that_is_too_large(self) -> None:
        config = replace(_base_config(), hole_radius=0.04)
        with self.assertRaisesRegex(ValueError, "Hole radius"):
            config.validate_geometry()

    def test_validate_geometry_rejects_non_positive_dimensions(self) -> None:
        config = replace(_base_config(), height=0.0)
        with self.assertRaisesRegex(ValueError, "Block height"):
            config.validate_geometry()

    def test_validate_geometry_rejects_surface_that_dips_below_base_plane(self) -> None:
        config = replace(
            _base_config(),
            height=0.005,
            surface=replace(default_surface_config(), amp=0.01, freq_x=50.0, freq_y=40.0),
        )
        with self.assertRaisesRegex(ValueError, "base plane"):
            config.validate_geometry()


if __name__ == "__main__":
    unittest.main()
