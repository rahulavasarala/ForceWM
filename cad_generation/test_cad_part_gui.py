from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from cad_generation.cad_part_gui import CadPartAuthoringGui


def _has_vhacdx() -> bool:
    try:
        import vhacdx  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


HAS_VHACDX = _has_vhacdx()


class CadPartGuiTests(unittest.TestCase):
    def _make_gui(self) -> CadPartAuthoringGui:
        temp_root = tempfile.TemporaryDirectory()
        self.addCleanup(temp_root.cleanup)
        root = Path(temp_root.name)
        gui = CadPartAuthoringGui(
            output_root=root / "generated_cad",
            models_dir=root / "models",
        )
        self.addCleanup(gui.close)
        return gui

    def test_gui_instantiates_with_default_preview(self) -> None:
        gui = self._make_gui()
        self.assertIsNotNone(gui.preview_data)
        self.assertIn("surface_freq_x", gui.visible_slider_keys())

    def test_family_switch_updates_visible_controls(self) -> None:
        gui = self._make_gui()
        self.assertIn("surface_freq_x", gui.visible_slider_keys())
        self.assertNotIn("surface_seed", gui.visible_slider_keys())

        gui.family_radio.set_active(1)

        self.assertIn("surface_seed", gui.visible_slider_keys())
        self.assertIn("surface_amp_gaussian", gui.visible_slider_keys())
        self.assertNotIn("surface_freq_x", gui.visible_slider_keys())
        self.assertNotIn("surface_amp_default", gui.visible_slider_keys())

    def test_slider_change_recomputes_preview(self) -> None:
        gui = self._make_gui()
        initial_top = np.array(gui.preview_data.top_surface.z, copy=True)

        gui.sliders["surface_amp_default"].set_val(0.006)

        updated_top = np.array(gui.preview_data.top_surface.z, copy=True)
        self.assertGreater(np.nanmax(np.abs(updated_top - initial_top)), 1e-6)

    @unittest.skipUnless(HAS_VHACDX, "vhacdx is required for GUI save tests.")
    def test_save_current_part_writes_artifacts(self) -> None:
        gui = self._make_gui()
        gui.part_name_box.set_val("gui_part")
        gui.sliders["nx"].set_val(16)
        gui.sliders["ny"].set_val(16)

        artifacts = gui.save_current_part()

        self.assertIsNotNone(artifacts)
        assert artifacts is not None
        self.assertTrue(artifacts.visual_mesh_path.exists())
        self.assertTrue(artifacts.metadata_path.exists())
        self.assertTrue(artifacts.surface_function_path.exists())
        self.assertTrue(artifacts.xml_path.exists())
        self.assertGreater(len(sorted(artifacts.convex_dir.glob("geometry_*.stl"))), 0)


if __name__ == "__main__":
    unittest.main()
