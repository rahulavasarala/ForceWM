from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cad_generation.cad_generator import (  # noqa: E402
    CadPartConfig,
    GeneratedPartArtifacts,
    PreviewWireframeData,
    build_preview_wireframe_data,
    default_surface_config,
    generate_part_assets,
    make_default_cad_part_config,
)
from policies.surface_models import SurfaceConfig  # noqa: E402


class CadPartAuthoringGui:
    def __init__(
        self,
        *,
        output_root: Path | str = REPO_ROOT / "generated_cad",
        models_dir: Path | str = REPO_ROOT / "models",
        initial_config: CadPartConfig | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.models_dir = Path(models_dir)
        self.initial_config = initial_config or make_default_cad_part_config()
        self.preview_data: PreviewWireframeData | None = None
        self.last_saved_artifacts: GeneratedPartArtifacts | None = None
        self._suspend_callbacks = False

        self.figure = plt.figure("CAD Part Authoring", figsize=(16.0, 9.5))
        self.preview_axis = self.figure.add_axes([0.05, 0.08, 0.58, 0.84], projection="3d")
        self.status_axis = self.figure.add_axes([0.67, 0.86, 0.28, 0.10])
        self.status_axis.axis("off")
        self.status_text = self.status_axis.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            transform=self.status_axis.transAxes,
            family="monospace",
        )

        self.part_name_axis = self.figure.add_axes([0.67, 0.80, 0.26, 0.04])
        self.family_axis = self.figure.add_axes([0.67, 0.69, 0.26, 0.09])
        self.part_name_box = TextBox(
            self.part_name_axis,
            "Part Name",
            initial=self.initial_config.part_name,
        )
        self.part_name_box.on_submit(self._on_text_submit)
        self.family_radio = RadioButtons(
            self.family_axis,
            ("default", "random_gaussian_two_peak"),
            active=0,
        )
        self.family_radio.on_clicked(self._on_family_change)

        self.slider_axes: dict[str, plt.Axes] = {}
        self.sliders: dict[str, Slider] = {}
        self.slider_families: dict[str, str | None] = {}
        self.buttons: dict[str, Button] = {}

        self._build_controls()
        self._apply_config_to_widgets(self.initial_config)
        self._update_family_visibility()
        self._refresh_preview()

    def _build_controls(self) -> None:
        slider_specs = [
            ("length", "Block Length", 0.04, 0.20, self.initial_config.length, None, None),
            ("width", "Block Width", 0.04, 0.20, self.initial_config.width, None, None),
            ("height", "Block Height", 0.01, 0.08, self.initial_config.height, None, None),
            ("hole_radius", "Hole Radius", 0.002, 0.045, self.initial_config.hole_radius, None, None),
            ("nx", "Export Nx", 16, 220, self.initial_config.nx, 4, None),
            ("ny", "Export Ny", 16, 220, self.initial_config.ny, 4, None),
            (
                "surface_amp_default",
                "Surface Amp",
                0.0,
                0.02,
                self.initial_config.surface.amp,
                None,
                "default",
            ),
            (
                "surface_freq_x",
                "Surface Freq X",
                0.0,
                160.0,
                self.initial_config.surface.freq_x,
                None,
                "default",
            ),
            (
                "surface_freq_y",
                "Surface Freq Y",
                0.0,
                160.0,
                self.initial_config.surface.freq_y,
                None,
                "default",
            ),
            (
                "surface_amp_gaussian",
                "Surface Amp",
                0.0,
                0.02,
                self.initial_config.surface.amp,
                None,
                "random_gaussian_two_peak",
            ),
            (
                "surface_seed",
                "Surface Seed",
                0,
                99,
                self.initial_config.surface.seed,
                1,
                "random_gaussian_two_peak",
            ),
            (
                "gaussian_curvature",
                "Gaussian Curv",
                0.001,
                0.05,
                self.initial_config.surface.gaussian_curvature,
                None,
                "random_gaussian_two_peak",
            ),
            (
                "gaussian_peak_offset",
                "Peak Offset",
                0.0,
                0.04,
                self.initial_config.surface.gaussian_peak_offset,
                None,
                "random_gaussian_two_peak",
            ),
        ]

        slider_left = 0.69
        slider_width = 0.24
        slider_height = 0.025
        slider_top = 0.63
        slider_gap = 0.038

        for index, (key, label, vmin, vmax, initial, valstep, family) in enumerate(slider_specs):
            axis = self.figure.add_axes(
                [slider_left, slider_top - index * slider_gap, slider_width, slider_height]
            )
            slider = Slider(
                ax=axis,
                label=label,
                valmin=vmin,
                valmax=vmax,
                valinit=initial,
                valstep=valstep,
            )
            slider.on_changed(self._on_slider_change)
            self.slider_axes[key] = axis
            self.sliders[key] = slider
            self.slider_families[key] = family

        reset_axis = self.figure.add_axes([0.69, 0.03, 0.10, 0.05])
        save_axis = self.figure.add_axes([0.81, 0.03, 0.12, 0.05])
        self.buttons["reset"] = Button(reset_axis, "Reset Defaults")
        self.buttons["reset"].on_clicked(self._on_reset)
        self.buttons["save"] = Button(save_axis, "Save Part")
        self.buttons["save"].on_clicked(self._on_save)

    def _surface_family(self) -> str:
        return str(self.family_radio.value_selected)

    def _slider_int(self, key: str) -> int:
        return int(round(float(self.sliders[key].val)))

    def visible_slider_keys(self) -> set[str]:
        return {
            key
            for key, axis in self.slider_axes.items()
            if axis.get_visible()
        }

    def _build_surface_config(self) -> SurfaceConfig:
        family = self._surface_family()
        base = default_surface_config()
        if family == "default":
            return SurfaceConfig(
                family=family,
                base_height=0.0,
                amp=float(self.sliders["surface_amp_default"].val),
                freq_x=float(self.sliders["surface_freq_x"].val),
                freq_y=float(self.sliders["surface_freq_y"].val),
                seed=base.seed,
                gaussian_curvature=float(self.sliders["gaussian_curvature"].val),
                gaussian_peak_offset=float(self.sliders["gaussian_peak_offset"].val),
                origin_x=0.0,
                origin_y=0.0,
            )

        return SurfaceConfig(
            family=family,
            base_height=0.0,
            amp=float(self.sliders["surface_amp_gaussian"].val),
            freq_x=base.freq_x,
            freq_y=base.freq_y,
            seed=self._slider_int("surface_seed"),
            gaussian_curvature=float(self.sliders["gaussian_curvature"].val),
            gaussian_peak_offset=float(self.sliders["gaussian_peak_offset"].val),
            origin_x=0.0,
            origin_y=0.0,
        )

    def current_config(self) -> CadPartConfig:
        return CadPartConfig(
            part_name=self.part_name_box.text,
            length=float(self.sliders["length"].val),
            width=float(self.sliders["width"].val),
            height=float(self.sliders["height"].val),
            hole_radius=float(self.sliders["hole_radius"].val),
            nx=self._slider_int("nx"),
            ny=self._slider_int("ny"),
            surface=self._build_surface_config(),
            body_pos=self.initial_config.body_pos,
            body_quat=self.initial_config.body_quat,
        )

    def _set_status(self, text: str, *, color: str = "black") -> None:
        self.status_text.set_text(text)
        self.status_text.set_color(color)
        self.figure.canvas.draw_idle()

    def _update_family_visibility(self) -> None:
        active_family = self._surface_family()
        for key, axis in self.slider_axes.items():
            family = self.slider_families[key]
            axis.set_visible(family is None or family == active_family)

    def _draw_preview(self, config: CadPartConfig, preview_data: PreviewWireframeData | None, error: str | None) -> None:
        axis = self.preview_axis
        axis.clear()

        if preview_data is None:
            axis.text2D(0.05, 0.95, error or "Preview unavailable.", transform=axis.transAxes, va="top", color="crimson")
            z_limit = max(config.height, 0.01)
        else:
            axis.plot_wireframe(
                preview_data.top_surface.x,
                preview_data.top_surface.y,
                preview_data.top_surface.z,
                rstride=2,
                cstride=2,
                color="#1f77b4",
                linewidth=0.7,
            )
            axis.plot_wireframe(
                preview_data.bottom_surface.x,
                preview_data.bottom_surface.y,
                preview_data.bottom_surface.z,
                rstride=3,
                cstride=3,
                color="#2ca02c",
                linewidth=0.5,
            )
            for wall in preview_data.side_walls:
                axis.plot_wireframe(
                    wall.x,
                    wall.y,
                    wall.z,
                    rstride=1,
                    cstride=max(1, wall.x.shape[1] // 8),
                    color="#7f7f7f",
                    linewidth=0.7,
                )
            axis.plot_wireframe(
                preview_data.hole_wall.x,
                preview_data.hole_wall.y,
                preview_data.hole_wall.z,
                rstride=1,
                cstride=6,
                color="#d62728",
                linewidth=0.7,
            )
            z_limit = max(preview_data.z_max, config.height, 0.01)

        axis.set_title("Analytic Part Preview")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.set_xlim(-0.5 * config.length, 0.5 * config.length)
        axis.set_ylim(-0.5 * config.width, 0.5 * config.width)
        axis.set_zlim(0.0, 1.05 * z_limit)
        axis.set_box_aspect((config.length, config.width, max(z_limit, config.height)))
        axis.view_init(elev=25, azim=-58)

    def _refresh_preview(self) -> None:
        config = self.current_config()

        try:
            preview_data = build_preview_wireframe_data(config)
            preview_error = None
        except ValueError as exc:
            preview_data = None
            preview_error = str(exc)

        try:
            normalized_name = config.validate_part_name()
            name_error = None
        except ValueError as exc:
            normalized_name = "<invalid>"
            name_error = str(exc)

        self.preview_data = preview_data
        self._draw_preview(config, preview_data, preview_error)

        status_lines = [
            f"family: {config.surface.family}",
            f"part: {normalized_name}",
            f"output: {self.output_root / normalized_name}",
        ]
        if preview_error:
            status_lines.append(f"preview error: {preview_error}")
        elif name_error:
            status_lines.append(f"save blocked: {name_error}")
        else:
            status_lines.append("preview ready")

        status_color = "crimson" if (preview_error or name_error) else "darkgreen"
        self._set_status("\n".join(status_lines), color=status_color)

    def _apply_config_to_widgets(self, config: CadPartConfig) -> None:
        self._suspend_callbacks = True
        self.part_name_box.set_val(config.part_name)
        family_index = 0 if config.surface.family == "default" else 1
        self.family_radio.set_active(family_index)

        self.sliders["length"].set_val(config.length)
        self.sliders["width"].set_val(config.width)
        self.sliders["height"].set_val(config.height)
        self.sliders["hole_radius"].set_val(config.hole_radius)
        self.sliders["nx"].set_val(config.nx)
        self.sliders["ny"].set_val(config.ny)
        self.sliders["surface_amp_default"].set_val(config.surface.amp)
        self.sliders["surface_freq_x"].set_val(config.surface.freq_x)
        self.sliders["surface_freq_y"].set_val(config.surface.freq_y)
        self.sliders["surface_amp_gaussian"].set_val(config.surface.amp)
        self.sliders["surface_seed"].set_val(config.surface.seed)
        self.sliders["gaussian_curvature"].set_val(config.surface.gaussian_curvature)
        self.sliders["gaussian_peak_offset"].set_val(config.surface.gaussian_peak_offset)
        self._suspend_callbacks = False

    def _on_slider_change(self, _value) -> None:
        if self._suspend_callbacks:
            return
        self._refresh_preview()

    def _on_text_submit(self, _text: str) -> None:
        if self._suspend_callbacks:
            return
        self._refresh_preview()

    def _on_family_change(self, _label: str) -> None:
        if self._suspend_callbacks:
            return
        self._update_family_visibility()
        self._refresh_preview()

    def _on_reset(self, _event) -> None:
        self._apply_config_to_widgets(self.initial_config)
        self._update_family_visibility()
        self._refresh_preview()

    def save_current_part(self) -> GeneratedPartArtifacts | None:
        config = self.current_config()
        try:
            artifacts = generate_part_assets(
                config,
                output_root=self.output_root,
                models_dir=self.models_dir,
            )
        except Exception as exc:
            self.last_saved_artifacts = None
            self._set_status(f"save failed:\n{exc}", color="crimson")
            return None

        self.last_saved_artifacts = artifacts
        self._set_status(
            "\n".join(
                [
                    f"saved: {artifacts.part_name}",
                    f"cad: {artifacts.output_dir}",
                    f"xml: {artifacts.xml_path}",
                ]
            ),
            color="darkgreen",
        )
        return artifacts

    def _on_save(self, _event) -> None:
        self.save_current_part()

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close(self.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the CAD part authoring GUI.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "generated_cad",
        help="Directory where generated CAD folders should be written.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory where generated MuJoCo XML should be written.",
    )
    return parser.parse_args()


def main() -> CadPartAuthoringGui:
    args = parse_args()
    gui = CadPartAuthoringGui(
        output_root=args.output_root,
        models_dir=args.models_dir,
    )
    gui.show()
    return gui


if __name__ == "__main__":
    main()
