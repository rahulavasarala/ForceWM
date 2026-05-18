#Generate cad for a hole given specific dimensions

# The dimensions will be length, width, height, diameter, and depth of the hole

# We need to be able to programatically generate the cad file, and generate the convex decomposition
# of these cad files --- 

import numpy as np
import trimesh
from pathlib import Path
from trimesh.decomposition import convex_decomposition
import argparse


def default_surface(x, y, amp=0.003, freq=8.0):
    # Non-flat top surface
    return amp * np.sin(freq * x) * np.cos(freq * y)


def make_two_peak_gaussian_surface(
    rng,
    amp=0.002,
    curvature=0.015,
    peak_offset=0.02,
):
    sigma = max(curvature, 1e-4)
    centers = rng.uniform(-peak_offset, peak_offset, size=(2, 2))
    weights = rng.uniform(0.7, 1.3, size=2)
    weights = weights / np.sum(weights)
    peak_amps = amp * weights

    def surface(x, y):
        z = np.zeros_like(x, dtype=float)
        for i in range(2):
            dx = x - centers[i, 0]
            dy = y - centers[i, 1]
            z += peak_amps[i] * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        return z

    return surface


def build_surface_function(args):
    if args.surface_family == "default":
        return lambda x, y: args.surface_amp * np.sin(args.surface_freq_x * x) * np.cos(args.surface_freq_y * y)

    rng = np.random.default_rng(args.surface_seed)
    return make_two_peak_gaussian_surface(
        rng=rng,
        amp=args.surface_amp,
        curvature=args.gaussian_curvature,
        peak_offset=args.gaussian_peak_offset,
    )


def _add_quad(faces, a, b, c, d):
    faces.append([a, b, c])
    faces.append([a, c, d])


def make_hole_block(
    L=0.08,
    W=0.08,
    H=0.03,
    hole_r=0.015,
    nx=160,
    ny=160,
    ntheta=128,
    surface_fn=default_surface,
):
    dx = L / nx
    dy = W / ny

    x_edges = np.linspace(-L / 2, L / 2, nx + 1)
    y_edges = np.linspace(-W / 2, W / 2, ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    top_surface = H + surface_fn(Xc, Yc)
    z_max = float(np.max(top_surface))

    dz = min(dx, dy)
    nz = max(2, int(np.ceil(z_max / dz)))
    z_edges = np.linspace(0.0, z_max, nz + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    occupancy_margin = 0.5 * (z_edges[1] - z_edges[0])

    occupancy = np.zeros((nx, ny, nz), dtype=bool)
    outside_hole = Xc * Xc + Yc * Yc >= hole_r * hole_r

    for k, zc in enumerate(z_centers):
        occupancy[:, :, k] = outside_hole & (zc <= top_surface + occupancy_margin)

    vertices = []
    faces = []
    vertex_ids = {}

    def vertex_index(ix, iy, iz):
        key = (ix, iy, iz)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append([x_edges[ix], y_edges[iy], z_edges[iz]])
        return vertex_ids[key]

    def emit_face(corners):
        a, b, c, d = [vertex_index(*corner) for corner in corners]
        _add_quad(faces, a, b, c, d)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not occupancy[i, j, k]:
                    continue

                if i == 0 or not occupancy[i - 1, j, k]:
                    emit_face(
                        [
                            (i, j, k),
                            (i, j, k + 1),
                            (i, j + 1, k + 1),
                            (i, j + 1, k),
                        ]
                    )
                if i == nx - 1 or not occupancy[i + 1, j, k]:
                    emit_face(
                        [
                            (i + 1, j, k),
                            (i + 1, j + 1, k),
                            (i + 1, j + 1, k + 1),
                            (i + 1, j, k + 1),
                        ]
                    )
                if j == 0 or not occupancy[i, j - 1, k]:
                    emit_face(
                        [
                            (i, j, k),
                            (i + 1, j, k),
                            (i + 1, j, k + 1),
                            (i, j, k + 1),
                        ]
                    )
                if j == ny - 1 or not occupancy[i, j + 1, k]:
                    emit_face(
                        [
                            (i, j + 1, k),
                            (i, j + 1, k + 1),
                            (i + 1, j + 1, k + 1),
                            (i + 1, j + 1, k),
                        ]
                    )
                if k == 0 or not occupancy[i, j, k - 1]:
                    emit_face(
                        [
                            (i, j, k),
                            (i, j + 1, k),
                            (i + 1, j + 1, k),
                            (i + 1, j, k),
                        ]
                    )
                if k == nz - 1 or not occupancy[i, j, k + 1]:
                    emit_face(
                        [
                            (i, j, k + 1),
                            (i + 1, j, k + 1),
                            (i + 1, j + 1, k + 1),
                            (i, j + 1, k + 1),
                        ]
                    )

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        raise ValueError("Generated hole mesh is not watertight")

    return mesh


def make_smooth_visual_hole_block(
    L=0.08,
    W=0.08,
    H=0.03,
    hole_r=0.015,
    nx=160,
    ny=160,
    surface_fn=default_surface,
):
    x_edges = np.linspace(-L / 2, L / 2, nx + 1)
    y_edges = np.linspace(-W / 2, W / 2, ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    occupied = Xc * Xc + Yc * Yc >= hole_r * hole_r

    Xv, Yv = np.meshgrid(x_edges, y_edges, indexing="ij")
    top_heights = H + surface_fn(Xv, Yv)

    vertices = []
    faces = []
    vertex_ids = {}

    def vertex_index(i, j, top):
        key = (i, j, top)
        if key not in vertex_ids:
            z = float(top_heights[i, j]) if top else 0.0
            vertex_ids[key] = len(vertices)
            vertices.append([float(x_edges[i]), float(y_edges[j]), z])
        return vertex_ids[key]

    def add_top_cell(i, j):
        a = vertex_index(i, j, True)
        b = vertex_index(i + 1, j, True)
        c = vertex_index(i + 1, j + 1, True)
        d = vertex_index(i, j + 1, True)
        _add_quad(faces, a, b, c, d)

    def add_bottom_cell(i, j):
        a = vertex_index(i, j, False)
        b = vertex_index(i, j + 1, False)
        c = vertex_index(i + 1, j + 1, False)
        d = vertex_index(i + 1, j, False)
        _add_quad(faces, a, b, c, d)

    def add_wall(i0, j0, i1, j1):
        a = vertex_index(i0, j0, False)
        b = vertex_index(i1, j1, False)
        c = vertex_index(i1, j1, True)
        d = vertex_index(i0, j0, True)
        _add_quad(faces, a, b, c, d)

    for i in range(nx):
        for j in range(ny):
            if not occupied[i, j]:
                continue

            add_top_cell(i, j)
            add_bottom_cell(i, j)

            if i == 0 or not occupied[i - 1, j]:
                add_wall(i, j, i, j + 1)
            if i == nx - 1 or not occupied[i + 1, j]:
                add_wall(i + 1, j + 1, i + 1, j)
            if j == 0 or not occupied[i, j - 1]:
                add_wall(i + 1, j, i, j)
            if j == ny - 1 or not occupied[i, j + 1]:
                add_wall(i, j + 1, i + 1, j + 1)

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        raise ValueError("Generated smooth visual hole mesh is not watertight")

    return mesh


def export_convex_parts(mesh, out_dir, prefix="part", max_hulls=32):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        parts = convex_decomposition(
            mesh,
            maxConvexHulls=max_hulls,
            resolution=400000,
            maxNumVerticesPerCH=64,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "vhacdx":
            raise ModuleNotFoundError(
                "Convex decomposition requires the optional 'vhacdx' package. "
                "Install it in the active environment to export convex parts."
            ) from exc
        raise

    for i, part in enumerate(parts):
        if isinstance(part, dict):
            part_mesh = trimesh.Trimesh(**part)
        else:
            part_mesh = part

        part_mesh.export(out_dir / f"{prefix}_{i}.stl")

    return parts


def generate_parametric_hole_xml(
    part_name,
    output_path,
    visual_mesh_path,
    collision_mesh_dir,
    body_pos=(0.4, 0.0, 0.3),
    body_quat=(1.0, 0.0, 0.0, 0.0),
):
    output_path = Path(output_path)
    collision_mesh_dir = Path(collision_mesh_dir)
    collision_meshes = sorted(collision_mesh_dir.glob("geometry_*.stl"))

    visual_rel = Path(
        "../../generated_cad",
        part_name,
        Path(visual_mesh_path).name,
    ).as_posix()

    asset_lines = [f'        <mesh name="holesim" file="{visual_rel}"/>']
    geom_lines = ['            <geom mesh="holesim" class="visual" rgba="0.55 0.35 0.2 1"/>']

    for mesh_path in collision_meshes:
        mesh_name = mesh_path.stem
        mesh_rel = Path(
            "../../generated_cad",
            part_name,
            "hole_block_convex",
            mesh_path.name,
        ).as_posix()
        asset_lines.append(f'        <mesh name="{mesh_name}" file="{mesh_rel}"/>')
        geom_lines.append(f'            <geom mesh="{mesh_name}" class="collision" friction="0 0 0"/>')

    pos_str = " ".join(str(v) for v in body_pos)
    quat_str = " ".join(str(v) for v in body_quat)

    xml_text = "\n".join(
        [
            '<mujoco model="hole">',
            "    <asset>",
            *asset_lines,
            "    </asset>",
            "",
            "    <worldbody>",
            f'        <body name="task" pos="{pos_str}" quat="{quat_str}">',
            *geom_lines,
            "        </body>",
            "    </worldbody>",
            "</mujoco>",
            "",
        ]
    )

    output_path.write_text(xml_text)
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CAD models with holes and export convex parts.")
    parser.add_argument("--part-name", type=str, default="default_part", help="Directory to save generated CAD files.")
    parser.add_argument("--nx", type=int, default=160, help="Number of x samples for the generated block.")
    parser.add_argument("--ny", type=int, default=160, help="Number of y samples for the generated block.")
    parser.add_argument(
        "--surface-family",
        type=str,
        choices=["default", "random_gaussian_two_peak"],
        default="default",
        help="Family of smooth surfaces to generate.",
    )
    parser.add_argument("--surface-amp", type=float, default=0.002, help="Overall amplitude of the top surface.")
    parser.add_argument("--surface-freq-x", type=float, default=80.0, help="X frequency for the default surface.")
    parser.add_argument("--surface-freq-y", type=float, default=60.0, help="Y frequency for the default surface.")
    parser.add_argument("--surface-seed", type=int, default=0, help="Seed used for randomized surface families.")
    parser.add_argument(
        "--gaussian-curvature",
        type=float,
        default=0.015,
        help="Gaussian spread for the two random peaks. Smaller values create sharper curvature.",
    )
    parser.add_argument(
        "--gaussian-peak-offset",
        type=float,
        default=0.02,
        help="Maximum absolute x/y offset of each random Gaussian peak from the center.",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    base_path = repo_root / "generated_cad"
    base_path.mkdir(exist_ok=True)

    part_name = Path(args.part_name)

    out = base_path / part_name

    out.mkdir(exist_ok=True)

    surface_fn = build_surface_function(args)

    surface_mesh = make_smooth_visual_hole_block(
        L=0.10,
        W=0.08,
        H=0.03,
        hole_r=0.018,
        nx=args.nx,
        ny=args.ny,
        surface_fn=surface_fn,
    )

    surface_mesh.export(out / "hole_block.stl")
    export_convex_parts(surface_mesh, out / "hole_block_convex", "geometry")
    generate_parametric_hole_xml(
        part_name=part_name.name,
        output_path=repo_root / "models" / f"{part_name.name}.xml",
        visual_mesh_path=out / "hole_block.stl",
        collision_mesh_dir=out / "hole_block_convex",
    )
