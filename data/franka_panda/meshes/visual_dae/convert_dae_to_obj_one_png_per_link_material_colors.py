import bpy
from pathlib import Path
import re
import math

in_dir = Path(".").resolve()
out_dir = in_dir.parent / "visual_obj_textured_png"
out_dir.mkdir(parents=True, exist_ok=True)

dae_files = sorted(in_dir.glob("*.dae"))

print(f"Input dir:  {in_dir}")
print(f"Output dir: {out_dir}")
print(f"Found {len(dae_files)} DAE files")


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_") or "material"


def get_material_color(mat):
    """
    Robustly get material base/diffuse color from Blender material.
    Important: force alpha to 1.0 to avoid transparent PyBullet materials.
    """
    fallback = (0.8, 0.8, 0.8, 1.0)

    if mat is None:
        return fallback

    # Try Principled BSDF base color if nodes exist
    if mat.use_nodes and mat.node_tree is not None:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                # Blender 4.x usually has "Base Color"
                if "Base Color" in node.inputs:
                    val = node.inputs["Base Color"].default_value
                    return (
                        clamp01(val[0]),
                        clamp01(val[1]),
                        clamp01(val[2]),
                        1.0,
                    )

    # Try material diffuse_color
    if hasattr(mat, "diffuse_color"):
        c = mat.diffuse_color
        return (clamp01(c[0]), clamp01(c[1]), clamp01(c[2]), 1.0)

    return fallback


def create_color_atlas_png(path: Path, colors, cell_size=256, padding=32):
    """
    Create one PNG per link.
    Each material gets one vertical color cell with padding.
    Larger cells + padding reduce PyBullet/OpenGL texture bleeding.
    """
    n = max(1, len(colors))
    width = cell_size * n
    height = cell_size

    img = bpy.data.images.new(path.stem, width=width, height=height, alpha=True)
    pixels = [0.0] * (width * height * 4)

    for y in range(height):
        for x in range(width):
            mat_id = min(n - 1, x // cell_size)
            r, g, b, a = colors[mat_id]
            idx = (y * width + x) * 4

            # Always opaque. Avoid transparent/black fringe.
            pixels[idx + 0] = r
            pixels[idx + 1] = g
            pixels[idx + 2] = b
            pixels[idx + 3] = 1.0

    img.pixels.foreach_set(pixels)
    img.file_format = "PNG"
    img.filepath_raw = str(path)
    img.save()

    return img, width, height


def assign_uv_by_material(obj, material_offset, material_count, cell_size=256):
    """
    Assign each polygon to a tiny non-degenerate UV patch inside its material cell.

    Why:
    - all UVs at exactly one point can create degenerate UV triangles
    - degenerate UVs can cause black grid / black seam artifacts in PyBullet/OpenGL
    - this keeps UVs far from color-cell boundaries while avoiding degenerate UVs
    """
    mesh = obj.data

    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    uv_layer = mesh.uv_layers.active.data
    n = max(1, material_count)

    # Tiny patch size inside one material cell.
    # Must be much smaller than 1/n to avoid crossing into neighbor material color.
    patch_u = 0.20 / n
    patch_v = 0.20

    for poly in mesh.polygons:
        global_mat_id = material_offset + poly.material_index
        global_mat_id = max(0, min(n - 1, global_mat_id))

        center_u = (global_mat_id + 0.5) / n
        center_v = 0.5

        # Non-degenerate local UV patch.
        coords = [
            (center_u - patch_u, center_v - patch_v),
            (center_u + patch_u, center_v - patch_v),
            (center_u + patch_u, center_v + patch_v),
            (center_u - patch_u, center_v + patch_v),
        ]

        for k, loop_idx in enumerate(poly.loop_indices):
            uv_layer[loop_idx].uv = coords[k % 4]

    mesh.update()

def make_single_texture_material(mesh_stem, image):
    """
    Create exactly one material using exactly one PNG texture.
    """
    mat = bpy.data.materials.new(f"{mesh_stem}_single_texture")
    mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.image = image
    tex_node.location = (-400, 100)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (-100, 100)

    # Force fully opaque
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 1.0

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (200, 100)

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat

def fix_obj_and_mtl(obj_path: Path, mtl_path: Path, png_name: str):
    """
    Make MTL simple and PyBullet-friendly:
    one material, one map_Kd, opaque.
    Also remove smoothing groups to reduce dark seam artifacts.
    """
    if mtl_path.exists():
        mtl_path.write_text(
            "\n".join([
                "newmtl material_0",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "Ke 0.000000 0.000000 0.000000",
                "Ns 1.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 1",
                f"map_Kd {png_name}",
                "",
            ])
        )

    if obj_path.exists():
        lines = obj_path.read_text(errors="ignore").splitlines()
        fixed = []

        inserted_s_off = False

        for line in lines:
            if line.startswith("usemtl "):
                fixed.append("usemtl material_0")
            elif line.startswith("s "):
                if not inserted_s_off:
                    fixed.append("s off")
                    inserted_s_off = True
            else:
                fixed.append(line)

        if not inserted_s_off:
            # Put s off near the top, after mtllib if possible.
            out = []
            added = False
            for line in fixed:
                out.append(line)
                if line.startswith("mtllib ") and not added:
                    out.append("s off")
                    added = True
            fixed = out if added else ["s off"] + fixed

        obj_path.write_text("\n".join(fixed) + "\n")


for dae_path in dae_files:
    mesh_stem = dae_path.stem
    print(f"\nConverting {dae_path.name}")

    # Clear scene
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import DAE.
    # Do not use import_units=True.
    # Do not apply transforms.
    bpy.ops.wm.collada_import(filepath=str(dae_path))

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not mesh_objects:
        print(f"  WARNING: no mesh objects imported from {dae_path.name}")
        continue

    for obj in mesh_objects:
        print(
            f"  object={obj.name}, "
            f"loc={tuple(round(x, 6) for x in obj.location)}, "
            f"rot={tuple(round(x, 6) for x in obj.rotation_euler)}, "
            f"scale={tuple(round(x, 6) for x in obj.scale)}"
        )

    # Collect all original materials/colors before modifying anything.
    all_colors = []
    material_offsets = {}

    for obj in mesh_objects:
        material_offsets[obj.name] = len(all_colors)

        if not obj.data.materials:
            mat = bpy.data.materials.new(f"{mesh_stem}_default")
            mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            obj.data.materials.append(mat)

        for mat in obj.data.materials:
            color = get_material_color(mat)
            all_colors.append(color)
            print(f"  material color {mat.name if mat else 'None'} -> {color}")

    if not all_colors:
        all_colors = [(0.8, 0.8, 0.8, 1.0)]

    # Create one PNG per link, same folder as OBJ/MTL.
    png_path = out_dir / f"{mesh_stem}.png"
    atlas_img, _, _ = create_color_atlas_png(png_path, all_colors, cell_size=256)

    # Assign UVs based on original face material indices.
    total_material_count = len(all_colors)
    for obj in mesh_objects:
        assign_uv_by_material(
            obj,
            material_offset=material_offsets[obj.name],
            material_count=total_material_count,
            cell_size=256,
        )

    # Replace materials with a single texture material.
    single_mat = make_single_texture_material(mesh_stem, atlas_img)

    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(single_mat)
        for poly in obj.data.polygons:
            poly.material_index = 0

    # Export OBJ.
    obj_path = out_dir / f"{mesh_stem}.obj"

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        export_colors=False,
        apply_modifiers=False,
        global_scale=1.0,

        # This is the least disruptive setting for URDF-style assets.
        # No transform is applied to the actual Blender objects.
        forward_axis="Y",
        up_axis="Z",

        path_mode="RELATIVE",
    )

    mtl_path = out_dir / f"{mesh_stem}.mtl"
    fix_obj_and_mtl(obj_path, mtl_path, f"{mesh_stem}.png")

    print(f"  wrote {obj_path.name}")
    print(f"  wrote {mtl_path.name}")
    print(f"  wrote {png_path.name}")

print("\nDone. Each link has exactly one PNG texture in the same folder as OBJ/MTL.")
