import bpy
from pathlib import Path
import re

in_dir = Path(".").resolve()
out_dir = in_dir.parent / "visual_obj_textured_png"
out_dir.mkdir(parents=True, exist_ok=True)

dae_files = sorted(in_dir.glob("*.dae"))

print(f"Input dir:  {in_dir}")
print(f"Output dir: {out_dir}")
print(f"Found {len(dae_files)} DAE files")


def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_") or "material"


def get_material_base_color(mat):
    if mat is None:
        return (0.8, 0.8, 0.8, 1.0)

    if hasattr(mat, "diffuse_color"):
        c = mat.diffuse_color
        return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))

    return (0.8, 0.8, 0.8, 1.0)


def ensure_uv(obj):
    """
    Ensure object has UVs for baking.
    This does not apply object transform, rotate, or scale the mesh.
    """
    if obj.type != "MESH":
        return

    if obj.data.uv_layers:
        return

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=1.15192,  # about 66 degrees
        island_margin=0.03,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def add_bake_target_to_material(mat, image):
    """
    Add an image texture node to material and make it active,
    so Blender bake writes into this image.
    """
    if mat is None:
        mat = bpy.data.materials.new("bake_material")

    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.name = "BAKE_TARGET_ONE_LINK_TEXTURE"
    tex_node.label = "BAKE_TARGET_ONE_LINK_TEXTURE"
    tex_node.image = image
    nodes.active = tex_node

    return mat


def make_single_texture_material(mesh_stem, image):
    """
    After baking, replace all materials with a single material
    that uses exactly one PNG texture.
    """
    mat = bpy.data.materials.new(f"{mesh_stem}_single_texture")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.image = image
    tex_node.location = (-400, 100)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (-100, 100)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (200, 100)

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    return mat


def fix_mtl_to_use_one_png(mtl_path: Path, png_name: str):
    """
    Make sure the exported MTL references exactly the single PNG
    in the same folder as the OBJ/MTL.
    """
    if not mtl_path.exists():
        return

    lines = mtl_path.read_text(errors="ignore").splitlines()

    fixed = []
    has_map_kd = False

    for line in lines:
        stripped = line.strip()

        # Remove extra texture maps that may point elsewhere.
        if stripped.startswith(("map_Ka", "map_Ks", "map_Bump", "bump", "map_d", "disp", "decal")):
            continue

        if stripped.startswith("map_Kd"):
            if not has_map_kd:
                fixed.append(f"map_Kd {png_name}")
                has_map_kd = True
            continue

        fixed.append(line)

    if not has_map_kd:
        fixed.append(f"map_Kd {png_name}")

    mtl_path.write_text("\n".join(fixed) + "\n")


for dae_path in dae_files:
    mesh_stem = dae_path.stem
    print(f"\nConverting {dae_path.name}")

    # Clear scene
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.ops.object.mode_set.poll() else None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import DAE.
    # IMPORTANT:
    # - no import_units=True
    # - no transform_apply
    # - no manual rotation/scale/location edits
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

    # One PNG per link, in the same output folder as OBJ/MTL.
    png_path = out_dir / f"{mesh_stem}.png"

    # Create bake image.
    bake_image = bpy.data.images.new(
        name=f"{mesh_stem}_baked_texture",
        width=1024,
        height=1024,
        alpha=True,
        float_buffer=False,
    )
    bake_image.file_format = "PNG"
    bake_image.filepath_raw = str(png_path)
    bake_image.filepath = str(png_path)

    # Ensure UVs and add bake target to every material.
    for obj in mesh_objects:
        ensure_uv(obj)

        if not obj.data.materials:
            mat = bpy.data.materials.new(f"{mesh_stem}_default_source_material")
            mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            obj.data.materials.append(mat)

        for i, mat in enumerate(obj.data.materials):
            if mat is None:
                mat = bpy.data.materials.new(f"{mesh_stem}_source_mat_{i:02d}")
                mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)

            obj.data.materials[i] = add_bake_target_to_material(mat, bake_image)

    # Select all mesh objects for baking.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    # Use Cycles for baking.
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1

    # Bake diffuse color only: this preserves material colors and existing base-color textures
    # into exactly one PNG per link.
    print(f"  baking one texture -> {png_path.name}")
    try:
        bpy.ops.object.bake(
            type="DIFFUSE",
            pass_filter={"COLOR"},
            margin=8,
            use_clear=True,
        )
        bake_image.save()
    except Exception as e:
        print(f"  WARNING: bake failed for {dae_path.name}: {e}")
        print("  Falling back to a solid color PNG from the first material.")

        first_mat = None
        for obj in mesh_objects:
            if obj.data.materials:
                first_mat = obj.data.materials[0]
                break

        rgba = get_material_base_color(first_mat)
        pixels = []
        for _ in range(1024 * 1024):
            pixels.extend([rgba[0], rgba[1], rgba[2], rgba[3]])

        bake_image.pixels.foreach_set(pixels)
        bake_image.save()

    # Replace all source materials with one single texture material.
    single_mat = make_single_texture_material(mesh_stem, bake_image)

    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(single_mat)

        for poly in obj.data.polygons:
            poly.material_index = 0

    # Export OBJ into the same folder.
    obj_path = out_dir / f"{mesh_stem}.obj"

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    # IMPORTANT:
    # forward_axis='Y', up_axis='Z' avoids the usual Blender OBJ axis remap.
    # global_scale=1.0 avoids scaling.
    # We do not apply transforms.
    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        export_colors=False,
        apply_modifiers=False,
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
        path_mode="RELATIVE",
    )

    mtl_path = out_dir / f"{mesh_stem}.mtl"
    fix_mtl_to_use_one_png(mtl_path, f"{mesh_stem}.png")

    print(f"  wrote {obj_path.name}")
    print(f"  wrote {mtl_path.name}")
    print(f"  wrote {png_path.name}")

print("\nDone. Each link has exactly: link.obj + link.mtl + link.png in the same folder.")
