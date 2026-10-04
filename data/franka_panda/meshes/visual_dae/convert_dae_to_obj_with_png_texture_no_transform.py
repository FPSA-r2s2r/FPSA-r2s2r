import bpy
from pathlib import Path
import re
import shutil

in_dir = Path(".").resolve()
out_dir = in_dir.parent / "visual_obj_textured_png"
tex_dir = out_dir / "textures"

out_dir.mkdir(parents=True, exist_ok=True)
tex_dir.mkdir(parents=True, exist_ok=True)

dae_files = sorted(in_dir.glob("*.dae"))

print(f"Input dir:  {in_dir}")
print(f"Output dir: {out_dir}")
print(f"Texture dir: {tex_dir}")
print(f"Found {len(dae_files)} DAE files")


def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_") or "material"


def get_material_base_color(mat):
    """Return RGBA color from material."""
    if mat is None:
        return (0.8, 0.8, 0.8, 1.0)

    if hasattr(mat, "diffuse_color"):
        c = mat.diffuse_color
        return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))

    return (0.8, 0.8, 0.8, 1.0)


def find_existing_image_texture(mat):
    """Try to find an existing image texture in material nodes."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None

    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node.image

    return None


def save_solid_color_png(path: Path, rgba, size=8):
    """Create a tiny solid PNG texture from RGBA."""
    img = bpy.data.images.new(path.stem, width=size, height=size, alpha=True)

    pixels = []
    for _ in range(size * size):
        pixels.extend([rgba[0], rgba[1], rgba[2], rgba[3]])

    img.pixels.foreach_set(pixels)
    img.file_format = "PNG"
    img.filepath_raw = str(path)
    img.save()

    return img


def save_image_as_png(image, path: Path):
    """Save/copy existing Blender image as PNG if possible."""
    # If image has an existing PNG file, copy it.
    src = None
    try:
        if image.filepath:
            src = Path(bpy.path.abspath(image.filepath))
    except Exception:
        src = None

    if src is not None and src.exists() and src.suffix.lower() == ".png":
        shutil.copy2(src, path)
        new_img = bpy.data.images.load(str(path), check_existing=False)
        return new_img

    # Otherwise try saving the loaded image pixels as PNG.
    image.file_format = "PNG"
    image.filepath_raw = str(path)
    image.save()

    new_img = bpy.data.images.load(str(path), check_existing=False)
    return new_img


def force_material_to_png_texture(mat, mesh_stem, idx):
    """
    Ensure material has a PNG texture connected to Base Color.
    If original DAE had an image texture, convert/copy it to PNG.
    If original DAE only had diffuse color, create a solid-color PNG.
    """
    if mat is None:
        mat = bpy.data.materials.new(f"{mesh_stem}_mat_{idx:02d}")

    original_name = safe_name(mat.name)
    mat.name = f"{mesh_stem}_{idx:02d}_{original_name}"

    png_path = tex_dir / f"{mat.name}.png"

    existing_img = find_existing_image_texture(mat)

    if existing_img is not None:
        try:
            tex_img = save_image_as_png(existing_img, png_path)
        except Exception as e:
            print(f"  WARNING: failed to save existing texture for {mat.name}: {e}")
            rgba = get_material_base_color(mat)
            tex_img = save_solid_color_png(png_path, rgba)
    else:
        rgba = get_material_base_color(mat)
        tex_img = save_solid_color_png(png_path, rgba)

    # Rebuild material node tree so OBJ/MTL exporter definitely writes map_Kd.
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.image = tex_img
    tex_node.location = (-400, 100)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (-100, 100)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (200, 100)

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    rgba = get_material_base_color(mat)
    mat.diffuse_color = rgba

    return mat, png_path


for dae_path in dae_files:
    print(f"\nConverting {dae_path.name}")

    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import DAE.
    # IMPORTANT: no transform_apply, no rotation, no scaling.
    # Do not use import_units=True to avoid unexpected unit scaling.
    bpy.ops.wm.collada_import(filepath=str(dae_path))

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not mesh_objects:
        print(f"  WARNING: no mesh objects imported from {dae_path.name}")
        continue

    # Do NOT apply transforms.
    # We keep original imported object transforms as-is.
    for obj in mesh_objects:
        print(
            f"  object={obj.name}, "
            f"loc={tuple(round(x, 6) for x in obj.location)}, "
            f"rot={tuple(round(x, 6) for x in obj.rotation_euler)}, "
            f"scale={tuple(round(x, 6) for x in obj.scale)}"
        )

    # Ensure each material has a PNG texture.
    for obj in mesh_objects:
        if not obj.data.materials:
            mat = bpy.data.materials.new(f"{dae_path.stem}_mat_00")
            obj.data.materials.append(mat)

        for i, mat in enumerate(obj.data.materials):
            new_mat, png_path = force_material_to_png_texture(mat, dae_path.stem, i)
            obj.data.materials[i] = new_mat
            print(f"  material {new_mat.name} -> {png_path.name}")

    # Select mesh objects only
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    obj_path = out_dir / f"{dae_path.stem}.obj"

    # Blender 4.x OBJ exporter.
    # forward_axis='Y', up_axis='Z' avoids the usual Blender OBJ axis remap.
    # global_scale=1.0 avoids scaling.
    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        export_colors=True,
        apply_modifiers=False,
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
        path_mode="RELATIVE",
    )

    print(f"  wrote {obj_path}")

print("\nDone. Converted DAE meshes to OBJ + MTL + PNG textures without applying transforms.")
