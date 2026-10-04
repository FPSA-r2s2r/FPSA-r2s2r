import bpy
import os
import math
import shutil
from pathlib import Path


# =========================
# User config
# =========================

env_mesh_path = "data/background/recon_background_simplied.ply"

OUT_DIR = "data/background/recon_background_baked"
OUT_OBJ = "recon_background_baked.obj"
OUT_TEXTURE = "recon_background_baked.png"

# The reference mesh has 523k faces.
# 8192 is a reliable default; use 16384 if artifacts remain and memory permits.
TEXTURE_SIZE = 8192

# Expand baked edges to reduce black-border sampling in PyBullet.
# For 523k faces, start conservatively because a large margin wastes atlas space.
BAKE_MARGIN_PX = 8

# UV-island spacing as a fraction of UV space.
# BAKE_MARGIN_PX / TEXTURE_SIZE is usually appropriate.
UV_ISLAND_MARGIN = BAKE_MARGIN_PX / TEXTURE_SIZE

# Ignore or repair PLY vertex alpha to prevent transparent regions from turning black in PyBullet.
FORCE_ALPHA_TO_ONE = True

# Whether to clean the mesh
CLEAN_MESH = True


# =========================
# Helpers
# =========================

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_ply(filepath):
    filepath = str(Path(filepath).resolve())

    # Blender 4.x
    try:
        bpy.ops.wm.ply_import(filepath=filepath)
        return bpy.context.object
    except Exception:
        pass

    # Blender 3.x / older
    try:
        bpy.ops.import_mesh.ply(filepath=filepath)
        return bpy.context.object
    except Exception as e:
        raise RuntimeError(f"Failed to import PLY: {filepath}") from e


def get_color_attribute(mesh):
    """
    Return the color attribute/layer name used for vertex colors.
    Compatible with Blender 3.x/4.x as much as possible.
    """
    # New API: mesh.color_attributes
    if hasattr(mesh, "color_attributes") and len(mesh.color_attributes) > 0:
        names = [a.name for a in mesh.color_attributes]
        preferred = ["Col", "Color", "color", "RGBA", "rgb"]

        for name in preferred:
            if name in names:
                return mesh.color_attributes[name]

        active = getattr(mesh.color_attributes, "active_color", None)
        if active is not None:
            return active

        return mesh.color_attributes[0]

    # Legacy API: mesh.vertex_colors
    if hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
        active = getattr(mesh.vertex_colors, "active", None)
        if active is not None:
            return active
        return mesh.vertex_colors[0]

    raise RuntimeError(
        "No vertex color / color attribute found on imported mesh. "
        "Please check whether Blender imported the PLY colors."
    )


def force_color_alpha_to_one(color_attr):
    """
    Set color alpha to 1.0 where possible.
    This avoids transparent/black artifacts during bake or in PyBullet.
    """
    if not hasattr(color_attr, "data"):
        return

    for item in color_attr.data:
        if hasattr(item, "color") and len(item.color) >= 4:
            item.color[3] = 1.0


def clean_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    # Basic cleanup
    try:
        bpy.ops.mesh.remove_doubles(threshold=1e-8)
    except Exception:
        pass

    try:
        bpy.ops.mesh.delete_loose()
    except Exception:
        pass

    bpy.ops.object.mode_set(mode="OBJECT")

    # Remove zero-area faces / degenerate geometry via bmesh would be more precise,
    # but this keeps dependencies minimal.


def smart_uv_project(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    # Different Blender versions have slightly different operator arguments.
    tried = False

    # Blender 4.x style
    try:
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66.0),
            margin_method="SCALED",
            island_margin=UV_ISLAND_MARGIN,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=True,
        )
        tried = True
    except TypeError:
        pass

    # Blender 3.x / older style
    if not tried:
        try:
            bpy.ops.uv.smart_project(
                angle_limit=math.radians(66.0),
                island_margin=UV_ISLAND_MARGIN,
                user_area_weight=0.0,
                use_aspect=True,
                stretch_to_bounds=True,
            )
            tried = True
        except TypeError:
            pass

    # Minimal fallback
    if not tried:
        bpy.ops.uv.smart_project()

    # Optional pack islands
    try:
        bpy.ops.uv.pack_islands(margin=UV_ISLAND_MARGIN)
    except Exception:
        pass

    bpy.ops.object.mode_set(mode="OBJECT")


def create_bake_material(obj, color_attr_name, image):
    mat = bpy.data.materials.new("bake_vertex_color_material")
    mat.use_nodes = True
    mat.blend_method = "OPAQUE"
    mat.use_screen_refraction = False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError("Cannot find Principled BSDF node.")

    # Attribute node reads Blender color attribute by name.
    attr_node = nodes.new(type="ShaderNodeAttribute")
    attr_node.attribute_name = color_attr_name
    attr_node.location = (-600, 150)

    # Image node is the bake target.
    img_node = nodes.new(type="ShaderNodeTexImage")
    img_node.image = image
    img_node.location = (-600, -150)

    # Important: active image texture node is where Blender bakes.
    nodes.active = img_node
    img_node.select = True

    # Vertex color -> Base Color
    links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])

    # Force opaque material
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 1.0

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 1.0

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    return mat, attr_node, img_node, bsdf


def bake_diffuse_color(obj, image):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Use Cycles for baking
    bpy.context.scene.render.engine = "CYCLES"

    # Keep bake deterministic and fast
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.use_denoising = False

    # Avoid color-management surprises when saving
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1

    # Bake only diffuse color, no lighting/shadows
    bpy.ops.object.bake(
        type="DIFFUSE",
        pass_filter={"COLOR"},
        margin=BAKE_MARGIN_PX,
        use_clear=True,
    )


def switch_material_to_texture(mat, img_node, bsdf):
    """
    After baking, make exported OBJ material use the baked texture,
    not the original vertex color attribute.
    """
    links = mat.node_tree.links

    # Remove existing links into Base Color
    if "Base Color" in bsdf.inputs:
        for link in list(bsdf.inputs["Base Color"].links):
            links.remove(link)

        links.new(img_node.outputs["Color"], bsdf.inputs["Base Color"])

    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 1.0


def export_obj(obj, out_obj_path):
    """
    Export OBJ without changing the original coordinate transform.

    Key idea:
    - Blender default OBJ export may convert axes.
    - For PyBullet / original PLY coordinate preservation, use:
        forward_axis='Y', up_axis='Z'
      or for older Blender:
        axis_forward='Y', axis_up='Z'
    - global_scale=1.0
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    out_obj_path = str(Path(out_obj_path).resolve())

    # Keep current object transform untouched
    original_matrix_world = obj.matrix_world.copy()
    original_location = obj.location.copy()
    original_rotation_euler = obj.rotation_euler.copy()
    original_scale = obj.scale.copy()

    try:
        # Blender 4.x
        try:
            bpy.ops.wm.obj_export(
                filepath=out_obj_path,
                export_selected_objects=True,
                export_uv=True,
                export_normals=True,
                export_materials=True,
                path_mode="RELATIVE",

                # Important: no coordinate axis conversion
                forward_axis="Y",
                up_axis="Z",
                global_scale=1.0,

                # Do not apply modifiers/transforms unexpectedly
                apply_modifiers=False,
            )
            return
        except TypeError:
            pass
        except Exception:
            pass

        # Blender 3.x / older
        try:
            bpy.ops.export_scene.obj(
                filepath=out_obj_path,
                use_selection=True,
                use_materials=True,
                use_uvs=True,
                use_normals=True,
                path_mode="RELATIVE",

                # Important: no coordinate axis conversion
                axis_forward="Y",
                axis_up="Z",
                global_scale=1.0,

                # Do not apply mesh modifiers unexpectedly
                use_mesh_modifiers=False,
            )
            return
        except TypeError:
            pass
        except Exception as e:
            raise RuntimeError(f"Failed to export OBJ: {out_obj_path}") from e

    finally:
        # Restore Blender-side object transform no matter what
        obj.matrix_world = original_matrix_world
        obj.location = original_location
        obj.rotation_euler = original_rotation_euler
        obj.scale = original_scale
        bpy.context.view_layer.update()

def write_simple_mtl_and_patch_obj(out_obj_path, texture_filename):
    """
    Blender's OBJ/MTL exporter behavior varies by version.
    This makes the material explicit and PyBullet-friendly.
    """
    out_obj_path = Path(out_obj_path)
    mtl_path = out_obj_path.with_suffix(".mtl")
    mtl_name = mtl_path.name
    material_name = "baked_vertex_color"

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Ns 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {texture_filename}\n")

    tmp_path = out_obj_path.with_suffix(".obj.tmp")

    inserted_usemtl = False

    with open(out_obj_path, "r", encoding="utf-8", errors="ignore") as src, \
         open(tmp_path, "w", encoding="utf-8") as dst:

        dst.write(f"mtllib {mtl_name}\n")

        for line in src:
            if line.startswith("mtllib "):
                continue
            if line.startswith("usemtl "):
                continue

            if not inserted_usemtl and line.startswith("f "):
                dst.write(f"usemtl {material_name}\n")
                inserted_usemtl = True

            dst.write(line)

    shutil.move(str(tmp_path), str(out_obj_path))


# =========================
# Main
# =========================

def main():
    in_path = Path(env_mesh_path).resolve()
    out_dir = Path(OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_obj_path = out_dir / OUT_OBJ
    out_tex_path = out_dir / OUT_TEXTURE

    print(f"[INFO] Input PLY: {in_path}")
    print(f"[INFO] Output OBJ: {out_obj_path}")
    print(f"[INFO] Output texture: {out_tex_path}")
    print(f"[INFO] Texture size: {TEXTURE_SIZE} x {TEXTURE_SIZE}")
    print(f"[INFO] Bake margin px: {BAKE_MARGIN_PX}")

    clear_scene()

    obj = import_ply(in_path)
    obj.name = "recon_background"

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh = obj.data
    print(f"[INFO] Imported mesh: {obj.name}")
    print(f"[INFO] Vertices: {len(mesh.vertices)}")
    print(f"[INFO] Polygons: {len(mesh.polygons)}")

    color_attr = get_color_attribute(mesh)
    color_attr_name = color_attr.name
    print(f"[INFO] Using color attribute: {color_attr_name}")

    if FORCE_ALPHA_TO_ONE:
        print("[INFO] Forcing vertex color alpha to 1.0")
        force_color_alpha_to_one(color_attr)

    if CLEAN_MESH:
        print("[INFO] Cleaning mesh")
        clean_mesh(obj)

    print("[INFO] Generating UVs")
    smart_uv_project(obj)

    print("[INFO] Creating bake image")
    image = bpy.data.images.new(
        name=OUT_TEXTURE,
        width=TEXTURE_SIZE,
        height=TEXTURE_SIZE,
        alpha=False,
        float_buffer=False,
    )
    image.filepath_raw = str(out_tex_path)
    image.file_format = "PNG"

    print("[INFO] Creating material from vertex color")
    mat, attr_node, img_node, bsdf = create_bake_material(
        obj=obj,
        color_attr_name=color_attr_name,
        image=image,
    )

    print("[INFO] Baking vertex color to texture")
    bake_diffuse_color(obj, image)

    print("[INFO] Saving texture")
    image.filepath_raw = str(out_tex_path)
    image.file_format = "PNG"
    image.save()

    print("[INFO] Switching material to baked texture")
    switch_material_to_texture(mat, img_node, bsdf)

    print("[INFO] Exporting OBJ")
    export_obj(obj, out_obj_path)

    print("[INFO] Writing PyBullet-friendly MTL and patching OBJ")
    write_simple_mtl_and_patch_obj(out_obj_path, OUT_TEXTURE)

    print("[DONE]")
    print(f"OBJ: {out_obj_path}")
    print(f"MTL: {out_obj_path.with_suffix('.mtl')}")
    print(f"PNG: {out_tex_path}")


if __name__ == "__main__":
    main()
