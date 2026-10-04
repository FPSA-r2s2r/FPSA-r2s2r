#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Convert USDZ / USDC / USD directory -> OBJ + MTL + textures
#
# Recommended for your unpacked AR Code format:
#
#   blender -b --python usdz_to_obj.py -- \
#     ./data/ugreen_scene/AR-Code-Object-Capture-app-1778140054 \
#     ./ugreen_obj \
#     --apply-scale \
#     --triangulate
#
# Or directly use the .usdc:
#
#   blender -b --python usdz_to_obj.py -- \
#     ./data/ugreen_scene/AR-Code-Object-Capture-app-1778140054/baked_mesh_f31081fd.usdc \
#     ./ugreen_obj \
#     --apply-scale \
#     --triangulate
#
# Output:
#
#   ugreen_obj/
#     <name>.obj
#     <name>.mtl
#     textures/
#       *_tex0.png
#       *_norm0.png
#       *_ao0.png

import bpy
import sys
import shutil
import zipfile
import traceback
from pathlib import Path


IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".exr",
    ".avif",
}


USD_EXTS = {
    ".usdz",
    ".usdc",
    ".usd",
    ".usda",
}


def parse_args():
    argv = sys.argv

    if "--" not in argv:
        raise SystemExit(
            "Usage:\n"
            "  blender -b --python usdz_to_obj.py -- input.usdz_or_usdc_or_dir output_dir "
            "[--apply-scale] [--triangulate]\n"
        )

    args = argv[argv.index("--") + 1:]

    if len(args) < 2:
        raise SystemExit(
            "Usage:\n"
            "  blender -b --python usdz_to_obj.py -- input.usdz_or_usdc_or_dir output_dir "
            "[--apply-scale] [--triangulate]\n"
        )

    input_path = Path(args[0]).expanduser().resolve()
    output_dir = Path(args[1]).expanduser().resolve()

    apply_scale = "--apply-scale" in args
    triangulate = "--triangulate" in args

    return input_path, output_dir, apply_scale, triangulate


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def sanitize_name(name: str) -> str:
    chars = []

    for c in name:
        if c.isalnum() or c in "._-":
            chars.append(c)
        else:
            chars.append("_")

    return "".join(chars).strip("._") or "file"


def find_import_file(input_path: Path) -> Path:
    """
    Accepts:
      1. file.usdz
      2. file.usdc / file.usd / file.usda
      3. unpacked directory containing .usdc/.usd/.usda
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in USD_EXTS:
            raise RuntimeError(f"Unsupported input file: {input_path}")
        return input_path

    if not input_path.is_dir():
        raise RuntimeError(f"Input path does not exist: {input_path}")

    # Prefer files in the root of the directory.
    candidates = []

    for ext in [".usdc", ".usd", ".usda", ".usdz"]:
        candidates.extend(sorted(input_path.glob(f"*{ext}")))

    if not candidates:
        for ext in [".usdc", ".usd", ".usda", ".usdz"]:
            candidates.extend(sorted(input_path.rglob(f"*{ext}")))

    if not candidates:
        raise RuntimeError(f"No USD/USDZ file found in directory: {input_path}")

    return candidates[0]


def copy_file_unique(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)

    target = dst_dir / sanitize_name(src.name)

    if target.exists():
        if src.resolve() == target.resolve():
            return target

        stem = target.stem
        suffix = target.suffix
        i = 1

        while True:
            candidate = dst_dir / f"{stem}_{i:03d}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            i += 1

    shutil.copy2(src, target)
    return target


def collect_textures_from_directory(root: Path, textures_dir: Path) -> dict[str, str]:
    """
    Recursively copy image textures from an unpacked USDZ directory.

    Returns:
      original basename -> relative texture path used by MTL
    """
    texture_map = {}

    if not root.exists() or not root.is_dir():
        return texture_map

    for src in sorted(root.rglob("*")):
        if not src.is_file():
            continue

        if src.suffix.lower() not in IMAGE_EXTS:
            continue

        target = copy_file_unique(src, textures_dir)
        rel = f"textures/{target.name}"

        texture_map[src.name] = rel
        texture_map[sanitize_name(src.name)] = rel

        print(f"[INFO] Copied texture: {src} -> {rel}")

    return texture_map


def collect_textures_from_usdz(usdz_path: Path, textures_dir: Path) -> dict[str, str]:
    """
    USDZ is a zip-like package. This extracts image files directly.
    """
    texture_map = {}

    if not usdz_path.is_file():
        return texture_map

    if usdz_path.suffix.lower() != ".usdz":
        return texture_map

    if not zipfile.is_zipfile(usdz_path):
        return texture_map

    with zipfile.ZipFile(usdz_path, "r") as z:
        for name in z.namelist():
            suffix = Path(name).suffix.lower()

            if suffix not in IMAGE_EXTS:
                continue

            basename = sanitize_name(Path(name).name)
            target = textures_dir / basename
            textures_dir.mkdir(parents=True, exist_ok=True)

            if target.exists():
                stem = target.stem
                ext = target.suffix
                i = 1

                while True:
                    candidate = textures_dir / f"{stem}_{i:03d}{ext}"
                    if not candidate.exists():
                        target = candidate
                        break
                    i += 1

            with z.open(name, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

            rel = f"textures/{target.name}"
            texture_map[Path(name).name] = rel
            texture_map[basename] = rel

            print(f"[INFO] Extracted texture from USDZ: {name} -> {rel}")

    return texture_map


def collect_textures(input_path: Path, import_file: Path, output_dir: Path) -> dict[str, str]:
    """
    Supports:
      1. unpacked directory input
      2. .usdc input with sibling 0/*.png
      3. .usdz input with sibling unpacked directory of same stem
      4. .usdz input with textures inside the package
    """
    textures_dir = output_dir / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)

    texture_map = {}

    search_roots = []

    if input_path.is_dir():
        search_roots.append(input_path)

    elif input_path.is_file():
        if input_path.suffix.lower() in {".usdc", ".usd", ".usda"}:
            search_roots.append(input_path.parent)

        if input_path.suffix.lower() == ".usdz":
            unpacked_dir = input_path.with_suffix("")
            if unpacked_dir.exists() and unpacked_dir.is_dir():
                search_roots.append(unpacked_dir)

    if import_file.parent not in search_roots and import_file.suffix.lower() != ".usdz":
        search_roots.append(import_file.parent)

    for root in search_roots:
        texture_map.update(
            collect_textures_from_directory(
                root=root,
                textures_dir=textures_dir,
            )
        )

    texture_map.update(
        collect_textures_from_usdz(
            usdz_path=input_path,
            textures_dir=textures_dir,
        )
    )

    print(f"[INFO] Texture entries available: {len(texture_map)}")

    return texture_map


def blender_operator_kwargs(op, **kwargs):
    allowed = set(op.get_rna_type().properties.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def import_usd(import_file: Path):
    print(f"[INFO] Importing USD file: {import_file}")

    if not hasattr(bpy.ops.wm, "usd_import"):
        raise RuntimeError("This Blender build does not support bpy.ops.wm.usd_import.")

    kwargs = blender_operator_kwargs(
        bpy.ops.wm.usd_import,
        filepath=str(import_file),
        import_cameras=False,
        import_lights=False,
        import_materials=True,
        read_mesh_uvs=True,
    )

    bpy.ops.wm.usd_import(**kwargs)

    mesh_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    if not mesh_objects:
        raise RuntimeError("No mesh objects were imported.")

    print(f"[INFO] Mesh objects imported: {len(mesh_objects)}")

    return mesh_objects


def select_meshes(mesh_objects):
    bpy.ops.object.select_all(action="DESELECT")

    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objects[0]


def apply_scale(mesh_objects):
    select_meshes(mesh_objects)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print("[INFO] Applied object scale.")


def triangulate_meshes(mesh_objects):
    for obj in mesh_objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        mod = obj.modifiers.new("Triangulate_for_OBJ", "TRIANGULATE")
        bpy.ops.object.modifier_apply(modifier=mod.name)

    print("[INFO] Triangulated meshes.")


def print_bbox(mesh_objects):
    import mathutils

    points = []

    for obj in mesh_objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ mathutils.Vector(corner))

    if not points:
        return

    min_v = mathutils.Vector(
        (
            min(p.x for p in points),
            min(p.y for p in points),
            min(p.z for p in points),
        )
    )

    max_v = mathutils.Vector(
        (
            max(p.x for p in points),
            max(p.y for p in points),
            max(p.z for p in points),
        )
    )

    size = max_v - min_v

    print("[INFO] Bounding box min:", tuple(round(v, 6) for v in min_v))
    print("[INFO] Bounding box max:", tuple(round(v, 6) for v in max_v))
    print("[INFO] Bounding box size XYZ:", tuple(round(v, 6) for v in size))


def find_principled_node(mat):
    if not mat or not getattr(mat, "use_nodes", False):
        return None

    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node

    return None


def find_image_from_socket(socket, visited=None):
    if visited is None:
        visited = set()

    if socket is None:
        return None

    for link in socket.links:
        node = link.from_node

        if node in visited:
            continue

        visited.add(node)

        if node.type == "TEX_IMAGE" and node.image is not None:
            return node.image.name

        for input_socket in node.inputs:
            found = find_image_from_socket(input_socket, visited)
            if found:
                return found

    return None


def texture_score(name: str) -> int:
    text = name.lower()
    score = 0

    positive = [
        "tex0",
        "basecolor",
        "base_color",
        "base color",
        "diffuse",
        "albedo",
        "color",
        "colour",
    ]

    negative = [
        "norm",
        "normal",
        "nrm",
        "bump",
        "ao",
        "occlusion",
        "rough",
        "roughness",
        "metal",
        "metallic",
        "height",
        "disp",
        "alpha",
        "opacity",
    ]

    for word in positive:
        if word in text:
            score += 10

    for word in negative:
        if word in text:
            score -= 100

    return score


def best_global_color_texture(texture_map: dict[str, str]) -> str | None:
    if not texture_map:
        return None

    unique_rels = sorted(set(texture_map.values()))
    unique_rels.sort(key=texture_score, reverse=True)

    if unique_rels and texture_score(unique_rels[0]) > -50:
        return unique_rels[0]

    return None


def choose_material_texture(mat, texture_map: dict[str, str]) -> str | None:
    """
    Prefer the image connected to Principled BSDF Base Color.
    Fallback to the best texture name, usually *_tex0.png.
    """
    principled = find_principled_node(mat)

    if principled is not None:
        base_input = principled.inputs.get("Base Color")
        image_name = find_image_from_socket(base_input)

        if image_name:
            basename = Path(image_name).name
            sanitized = sanitize_name(basename)

            if basename in texture_map:
                return texture_map[basename]

            if sanitized in texture_map:
                return texture_map[sanitized]

    return best_global_color_texture(texture_map)


def build_material_texture_map(texture_map: dict[str, str]) -> dict[str, str]:
    material_map = {}

    for mat in bpy.data.materials:
        tex = choose_material_texture(mat, texture_map)

        if tex:
            material_map[mat.name] = tex
            print(f"[INFO] Material {mat.name!r} -> map_Kd {tex}")

    return material_map


def export_obj(output_obj: Path):
    print(f"[INFO] Exporting OBJ: {output_obj}")

    if hasattr(bpy.ops.wm, "obj_export"):
        kwargs = blender_operator_kwargs(
            bpy.ops.wm.obj_export,
            filepath=str(output_obj),
            export_selected_objects=True,
            export_materials=True,
            export_uv=True,
            export_normals=True,
            path_mode="RELATIVE",
            apply_modifiers=True,
        )
        bpy.ops.wm.obj_export(**kwargs)
        return

    if hasattr(bpy.ops.export_scene, "obj"):
        kwargs = blender_operator_kwargs(
            bpy.ops.export_scene.obj,
            filepath=str(output_obj),
            use_selection=True,
            use_materials=True,
            use_uvs=True,
            use_normals=True,
            path_mode="RELATIVE",
        )
        bpy.ops.export_scene.obj(**kwargs)
        return

    raise RuntimeError("No OBJ exporter found in this Blender build.")


def patch_mtl(output_obj: Path, material_map: dict[str, str], fallback_texture: str | None):
    """
    Force MTL map_Kd to point to the real copied diffuse texture.
    """
    mtl_path = output_obj.with_suffix(".mtl")

    if not mtl_path.exists():
        print(f"[WARN] MTL not found: {mtl_path}")
        return

    lines = mtl_path.read_text(errors="replace").splitlines(keepends=True)

    out = []
    current_mat = None
    inserted = False

    def texture_for_current():
        if current_mat in material_map:
            return material_map[current_mat]
        return fallback_texture

    def insert_map_kd_if_needed():
        nonlocal inserted

        tex = texture_for_current()

        if current_mat and tex and not inserted:
            out.append(f"map_Kd {tex}\n")
            inserted = True

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("newmtl "):
            insert_map_kd_if_needed()

            current_mat = stripped.split(maxsplit=1)[1]
            inserted = False
            out.append(line)
            continue

        if stripped.startswith("map_Kd "):
            tex = texture_for_current()

            if tex and not inserted:
                out.append(f"map_Kd {tex}\n")
                inserted = True

            continue

        out.append(line)

    insert_map_kd_if_needed()

    mtl_path.write_text("".join(out))
    print(f"[INFO] Patched MTL: {mtl_path}")


def print_summary(output_obj: Path, output_dir: Path):
    mtl_path = output_obj.with_suffix(".mtl")
    textures_dir = output_dir / "textures"

    print()
    print("[DONE]")
    print(f"OBJ: {output_obj}")
    print(f"MTL: {mtl_path}")
    print(f"Textures dir: {textures_dir}")

    if mtl_path.exists():
        print("[INFO] map_Kd lines:")
        for line in mtl_path.read_text(errors="replace").splitlines():
            if line.strip().startswith("map_Kd "):
                print(f"  {line.strip()}")

    if textures_dir.exists():
        files = sorted(p for p in textures_dir.iterdir() if p.is_file())
        print(f"[INFO] Texture files: {len(files)}")
        for p in files:
            print(f"  {p.name} ({p.stat().st_size} bytes)")


def main():
    input_path, output_dir, do_apply_scale, do_triangulate = parse_args()

    output_dir.mkdir(parents=True, exist_ok=True)
    textures_dir = output_dir / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)

    import_file = find_import_file(input_path)

    if input_path.is_dir():
        output_name = sanitize_name(input_path.name)
    else:
        output_name = sanitize_name(input_path.stem)

    output_obj = output_dir / f"{output_name}.obj"

    clear_scene()

    texture_map = collect_textures(
        input_path=input_path,
        import_file=import_file,
        output_dir=output_dir,
    )

    mesh_objects = import_usd(import_file)

    select_meshes(mesh_objects)

    if do_apply_scale:
        apply_scale(mesh_objects)

    if do_triangulate:
        triangulate_meshes(mesh_objects)

    select_meshes(mesh_objects)
    print_bbox(mesh_objects)

    material_map = build_material_texture_map(texture_map)
    fallback_texture = best_global_color_texture(texture_map)

    if fallback_texture:
        print(f"[INFO] Fallback map_Kd texture: {fallback_texture}")
    else:
        print("[WARN] No diffuse/color texture found for map_Kd.")

    export_obj(output_obj)

    patch_mtl(
        output_obj=output_obj,
        material_map=material_map,
        fallback_texture=fallback_texture,
    )

    print_summary(
        output_obj=output_obj,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)