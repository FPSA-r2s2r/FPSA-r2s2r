import bpy
from pathlib import Path

in_dir = Path(".").resolve()
out_dir = in_dir.parent / "visual_obj_colored"
out_dir.mkdir(parents=True, exist_ok=True)

dae_files = sorted(in_dir.glob("*.dae"))

print(f"Input dir:  {in_dir}")
print(f"Output dir: {out_dir}")
print(f"Found {len(dae_files)} DAE files")

for dae_path in dae_files:
    print(f"\nConverting {dae_path.name}")

    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import DAE / Collada. This exists in Blender 4.x, not Blender 5.x.
    bpy.ops.wm.collada_import(
        filepath=str(dae_path),
        import_units=True,
    )

    # Select imported mesh objects
    bpy.ops.object.select_all(action="DESELECT")
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not mesh_objects:
        print(f"  WARNING: no mesh objects imported from {dae_path.name}")
        continue

    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objects[0]

    obj_path = out_dir / f"{dae_path.stem}.obj"

    # Blender 4.x OBJ exporter
    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        path_mode="COPY",  # copy texture files referenced by materials if present
    )

    print(f"  wrote {obj_path}")

print("\nDone. Converted the DAE mesh files to OBJ format using Blender.")