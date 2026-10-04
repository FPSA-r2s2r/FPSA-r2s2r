from __future__ import annotations

from icecream import ic

from FPSA import ShapeAugmentor
import numpy as np
# ----------------------------
# IO helpers
# ----------------------------

import yaml
from typing import Any, Dict, Optional, Sequence
from pathlib import Path
import json


def load_meta(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML meta files. Install with: pip install pyyaml")
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported meta file suffix: {path.suffix}. Use .yaml/.yml/.json")


def dump_yaml_or_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_builtin(data)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            path = path.with_suffix(".json")
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _first_present(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _matrix4(data: Dict[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    value = _first_present(data, keys)
    if value is None:
        return None

    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{keys[0]} must be a 4x4 matrix, got {mat.shape}")
    return mat


def write_grasp_yaml(path: str | Path, grasp_result: Dict[str, Any], mesh_path: str | Path) -> None:
    """Write the transferred grasp in the format consumed by load_initial_grasp_pose()."""
    if not isinstance(grasp_result, dict):
        raise TypeError(f"grasp_result must be a dict, got {type(grasp_result).__name__}")

    T_mesh_hand_tcp = _matrix4(grasp_result, ["T_mesh_hand_tcp"])
    T_mesh_hand = _matrix4(grasp_result, ["T_mesh_hand"])

    # The user's loader expects both keys to exist, but the FPSA transfer result may
    # only return T_mesh_hand_tcp. Do not compute an extra transform here; just mirror
    # the available 4x4 matrix so the YAML remains loadable by the existing reader.
    if T_mesh_hand_tcp is None and T_mesh_hand is None:
        raise KeyError("grasp_result is missing T_mesh_hand_tcp / T_mesh_hand")
    if T_mesh_hand_tcp is None:
        T_mesh_hand_tcp = T_mesh_hand
    if T_mesh_hand is None:
        T_mesh_hand = T_mesh_hand_tcp

    opening = float(_first_present(grasp_result, ["opening_width_m", "opening"], 0.06))

    record = {
        "mesh_path": str(mesh_path),
        "reference_frame": str(_first_present(grasp_result, ["reference_frame"], "mesh_local_frame")),
        "hand_frame": str(_first_present(grasp_result, ["hand_frame"], "panda_hand")),
        "opening_width_m": opening,
        "pregrasp_opening_width_m": float(
            _first_present(grasp_result, ["pregrasp_opening_width_m", "pregrasp_opening"], opening)
        ),
        "finger_joint_m": float(_first_present(grasp_result, ["finger_joint_m", "finger_joint"], opening / 2.0)),
        "T_mesh_hand": T_mesh_hand,
        "T_mesh_hand_tcp": T_mesh_hand_tcp,
    }
    dump_yaml_or_json(path, record)


def to_builtin(x: Any) -> Any:
    """Convert numpy objects to JSON/YAML-safe Python objects."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    return x

if __name__ == "__main__":
    obj_path = "./data/objects/assembly/tool/tool.obj"
    initial_grasp_path = "./data/objects/assembly/tool/tool_grasp.yaml"

    augmentor = ShapeAugmentor(
        obj_path=obj_path,
        initial_grasp_path=initial_grasp_path,
    )
    # Initial TCP
    T_init_tcp = np.eye(4)
    T_init_tcp[:3, 3] = np.array([0.06989, 0.0, 0.0], dtype=float)

    # Same constraint set as your original main.py.
    constraint_ids = [17, 15, 19, 13, 208, 164, 105, 106, 107, 108, 103, 118, 119,
      120, 121, 122, 128, 78, 93, 92, 91, 90, 89, 88, 123, 18, 16, 14, 12, 81, 82,
      111, 112, 109, 101, 102, 100, 85, 285, 115, 117, 116, 75, 77, 76, 10, 21, 20,
      126, 66, 39, 184, 173, 48, 46, 45, 6, 149, 148, 147, 57, 40, 41, 28, 29, 1,
      169, 165, 26, 51, 191, 273, 281, 278, 262, 4, 5, 52, 9, 11, 23, 22, 71, 72,
      73, 133, 134, 135, 355, 354]

    T_origin = np.eye(4)
    T_origin[:3, 3] = [0.0, 0.0, 0.0]
    # ============================================================
    # Step 1: ARAP refinement
    # ============================================================
    # This call starts from Step 1's result, not from the original mesh.
    # The displacements below are therefore incremental displacements on top of
    # the slippage output.

    jaw_move_ids = [9, 11, 23, 22, 71, 72, 73, 133, 134, 135, 355, 354]
    jaw_slippage_displacements = np.array([
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],

    ])

    V_after_slippage = augmentor.displacement_reshape(
        constraint_ids=constraint_ids,
        displace_idxs=jaw_move_ids,
        displacements=jaw_slippage_displacements,
        max_iters=200,
        reshape_method="ARAP",
        input_name="step01_arap",
    )

    augmentor.write_augment_obj(
        output_path="step01_arap.obj",
        write_coacd=False,
    )

    augmentor.visualize_reshaped_mesh()

    print("V_after_slippage:", V_after_slippage.shape)

    # ============================================================
    # Step 2: slippage reshaping
    # ============================================================
    # This starts from the original mesh. After this call, augmentor.V_work is
    # automatically updated to the slippage result, so the next ARAP call will
    # use the slippage-deformed mesh as input.
    constraint_ids = [17, 15, 19, 13, 208, 164, 105, 106, 107, 108, 103, 118, 119,
      120, 121, 122, 128, 78, 93, 92, 91, 90, 89, 88, 123, 18, 16, 14, 12, 81, 82,
      111, 112, 109, 101, 102, 100, 85, 285, 115, 117, 116, 75, 77, 76, 10, 21, 20,
      126, 66, 39, 184, 173, 48, 46, 45, 6, 149, 148, 147, 57, 40, 41, 28, 29, 1,
      169, 165, 26, 51, 191, 273, 281, 278, 262, 4, 5, 52, 9, 11, 23, 22, 71, 72,
      73, 133, 134, 135, 355, 354]
    arap_move_ids = [9, 11, 23, 22, 71, 72, 73, 133, 134, 135, 355, 354]
    arap_refine_displacements = np.array([
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.03, 0.0],
    ])

    V_final = augmentor.displacement_reshape(
        constraint_ids=constraint_ids,
        displace_idxs=arap_move_ids,
        displacements=arap_refine_displacements,
        max_iters=120,
        reshape_method="slippage",
        input_name="step02_slippage",
    )

    augmentor.write_augment_obj(
        output_path="step02_slippage.obj",
        write_coacd=True,
    )

    print("V_final:", V_final.shape)

    # ============================================================
    # Grasp transfer and visualization
    # ============================================================
    # self.V is still the original reference mesh. self.V_opt is the final mesh
    # after ARAP -> slippage, so grasp transfer maps the original grasp to the
    # final chained deformation.


    T_new, anchor, debug = augmentor.transfer_initial_grasp_guess(
        k_ring=3,
        use_distance_weights=True,
        quat_order="xyzw",
        patch_method="k_ring",
    )
    ic(T_new)
    augmentor.visualize_deformed_grasp_pose(
        T_grasp_new=T_new["T_mesh_hand_tcp"],
        anchor=anchor,
        debug_info=debug,
        show_anchor=True,
        show_patch=True,
        show_old_grasp=True,
        T_grasp_old=T_origin,
    )
    

    write_grasp_yaml("transferred_grasp.yaml",T_new, mesh_path="step02_slippage.obj" )