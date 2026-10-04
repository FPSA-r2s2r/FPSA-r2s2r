import json
import yaml
import os
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import coacd
import trimesh

def load_initial_grasp_pose(path: str):
    """
    Read grasp_pose.yaml or grasp_pose.json.

    Return a dictionary containing:
        mesh_path: str
        reference_frame: str
        hand_frame: str
        t: np.ndarray (3,)
        R: np.ndarray (3,3)
        opening: float
        pregrasp_opening: float
        finger_joint: float
        T_mesh_hand: np.ndarray (4,4)
        T_mesh_hand_tcp: np.ndarray (4,4)

    Precedence:
      1. If T_mesh_hand is present, use it directly to recover t and R.
      2. Otherwise, use position_m and quaternion_xyzw.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"抓取文件不存在: {path}")

    suffix = Path(path).suffix.lower()
    if suffix in [".yaml", ".yml"]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {Path(path).suffix}，只支持 .yaml/.yml/.json")

    mesh_path = str(data.get("mesh_path", ""))
    reference_frame = str(data.get("reference_frame", "mesh_local_frame"))
    hand_frame = str(data.get("hand_frame", "panda_hand"))
    opening = float(data.get("opening_width_m", 0.06))
    pregrasp_opening = float(data.get("pregrasp_opening_width_m", opening))
    finger_joint = float(data.get("finger_joint_m", opening / 2.0))

    if "T_mesh_hand" in data and data["T_mesh_hand"] is not None:
        T_mesh_hand = np.asarray(data["T_mesh_hand"], dtype=float)
        if T_mesh_hand.shape != (4, 4):
            raise ValueError("T_mesh_hand 维度错误，应为 4x4")

    if "T_mesh_hand_tcp" in data and data["T_mesh_hand_tcp"] is not None:
        T_mesh_hand_tcp = np.asarray(data["T_mesh_hand_tcp"], dtype=float)
        if T_mesh_hand_tcp.shape != (4, 4):
            raise ValueError("T_mesh_hand_tcp 维度错误，应为 4x4")
        R = T_mesh_hand_tcp[:3, :3]
        t = T_mesh_hand_tcp[:3, 3]
        quat = Rotation.from_matrix(R).as_quat()

    return {
        "mesh_path": mesh_path,
        "reference_frame": reference_frame,
        "hand_frame": hand_frame,
        "t": t,
        "R": R,
        "quat": quat,
        "opening": opening,
        "pregrasp_opening": pregrasp_opening,
        "finger_joint": finger_joint,
        "T_mesh_hand": T_mesh_hand,
        "T_mesh_hand_tcp": T_mesh_hand_tcp,
    }


def coacd_convex_decomposition(obj_filename,threshold = 0.03, preprocess_resolution = 100):
    """
    COACD convex decomposition
    """
    output_file = os.path.splitext(obj_filename)[0] + "_coacd.obj"

    if os.path.exists(output_file):
        print(f"[COACD]: Convex decomposition mesh {output_file} exists")
        return output_file
    
    mesh = trimesh.load(obj_filename, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    #result = coacd.run_coacd(mesh, threshold= 0.02, mcts_max_depth= 5, mcts_nodes= 30, preprocess_mode= "False") # a list of convex hulls.
    result = coacd.run_coacd(mesh, threshold = threshold, preprocess_resolution =preprocess_resolution) # a list of convex hulls.

    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    scene.export(output_file)

    return output_file




def save_transform_yaml(
    yaml_path,
    transform,
    key,
) -> None:
    """
    Save a 4x4 homogeneous transformation matrix to a YAML file.
    """
    transform = np.asarray(transform, dtype=float)

    if transform.shape != (4, 4):
        raise ValueError(
            f"transform 必须是 4x4 矩阵，当前形状为 {transform.shape}"
        )

    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        key: transform.tolist(),
    }

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    print(f"Transform saved to: {yaml_path.resolve()}")


def load_transform_yaml(
    yaml_path,
    key,
) -> np.ndarray:
    """
    Read a 4x4 homogeneous transformation matrix from a YAML file.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到 YAML 文件: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict) or key not in data:
        raise KeyError(f"YAML 文件中不存在字段: {key}")

    transform = np.asarray(data[key], dtype=float)

    if transform.shape != (4, 4):
        raise ValueError(
            f"读取到的 transform 必须是 4x4 矩阵，当前形状为 {transform.shape}"
        )

    return transform
