import math
from pathlib import Path
from typing import Dict
import numpy as np
import json
import yaml
import os
from scipy.spatial.transform import Rotation
import open3d as o3d
import cv2
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

def convert_save_ply(points, filename):
    """
    Convert a point cloud (NumPy array or PyTorch tensor) to a PLY file using Open3D.
    
    :param points: NumPy array or PyTorch tensor of shape (N, 3) or (N, 6) 
                where N is the number of points.
    :param filename: Name of the output PLY file.
    """
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points. Assuming the first 3 columns are x, y, z coordinates
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # If the points array has 6 columns, assume the last 3 are RGB values
    if points.shape[1] == 6:
        # Normalize color values to [0, 1] if they are not already
        colors = points[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Write to a PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to '{filename}'.")

def convert_save_glb(points, filename):
    import numpy as np
    import trimesh

    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] not in [3, 6]:
        raise ValueError(f"points must have shape (N, 3) or (N, 6), got {points.shape}")

    valid_mask = np.isfinite(points[:, :3]).all(axis=1)
    points = points[valid_mask]

    xyz = points[:, :3].astype(np.float32)

    if points.shape[1] == 6:
        colors = points[:, 3:6]

        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = np.clip(colors, 0, 255).astype(np.uint8)

        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        colors = np.concatenate([colors, alpha], axis=1)
    else:
        colors = None

    cloud = trimesh.points.PointCloud(vertices=xyz, colors=colors)
    cloud.export(filename)

    print(f"Point cloud saved to '{filename}'.")


def downsample_points(points, voxel_size=0.01):
    """
    Downsample scene-level point cloud using voxel grid.

    Args:
        points: (N, 3) or (N, 6), XYZ or XYZRGB
        voxel_size: voxel size for downsampling

    Returns:
        downsampled points: (M, 3) or (M, 6)
    """
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] not in [3, 6]:
        raise ValueError(f"points must be (N, 3) or (N, 6), got {points.shape}")

    # remove nan / inf
    valid_mask = np.isfinite(points[:, :3]).all(axis=1)
    points = points[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    has_color = points.shape[1] == 6
    if has_color:
        colors = points[:, 3:6].astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    xyz = np.asarray(pcd_down.points)

    if has_color:
        rgb = np.asarray(pcd_down.colors)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return np.concatenate([xyz, rgb], axis=1)

    return xyz

def build_scene_level_points(pointmaps, rgb=None, extrinsics=None, max_camera_distance=None):
    """
    Merge multi-view pointmaps into one scene-level point cloud.

    Args:
        pointmaps: (N, H, W, 3), camera-space pointmaps
        rgb: optional (N, H, W, 3), uint8 RGB
        extrinsics: optional (N, 4, 4), camera-to-world or world transform
        max_camera_distance: optional float. If set, remove points whose
            Euclidean distance from the camera origin is larger than this value.
            Filtering is done before transforming to world coordinates.

    Returns:
        scene_points: (M, 3) or (M, 6)
    """
    scene_points = []

    n = pointmaps.shape[0]

    for i in range(n):
        pts = pointmaps[i].reshape(-1, 3)

        valid_mask = np.isfinite(pts).all(axis=1)

        if max_camera_distance is not None:
            max_camera_distance = float(max_camera_distance)
            dist = np.linalg.norm(pts, axis=1)
            valid_mask &= dist <= max_camera_distance

        pts = pts[valid_mask]

        if extrinsics is not None:
            pts = transform_pcd(pts, extrinsics[i])

        if rgb is not None:
            colors = rgb[i].reshape(-1, 3)[valid_mask]
            pts = np.concatenate([pts, colors], axis=1)

        scene_points.append(pts)

    scene_points = np.concatenate(scene_points, axis=0)
    return scene_points



def depth_to_pointmap(
    depth: np.ndarray, 
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Convert depth map to pointmap (3D coordinates in camera space).
    
    NOTE: This outputs in STANDARD CAMERA SPACE (same as MoGe raw output):
        - x: right direction
        - y: down direction  
        - z: forward direction (away from camera, positive depth)
    
    SAM3D's compute_pointmap() will apply the PyTorch3D coordinate transform
    internally, so we should NOT do the transform here.
    
    Args:
        depth: (H, W) depth map, values are distances from camera
        intrinsics: (3, 3) camera intrinsic matrix
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    
    Returns:
        pointmap: (H, W, 3) point cloud map, each pixel is (x, y, z) coordinate
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel coordinate grids
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Unproject to 3D (standard camera space)
    # z is positive (depth values are positive, pointing away from camera)
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    pointmap = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    return pointmap

def pointmap_to_sam3d_format(pointmap: np.ndarray) -> np.ndarray:
    """
    Convert pointmap to SAM3D expected format.
    
    Args:
        pointmap: (H, W, 3) pointmap in PyTorch3D coordinates
        
    Returns:
        pointmap_sam3d: (3, H, W) pointmap ready for SAM3D
    """
    # SAM3D expects (3, H, W) format (channel-first)
    return pointmap.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)

def visualize_point_cloud(data):
    """
    Visualizes a point cloud using Open3D. Supports N*3 and N*6 point clouds,
    and accepts both NumPy arrays and PyTorch tensors.

    :param data: A NumPy array or PyTorch tensor of shape (N, 3) or (N, 6).
                 For (N, 3), it represents the (x, y, z) coordinates of the points.
                 For (N, 6), it represents the (x, y, z, r, g, b) coordinates and colors of the points.
    """
    if data.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])

    if data.shape[1] == 6:
        point_cloud.colors = o3d.utility.Vector3dVector(data[:, 3:])
    
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([point_cloud,world_frame])

def transform_pcd(pcd, transformation):
        # Validate input shapes
    if pcd.shape[1] not in [3, 6] :
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")
    
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    source_o3d.transform(transformation)

    return  np.array(source_o3d.points)


def resize_rgb(img, out_size=224):
    """
    img: RGB image, shape [H, W, 3]
    return: RGB image, shape [out_size, out_size, 3]
    """
    # Resize to 224x224.
    img_224 = cv2.resize(
        img,
        (out_size, out_size),
        interpolation=cv2.INTER_AREA
    )

    return img_224

def compare_rgb(a, b):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    print("shape:", a.shape, b.shape)
    print("mean diff:", diff.mean())
    print("max diff:", diff.max())
    print("same pixels:", np.mean(np.all(diff == 0, axis=-1)) * 100, "%")

def normalize_vector(vec, eps=1e-8):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def quat_from_rotation_matrix(rot):
    """Convert a 3x3 rotation matrix to a PyBullet xyzw quaternion."""
    m = np.asarray(rot, dtype=float).reshape(3, 3)
    trace = np.trace(m)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=float)
    return quat / np.linalg.norm(quat)

def _uniform_shrink_about_point(mesh: trimesh.Trimesh, scale: float, center: np.ndarray) -> trimesh.Trimesh:
    # v' = c + s*(v - c)
    V = mesh.vertices
    V2 = center + scale * (V - center)
    m2 = trimesh.Trimesh(V2, mesh.faces, process=False)
    return m2

def shrink_mesh_uniform(obj_path: str, scale: float = 0.99, about: str = "centroid") -> str:
    """
    Uniformly scale the mesh by a factor below 1.0 and return the cached sidecar OBJ path.
    about: "centroid" | "center_mass" | "bbox"
    """
    assert 0.0 < scale < 1.0, "scale 必须在 (0,1)"
    mesh = trimesh.load(obj_path, force="mesh")
    if about == "center_mass":
        c = mesh.center_mass
    elif about == "bbox":
        c = mesh.bounding_box.centroid
    else:
        c = mesh.centroid

    out_path = os.path.splitext(obj_path)[0] + f"_shrink{int(scale*1000):03d}.obj"
    if os.path.exists(out_path):
        return out_path

    m2 = _uniform_shrink_about_point(mesh, scale, c)
    m2.export(out_path)
    return out_path

def shrink_mesh_by_margin(obj_path: str, margin_m: float = 0.003, about: str = "centroid") -> str:
    """
    Shrink the mesh inward by margin_m metres in the radial sense.
    This is equivalent to scale = 1 - margin / R_max, where R_max = max(||v - c||).
    """
    mesh = trimesh.load(obj_path, force="mesh")
    if about == "center_mass":
        c = mesh.center_mass
    elif about == "bbox":
        c = mesh.bounding_box.centroid
    else:
        c = mesh.centroid

    R = np.linalg.norm(mesh.vertices - c, axis=1).max()
    if R <= 1e-9:
        # Degenerate case: the mesh is effectively a point.
        return obj_path
    scale = max(0.01, 1.0 - float(margin_m) / float(R))  # Prevent collapse to zero.
    out_path = os.path.splitext(obj_path)[0] + f"_shrink{int(scale*1000):03d}.obj"
    if os.path.exists(out_path):
        return out_path

    m2 = _uniform_shrink_about_point(mesh, scale, c)
    m2.export(out_path)
    return out_path

def shrink_mesh(collision_asset_path: str,
                shrink_mode: str = "by_margin",  # "by_margin" | "by_scale" | "none"
                margin_m: float = 0.003,
                scale: float = 0.99,
                shrink_center: str = "centroid",
                ) -> str:
    """
    1) Ensure the input is an OBJ sidecar.
    2) Optionally shrink it, preferring by_margin over by_scale.
    Return the COACD output OBJ path.
    """
    # 1) Ensure the asset is an OBJ.
    obj_path = collision_asset_path

    # 2) Shrink it if requested.
    if shrink_mode == "by_margin":
        obj_shrunk = shrink_mesh_by_margin(obj_path, margin_m=margin_m, about=shrink_center)
    elif shrink_mode == "by_scale":
        obj_shrunk = shrink_mesh_uniform(obj_path, scale=scale, about=shrink_center)
    else:
        obj_shrunk = obj_path

    return obj_shrunk
