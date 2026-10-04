"""
manual_panda_grasp_editor.py

A lightweight manual grasp-pose editor that:
- loads meshes in OBJ, STL, PLY, GLB, GLTF, or OFF format;
- adjusts a Franka Panda hand grasp pose in the mesh-local frame;
- displays a simplified Panda hand (palm and two fingers);
- supports translation, rotation, and gripper-opening adjustments; and
- exports JSON or YAML for use with MoveIt, Isaac Sim, or a custom planner.

Required:
    pip install open3d numpy
Optional:
    pip install pyyaml

Usage:
    python get_initial_grasp_guess.py /path/to/object_mesh.stl
    python get_initial_grasp_guess.py data/objects/assembly/tool/tool.obj --out tool_grasp.yaml

    python get_initial_grasp_guess.py data/objects/gear_extraction/gear/gear.obj --out tool_grasp.yaml

    python get_initial_grasp_guess.py ./data/objects/banana/banana.obj --out ./data/objects/banana/grasp.yaml
Keyboard shortcuts (the terminal continuously prints the current pose):
    Translation (along mesh-frame axes)
      A / D : -X / +X
      S / W : -Y / +Y
      Q / E : -Z / +Z

    Rotation (about hand-local axes)
      J / L : -Rx / +Rx
      K / I : -Ry / +Ry
      U / O : -Rz / +Rz

    Gripper
      [ / ] : decrease / increase the total opening width in metres

    Step size
      , / . : halve / double the translation step
      ; / ' : halve / double the rotation step

    Other
      R     : reset to the initial pose
      P     : print the current pose
      X     : export to the file specified by --out
      H     : print help
      ESC   : quit
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple
import copy

import numpy as np
import open3d as o3d

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False


# ---------------------------
# Linear-algebra helpers
# ---------------------------

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def to_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def rpy_from_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    # intrinsic xyz
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw

def quat_xyzw_from_matrix(R: np.ndarray) -> np.ndarray:
    # returns x, y, z, w
    q = np.empty(4, dtype=float)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    q /= np.linalg.norm(q)
    return q

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------
# Franka hand visualization configuration
# Coordinate conventions:
#   hand frame: +Z points forward along the grasp, +Y toward the left finger,
#   and +X across the palm.
#   franka_hand_opened.obj must already be expressed in the panda_hand local frame.
# ---------------------------

FRANKA_HAND_OPENED_MESH_PATH = "franka_panda/franka_hand_opened.obj"

MAX_OPENING = 0.080
DEFAULT_OPENING = 0.040

# Typical +Z offset from panda_hand to hand_tcp in Panda models.
HAND_TCP_OFFSET_Z = 0.1034


@dataclass
class ExportPose:
    mesh_path: str
    reference_frame: str
    hand_frame: str
    position_m: Dict[str, float]
    quaternion_xyzw: Dict[str, float]
    rpy_rad: Dict[str, float]
    rpy_deg: Dict[str, float]
    opening_width_m: float
    finger_joint_m: float
    pregrasp_opening_width_m: float
    hand_tcp_offset_m: Dict[str, float]
    T_mesh_hand: list
    T_mesh_hand_tcp: list


class PandaHandEditor:
    def __init__(
        self,
        mesh_path: str,
        out_path: str,
        init_xyz: Tuple[float, float, float],
        init_rpy_deg: Tuple[float, float, float],
        init_opening: float,
        pregrasp_opening: float,
        show_bbox: bool = True,
        frame_size: float = 0.06,
    ) -> None:
        self.mesh_path = str(Path(mesh_path).expanduser().resolve())
        self.out_path = str(Path(out_path).expanduser().resolve())

        self.t = np.array(init_xyz, dtype=float)
        rr = np.deg2rad(init_rpy_deg[0])
        rp = np.deg2rad(init_rpy_deg[1])
        ry = np.deg2rad(init_rpy_deg[2])
        self.R = rot_z(ry) @ rot_y(rp) @ rot_x(rr)
        self.opening = clamp(init_opening, 0.0, MAX_OPENING)
        self.pregrasp_opening = clamp(pregrasp_opening, 0.0, MAX_OPENING)

        self.reset_state = (
            self.t.copy(),
            self.R.copy(),
            self.opening,
        )

        self.trans_step = 0.005
        self.rot_step = math.radians(5.0)
        self.frame_size = frame_size
        self.show_bbox = show_bbox

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.mesh = self._load_mesh(self.mesh_path)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.75, 0.75, 0.78])

        self.hand_mesh_template = self._load_hand_mesh_template()

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_size)
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_size * 0.7)

        self.bbox = self.mesh.get_axis_aligned_bounding_box()
        self.bbox.color = (0.1, 0.7, 0.1)

        self.hand_geoms = []
        self._build_static_scene()

    def _load_mesh(self, path: str) -> o3d.geometry.TriangleMesh:
        mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
        if mesh.is_empty():
            raise RuntimeError(f"无法读取 mesh: {path}")
        return mesh


    def _load_hand_mesh_template(self) -> o3d.geometry.TriangleMesh:
        hand_mesh_path = Path(FRANKA_HAND_OPENED_MESH_PATH)
        if not hand_mesh_path.is_absolute():
            hand_mesh_path = (Path.cwd() / hand_mesh_path).resolve()

        mesh = self._load_mesh(str(hand_mesh_path))
        mesh.compute_vertex_normals()
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.78, 0.78, 0.82])
        return mesh

    def _build_static_scene(self) -> None:
        self.vis.create_window(
            window_name="Manual Panda Grasp Editor",
            width=1600,
            height=960,
            visible=True,
        )
        self.vis.add_geometry(self.mesh)
        self.vis.add_geometry(self.mesh_frame)
        self.vis.add_geometry(self.world_frame)
        if self.show_bbox:
            self.vis.add_geometry(self.bbox)

        self._register_keys()
        self._refresh_hand_geometries(first_time=True)
        self._print_help()
        self._print_pose()

        # Aim the camera at the object center.
        ctr = self.vis.get_view_control()
        center = self.mesh.get_center()
        ctr.set_lookat(center)
        ctr.set_front([0.35, -0.3, -0.88])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.8)

    def _register_keys(self) -> None:
        def reg(ch: str, fn):
            self.vis.register_key_callback(ord(ch), fn)

        # Translation
        reg("A", lambda v: self._translate(np.array([-self.trans_step, 0, 0])))
        reg("D", lambda v: self._translate(np.array([+self.trans_step, 0, 0])))
        reg("S", lambda v: self._translate(np.array([0, -self.trans_step, 0])))
        reg("W", lambda v: self._translate(np.array([0, +self.trans_step, 0])))
        reg("Q", lambda v: self._translate(np.array([0, 0, -self.trans_step])))
        reg("E", lambda v: self._translate(np.array([0, 0, +self.trans_step])))

        # Rotation about hand-local axes
        reg("J", lambda v: self._rotate_local(rot_x(-self.rot_step)))
        reg("L", lambda v: self._rotate_local(rot_x(+self.rot_step)))
        reg("K", lambda v: self._rotate_local(rot_y(-self.rot_step)))
        reg("I", lambda v: self._rotate_local(rot_y(+self.rot_step)))
        reg("U", lambda v: self._rotate_local(rot_z(-self.rot_step)))
        reg("O", lambda v: self._rotate_local(rot_z(+self.rot_step)))

        # Gripper
        self.vis.register_key_callback(ord("["), lambda v: self._change_opening(-0.002))
        self.vis.register_key_callback(ord("]"), lambda v: self._change_opening(+0.002))

        # Step size
        self.vis.register_key_callback(ord(","), lambda v: self._set_trans_step(self.trans_step / 2.0))
        self.vis.register_key_callback(ord("."), lambda v: self._set_trans_step(self.trans_step * 2.0))
        self.vis.register_key_callback(ord(";"), lambda v: self._set_rot_step(self.rot_step / 2.0))
        self.vis.register_key_callback(ord("'"), lambda v: self._set_rot_step(self.rot_step * 2.0))

        # Other controls
        reg("R", lambda v: self._reset())
        reg("P", lambda v: self._print_pose(True))
        reg("X", lambda v: self._export_pose())
        reg("H", lambda v: self._print_help())

    def _build_hand_parts_local(self):
        hand_mesh = copy.deepcopy(self.hand_mesh_template)

        tcp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_size * 0.8)
        tcp_frame.translate([0.0, 0.0, HAND_TCP_OFFSET_Z])

        hand_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_size)

        return [hand_mesh, hand_frame, tcp_frame]

    def _refresh_hand_geometries(self, first_time: bool = False) -> None:
        if not first_time:
            for g in self.hand_geoms:
                self.vis.remove_geometry(g, reset_bounding_box=False)

        local_parts = self._build_hand_parts_local()
        T = self.T_mesh_hand
        self.hand_geoms = []
        for g in local_parts:
            g.transform(T)
            self.hand_geoms.append(g)
            self.vis.add_geometry(g, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()

    @property
    def T_mesh_hand(self) -> np.ndarray:
        return to_transform(self.R, self.t)

    @property
    def T_mesh_hand_tcp(self) -> np.ndarray:
        T_hand_tcp = to_transform(np.eye(3), np.array([0.0, 0.0, HAND_TCP_OFFSET_Z]))
        return self.T_mesh_hand @ T_hand_tcp

    @property
    def finger_joint(self) -> float:
        # Each Panda finger prismatic joint is typically positioned at opening / 2.
        return self.opening / 2.0

    def _translate(self, delta_mesh: np.ndarray):
        self.t = self.t + delta_mesh
        self._refresh_hand_geometries()
        self._print_pose()

    def _rotate_local(self, dR_local: np.ndarray):
        self.R = self.R @ dR_local
        self._refresh_hand_geometries()
        self._print_pose()

    def _change_opening(self, delta: float):
        self.opening = clamp(self.opening + delta, 0.0, MAX_OPENING)
        self._refresh_hand_geometries()
        self._print_pose()

    def _set_trans_step(self, v: float):
        self.trans_step = clamp(v, 0.00025, 0.10)
        print(f"[step] trans_step = {self.trans_step:.6f} m")
        return False

    def _set_rot_step(self, v: float):
        self.rot_step = clamp(v, math.radians(0.25), math.radians(45.0))
        print(f"[step] rot_step = {math.degrees(self.rot_step):.3f} deg")
        return False

    def _reset(self):
        t, R, opening = self.reset_state
        self.t = t.copy()
        self.R = R.copy()
        self.opening = opening
        self._refresh_hand_geometries()
        self._print_pose(True)
        return False

    def export_data(self) -> ExportPose:
        q = quat_xyzw_from_matrix(self.R)
        rr, rp, ry = rpy_from_matrix(self.R)
        tcp = self.T_mesh_hand_tcp[:3, 3]

        return ExportPose(
            mesh_path=self.mesh_path,
            reference_frame="mesh_local_frame",
            hand_frame="panda_hand",
            position_m={"x": float(self.t[0]), "y": float(self.t[1]), "z": float(self.t[2])},
            quaternion_xyzw={"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
            rpy_rad={"r": float(rr), "p": float(rp), "y": float(ry)},
            rpy_deg={"r": float(np.rad2deg(rr)), "p": float(np.rad2deg(rp)), "y": float(np.rad2deg(ry))},
            opening_width_m=float(self.opening),
            finger_joint_m=float(self.finger_joint),
            pregrasp_opening_width_m=float(self.pregrasp_opening),
            hand_tcp_offset_m={"x": 0.0, "y": 0.0, "z": HAND_TCP_OFFSET_Z},
            T_mesh_hand=np.round(self.T_mesh_hand, 9).tolist(),
            T_mesh_hand_tcp=np.round(self.T_mesh_hand_tcp, 9).tolist(),
        )

    def _export_pose(self):
        data = asdict(self.export_data())
        out = Path(self.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            if not _HAS_YAML:
                print("[warn] 未安装 pyyaml，改为导出 JSON。")
                out = out.with_suffix(".json")
                out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[saved] {out}")
        return False

    def _print_pose(self, detailed: bool = False):
        q = quat_xyzw_from_matrix(self.R)
        rr, rp, ry = rpy_from_matrix(self.R)
        print(
            f"[pose] t=({self.t[0]:+.4f}, {self.t[1]:+.4f}, {self.t[2]:+.4f}) m | "
            f"rpy=({np.rad2deg(rr):+.1f}, {np.rad2deg(rp):+.1f}, {np.rad2deg(ry):+.1f}) deg | "
            f"q=({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f}) | "
            f"opening={self.opening:.4f} m | joint={self.finger_joint:.4f} m"
        )
        if detailed:
            print("[T_mesh_hand]")
            print(np.array_str(self.T_mesh_hand, precision=5, suppress_small=True))
            print("[T_mesh_hand_tcp]")
            print(np.array_str(self.T_mesh_hand_tcp, precision=5, suppress_small=True))
        return False

    def _print_help(self):
        print("=" * 78)
        print("Manual Panda Grasp Editor")
        print("Mesh frame = 物体 mesh 的本地坐标系；你调出来的是 T_mesh_hand。")
        print("")
        print("平移（沿 mesh 坐标系）:  A/D = -/+X,  S/W = -/+Y,  Q/E = -/+Z")
        print("旋转（绕 hand 局部轴）: J/L = -/+Rx, K/I = -/+Ry, U/O = -/+Rz")
        print("夹爪开口: [ / ]")
        print("步长调整: , / .  调平移步长   ; / '  调旋转步长")
        print("其他: R=reset, P=print pose, X=export, H=help, ESC=quit")
        print("=" * 78)
        return False

    def run(self) -> None:
        self.vis.run()
        self.vis.destroy_window()


def auto_init_from_bbox(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()

    # Start slightly above the object, looking down along -Z.
    offset = np.array([0.0, 0.0, max(extent) * 0.6 + 0.10], dtype=float)
    t = center + offset

    # Point hand +Z toward the object center using a common top-grasp orientation.
    # hand local +Z -> mesh -Z
    R = rot_x(math.pi)
    return t, R


def parse_args():
    p = argparse.ArgumentParser(description="Manual Panda grasp pose editor for meshes")
    p.add_argument("mesh", help="mesh path, e.g. .stl/.obj/.ply/.glb")
    p.add_argument("--out", default="grasp_pose.yaml", help="export path (.json/.yaml)")
    p.add_argument("--x", type=float, default=None, help="initial x in mesh frame")
    p.add_argument("--y", type=float, default=None, help="initial y in mesh frame")
    p.add_argument("--z", type=float, default=None, help="initial z in mesh frame")
    p.add_argument("--roll", type=float, default=180.0, help="initial roll in deg")
    p.add_argument("--pitch", type=float, default=0.0, help="initial pitch in deg")
    p.add_argument("--yaw", type=float, default=0.0, help="initial yaw in deg")
    p.add_argument("--opening", type=float, default=DEFAULT_OPENING, help="opening width in m")
    p.add_argument("--pregrasp-opening", type=float, default=0.06, help="pregrasp opening width in m")
    p.add_argument("--no-bbox", action="store_true", help="hide mesh bbox")
    p.add_argument("--frame-size", type=float, default=0.06, help="coord frame size")
    return p.parse_args()


def main():
    args = parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh, enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError(f"无法读取 mesh: {args.mesh}")

    if args.x is None or args.y is None or args.z is None:
        t0, R0 = auto_init_from_bbox(mesh)
        rr, rp, ry = rpy_from_matrix(R0)
        init_xyz = (float(t0[0]), float(t0[1]), float(t0[2]))
        init_rpy_deg = (float(np.rad2deg(rr)), float(np.rad2deg(rp)), float(np.rad2deg(ry)))
    else:
        init_xyz = (args.x, args.y, args.z)
        init_rpy_deg = (args.roll, args.pitch, args.yaw)

    editor = PandaHandEditor(
        mesh_path=args.mesh,
        out_path=args.out,
        init_xyz=init_xyz,
        init_rpy_deg=init_rpy_deg,
        init_opening=args.opening,
        pregrasp_opening=args.pregrasp_opening,
        show_bbox=not args.no_bbox,
        frame_size=args.frame_size,
    )
    editor.run()


if __name__ == "__main__":
    main()
