#!/usr/bin/env python3
# franka_export_mesh_by_joint_no_urdfpy.py
# Version: no_urdfpy_2026_05_09
#
# Export Franka Panda URDF pose to one merged .ply mesh.
# This script does NOT import urdfpy, pyrender, or PyOpenGL.
#
# Install:
#   pip install numpy trimesh pycollada
#
# Run:
#   python franka_export_mesh_by_joint_no_urdfpy.py \
#       --urdf franka_panda/panda.urdf \
#       --output franka_panda_pose.ply

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh


VERSION = "no_urdfpy_2026_05_09"

DEFAULT_JOINTS = [
    -0.07630907275429047,
    0.16981874418467807,
    -0.5780407272806453,
    -1.427642524985215,
    0.055147870278131755,
    1.5863168275091382,
    -1.0853228909191157,
]

DEFAULT_GRIPPER_WIDTH = 0.08021380007266998

DEFAULT_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

DEFAULT_FINGER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]


@dataclass
class Geometry:
    kind: str
    filename: Optional[str] = None
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=float))
    size: Optional[np.ndarray] = None
    radius: Optional[float] = None
    length: Optional[float] = None


@dataclass
class VisualOrCollision:
    origin: np.ndarray
    geometry: Geometry


@dataclass
class Link:
    name: str
    visuals: List[VisualOrCollision] = field(default_factory=list)
    collisions: List[VisualOrCollision] = field(default_factory=list)


@dataclass
class JointLimit:
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limit: JointLimit


@dataclass
class Robot:
    name: str
    links: Dict[str, Link]
    joints: Dict[str, Joint]
    parent_to_joints: Dict[str, List[str]]
    root_link: str


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def children(elem: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in list(elem) if local_name(c.tag) == name]


def child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    xs = children(elem, name)
    return xs[0] if xs else None


def parse_vec(text: Optional[str], default: Iterable[float], n: int) -> np.ndarray:
    if text is None or text.strip() == "":
        return np.array(list(default), dtype=float)
    values = [float(x) for x in text.split()]
    if len(values) != n:
        raise ValueError(f"Expected {n} floats, got {len(values)} in {text!r}")
    return np.array(values, dtype=float)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, cr, -sr],
         [0.0, sr, cr]],
        dtype=float,
    )
    ry = np.array(
        [[cp, 0.0, sp],
         [0.0, 1.0, 0.0],
         [-sp, 0.0, cp]],
        dtype=float,
    )
    rz = np.array(
        [[cy, -sy, 0.0],
         [sy, cy, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def parse_origin(elem: Optional[ET.Element]) -> np.ndarray:
    if elem is None:
        return np.eye(4, dtype=float)
    xyz = parse_vec(elem.attrib.get("xyz"), [0.0, 0.0, 0.0], 3)
    rpy = parse_vec(elem.attrib.get("rpy"), [0.0, 0.0, 0.0], 3)
    return transform_from_xyz_rpy(xyz, rpy)


def axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = axis / norm

    x, y, z = axis
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    C = 1.0 - c

    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def translation_transform(vec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = vec
    return T


def parse_geometry(geometry_elem: ET.Element) -> Geometry:
    mesh_elem = child(geometry_elem, "mesh")
    if mesh_elem is not None:
        filename = mesh_elem.attrib.get("filename")
        if not filename:
            raise ValueError("<mesh> missing filename")
        scale = parse_vec(mesh_elem.attrib.get("scale"), [1.0, 1.0, 1.0], 3)
        return Geometry(kind="mesh", filename=filename, scale=scale)

    box_elem = child(geometry_elem, "box")
    if box_elem is not None:
        return Geometry(kind="box", size=parse_vec(box_elem.attrib.get("size"), [1.0, 1.0, 1.0], 3))

    cylinder_elem = child(geometry_elem, "cylinder")
    if cylinder_elem is not None:
        return Geometry(
            kind="cylinder",
            radius=float(cylinder_elem.attrib["radius"]),
            length=float(cylinder_elem.attrib["length"]),
        )

    sphere_elem = child(geometry_elem, "sphere")
    if sphere_elem is not None:
        return Geometry(kind="sphere", radius=float(sphere_elem.attrib["radius"]))

    raise ValueError("Unsupported geometry: expected mesh / box / cylinder / sphere")


def parse_visual_or_collision(elem: ET.Element) -> VisualOrCollision:
    geometry_elem = child(elem, "geometry")
    if geometry_elem is None:
        raise ValueError(f"<{local_name(elem.tag)}> missing <geometry>")
    return VisualOrCollision(
        origin=parse_origin(child(elem, "origin")),
        geometry=parse_geometry(geometry_elem),
    )


def parse_urdf(urdf_path: Path) -> Robot:
    root = ET.parse(urdf_path).getroot()
    if local_name(root.tag) != "robot":
        raise ValueError(f"{urdf_path} does not have <robot> as root")

    links: Dict[str, Link] = {}
    for link_elem in children(root, "link"):
        name = link_elem.attrib["name"]
        link = Link(name=name)
        link.visuals = [parse_visual_or_collision(v) for v in children(link_elem, "visual")]
        link.collisions = [parse_visual_or_collision(c) for c in children(link_elem, "collision")]
        links[name] = link

    joints: Dict[str, Joint] = {}
    child_links = set()
    parent_to_joints: Dict[str, List[str]] = {}

    for joint_elem in children(root, "joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "fixed")

        parent_elem = child(joint_elem, "parent")
        child_elem = child(joint_elem, "child")
        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint {name} missing parent/child")

        parent = parent_elem.attrib["link"]
        child_link = child_elem.attrib["link"]
        child_links.add(child_link)

        axis_elem = child(joint_elem, "axis")
        axis = parse_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, [1.0, 0.0, 0.0], 3)

        limit_elem = child(joint_elem, "limit")
        limit = JointLimit()
        if limit_elem is not None:
            if "lower" in limit_elem.attrib:
                limit.lower = float(limit_elem.attrib["lower"])
            if "upper" in limit_elem.attrib:
                limit.upper = float(limit_elem.attrib["upper"])

        joints[name] = Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child_link,
            origin=parse_origin(child(joint_elem, "origin")),
            axis=axis,
            limit=limit,
        )
        parent_to_joints.setdefault(parent, []).append(name)

    root_candidates = [link_name for link_name in links if link_name not in child_links]
    if not root_candidates:
        raise ValueError("Cannot find URDF root link")
    root_link = root_candidates[0]

    return Robot(
        name=root.attrib.get("name", "robot"),
        links=links,
        joints=joints,
        parent_to_joints=parent_to_joints,
        root_link=root_link,
    )


def parse_package_args(values: Optional[List[str]]) -> Dict[str, Path]:
    out = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Bad --package {value!r}; expected package_name=/path")
        package, path = value.split("=", 1)
        out[package] = Path(path).expanduser().resolve()
    return out


def default_search_roots(urdf_path: Path) -> List[Path]:
    roots = [Path.cwd(), urdf_path.parent, urdf_path.parent.parent, urdf_path.parent.parent.parent]
    out = []
    seen = set()
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            continue
        if rr.exists() and rr not in seen:
            out.append(rr)
            seen.add(rr)
    return out


def resolve_package_uri(uri: str, package_map: Dict[str, Path], search_roots: Iterable[Path]) -> Path:
    rest = uri[len("package://"):]
    package, rel = rest.split("/", 1)

    candidates = []
    if package in package_map:
        candidates.append(package_map[package] / rel)
    for root in search_roots:
        candidates.append(root / package / rel)

    for c in candidates:
        cc = c.expanduser().resolve()
        if cc.exists():
            return cc

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Cannot resolve {uri}\nTried:\n{tried}\n"
        f"Use: --package {package}=/absolute/path/to/{package}"
    )


def resolve_mesh_path(filename: str, urdf_dir: Path, package_map: Dict[str, Path], search_roots: Iterable[Path]) -> Path:
    if filename.startswith("package://"):
        return resolve_package_uri(filename, package_map, search_roots)
    if filename.startswith("file://"):
        return Path(filename[len("file://"):]).expanduser().resolve()
    if "://" in filename:
        raise ValueError(f"Unsupported remote mesh URI: {filename}")

    p = Path(filename).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (urdf_dir / p).resolve()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"No geometry in scene: {path}")
        parts = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name].copy()
            geom.apply_transform(transform)
            parts.append(geom)
        if not parts:
            parts = [g.copy() for g in loaded.geometry.values()]
        return trimesh.util.concatenate(parts)

    raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)}")


def mesh_from_geometry(
    geometry: Geometry,
    urdf_dir: Path,
    package_map: Dict[str, Path],
    search_roots: Iterable[Path],
    cache: Dict[Tuple[str, Tuple[float, float, float]], trimesh.Trimesh],
) -> trimesh.Trimesh:
    if geometry.kind == "mesh":
        assert geometry.filename is not None
        p = resolve_mesh_path(geometry.filename, urdf_dir, package_map, search_roots)
        key = (str(p), tuple(float(x) for x in geometry.scale))
        if key not in cache:
            m = load_mesh(p)
            if not np.allclose(geometry.scale, np.ones(3)):
                S = np.eye(4)
                S[0, 0], S[1, 1], S[2, 2] = geometry.scale
                m.apply_transform(S)
            cache[key] = m
        return cache[key].copy()

    if geometry.kind == "box":
        assert geometry.size is not None
        return trimesh.creation.box(extents=geometry.size)

    if geometry.kind == "cylinder":
        assert geometry.radius is not None and geometry.length is not None
        return trimesh.creation.cylinder(radius=geometry.radius, height=geometry.length, sections=48)

    if geometry.kind == "sphere":
        assert geometry.radius is not None
        return trimesh.creation.uv_sphere(radius=geometry.radius)

    raise ValueError(f"Unsupported geometry kind {geometry.kind}")


def joint_transform(joint: Joint, value: float) -> np.ndarray:
    if joint.joint_type == "fixed":
        return np.eye(4)

    if joint.joint_type in ("revolute", "continuous"):
        return axis_angle_transform(joint.axis, value)

    if joint.joint_type == "prismatic":
        axis = joint.axis.astype(float)
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
        return translation_transform(axis * value)

    # Panda should not need these for visual export.
    if joint.joint_type in ("floating", "planar"):
        return np.eye(4)

    raise ValueError(f"Unsupported joint type {joint.joint_type} for {joint.name}")


def clamp_value(robot: Robot, joint_name: str, value: float, clamp: bool) -> float:
    if not clamp or joint_name not in robot.joints:
        return float(value)

    lim = robot.joints[joint_name].limit
    out = float(value)
    if lim.lower is not None:
        out = max(out, lim.lower)
    if lim.upper is not None:
        out = min(out, lim.upper)

    if not np.isclose(out, value):
        print(f"[warn] Clamped {joint_name}: requested {value:.12g}, using {out:.12g}")
    return out


def build_joint_cfg(
    robot: Robot,
    joints: List[float],
    gripper_width: float,
    arm_joint_names: List[str],
    finger_joint_names: List[str],
    clamp: bool,
) -> Dict[str, float]:
    cfg = {}

    if len(joints) != 7:
        raise ValueError(f"Expected 7 arm joint values, got {len(joints)}")

    for name, value in zip(arm_joint_names, joints):
        if name not in robot.joints:
            print(f"[warn] Joint {name} not found in URDF, skipping")
            continue
        cfg[name] = clamp_value(robot, name, value, clamp)

    finger_value = float(gripper_width) / 2.0
    for name in finger_joint_names:
        if name not in robot.joints:
            print(f"[warn] Finger joint {name} not found in URDF, skipping")
            continue
        cfg[name] = clamp_value(robot, name, finger_value, clamp)

    return cfg


def compute_link_transforms(robot: Robot, joint_cfg: Dict[str, float]) -> Dict[str, np.ndarray]:
    transforms = {robot.root_link: np.eye(4)}
    stack = [robot.root_link]

    while stack:
        parent_link = stack.pop()
        parent_T = transforms[parent_link]

        for joint_name in robot.parent_to_joints.get(parent_link, []):
            j = robot.joints[joint_name]
            q = joint_cfg.get(j.name, 0.0)
            transforms[j.child] = parent_T @ j.origin @ joint_transform(j, q)
            stack.append(j.child)

    return transforms


def export_mesh(args: argparse.Namespace) -> None:
    urdf_path = args.urdf.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Script version: {VERSION}")
    print("[info] This script does not use urdfpy / pyrender / PyOpenGL.")

    package_map = parse_package_args(args.package)
    search_roots = default_search_roots(urdf_path)
    for r in args.search_root or []:
        rr = r.expanduser().resolve()
        if rr.exists() and rr not in search_roots:
            search_roots.append(rr)

    print("[info] Package search roots:")
    for r in search_roots:
        print(f"  {r}")

    robot = parse_urdf(urdf_path)
    print(f"[info] Loaded robot: {robot.name}")
    print(f"[info] Root link: {robot.root_link}")
    print(f"[info] Links: {len(robot.links)}, joints: {len(robot.joints)}")
    print(f"[info] Joint names: {', '.join(robot.joints.keys())}")

    joint_cfg = build_joint_cfg(
        robot=robot,
        joints=list(args.joints),
        gripper_width=float(args.gripper_width),
        arm_joint_names=list(args.arm_joint_names),
        finger_joint_names=list(args.finger_joint_names),
        clamp=not args.no_limit_clamp,
    )

    print("[info] Joint config used:")
    for k in sorted(joint_cfg):
        print(f"  {k}: {joint_cfg[k]:.12g}")

    link_T = compute_link_transforms(robot, joint_cfg)
    cache = {}
    parts = []

    for link_name, link in robot.links.items():
        if link_name not in link_T:
            print(f"[warn] Link {link_name} is unreachable from root; skipping")
            continue

        items = link.visuals if args.mesh_source == "visual" else link.collisions
        for item in items:
            m = mesh_from_geometry(
                item.geometry,
                urdf_dir=urdf_path.parent,
                package_map=package_map,
                search_roots=search_roots,
                cache=cache,
            )
            m.apply_transform(link_T[link_name] @ item.origin)
            parts.append(m)

    if not parts:
        raise RuntimeError(f"No {args.mesh_source} geometries found in URDF")

    merged = trimesh.util.concatenate(parts)
    try:
        merged.remove_duplicate_faces()
    except Exception:
        pass
    merged.remove_unreferenced_vertices()

    merged.export(str(output_path))
    print(f"[done] Exported: {output_path}")
    print(f"[done] parts={len(parts)}, vertices={len(merged.vertices)}, faces={len(merged.faces)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", type=Path, default=Path("franka_panda/panda.urdf"))
    p.add_argument("--output", type=Path, default=Path("franka_panda_pose.ply"))
    p.add_argument("--joints", type=float, nargs=7, default=DEFAULT_JOINTS)
    p.add_argument("--gripper-width", type=float, default=DEFAULT_GRIPPER_WIDTH)
    p.add_argument("--arm-joint-names", nargs=7, default=DEFAULT_ARM_JOINT_NAMES)
    p.add_argument("--finger-joint-names", nargs="*", default=DEFAULT_FINGER_JOINT_NAMES)
    p.add_argument("--mesh-source", choices=["visual", "collision"], default="visual")
    p.add_argument("--package", action="append")
    p.add_argument("--search-root", type=Path, action="append", default=[])
    p.add_argument("--no-limit-clamp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    export_mesh(parse_args())
