"""
Open3D GUI helper for manually annotating FPSA deformation labels.

This version is intentionally simple and stable:
  1. The full triangle mesh, vertex point cloud, and a small world XYZ frame are shown together.
  2. Picked vertices show their original mesh vertex id as a 3D label.
  3. Direction selection is button-based; there is no large center axis selector in the scene.
  4. Direction buttons include +X/-X/+Y/-Y/+Z/-Z plus diagonal plane directions such as +XY, +X-Y, -X+Y, -XY.
  5. Every handle has a direction. Newly selected handles default to +X.
  6. The output YAML is directly compatible with grasp_randomizer.py and uses
     per-handle `reshaped_vector` and per-handle `range` lists.

Typical usage:
    python FPSA_annotation_helper_gui_axis.py \
        --mesh ./data/objects/gear_extraction/gear/gear.obj \
        --label gear_streching \
        --out gear_streching.yaml

Controls:
    C             : constrained/fixed vertex picking mode
    H             : reshaped/handle vertex picking mode
    D             : direction mode
    Left click    : in C/H mode, toggle nearest vertex; in D mode, select a handle
    Direction buttons: set selected handle direction to axis or diagonal directions
    X/Y/Z         : set selected handle direction to +x/+y/+z
    1/2/3         : set selected handle direction to -x/-y/-z
    A             : apply selected handle direction to all handles
    Backspace     : remove selected vertex/direction
    S             : save YAML
    Q/Esc         : quit

Notes:
    - In D mode, click a red handle point first, then click a direction button or press X/Y/Z/1/2/3.
    - If no handle direction is manually changed, it remains the default +X.
    - `range` is written as one editable placeholder per handle: [[0.0, 0.0], ...].
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import trimesh

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except Exception as exc:  # pragma: no cover
    gui = None
    rendering = None
    _GUI_IMPORT_ERROR = exc
else:
    _GUI_IMPORT_ERROR = None


DIRECTION_BUTTONS: List[Tuple[str, List[float]]] = [
    ("+X", [1.0, 0.0, 0.0]),
    ("-X", [-1.0, 0.0, 0.0]),
    ("+Y", [0.0, 1.0, 0.0]),
    ("-Y", [0.0, -1.0, 0.0]),
    ("+Z", [0.0, 0.0, 1.0]),
    ("-Z", [0.0, 0.0, -1.0]),
    ("+XY", [1.0, 1.0, 0.0]),
    ("+X-Y", [1.0, -1.0, 0.0]),
    ("-X+Y", [-1.0, 1.0, 0.0]),
    ("-XY", [-1.0, -1.0, 0.0]),
    ("+XZ", [1.0, 0.0, 1.0]),
    ("+X-Z", [1.0, 0.0, -1.0]),
    ("-X+Z", [-1.0, 0.0, 1.0]),
    ("-XZ", [-1.0, 0.0, -1.0]),
    ("+YZ", [0.0, 1.0, 1.0]),
    ("+Y-Z", [0.0, 1.0, -1.0]),
    ("-Y+Z", [0.0, -1.0, 1.0]),
    ("-YZ", [0.0, -1.0, -1.0]),
]
AXIS_DIRECTIONS: Dict[str, List[float]] = dict(DIRECTION_BUTTONS)
DEFAULT_DIRECTION = [1.0, 0.0, 0.0]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def unique_ints(xs: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for x in xs:
        x = int(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_vector(v: Sequence[float]) -> List[float]:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise ValueError(f"Direction vector is too small: {arr.tolist()}")
    return (arr / n).tolist()


def mesh_scale(V: np.ndarray) -> float:
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    return diag if diag > 1e-12 else 1.0


def to_builtin(x: Any) -> Any:
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


def key_name(key: Any) -> str:
    s = str(key)
    if "." in s:
        s = s.split(".")[-1]
    return s.upper()


# -----------------------------------------------------------------------------
# Mesh IO / adjacency
# -----------------------------------------------------------------------------


def load_mesh_keep_vertex_order(path: str | Path) -> Tuple[np.ndarray, np.ndarray, o3d.geometry.TriangleMesh]:
    """Load mesh with trimesh maintain_order=True, then convert to Open3D."""
    path = Path(path)
    mesh = trimesh.load(path, process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected one mesh, got {type(mesh)} from {path}")

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"Mesh must be triangulated, got faces shape {F.shape}")

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(V)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(F)
    mesh_o3d.compute_vertex_normals()
    return V, F, mesh_o3d


def build_vertex_adjacency(F: np.ndarray, n_vertices: int) -> List[List[int]]:
    adj = [set() for _ in range(n_vertices)]
    for tri in np.asarray(F, dtype=np.int64):
        i, j, k = map(int, tri)
        adj[i].update([j, k])
        adj[j].update([i, k])
        adj[k].update([i, j])
    return [sorted(x) for x in adj]


def expand_k_ring(seed_ids: Sequence[int], F: np.ndarray, n_vertices: int, k_ring: int) -> List[int]:
    seed_ids = unique_ints(seed_ids)
    if k_ring <= 0 or not seed_ids:
        return seed_ids

    adj = build_vertex_adjacency(F, n_vertices)
    visited = set(seed_ids)
    q = deque((int(v), 0) for v in seed_ids)

    while q:
        v, depth = q.popleft()
        if depth >= k_ring:
            continue
        for nb in adj[v]:
            if nb not in visited:
                visited.add(nb)
                q.append((nb, depth + 1))

    return sorted(visited)


# -----------------------------------------------------------------------------
# Geometry construction
# -----------------------------------------------------------------------------


def make_vertex_point_cloud(
    V: np.ndarray,
    constrained_ids: Sequence[int],
    reshaped_ids: Sequence[int],
    selected_id: Optional[int] = None,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64))

    colors = np.tile(np.array([[0.86, 0.86, 0.86]], dtype=np.float64), (len(V), 1))
    for vid in constrained_ids:
        colors[int(vid)] = np.array([0.05, 0.25, 1.0])  # constrained: blue
    for vid in reshaped_ids:
        colors[int(vid)] = np.array([1.0, 0.08, 0.03])  # handle: red
    if selected_id is not None:
        colors[int(selected_id)] = np.array([1.0, 0.9, 0.05])  # selected: yellow

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_direction_lines(V: np.ndarray, directions: Dict[int, Sequence[float]], length: float) -> o3d.geometry.LineSet:
    points: List[List[float]] = []
    lines: List[List[int]] = []
    colors: List[List[float]] = []

    for vid, direction in directions.items():
        if vid < 0 or vid >= len(V):
            continue
        try:
            d = np.asarray(normalize_vector(direction), dtype=np.float64)
        except ValueError:
            continue
        p0 = np.asarray(V[int(vid)], dtype=np.float64)
        p1 = p0 + float(length) * d
        points.extend([p0.tolist(), p1.tolist()])
        i = len(points) - 2
        lines.append([i, i + 1])
        colors.append([0.1, 0.85, 0.1])

    ls = o3d.geometry.LineSet()
    if points:
        ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


def make_arrow_tip_points(V: np.ndarray, directions: Dict[int, Sequence[float]], length: float) -> o3d.geometry.PointCloud:
    pts: List[List[float]] = []
    for vid, direction in directions.items():
        try:
            d = np.asarray(normalize_vector(direction), dtype=np.float64)
        except ValueError:
            continue
        pts.append((np.asarray(V[int(vid)], dtype=np.float64) + float(length) * d).tolist())

    pcd = o3d.geometry.PointCloud()
    if pts:
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.1, 0.95, 0.1]]), (len(pts), 1)))
    return pcd


def make_world_axes(scale: float, origin: Sequence[float] = (0.0, 0.0, 0.0)) -> o3d.geometry.TriangleMesh:
    # Keep only a small world coordinate reference. Direction selection is done by GUI buttons.
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10 * float(scale), origin=np.asarray(origin, dtype=np.float64))


# -----------------------------------------------------------------------------
# YAML writer
# -----------------------------------------------------------------------------


class _FlowStyleList(list):
    """Marker list that PyYAML should render as `[a, b, c]`."""


if yaml is not None:
    class _IndentedSafeDumper(yaml.SafeDumper):
        """Safe dumper that indents block sequences under mapping keys."""

        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)


    def _flow_style_list_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", list(data), flow_style=True)


    _IndentedSafeDumper.add_representer(_FlowStyleList, _flow_style_list_representer)
else:  # pragma: no cover
    _IndentedSafeDumper = None


def _format_fpsa_yaml_lists(x: Any) -> Any:
    """Format selected FPSA YAML fields with compact inline list rows.

    Desired output example:
        constrained_ids: [330, 344, 345]
        reshaped_ids: [378, 20]
        reshaped_vector:
          - [1.0, 0.0, 0.0]
          - [1.0, 0.0, 0.0]
        range:
          - [0.0, 0.0]
          - [0.0, 0.0]
    """
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in {"constrained_ids", "reshaped_ids"}:
                out[k] = _FlowStyleList(to_builtin(v))
            elif k in {"reshaped_vector", "range"}:
                out[k] = [_FlowStyleList(to_builtin(row)) for row in v]
            else:
                out[k] = _format_fpsa_yaml_lists(v)
        return out
    if isinstance(x, list):
        return [_format_fpsa_yaml_lists(v) for v in x]
    return x


def deformation_record(
    label: str,
    constrained_ids: Sequence[int],
    reshaped_ids: Sequence[int],
    reshaped_vectors: Sequence[Sequence[float]],
    description: str,
    coupled: bool,
    normalize_vector_flag: bool,
    distribution: str,
    weight: float,
) -> dict:
    reshaped_ids = unique_ints(reshaped_ids)
    constrained_ids = unique_ints([*constrained_ids, *reshaped_ids])
    if len(reshaped_vectors) != len(reshaped_ids):
        raise ValueError(
            f"reshaped_vectors length ({len(reshaped_vectors)}) must match reshaped_ids length ({len(reshaped_ids)})"
        )

    return {
        "label": str(label),
        "type": "displacement",
        "description": str(description),
        # FPSA expects handles to be part of the constrained set too.
        "constrained_ids": constrained_ids,
        "reshaped_ids": reshaped_ids,
        "reshaped_vector": [[float(x) for x in normalize_vector(v)] for v in reshaped_vectors],
        # One editable placeholder per handle. Fill these manually.
        "range": [[0.0, 0.0] for _ in reshaped_ids],
        "coupled": bool(coupled),
        "normalize_vector": bool(normalize_vector_flag),
        "distribution": str(distribution),
        "weight": float(weight),
    }


def write_deformation_yaml(path: str | Path, record: dict, wrap_deformations: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"deformations": [record]} if wrap_deformations else record
    data = _format_fpsa_yaml_lists(to_builtin(data))

    header = (
        "# FPSA manual annotation.\n"
        "# Edit `range` manually before running grasp_randomizer.py.\n"
        "# `constrained_ids` includes `reshaped_ids`.\n"
        "# `reshaped_vector` and `range` are per-handle and match the order of `reshaped_ids`.\n"
    )
    if yaml is not None:
        text = yaml.dump(
            data,
            Dumper=_IndentedSafeDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    else:
        text = json.dumps(to_builtin(data), indent=2, ensure_ascii=False)
    path.write_text(header + text, encoding="utf-8")


# -----------------------------------------------------------------------------
# GUI app
# -----------------------------------------------------------------------------


class FPSAAnnotationApp:
    MODE_CONSTRAINED = "constrained"
    MODE_RESHAPED = "reshaped"
    MODE_DIRECTION = "direction"

    def __init__(self, args: argparse.Namespace):
        if gui is None or rendering is None:
            raise RuntimeError(
                "open3d.visualization.gui/rendering is not available in this Open3D build. "
                f"Original import error: {_GUI_IMPORT_ERROR}"
            )

        self.args = args
        self.V, self.F, self.mesh = load_mesh_keep_vertex_order(args.mesh)
        self.scale = mesh_scale(self.V)
        self.pick_radius = float(args.pick_radius) * self.scale
        self.arrow_len = float(args.arrow_length) * self.scale
        self.mode = self.MODE_CONSTRAINED

        self.constrained_seed: List[int] = []
        self.reshaped_seed: List[int] = []
        self.directions: Dict[int, List[float]] = {}
        self.selected_id: Optional[int] = None
        self.status_text = ""
        self._labels: List[Any] = []

        self.app = gui.Application.instance
        self.window = self.app.create_window("FPSA manual annotation - mesh + ids + button directions", 1500, 980)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_on_mouse(self._on_mouse_safe)
        self.scene_widget.set_on_key(self._on_key_safe)

        self.panel = gui.Vert(0.25 * self.window.theme.font_size, gui.Margins(8, 8, 8, 8))
        self.label_status = gui.Label("")
        self.label_help = gui.Label(
            "C constrained | H handle | D direction | click handle then direction button | X/Y/Z, 1/2/3 shortcuts | S save | Q/Esc quit"
        )
        self.panel.add_child(self.label_status)
        self.panel.add_child(self.label_help)
        self._add_axis_buttons()

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self._setup_scene()
        self._update_scene_geometry()
        self._update_status("loaded mesh")

    def _add_axis_buttons(self) -> None:
        rows = [
            ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
            ["+XY", "+X-Y", "-X+Y", "-XY"],
            ["+XZ", "+X-Z", "-X+Z", "-XZ"],
            ["+YZ", "+Y-Z", "-Y+Z", "-YZ"],
        ]
        for names in rows:
            row = gui.Horiz(0.25 * self.window.theme.font_size)
            for name in names:
                btn = gui.Button(name)
                btn.set_on_clicked(lambda n=name: self._set_direction_button_safe(n))
                row.add_child(btn)
            self.panel.add_child(row)

        row = gui.Horiz(0.25 * self.window.theme.font_size)
        btn_all = gui.Button("Apply selected dir to all handles")
        btn_all.set_on_clicked(self._apply_selected_direction_to_all_safe)
        row.add_child(btn_all)
        self.panel.add_child(row)

    def _on_layout(self, ctx: gui.LayoutContext) -> None:
        r = self.window.content_rect
        panel_h = 170
        self.panel.frame = gui.Rect(r.x, r.y, r.width, panel_h)
        self.scene_widget.frame = gui.Rect(r.x, r.y + panel_h, r.width, r.height - panel_h)

    def _setup_scene(self) -> None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(self.V.min(axis=0), self.V.max(axis=0))
        center = bbox.get_center()
        extent = max(float(np.linalg.norm(bbox.get_extent())), 1e-6)

        self.scene_widget.scene.set_background([0.03, 0.03, 0.035, 1.0])
        self.scene_widget.scene.camera.look_at(
            center,
            center + np.array([0.0, -1.8 * extent, 0.8 * extent]),
            [0.0, 0.0, 1.0],
        )

        self.scene_widget.scene.scene.set_sun_light([0.2, -0.7, -0.5], [1.0, 1.0, 1.0], 65000)
        self.scene_widget.scene.scene.enable_sun_light(True)

    def _material_mesh(self):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency"
        mat.base_color = [0.65, 0.65, 0.65, float(self.args.mesh_alpha)]
        return mat

    def _material_points(self, point_size: Optional[float] = None):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = float(point_size if point_size is not None else self.args.point_size)
        return mat

    def _material_lines(self, width: Optional[float] = None):
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = float(width if width is not None else self.args.line_width)
        return mat

    def _material_default(self):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        return mat

    def _remove_if_exists(self, name: str) -> None:
        try:
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.remove_geometry(name)
        except Exception:
            pass

    def _clear_3d_labels(self) -> None:
        for label in self._labels:
            try:
                self.scene_widget.remove_3d_label(label)
            except Exception:
                pass
        self._labels = []

    def _add_3d_label_safe(self, pos: Sequence[float], text: str, color: Sequence[float]) -> None:
        if not bool(self.args.show_ids):
            return
        if not hasattr(self.scene_widget, "add_3d_label"):
            return
        try:
            label = self.scene_widget.add_3d_label(np.asarray(pos, dtype=np.float64), text)
            if hasattr(label, "color"):
                label.color = gui.Color(float(color[0]), float(color[1]), float(color[2]))
            if hasattr(label, "scale"):
                label.scale = float(self.args.id_label_scale)
            self._labels.append(label)
        except Exception:
            # Some Open3D builds have partial 3D-label support. Ignore instead of crashing.
            pass

    def _update_vertex_id_labels(self) -> None:
        self._clear_3d_labels()
        offset = np.array([0.0, 0.0, float(self.args.id_label_offset) * self.scale], dtype=np.float64)
        for vid in unique_ints(self.constrained_seed):
            self._add_3d_label_safe(self.V[int(vid)] + offset, f"C {int(vid)}", [0.25, 0.45, 1.0])
        for vid in unique_ints(self.reshaped_seed):
            prefix = "H*" if self.selected_id == int(vid) else "H"
            vec = self.directions.get(int(vid), DEFAULT_DIRECTION)
            vtxt = self._axis_name_for_vector(vec)
            self._add_3d_label_safe(self.V[int(vid)] + 1.4 * offset, f"{prefix} {int(vid)} {vtxt}", [1.0, 0.25, 0.15])

    def _axis_name_for_vector(self, vec: Sequence[float]) -> str:
        arr = np.asarray(normalize_vector(vec), dtype=np.float64)
        best_name = "+X"
        best_dot = -1.0
        for name, direction in AXIS_DIRECTIONS.items():
            dot = float(np.dot(arr, np.asarray(direction, dtype=np.float64)))
            if dot > best_dot:
                best_dot = dot
                best_name = name
        return best_name

    def _update_scene_geometry(self) -> None:
        for name in ["mesh", "vertices", "directions", "arrow_tips", "world_axes"]:
            self._remove_if_exists(name)

        mesh_show = o3d.geometry.TriangleMesh(self.mesh)
        mesh_show.compute_vertex_normals()
        self.scene_widget.scene.add_geometry("mesh", mesh_show, self._material_mesh())

        pcd = make_vertex_point_cloud(self.V, self.constrained_seed, self.reshaped_seed, self.selected_id)
        self.scene_widget.scene.add_geometry("vertices", pcd, self._material_points())

        world_axes = make_world_axes(self.scale, origin=(0.0, 0.0, 0.0))
        self.scene_widget.scene.add_geometry("world_axes", world_axes, self._material_default())

        if self.directions:
            lines = make_direction_lines(self.V, self.directions, self.arrow_len)
            self.scene_widget.scene.add_geometry("directions", lines, self._material_lines())
            tips = make_arrow_tip_points(self.V, self.directions, self.arrow_len)
            self.scene_widget.scene.add_geometry("arrow_tips", tips, self._material_points(point_size=max(self.args.point_size * 1.2, 10.0)))

        self._update_vertex_id_labels()
        self.window.post_redraw()

    def _update_status(self, extra: str = "") -> None:
        self.status_text = (
            f"Mode: {self.mode.upper()} | "
            f"constrained: {len(self.constrained_seed)} | "
            f"handles: {len(self.reshaped_seed)} | "
            f"directions: {len(self.directions)}/{len(self.reshaped_seed)} | "
            f"selected: {self.selected_id}"
        )
        if extra:
            self.status_text += f" | {extra}"
        self.label_status.text = self.status_text

    def _local_xy(self, event) -> Tuple[int, int]:
        x = int(event.x - self.scene_widget.frame.x)
        y = int(event.y - self.scene_widget.frame.y)
        x = max(0, min(x, int(self.scene_widget.frame.width) - 1))
        y = max(0, min(y, int(self.scene_widget.frame.height) - 1))
        return x, y

    def _unproject_with_depth(self, event, callback) -> None:
        x, y = self._local_xy(event)
        width = max(1, int(self.scene_widget.frame.width))
        height = max(1, int(self.scene_widget.frame.height))

        def depth_callback(depth_image):
            try:
                depth = np.asarray(depth_image)
                if depth.size == 0:
                    callback(None)
                    return
                yy = max(0, min(y, depth.shape[0] - 1))
                xx = max(0, min(x, depth.shape[1] - 1))
                z = float(depth[yy, xx])
                if not np.isfinite(z) or z >= 1.0:
                    callback(None)
                    return
                world = self.scene_widget.scene.camera.unproject(x, y, z, width, height)
                callback(np.asarray(world, dtype=np.float64))
            except Exception:
                traceback.print_exc()
                callback(None)

        self.scene_widget.scene.scene.render_to_depth_image(depth_callback)

    def _nearest_vertex(self, world: np.ndarray) -> Optional[int]:
        d = np.linalg.norm(self.V - world.reshape(1, 3), axis=1)
        idx = int(np.argmin(d))
        if d[idx] > self.pick_radius:
            return None
        return idx

    def _nearest_handle(self, world: np.ndarray) -> Optional[int]:
        if not self.reshaped_seed:
            return None
        ids = np.asarray(self.reshaped_seed, dtype=np.int64)
        d = np.linalg.norm(self.V[ids] - world.reshape(1, 3), axis=1)
        j = int(np.argmin(d))
        if d[j] > max(self.pick_radius, 2.5 * self.args.point_size * 1e-4 * self.scale):
            return None
        return int(ids[j])

    def _toggle_in_current_set(self, vid: int) -> None:
        if self.mode == self.MODE_CONSTRAINED:
            target = self.constrained_seed
        elif self.mode == self.MODE_RESHAPED:
            target = self.reshaped_seed
        else:
            return

        if vid in target:
            target.remove(vid)
            if vid in self.directions:
                del self.directions[vid]
            if self.selected_id == vid:
                self.selected_id = None
        else:
            target.append(vid)
            target[:] = unique_ints(target)
            if self.mode == self.MODE_RESHAPED:
                self.directions.setdefault(vid, DEFAULT_DIRECTION.copy())
                self.selected_id = vid

    def _set_direction_for_selected(self, direction: Sequence[float], apply_all: bool = False) -> None:
        d = normalize_vector(direction)
        if apply_all:
            for vid in self.reshaped_seed:
                self.directions[int(vid)] = d.copy()
            self._update_scene_geometry()
            self._update_status(f"applied {self._axis_name_for_vector(d)} to all handles")
            return

        if self.selected_id is None:
            # Reasonable fallback: if no handle is selected, apply to all handles.
            for vid in self.reshaped_seed:
                self.directions[int(vid)] = d.copy()
            self._update_scene_geometry()
            self._update_status(f"no selected handle; applied {self._axis_name_for_vector(d)} to all handles")
            return

        if int(self.selected_id) not in self.reshaped_seed:
            self._update_status("selected id is not a handle")
            return

        self.directions[int(self.selected_id)] = d.copy()
        self._update_scene_geometry()
        self._update_status(f"handle {self.selected_id} direction = {self._axis_name_for_vector(d)}")

    def _set_direction_by_axis_name(self, axis_name: str, apply_all: bool = False) -> None:
        if axis_name not in AXIS_DIRECTIONS:
            self._update_status(f"unknown axis {axis_name}")
            return
        self._set_direction_for_selected(AXIS_DIRECTIONS[axis_name], apply_all=apply_all)

    def _set_direction_button_safe(self, axis_name: str) -> None:
        try:
            self._set_direction_by_axis_name(axis_name, apply_all=False)
        except Exception:
            traceback.print_exc()
            self._update_status("button callback error")

    def _apply_selected_direction_to_all_safe(self) -> None:
        try:
            if self.selected_id is not None and self.selected_id in self.directions:
                self._set_direction_for_selected(self.directions[self.selected_id], apply_all=True)
            else:
                self._set_direction_for_selected(DEFAULT_DIRECTION, apply_all=True)
        except Exception:
            traceback.print_exc()
            self._update_status("apply-all callback error")

    def _safe_callback_result(self) -> int:
        return gui.Widget.EventCallbackResult.HANDLED

    def _on_mouse_safe(self, event) -> int:
        try:
            return self._on_mouse(event)
        except Exception:
            traceback.print_exc()
            self._update_status("mouse callback error; see terminal traceback")
            return self._safe_callback_result()

    def _on_key_safe(self, event) -> int:
        try:
            return self._on_key(event)
        except Exception:
            traceback.print_exc()
            self._update_status("key callback error; see terminal traceback")
            return self._safe_callback_result()

    def _on_mouse(self, event) -> int:
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            def cb(world):
                if world is None:
                    self._update_status("click missed geometry; zoom in or increase --point-size")
                    return

                if self.mode in {self.MODE_CONSTRAINED, self.MODE_RESHAPED}:
                    vid = self._nearest_vertex(world)
                    if vid is None:
                        self._update_status("no nearby vertex; zoom in or increase --pick-radius")
                        return
                    self._toggle_in_current_set(vid)
                    self._update_scene_geometry()
                    self._update_status(f"toggled vertex {vid}")
                    return

                if self.mode == self.MODE_DIRECTION:
                    hid = self._nearest_handle(world)
                    if hid is None:
                        self._update_status("click a red handle, then choose a direction button")
                        return
                    self.selected_id = hid
                    self.directions.setdefault(hid, DEFAULT_DIRECTION.copy())
                    self._update_scene_geometry()
                    self._update_status(f"selected handle {hid}; choose a direction button")

            self._unproject_with_depth(event, cb)
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key(self, event) -> int:
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        key = event.key
        kname = key_name(key)

        if key == gui.KeyName.C or kname == "C":
            self.mode = self.MODE_CONSTRAINED
        elif key == gui.KeyName.H or kname == "H":
            self.mode = self.MODE_RESHAPED
        elif key == gui.KeyName.D or kname == "D":
            self.mode = self.MODE_DIRECTION
        elif key == gui.KeyName.Q or kname in {"Q", "ESCAPE"}:
            self.window.close()
            return gui.Widget.EventCallbackResult.HANDLED
        elif key == gui.KeyName.S or kname == "S":
            self.save()
            return gui.Widget.EventCallbackResult.HANDLED
        elif key == gui.KeyName.A or kname == "A":
            if self.selected_id is not None and self.selected_id in self.directions:
                self._set_direction_for_selected(self.directions[self.selected_id], apply_all=True)
            else:
                self._set_direction_for_selected(DEFAULT_DIRECTION, apply_all=True)
            return gui.Widget.EventCallbackResult.HANDLED
        elif key == gui.KeyName.BACKSPACE or kname == "BACKSPACE":
            self._delete_selected()
            return gui.Widget.EventCallbackResult.HANDLED
        elif kname == "X" or key == gui.KeyName.X:
            self._set_direction_by_axis_name("+X")
            return gui.Widget.EventCallbackResult.HANDLED
        elif kname == "Y" or key == gui.KeyName.Y:
            self._set_direction_by_axis_name("+Y")
            return gui.Widget.EventCallbackResult.HANDLED
        elif kname == "Z" or key == gui.KeyName.Z:
            self._set_direction_by_axis_name("+Z")
            return gui.Widget.EventCallbackResult.HANDLED
        elif kname in {"1", "ONE", "N1", "NUMPAD_1", "KP_1"} or kname.endswith("ONE"):
            self._set_direction_by_axis_name("-X")
            return gui.Widget.EventCallbackResult.HANDLED
        elif kname in {"2", "TWO", "N2", "NUMPAD_2", "KP_2"} or kname.endswith("TWO"):
            self._set_direction_by_axis_name("-Y")
            return gui.Widget.EventCallbackResult.HANDLED
        elif kname in {"3", "THREE", "N3", "NUMPAD_3", "KP_3"} or kname.endswith("THREE"):
            self._set_direction_by_axis_name("-Z")
            return gui.Widget.EventCallbackResult.HANDLED
        else:
            return gui.Widget.EventCallbackResult.IGNORED

        self._update_scene_geometry()
        self._update_status()
        return gui.Widget.EventCallbackResult.HANDLED

    def _delete_selected(self) -> None:
        if self.selected_id is None:
            return
        vid = int(self.selected_id)
        if self.mode == self.MODE_CONSTRAINED and vid in self.constrained_seed:
            self.constrained_seed.remove(vid)
        elif self.mode in {self.MODE_RESHAPED, self.MODE_DIRECTION} and vid in self.reshaped_seed:
            self.reshaped_seed.remove(vid)
            self.directions.pop(vid, None)
        self.selected_id = None
        self._update_scene_geometry()
        self._update_status("deleted selected")

    def save(self) -> None:
        constrained_ids = expand_k_ring(self.constrained_seed, self.F, len(self.V), self.args.constraint_k_ring)
        reshaped_ids = expand_k_ring(self.reshaped_seed, self.F, len(self.V), self.args.reshaped_k_ring)

        if not constrained_ids:
            self._update_status("cannot save: no constrained vertices")
            print("Cannot save: no constrained vertices")
            return
        if not reshaped_ids:
            self._update_status("cannot save: no reshaped vertices")
            print("Cannot save: no reshaped vertices")
            return

        # For k-ring expansion: selected handle seeds define vectors. Expanded neighbors inherit nearest seed direction.
        vectors: List[List[float]] = []
        seed_arr = np.asarray(self.reshaped_seed, dtype=np.int64)
        for vid in reshaped_ids:
            if int(vid) in self.directions:
                vectors.append(normalize_vector(self.directions[int(vid)]))
            elif len(seed_arr) > 0:
                j = int(np.argmin(np.linalg.norm(self.V[seed_arr] - self.V[int(vid)], axis=1)))
                vectors.append(normalize_vector(self.directions.get(int(seed_arr[j]), DEFAULT_DIRECTION)))
            else:
                vectors.append(DEFAULT_DIRECTION.copy())

        record = deformation_record(
            label=self.args.label,
            constrained_ids=constrained_ids,
            reshaped_ids=reshaped_ids,
            reshaped_vectors=vectors,
            description=self.args.description,
            coupled=self.args.coupled,
            normalize_vector_flag=self.args.normalize_vector,
            distribution=self.args.distribution,
            weight=self.args.weight,
        )
        write_deformation_yaml(self.args.out, record, wrap_deformations=not self.args.single_record)

        msg = f"saved {Path(self.args.out).resolve()}"
        self._update_status(msg)
        print("\nSaved deformation annotation:")
        print(Path(self.args.out).resolve())
        print(f"  constrained_ids: {len(record['constrained_ids'])}")
        print(f"  reshaped_ids   : {len(record['reshaped_ids'])}")
        print("  range          : one [0.0, 0.0] placeholder per handle; edit manually")
        print("  coupled        :", record["coupled"])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open3D GUI FPSA annotation helper with mesh overlay, vertex ids, and axis directions")
    parser.add_argument("--mesh", required=True, help="Input OBJ mesh path. Vertex order is preserved with trimesh.")
    parser.add_argument("--label", required=True, help="Deformation label name, e.g. bracket_x_stretch")
    parser.add_argument("--out", required=True, help="Output YAML path")
    parser.add_argument("--description", default="Manually annotated FPSA deformation label")

    parser.add_argument("--constraint-k-ring", type=int, default=0, help="Expand picked constrained seed vertices by k-ring when saving")
    parser.add_argument("--reshaped-k-ring", type=int, default=0, help="Expand picked reshaped seed vertices by k-ring when saving")

    parser.add_argument("--mesh-alpha", type=float, default=0.35, help="Transparent mesh alpha in the GUI")
    parser.add_argument("--point-size", type=float, default=9.0, help="Vertex point cloud display size")
    parser.add_argument("--line-width", type=float, default=5.0, help="Direction/axis line width")
    parser.add_argument("--arrow-length", type=float, default=0.18, help="Handle direction line length as fraction of mesh bbox diagonal")
    parser.add_argument("--axis-selector-length", type=float, default=0.28, help=argparse.SUPPRESS)  # deprecated no-op
    parser.add_argument("--pick-radius", type=float, default=0.025, help="Nearest-vertex/handle pick radius as fraction of mesh bbox diagonal")
    parser.add_argument("--show-ids", action=argparse.BooleanOptionalAction, default=True, help="Show vertex ids next to picked vertices")
    parser.add_argument("--id-label-offset", type=float, default=0.012, help="3D label z offset as fraction of mesh bbox diagonal")
    parser.add_argument("--id-label-scale", type=float, default=1.0, help="3D label scale, if supported by this Open3D build")

    parser.add_argument("--coupled", action=argparse.BooleanOptionalAction, default=False, help="Default false so output range is naturally per-handle")
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distribution", default="uniform")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--single-record", action="store_true", help="Write a single deformation dict instead of {'deformations': [dict]}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if gui is None:
        raise RuntimeError(
            "This helper requires Open3D's modern GUI module. "
            "Try upgrading Open3D or use FPSA_annotation_helper_pcd_picker.py. "
            f"Import error: {_GUI_IMPORT_ERROR}"
        )

    app = gui.Application.instance
    app.initialize()
    FPSAAnnotationApp(args)
    app.run()


if __name__ == "__main__":
    main()
