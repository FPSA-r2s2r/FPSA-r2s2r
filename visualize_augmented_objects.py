#!/usr/bin/env python3
"""Create a compact, publication-style overview of FPSA augmented meshes.

The script does not render every OBJ.  It groups samples by deformation label
and uses seeded maximin sampling in normalized deformation-parameter space.

Example:
    python visualize_augmented_objects.py \
        --root ./data/objects/assembly/tool/fpsa_aug_outputs \
        --samples-per-label 5 --seed 7 --pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fpsa")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fpsa-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PALETTE = ("#3B82F6", "#F97316", "#10B981", "#A855F7", "#E11D48", "#06B6D4")


@dataclass
class Sample:
    name: str
    label: str
    sample_id: int
    obj_path: Path
    meta_path: Optional[Path]
    feature: np.ndarray
    frame: Optional[np.ndarray] = None


def _natural_key(path: Path) -> list[Any]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def _read_yaml(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data if isinstance(data, dict) else {}


def _resolve_record_path(root: Path, raw: str, sample_name: str, suffix: str) -> Optional[Path]:
    candidates = []
    if raw:
        value = Path(raw).expanduser()
        candidates.extend((value, root / value, root.parent / value))
    candidates.append(root / sample_name / f"{sample_name}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        return []
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if str(row.get("ok", "true")).lower() in {"1", "true", "yes"}]


def _fallback_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for obj_path in sorted(root.rglob("*.obj"), key=_natural_key):
        if obj_path.stem.endswith("_coacd"):
            continue
        name = obj_path.stem
        rows.append(
            {
                "sample_name": name,
                "sample_id": re.search(r"(\d+)$", name).group(1) if re.search(r"(\d+)$", name) else "0",
                "label": "unknown",
                "obj_path": str(obj_path),
                "meta_path": str(obj_path.with_name(f"{name}_sample.yaml")),
            }
        )
    return rows


def _sample_feature(meta: dict[str, Any], fallback_id: int) -> np.ndarray:
    indices = np.asarray(meta.get("grid_indices", []), dtype=float).reshape(-1)
    counts = np.asarray(meta.get("grid_counts", []), dtype=float).reshape(-1)
    if indices.size and indices.size == counts.size:
        denom = np.maximum(counts - 1.0, 1.0)
        return 2.0 * indices / denom - 1.0

    values = []
    for stage in meta.get("stages", []):
        magnitudes = np.asarray(stage.get("magnitudes", []), dtype=float).reshape(-1)
        if magnitudes.size:
            values.append(float(np.mean(magnitudes)))
    if values:
        return np.asarray(values, dtype=float)
    return np.asarray([float(fallback_id)], dtype=float)


def _load_task_frame(sample_dir: Path, name: str) -> Optional[np.ndarray]:
    candidates = [
        sample_dir / f"{name}_grasp.yaml",
        sample_dir / f"{name}_wrench_to_tcp.yaml",
    ]
    for path in candidates:
        data = _read_yaml(path)
        for key in ("T_mesh_hand_tcp", "T_mesh_hand", "wrench_to_tcp_T"):
            value = np.asarray(data.get(key, []), dtype=float)
            if value.shape == (4, 4):
                return value
    return None


def discover_samples(root: Path) -> list[Sample]:
    rows = _manifest_rows(root) or _fallback_rows(root)
    samples = []
    for row in rows:
        name = str(row.get("sample_name") or Path(row.get("obj_path", "sample")).stem)
        obj_path = _resolve_record_path(root, row.get("obj_path", ""), name, ".obj")
        if obj_path is None or obj_path.stem.endswith("_coacd"):
            continue
        meta_path = _resolve_record_path(root, row.get("meta_path", ""), name, "_sample.yaml")
        meta = _read_yaml(meta_path)
        sample_id = int(meta.get("sample_id", row.get("sample_id") or 0))
        label = str(meta.get("label", row.get("label") or "unknown"))
        samples.append(
            Sample(
                name=name,
                label=label,
                sample_id=sample_id,
                obj_path=obj_path,
                meta_path=meta_path,
                feature=_sample_feature(meta, sample_id),
                frame=_load_task_frame(obj_path.parent, name),
            )
        )
    return samples


def _feature_matrix(samples: list[Sample]) -> np.ndarray:
    width = max(item.feature.size for item in samples)
    matrix = np.zeros((len(samples), width), dtype=float)
    for row, sample in enumerate(samples):
        matrix[row, : sample.feature.size] = sample.feature
    lo, hi = np.nanmin(matrix, axis=0), np.nanmax(matrix, axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    return (matrix - lo) / scale


def diverse_subset(
    samples: list[Sample], count: int, rng: np.random.Generator
) -> list[Sample]:
    """Seeded maximin selection with a small randomized farthest-candidate pool."""
    if len(samples) <= count:
        return sorted(samples, key=lambda item: item.sample_id)
    features = _feature_matrix(samples)
    selected = [int(rng.integers(len(samples)))]
    distances = np.linalg.norm(features - features[selected[0]], axis=1)
    while len(selected) < count:
        distances[selected] = -1.0
        available = np.flatnonzero(distances >= 0.0)
        ranked = available[np.argsort(distances[available])[::-1]]
        pool_size = min(len(ranked), max(3, int(np.ceil(0.15 * len(ranked)))))
        next_index = int(rng.choice(ranked[:pool_size]))
        selected.append(next_index)
        distances = np.minimum(distances, np.linalg.norm(features - features[next_index], axis=1))
    selected.sort(key=lambda index: tuple(features[index]))
    return [samples[index] for index in selected]


def _to_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No triangular geometry in {path}")
    return trimesh.util.concatenate(meshes)


def _plot_frame(ax: Any, transform: np.ndarray, centre: np.ndarray, scale: float) -> None:
    origin = (transform[:3, 3] - centre) / scale
    length = 0.34
    for axis, color in zip(range(3), ("#EF4444", "#22C55E", "#3B82F6")):
        direction = transform[:3, axis]
        ax.quiver(*origin, *(direction * length), color=color, linewidth=1.35, arrow_length_ratio=0.22)


def _render_mesh(ax: Any, sample: Sample, color: str, max_faces: int, elev: float, azim: float) -> None:
    mesh = _to_mesh(sample.obj_path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    centre = (lower + upper) / 2.0
    scale = max(float(np.max(upper - lower)) / 2.0, 1e-9)
    vertices = (vertices - centre) / scale

    if len(faces) > max_faces:
        keep = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
        faces = faces[keep]
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.asarray([0.35, -0.45, 0.82])
    intensity = np.clip(0.56 + 0.44 * np.abs(normals @ light), 0.0, 1.0)
    base = np.asarray(matplotlib.colors.to_rgb(color))
    face_colors = np.clip(base[None, :] * intensity[:, None] + 0.10 * (1.0 - intensity[:, None]), 0, 1)
    collection = Poly3DCollection(triangles, facecolors=face_colors, edgecolors="none", rasterized=True)
    ax.add_collection3d(collection)

    if sample.frame is not None:
        _plot_frame(ax, sample.frame, centre, scale)
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), zlim=(-1.08, 1.08))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _pretty_label(label: str) -> str:
    return label.replace("_", " ").replace("streching", "stretching").title()


def create_figure(
    samples: list[Sample],
    output: Path,
    samples_per_label: int,
    labels: Optional[set[str]],
    max_faces: int,
    elev: float,
    azim: float,
    dpi: int,
    save_pdf: bool,
    seed: int,
) -> list[Sample]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        if labels is None or sample.label in labels:
            grouped.setdefault(sample.label, []).append(sample)
    if not grouped:
        raise ValueError("No samples match the requested labels")

    rng = np.random.default_rng(seed)
    selected = {
        label: diverse_subset(items, samples_per_label, rng)
        for label, items in grouped.items()
    }
    columns = max(len(items) for items in selected.values())
    rows = len(selected)
    fig = plt.figure(
        figsize=(2.45 * columns, 2.15 * rows + 0.75), facecolor="#F8FAFC"
    )
    grid = fig.add_gridspec(
        rows,
        columns,
        left=0.055,
        right=0.995,
        bottom=0.015,
        top=0.895,
        wspace=-0.08,
        hspace=-0.04,
    )

    flattened = []
    for row, (label, items) in enumerate(selected.items()):
        color = PALETTE[row % len(PALETTE)]
        for column in range(columns):
            ax = fig.add_subplot(grid[row, column], projection="3d")
            ax.set_facecolor("#F8FAFC")
            if column >= len(items):
                ax.set_axis_off()
                continue
            sample = items[column]
            flattened.append(sample)
            _render_mesh(ax, sample, color, max_faces, elev, azim)
            ax.set_title(f"#{sample.sample_id:04d}", fontsize=8.5, color="#64748B", pad=-2)
            if column == 0:
                ax.text2D(-0.09, 0.50, _pretty_label(label), transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=10.0, fontweight="bold", color=color)

    fig.text(0.055, 0.965, "FPSA augmentations", fontsize=19, fontweight="bold", color="#0F172A", ha="left", va="top")
    fig.text(0.995, 0.958, f"seed {seed}", fontsize=8.5, color="#94A3B8", ha="right", va="top")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if save_pdf:
        fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="FPSA output root containing manifest.csv and sample folders")
    parser.add_argument("--output", type=Path, help="Output image (default: <root>/augmented_objects_overview.png)")
    parser.add_argument("--samples-per-label", type=int, default=5, help="Representative meshes per deformation label")
    parser.add_argument("--labels", help="Optional comma-separated label filter")
    parser.add_argument("--max-faces", type=int, default=12000, help="Maximum rendered triangles per panel")
    parser.add_argument("--elev", type=float, default=22.0, help="Camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=-58.0, help="Camera azimuth in degrees")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0, help="Reproducible representative-sample selection seed")
    parser.add_argument("--pdf", action="store_true", help="Also save a vector-text PDF companion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"FPSA output root does not exist: {root}")
    if args.samples_per_label < 1:
        raise SystemExit("--samples-per-label must be at least 1")
    samples = discover_samples(root)
    if not samples:
        raise SystemExit(f"No augmented OBJ files found below {root}")
    labels = {item.strip() for item in args.labels.split(",") if item.strip()} if args.labels else None
    output = args.output.expanduser() if args.output else root / "augmented_objects_overview.png"
    chosen = create_figure(
        samples, output, args.samples_per_label, labels, args.max_faces,
        args.elev, args.azim, args.dpi, args.pdf, args.seed,
    )
    print(f"Saved {output.resolve()}")
    if args.pdf:
        print(f"Saved {output.with_suffix('.pdf').resolve()}")
    print("Selected samples:")
    for sample in chosen:
        print(f"  {sample.label:32s} #{sample.sample_id:04d}  {sample.obj_path}")


if __name__ == "__main__":
    main()
