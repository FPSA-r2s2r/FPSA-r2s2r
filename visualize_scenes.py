#!/usr/bin/env python3
"""Visualize a collected FPSA run as a compact, publication-style data atlas.

The input is a collection run containing ``episodes/episode_*``.  Rather than
decoding thousands of videos, the script scans lightweight trajectory files,
chooses diverse episodes with seeded maximin sampling, and decodes only a few
phase-aligned frames from those episodes.

Example:
    python visualize_scenes.py --root /path/to/collection/run --seed 7 --pdf
"""

from __future__ import annotations

import argparse
import json
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

try:
    import cv2
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise SystemExit("visualize_scenes.py requires opencv-python") from exc


COLORS = ("#2563EB", "#F97316", "#10B981", "#A855F7", "#E11D48", "#0891B2")


@dataclass
class Episode:
    path: Path
    episode_id: int
    success: bool
    steps: int
    fps: float
    feature: np.ndarray


def _episode_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def _episode_root(root: Path) -> Path:
    nested = root / "episodes"
    return nested if nested.is_dir() else root


def _load_meta(path: Path) -> dict[str, Any]:
    meta_path = path / "episode_meta.json"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data if isinstance(data, dict) else {}


def _trajectory_feature(lowdim: Any, success: bool) -> tuple[np.ndarray, int]:
    position = np.asarray(lowdim["robot0_eef_pos"], dtype=float)
    action = np.asarray(lowdim["action"], dtype=float)
    gripper = np.asarray(lowdim["robot0_gripper_qpos"], dtype=float)
    steps = len(position)
    if steps == 0:
        raise ValueError("empty trajectory")
    delta = np.diff(position, axis=0)
    path_length = float(np.linalg.norm(delta, axis=1).sum()) if len(delta) else 0.0
    displacement = float(np.linalg.norm(position[-1] - position[0]))
    extent = np.ptp(position, axis=0)
    action_norm = np.linalg.norm(action.reshape(len(action), -1), axis=1) if len(action) else np.zeros(1)
    grip_flat = gripper.reshape(len(gripper), -1)
    feature = np.asarray(
        [
            np.log1p(steps), path_length, displacement,
            *extent[:3], float(np.mean(action_norm)), float(np.max(action_norm)),
            float(np.ptp(grip_flat)), float(success),
        ],
        dtype=float,
    )
    return feature, steps


def scan_episodes(root: Path, camera: str, scan_limit: Optional[int]) -> tuple[list[Episode], int]:
    paths = sorted((path for path in _episode_root(root).glob("episode_*") if path.is_dir()), key=_episode_id)
    total = len(paths)
    if scan_limit and len(paths) > scan_limit:
        indices = np.linspace(0, len(paths) - 1, scan_limit, dtype=int)
        paths = [paths[index] for index in indices]

    episodes = []
    for path in paths:
        lowdim_path = path / "lowdim.npz"
        has_agent = (path / "agentview.mp4").is_file()
        has_wrist = (path / "eye_in_hand.mp4").is_file()
        camera_ok = has_agent if camera == "agentview" else has_wrist if camera == "eye_in_hand" else (has_agent or has_wrist)
        if not lowdim_path.is_file() or not camera_ok:
            continue
        try:
            meta = _load_meta(path)
            with np.load(lowdim_path, allow_pickle=False) as lowdim:
                feature, steps = _trajectory_feature(lowdim, bool(meta.get("success", False)))
            episodes.append(
                Episode(
                    path=path,
                    episode_id=_episode_id(path),
                    success=bool(meta.get("success", False)),
                    steps=steps,
                    fps=float(meta.get("fps", 20.0)),
                    feature=feature,
                )
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[skip] {path.name}: {exc}")
    return episodes, total


def representative_episodes(
    episodes: list[Episode], count: int, seed: int
) -> list[Episode]:
    if len(episodes) <= count:
        return episodes
    matrix = np.stack([episode.feature for episode in episodes])
    median = np.nanmedian(matrix, axis=0)
    scale = np.nanpercentile(matrix, 90, axis=0) - np.nanpercentile(matrix, 10, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (matrix - median) / scale
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(episodes)))]
    distance = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    while len(selected) < count:
        distance[selected] = -1.0
        available = np.flatnonzero(distance >= 0.0)
        ranked = available[np.argsort(distance[available])[::-1]]
        pool_size = min(len(ranked), max(3, int(np.ceil(0.15 * len(ranked)))))
        index = int(rng.choice(ranked[:pool_size]))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(normalized - normalized[index], axis=1))
    return [episodes[index] for index in selected]


def _read_video_frames(path: Path, phases: np.ndarray) -> list[Optional[np.ndarray]]:
    if not path.is_file():
        return [None] * len(phases)
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[Optional[np.ndarray]] = []
    for phase in phases:
        index = int(round(float(phase) * max(count - 1, 0)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None)
    capture.release()
    return frames


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _camera_tiles(episode: Episode, phases: np.ndarray, camera: str) -> list[Optional[np.ndarray]]:
    agent = _read_video_frames(episode.path / "agentview.mp4", phases)
    wrist = _read_video_frames(episode.path / "eye_in_hand.mp4", phases)
    if camera == "agentview":
        return agent
    if camera == "eye_in_hand":
        return wrist
    output = []
    for main, inset in zip(agent, wrist):
        main = main if main is not None else inset
        if main is None:
            output.append(None)
            continue
        canvas = main.copy()
        if inset is not None:
            inset_height = max(32, int(canvas.shape[0] * 0.35))
            small = _resize_to_height(inset, inset_height)
            margin = max(5, canvas.shape[0] // 50)
            max_width = max(1, canvas.shape[1] - 2 * margin)
            if small.shape[1] > max_width:
                small = cv2.resize(small, (max_width, max(1, int(small.shape[0] * max_width / small.shape[1]))))
            y0, x0 = margin, canvas.shape[1] - small.shape[1] - margin
            canvas[max(0, y0 - 2): y0 + small.shape[0] + 2, max(0, x0 - 2): x0 + small.shape[1] + 2] = 255
            canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        output.append(canvas)
    return output


def _load_lowdim(episode: Episode) -> dict[str, np.ndarray]:
    with np.load(episode.path / "lowdim.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _normalized_curve(values: np.ndarray, points: int = 200) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(points, values[0])
    source = np.linspace(0.0, 1.0, len(values))
    return np.interp(np.linspace(0.0, 1.0, points), source, values)


def _style_metric_axis(ax: Any, ylabel: str) -> None:
    ax.set_facecolor("#F8FAFC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.tick_params(labelsize=7.5, colors="#64748B")
    ax.set_xlabel("normalized episode time", fontsize=8, color="#475569")
    ax.set_ylabel(ylabel, fontsize=8, color="#475569")


def create_figure(
    selected: list[Episode],
    output: Path,
    frame_count: int,
    camera: str,
    dpi: int,
    save_pdf: bool,
    seed: int,
) -> None:
    phases = np.linspace(0.0, 1.0, frame_count)
    nrows = len(selected)
    fig = plt.figure(
        figsize=(2.65 * frame_count, 1.82 * nrows + 3.8), facecolor="#FFFFFF"
    )
    grid = fig.add_gridspec(
        nrows + 1,
        frame_count,
        height_ratios=[1.0] * nrows + [1.58],
        left=0.048,
        right=0.995,
        bottom=0.055,
        top=0.915,
        wspace=0.012,
        hspace=0.065,
    )

    trajectories = []
    gripper_curves = []
    action_curves = []
    for row, episode in enumerate(selected):
        tiles = _camera_tiles(episode, phases, camera)
        lowdim = _load_lowdim(episode)
        position = np.asarray(lowdim["robot0_eef_pos"], dtype=float)
        trajectories.append(position)
        gripper = np.asarray(lowdim["robot0_gripper_qpos"], dtype=float).reshape(len(position), -1)
        opening = np.sum(gripper, axis=1) if gripper.shape[1] > 1 else gripper[:, 0]
        action = np.asarray(lowdim["action"], dtype=float).reshape(len(position), -1)
        action_norm = np.linalg.norm(action, axis=1)
        gripper_curves.append(_normalized_curve(opening * 1000.0))
        action_curves.append(_normalized_curve(action_norm))

        color = COLORS[row % len(COLORS)]
        for column, (phase, tile) in enumerate(zip(phases, tiles)):
            ax = fig.add_subplot(grid[row, column])
            if tile is None:
                ax.set_facecolor("#E2E8F0")
                ax.text(0.5, 0.5, "frame unavailable", ha="center", va="center", fontsize=8, color="#64748B")
            else:
                ax.imshow(tile)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.35 if column == 0 else 0.4)
                spine.set_color(color if column == 0 else "#E2E8F0")
            if row == 0:
                ax.set_title(f"{phase * 100:.0f}%", fontsize=9, color="#475569", pad=5)
            if column == 0:
                ax.text(-0.025, 0.5, f"EP {episode.episode_id:04d}", transform=ax.transAxes, ha="right", va="center", fontsize=8.5, fontweight="bold", color=color)

    bottom = grid[nrows, :].subgridspec(
        1, 3, wspace=0.22, width_ratios=(1.12, 1.0, 1.0)
    )
    trajectory_ax = fig.add_subplot(bottom[0], projection="3d")
    for index, trajectory in enumerate(trajectories):
        xyz = trajectory * 100.0
        color = COLORS[index % len(COLORS)]
        trajectory_ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=1.65, alpha=0.92)
        trajectory_ax.scatter(*xyz[0], color=color, marker="o", s=15)
        trajectory_ax.scatter(*xyz[-1], color=color, marker="X", s=24)
    trajectory_ax.set_title(
        "EEF trajectories",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        y=0.94,
        pad=0,
    )
    trajectory_ax.set_xlabel("x [cm]", fontsize=7.5)
    trajectory_ax.set_ylabel("y [cm]", fontsize=7.5)
    trajectory_ax.set_zlabel("z [cm]", fontsize=7.5)
    trajectory_ax.tick_params(labelsize=6.5)
    trajectory_ax.grid(True, color="#E2E8F0", linewidth=0.5)
    trajectory_ax.view_init(elev=25, azim=-55)

    x = np.linspace(0.0, 1.0, 200)
    gripper_ax = fig.add_subplot(bottom[1])
    action_ax = fig.add_subplot(bottom[2])
    for index, (gripper, action) in enumerate(zip(gripper_curves, action_curves)):
        color = COLORS[index % len(COLORS)]
        gripper_ax.plot(x, gripper, color=color, alpha=0.50, linewidth=1.0)
        action_ax.plot(x, action, color=color, alpha=0.50, linewidth=1.0)
    for ax, curves, title, ylabel in (
        (gripper_ax, gripper_curves, "Gripper", "opening [mm]"),
        (action_ax, action_curves, "Action magnitude", "L2 norm"),
    ):
        stack = np.stack(curves)
        median = np.median(stack, axis=0)
        lower, upper = np.percentile(stack, [10, 90], axis=0)
        ax.fill_between(x, lower, upper, color="#94A3B8", alpha=0.18, linewidth=0)
        ax.plot(x, median, color="#0F172A", linewidth=2.1)
        ax.set_title(
            title,
            fontsize=10.5,
            fontweight="bold",
            color="#0F172A",
            y=0.94,
            pad=0,
        )
        _style_metric_axis(ax, ylabel)

    fig.text(0.048, 0.970, "Generated scenes", fontsize=19, fontweight="bold", color="#0F172A", ha="left", va="top")
    fig.text(0.995, 0.963, f"seed {seed}", fontsize=8.5, color="#94A3B8", ha="right", va="top")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if save_pdf:
        fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Collection run directory, with or without the episodes/ suffix")
    parser.add_argument("--output", type=Path, help="Output image (default: <root>/scene_atlas.png)")
    parser.add_argument("--episodes", type=int, default=4, help="Number of representative episodes to decode")
    parser.add_argument("--frames", type=int, default=5, help="Phase-aligned frames per selected episode")
    parser.add_argument("--camera", choices=("both", "agentview", "eye_in_hand"), default="both", help="Camera layout; both puts wrist view in an inset")
    parser.add_argument("--scan-limit", type=int, help="Optionally cap lightweight trajectory scanning for very large runs")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0, help="Reproducible representative-episode selection seed")
    parser.add_argument("--pdf", action="store_true", help="Also save a vector-text PDF companion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Collection root does not exist: {root}")
    if args.episodes < 1 or args.frames < 2:
        raise SystemExit("--episodes must be >= 1 and --frames must be >= 2")
    episodes, _total = scan_episodes(root, args.camera, args.scan_limit)
    if not episodes:
        raise SystemExit(f"No complete episodes with the requested camera were found below {_episode_root(root)}")
    chosen = representative_episodes(episodes, args.episodes, args.seed)
    output = args.output.expanduser() if args.output else root / "scene_atlas.png"
    create_figure(
        chosen, output, args.frames, args.camera, args.dpi, args.pdf, args.seed
    )
    print(f"Saved {output.resolve()}")
    if args.pdf:
        print(f"Saved {output.with_suffix('.pdf').resolve()}")
    print("Selected episodes: " + ", ".join(f"{episode.episode_id:04d}" for episode in chosen))


if __name__ == "__main__":
    main()
