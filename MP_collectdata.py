"""Collect any simulation task using one ``Config_*.yaml`` file.

Example:
    python MP_collectdata.py sim_world/Config_Assembly.yaml --num-episodes 3000
    python MP_collectdata.py sim_world/Config_Assembly.yaml --num-episodes 3000 --resume
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pybullet as pb
import pybullet_data as pd
from multiprocessing.connection import Connection
from pybullet_utils import bullet_client
from tqdm import tqdm

from sim_world.src.episode_writer import EpisodeWriter
from sim_world.src.task_config import load_task_config, parse_number
from sim_world.src.task_runtime import build_simulation


_RESET = 1
_COLLECT = 2
_CLOSE = 3


@dataclass(frozen=True)
class CollectorConfig:
    task_config: Dict[str, Any]
    num_episodes: int
    num_processes: int
    base_seed: int
    base_dir: Path
    time_step: float
    fps: int
    max_steps: int
    record_every_n_sim_steps: int
    use_gui: bool
    overwrite: bool
    restart_every: int
    resume: bool


_EPISODE_DIR_RE = re.compile(r"^episode_(\d+)$")


def find_latest_collection_episodes_dir(output_root: Path) -> Path:
    """Return the episodes directory from the most recently modified run."""
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Cannot resume: output_root does not exist: {output_root}"
        )

    candidates = [
        run_dir / "episodes"
        for run_dir in output_root.iterdir()
        if run_dir.is_dir() and (run_dir / "episodes").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot resume: no collection folder containing episodes/ under {output_root}"
        )

    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
    )


def validate_episode_dir(
    episode_dir: Path,
    observation_config: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Check that an episode reached EpisodeWriter.close() successfully."""
    meta_path = episode_dir / "episode_meta.json"
    lowdim_path = episode_dir / "lowdim.npz"

    if not meta_path.is_file():
        return False, "missing episode_meta.json"
    if not lowdim_path.is_file() or lowdim_path.stat().st_size == 0:
        return False, "missing or empty lowdim.npz"

    try:
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
        num_steps = int(meta["num_steps"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return False, f"invalid episode_meta.json: {exc}"

    if num_steps <= 0:
        return False, f"invalid num_steps={num_steps}"

    required_lowdim = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "action",
        "timestamp",
    )
    try:
        with np.load(lowdim_path, allow_pickle=False) as lowdim:
            for key in required_lowdim:
                if key not in lowdim:
                    return False, f"lowdim.npz missing {key}"
                if len(lowdim[key]) != num_steps:
                    return False, (
                        f"lowdim.npz {key} has {len(lowdim[key])} steps, "
                        f"expected {num_steps}"
                    )
    except (OSError, ValueError, TypeError, EOFError) as exc:
        return False, f"invalid lowdim.npz: {exc}"

    required_videos = []
    if observation_config.get("use_agent_cam", True):
        required_videos.append("agentview.mp4")
    if observation_config.get("use_eye_in_hand", False):
        required_videos.append("eye_in_hand.mp4")
    for filename in required_videos:
        video_path = episode_dir / filename
        if not video_path.is_file() or video_path.stat().st_size == 0:
            return False, f"missing or empty {filename}"

    return True, "complete"


def clean_incomplete_resume_episodes(cfg: CollectorConfig) -> List[int]:
    """Delete incomplete episode directories and return valid episode IDs."""
    observation_config = dict(cfg.task_config["collect_observation"])
    valid_ids: List[int] = []
    removed: List[Tuple[Path, str]] = []

    for episode_dir in sorted(cfg.base_dir.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        match = _EPISODE_DIR_RE.fullmatch(episode_dir.name)
        if match is None:
            continue

        episode_id = int(match.group(1))
        valid, reason = validate_episode_dir(episode_dir, observation_config)
        if valid:
            if episode_id < cfg.num_episodes:
                valid_ids.append(episode_id)
            continue

        shutil.rmtree(episode_dir)
        removed.append((episode_dir, reason))

    print(
        f"[resume] found {len(valid_ids)} complete episode(s); "
        f"removed {len(removed)} incomplete episode directory/directories"
    )
    for episode_dir, reason in removed:
        print(f"[resume] removed {episode_dir.name}: {reason}")
    return valid_ids


def setup_world(client: bullet_client.BulletClient, time_step: float) -> None:
    client.setAdditionalSearchPath(pd.getDataPath())
    client.setTimeStep(time_step)
    client.setGravity(0.0, 0.0, -9.8)
    client.setPhysicsEngineParameter(solverResidualThreshold=0)


def collect_one_episode(
    client: bullet_client.BulletClient,
    sim: Any,
    task_config: Mapping[str, Any],
    episode_dir: Path,
    meta_seed: int,
    fps: int,
    max_steps: int,
    record_every_n_sim_steps: int,
) -> bool:
    observation_config = dict(task_config["collect_observation"])
    writer = EpisodeWriter(
        episode_dir,
        fps=fps,
        if_agent_view=observation_config.get("use_agent_cam", True),
        if_eye_in_hand=observation_config.get("use_eye_in_hand", False),
        extra_meta={
            "meta_seed": meta_seed,
            "simulation": task_config["simulation"],
        },
    )

    sim.done = False
    sim_step = 0
    record_idx = 0
    try:
        while not sim.done and sim_step < max_steps:
            should_record = sim_step % record_every_n_sim_steps == 0
            if should_record:
                observation = sim.collect_observation(**observation_config)

            sim.step()

            if should_record:
                action = sim.collect_action()
                writer.add_step(observation, action, record_idx / float(fps))
                record_idx += 1

            client.stepSimulation()
            sim_step += 1

        success = bool(sim.is_success())
        writer.close(success=success)
        return success
    except Exception:
        try:
            writer.close(success=False)
        except Exception:
            pass
        raise


def prepare_episode_dir(episode_dir: Path, overwrite: bool) -> Tuple[bool, str]:
    if episode_dir.exists():
        if not overwrite:
            return False, "exists"
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=False)
    return True, "new"


def collect_episode_in_worker(
    rank: int,
    client: bullet_client.BulletClient,
    cfg: CollectorConfig,
    episode_id: int,
) -> Dict[str, Any]:
    seed = cfg.base_seed + episode_id + 1
    episode_dir = cfg.base_dir / f"episode_{episode_id:06d}"
    should_collect, reason = prepare_episode_dir(episode_dir, cfg.overwrite)
    if not should_collect:
        return {
            "rank": rank,
            "episode_id": episode_id,
            "seed": seed,
            "status": "skipped",
            "reason": reason,
            "success": None,
        }

    client.resetSimulation()
    setup_world(client, cfg.time_step)
    try:
        client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 0)
    except Exception:
        pass

    np.random.seed(seed)
    sim = build_simulation(
        cfg.task_config,
        client,
        client._client,
        seed=seed,
        control_dt=cfg.time_step,
    )
    sim.make_scene(**cfg.task_config["make_scene"])
    sim.enable_high_quality_rendering()

    try:
        client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 1)
    except Exception:
        pass

    success = collect_one_episode(
        client=client,
        sim=sim,
        task_config=cfg.task_config,
        episode_dir=episode_dir,
        meta_seed=seed,
        fps=cfg.fps,
        max_steps=cfg.max_steps,
        record_every_n_sim_steps=cfg.record_every_n_sim_steps,
    )
    return {
        "rank": rank,
        "episode_id": episode_id,
        "seed": seed,
        "status": "ok",
        "success": success,
    }


def collector_worker(rank: int, child_pipe: Connection, cfg: CollectorConfig) -> None:
    log_dir = cfg.base_dir.parent / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / f"worker_{rank}.log", "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"\n===== worker {rank} started, pid={os.getpid()} =====", flush=True)
    client: Optional[bullet_client.BulletClient] = None
    episodes_since_restart = 0

    while True:
        try:
            message, payload = child_pipe.recv()
        except (EOFError, KeyboardInterrupt):
            break

        if message == _RESET:
            try:
                mode = pb.GUI if cfg.use_gui and rank == 0 else pb.DIRECT
                client = bullet_client.BulletClient(connection_mode=mode)
                setup_world(client, cfg.time_step)
                episodes_since_restart = 0
                child_pipe.send({"rank": rank, "status": "reset_ok"})
            except Exception:
                child_pipe.send({
                    "rank": rank,
                    "status": "reset_error",
                    "traceback": traceback.format_exc(),
                })
            continue

        if message == _COLLECT:
            assert client is not None, "Worker must receive _RESET before _COLLECT."
            episode_id = int(payload["episode_id"])
            try:
                result = collect_episode_in_worker(rank, client, cfg, episode_id)
            except Exception:
                try:
                    client.resetSimulation()
                    setup_world(client, cfg.time_step)
                except Exception:
                    pass
                result = {
                    "rank": rank,
                    "episode_id": episode_id,
                    "seed": cfg.base_seed + episode_id + 1,
                    "status": "error",
                    "success": False,
                    "traceback": traceback.format_exc(),
                }

            if result.get("status") != "skipped":
                episodes_since_restart += 1
            result["episodes_since_restart"] = episodes_since_restart
            result["restart_worker"] = (
                cfg.restart_every > 0 and episodes_since_restart >= cfg.restart_every
            )
            child_pipe.send(result)
            continue

        if message == _CLOSE:
            try:
                if client is not None:
                    client.disconnect()
            except Exception:
                pass
            child_pipe.send({"rank": rank, "status": "close_ok"})
            break

    child_pipe.close()


def start_one_worker(rank: int, cfg: CollectorConfig) -> Tuple[mp.Process, Connection]:
    parent_pipe, child_pipe = mp.Pipe()
    process = mp.Process(target=collector_worker, args=(rank, child_pipe, cfg), daemon=False)
    process.start()
    return process, parent_pipe


def reset_one_worker(rank: int, pipe: Connection) -> None:
    pipe.send((_RESET, None))
    message = pipe.recv()
    if message.get("status") != "reset_ok":
        raise RuntimeError(f"Worker {rank} reset failed: {message}")


def close_one_worker(process: mp.Process, pipe: Connection) -> None:
    try:
        pipe.send((_CLOSE, None))
    except Exception:
        pass
    try:
        if pipe.poll(2.0):
            pipe.recv()
    except Exception:
        pass
    try:
        pipe.close()
    except Exception:
        pass
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)


def restart_one_worker(
    rank: int,
    cfg: CollectorConfig,
    processes: List[mp.Process],
    parent_pipes: List[Connection],
) -> None:
    close_one_worker(processes[rank], parent_pipes[rank])
    processes[rank], parent_pipes[rank] = start_one_worker(rank, cfg)
    reset_one_worker(rank, parent_pipes[rank])


def start_workers(cfg: CollectorConfig) -> Tuple[List[mp.Process], List[Connection]]:
    processes: List[mp.Process] = []
    parent_pipes: List[Connection] = []
    for rank in range(cfg.num_processes):
        process, pipe = start_one_worker(rank, cfg)
        processes.append(process)
        parent_pipes.append(pipe)
    return processes, parent_pipes


def run_parent_scheduler(cfg: CollectorConfig) -> List[Dict[str, Any]]:
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = set(clean_incomplete_resume_episodes(cfg)) if cfg.resume else set()
    pending_episode_ids = [
        episode_id
        for episode_id in range(cfg.num_episodes)
        if episode_id not in completed_ids
    ]
    results: List[Dict[str, Any]] = [
        {
            "episode_id": episode_id,
            "seed": cfg.base_seed + episode_id + 1,
            "status": "skipped",
            "reason": "resume_complete",
            "success": None,
        }
        for episode_id in sorted(completed_ids)
    ]

    if not pending_episode_ids:
        print("[resume] collection is already complete")
        return results

    processes, parent_pipes = start_workers(cfg)
    try:
        for rank, pipe in enumerate(parent_pipes):
            reset_one_worker(rank, pipe)

        next_pending_index = 0
        active: Dict[int, int] = {}
        for rank, pipe in enumerate(parent_pipes):
            if next_pending_index >= len(pending_episode_ids):
                break
            episode_id = pending_episode_ids[next_pending_index]
            pipe.send((_COLLECT, {"episode_id": episode_id}))
            active[rank] = episode_id
            next_pending_index += 1

        with tqdm(
            total=cfg.num_episodes,
            initial=len(completed_ids),
            desc="Collecting episodes",
        ) as progress:
            while active:
                for rank, pipe in enumerate(parent_pipes):
                    if rank not in active or not pipe.poll(0.05):
                        continue
                    result = pipe.recv()
                    results.append(result)
                    active.pop(rank, None)
                    progress.update(1)

                    if result.get("status") == "error":
                        progress.write(
                            f"[worker {rank}] episode {result.get('episode_id')} failed"
                        )
                    if (
                        result.get("restart_worker")
                        and next_pending_index < len(pending_episode_ids)
                    ):
                        restart_one_worker(rank, cfg, processes, parent_pipes)
                        pipe = parent_pipes[rank]
                    if next_pending_index < len(pending_episode_ids):
                        episode_id = pending_episode_ids[next_pending_index]
                        pipe.send((_COLLECT, {"episode_id": episode_id}))
                        active[rank] = episode_id
                        next_pending_index += 1
    finally:
        for process, pipe in zip(processes, parent_pipes):
            close_one_worker(process, pipe)
    return results


def parse_args() -> CollectorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Task YAML, e.g. Config_Gear.yaml")
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--time-step", type=float)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--record-every-n-sim-steps", type=int)
    parser.add_argument("--use-gui", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--restart-every", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume the most recent collection under mp_collect.output_root, "
            "deleting incomplete episode directories first"
        ),
    )
    args = parser.parse_args()

    task_config = load_task_config(args.config)
    collector = task_config.get("mp_collect", {})
    if not isinstance(collector, Mapping):
        raise ValueError("Task config section 'mp_collect' must be a mapping")

    def value(name: str, fallback: Any) -> Any:
        cli_value = getattr(args, name)
        return collector.get(name, fallback) if cli_value is None else cli_value

    output_root = collector.get("output_root")
    if args.resume and args.base_dir is not None:
        parser.error("--resume cannot be combined with --base-dir")

    base_dir = args.base_dir
    if args.resume:
        if not output_root:
            raise ValueError("mp_collect.output_root is required with --resume")
        base_dir = find_latest_collection_episodes_dir(Path(output_root))
        print(f"[resume] using latest collection: {base_dir.parent}")
    elif base_dir is None:
        if not output_root:
            raise ValueError("mp_collect.output_root or --base-dir is required")
        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(output_root) / run_time / "episodes"

    config = CollectorConfig(
        task_config=task_config,
        num_episodes=int(value("num_episodes", 5)),
        num_processes=int(value("num_processes", max(1, min(8, os.cpu_count() or 1)))),
        base_seed=int(value("base_seed", 43)),
        base_dir=base_dir,
        time_step=parse_number(
            value("time_step", 1.0 / 120.0),
            field="mp_collect.time_step",
        ),
        fps=int(value("fps", 10)),
        max_steps=int(value("max_steps", 2000)),
        record_every_n_sim_steps=int(value("record_every_n_sim_steps", 12)),
        use_gui=bool(value("use_gui", False)),
        overwrite=bool(value("overwrite", False)),
        restart_every=int(value("restart_every", 50)),
        resume=bool(args.resume),
    )
    if config.num_processes < 1 or config.num_episodes < 1:
        raise ValueError("num_processes and num_episodes must both be >= 1")
    if config.max_steps < 1 or config.record_every_n_sim_steps < 1:
        raise ValueError("max_steps and record_every_n_sim_steps must both be >= 1")
    if config.use_gui and config.num_processes > 1:
        print("[warning] Only worker 0 uses GUI; other workers remain DIRECT.")
    return config


def print_summary(results: List[Dict[str, Any]]) -> None:
    ok = [result for result in results if result.get("status") == "ok"]
    skipped = [result for result in results if result.get("status") == "skipped"]
    errors = [result for result in results if result.get("status") == "error"]
    successes = [result for result in ok if result.get("success") is True]
    print("\n=== Multiprocessing collection summary ===")
    print(f"episodes returned : {len(results)}")
    print(f"collected ok      : {len(ok)}")
    print(f"successful tasks  : {len(successes)} / {len(ok)}")
    print(f"skipped existing  : {len(skipped)}")
    print(f"errors            : {len(errors)}")
    if errors:
        print("\nFirst error traceback:")
        print(errors[0].get("traceback", "<no traceback>"))


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    collector_config = parse_args()
    print(f"simulation: {collector_config.task_config['simulation']}")
    print(f"base_dir: {collector_config.base_dir}")
    print_summary(run_parent_scheduler(collector_config))
    print(f"\ncollection directory: {collector_config.base_dir.parent}")