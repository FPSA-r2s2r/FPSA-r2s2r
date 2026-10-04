"""Run any simulation task from one ``Config_*.yaml`` file.

Example:
    python demo.py Config_Assembly.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pybullet as p
import pybullet_data

from sim_world.src.task_config import load_task_config, parse_number
from sim_world.src.task_runtime import build_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    task_config = load_task_config(args.config)
    demo_config = dict(task_config.get("demo", {}))
    time_step = parse_number(
        demo_config.get("time_step", 1.0 / 120.0),
        field="demo.time_step",
    )
    max_steps = int(demo_config.get("max_steps", 0))
    record_every = int(demo_config.get("record_every_n_sim_steps", 12))
    sleep_seconds = float(demo_config.get("sleep_seconds", 0.0))
    seed = int(demo_config.get("seed", 42))

    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(time_step)
    p.setGravity(0.0, 0.0, -9.8)

    sim = build_simulation(task_config, p, client_id, seed=seed, control_dt=time_step)
    print(f"simulation: {task_config['simulation']}, seed: {seed}")

    try:
        sim.make_scene(**task_config["make_scene"])
        sim.enable_high_quality_rendering()

        sim_step = 0
        record_idx = 0
        while not sim.done and (max_steps == 0 or sim_step < max_steps):
            p.stepSimulation()
            sim.step()
            sim_step += 1

            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)

            if sim_step % record_every == 0:
                started = time.time()
                observation = sim.collect_observation(**task_config["collect_observation"])
                cv2.imwrite("temp_fisheye.png", cv2.cvtColor(observation["robot0_eye_in_hand_image"], cv2.COLOR_RGB2BGR))
                cv2.imwrite("temp_agentview.png", cv2.cvtColor(observation["agentview_image"], cv2.COLOR_RGB2BGR))
                record_idx += 1
                print(
                    f"record={record_idx}, step={sim_step}, state={getattr(sim, 'state', None)}, "
                    f"success={bool(sim.is_success())}, render={time.time() - started:.4f}s"
                )

        print(f"done={sim.done}, success={bool(sim.is_success())}, steps={sim_step}")
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        p.disconnect(client_id)


if __name__ == "__main__":
    main()
