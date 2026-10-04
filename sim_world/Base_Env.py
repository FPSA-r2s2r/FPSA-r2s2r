"""Gym environment shared by the configured simulation tasks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Union

import gym
from gym import spaces
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client

from sim_world.src.task_config import load_task_config, parse_number
from sim_world.src.task_runtime import build_simulation


class ConfiguredTaskEnv(gym.Env):
    """Build a task entirely from its ``sim_world/Config_*.yaml`` file."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        config_name: str,
        *,
        sim_steps_per_action: int = 1,
        connection_mode: int = p.DIRECT,
        seed: int = 46,
        use_agent_cam: bool = True,
        use_fisheye_cam: bool = True,
        control_gripper: bool = True,
        action_input_frame: Optional[str] = "parent_tcp",
        scene_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._seed = seed
        self.use_agent_cam = use_agent_cam
        self.use_fisheye_cam = use_fisheye_cam
        self.control_gripper = control_gripper
        self.action_input_frame = action_input_frame
        self.sim_steps_per_action = sim_steps_per_action

        self._task_config = load_task_config(config_name)
        if scene_overrides:
            self._task_config["make_scene"].update(deepcopy(scene_overrides))
        self._observation_kwargs = dict(self._task_config["collect_observation"])
        self._time_step = parse_number(
            self._task_config.get("env", {}).get(
                "time_step",
                self._task_config.get("demo", {}).get("time_step", "1 / 120"),
            ),
            field="env.time_step",
        )

        self._pybullet_client = bullet_client.BulletClient(
            connection_mode=connection_mode
        )
        self.sim = None
        self._last_obs = None

        self.action_space = spaces.Box(
            low=np.full(8, -np.inf, dtype=np.float32),
            high=np.full(8, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        observation_spaces = {
            "robot0_eef_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "robot0_eef_quat": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            "robot0_gripper_qpos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
        }
        if use_agent_cam:
            observation_spaces["agentview_image"] = spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            )
        if use_fisheye_cam:
            observation_spaces["robot0_eye_in_hand_image"] = spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(observation_spaces)

    def seed(self, seed: int) -> None:
        self._seed = seed

    def _build_sim(self) -> None:
        client = self._pybullet_client
        client.resetSimulation()
        client.setTimeStep(self._time_step)
        client.setGravity(0, 0, -9.8)

        self.sim = build_simulation(
            self._task_config,
            client,
            client._client,
            seed=self._seed,
            control_dt=self._time_step,
        )
        self.sim.enable_high_quality_rendering()
        self.sim.make_scene(**deepcopy(self._task_config["make_scene"]))

    def _collect_observation(self):
        kwargs = dict(self._observation_kwargs)
        kwargs["use_agent_cam"] = self.use_agent_cam
        kwargs["use_eye_in_hand"] = self.use_fisheye_cam
        return self.sim.collect_observation(**kwargs)

    def reset(self):
        self._build_sim()
        observation = self._collect_observation()
        self._last_obs = observation
        return observation

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (8,):
            raise ValueError(f"action must have shape (8,), got {action.shape}")

        target_pos = action[:3]
        target_orn = action[3:7]
        gripper = float(action[7])
        ik_kwargs = {}
        if self.action_input_frame is not None:
            ik_kwargs["input_frame"] = self.action_input_frame
        self.sim.solve_ik_and_apply(target_pos, target_orn, **ik_kwargs)
        if self.control_gripper:
            self.sim.set_gripper(gripper)

        for _ in range(self.sim_steps_per_action):
            self._pybullet_client.stepSimulation()

        observation = self._collect_observation()
        self._last_obs = observation
        done = bool(self.sim.is_success())
        info = {
            "target_pos": target_pos,
            "target_orn": target_orn,
            "gripper": gripper,
        }
        return observation, done, done, info

    def render(self, mode="rgb_array"):
        if self._last_obs is None:
            return None
        images = [
            self._last_obs[key]
            for key in ("agentview_image", "robot0_eye_in_hand_image")
            if key in self._last_obs
        ]
        if not images:
            return None
        if len(images) == 1:
            return images[0]
        spacer = np.full((images[0].shape[0], 16, 3), 255, dtype=np.uint8)
        return np.concatenate((images[0], spacer, images[1]), axis=1)

    def close(self) -> None:
        if self._pybullet_client is not None:
            self._pybullet_client.disconnect()
            self._pybullet_client = None
