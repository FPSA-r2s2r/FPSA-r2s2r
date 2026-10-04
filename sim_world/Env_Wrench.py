"""Configured Gym environment for the wrench-engagement task."""

import pybullet as p

from sim_world.Base_Env import ConfiguredTaskEnv


class WrenchEnv(ConfiguredTaskEnv):
    def __init__(
        self,
        sim_steps_per_action=1,
        connection_mode=p.DIRECT,
        seed=46,
        use_agent_cam=True,
        use_fisheye_cam=True,
        scene_overrides=None,
        **_legacy_options,
    ):
        super().__init__(
            "Config_Wrench.yaml",
            sim_steps_per_action=sim_steps_per_action,
            connection_mode=connection_mode,
            seed=seed,
            use_agent_cam=use_agent_cam,
            use_fisheye_cam=use_fisheye_cam,
            control_gripper=False,
            scene_overrides=scene_overrides,
        )
