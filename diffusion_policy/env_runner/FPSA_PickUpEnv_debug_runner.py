import numpy as np
import torch
import pybullet as p

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply
from sim_world.Env_PickUp import PickUpEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from icecream import ic
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

class PickUpEnvDebugRunner(BaseImageRunner):
    def __init__(self, output_dir,            
                 max_steps=2000,
                 _seed = 43,
                 randomize_image_noise = True,
                 randomize_lighting = True,
                 randomize_objpose = True,
                 randomize_distractors = True,
                 randomize_outlscene = True,
                 randomize_plane_height = True,
                 randomize_campose = True,
                 n_obs_steps=2,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 video_file_path = None, # None to disable video recording
                 ):
        super().__init__(output_dir)

        self.n_action_steps = n_action_steps
        ic(n_obs_steps, n_action_steps)
        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PickUpEnv(
                        sim_steps_per_action=12,
                        connection_mode=p.GUI,
                        seed = _seed,
                        randomize_image_noise=randomize_image_noise,
                        randomize_lighting=randomize_lighting,
                        randomize_objpose=randomize_objpose,
                        randomize_distractors=randomize_distractors,
                        randomize_outlscene = randomize_outlscene,
                        randomize_plane_height = randomize_plane_height,
                        randomize_campose = randomize_campose,
                        ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path= video_file_path,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        self.env_fn = env_fn
        self.env = None

    def _prepare_obs(self, obs):
        np_obs_dict = dict(obs)

        # image: uint8 NHWC -> float32 NCHW, range [0, 1]
        for key in ['agentview_image']:
            if key in np_obs_dict:
                np_obs_dict[key] = np.moveaxis(
                    np_obs_dict[key], -1, -3
                ).astype(np.float32) / 255.0

        for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
            if key in np_obs_dict:
                np_obs_dict[key] = np_obs_dict[key].astype(np.float32)

        return np_obs_dict


    def close(self):
        if self.env is not None:
            try:
                self.env.env.video_recoder.stop()
            finally:
                super().close()

    @torch.no_grad()
    def run(self, policy: BaseImagePolicy):
        self.env = self.env_fn()
        try:
            obs = self.env.reset()
            policy.reset()
            done = False
            while not done:
                obs_dict = dict_apply(self._prepare_obs(obs),
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(policy.device))
                action = policy.predict_action(obs_dict)['action'].cpu().numpy()
                action = action[:, :self.n_action_steps, :].squeeze(0)
                obs, _, done, _ = self.env.step(action)
                done = np.all(done)
        finally:
            self.close()
