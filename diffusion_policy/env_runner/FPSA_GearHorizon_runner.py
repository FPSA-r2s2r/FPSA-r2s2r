import pybullet as p

from sim_world.Env_GearHorizon import GearHorizonEnv
from diffusion_policy.env_runner.fpsa_image_runner import FPSAImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder


class GearEnvRunner(FPSAImageRunner):
    def __init__(self, output_dir,            
                 max_steps=2000,
                 n_obs_steps=2,
                 n_action_steps=8,
                 use_agent_cam = True,
                 use_fisheye_cam = True,
                 # How many envs as env runners
                 train_start_seed=43,
                 test_start_seed=20043,
                 n_train=10,
                 n_train_vis=3,
                 n_test=12,
                 n_test_vis=6,
                 fps=10,
                 crf=22,
                 video_file_path = None, # None to disable video recording
                 n_envs=None,
                 tqdm_interval_sec=5.0,
                 ):

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    GearHorizonEnv(
                        sim_steps_per_action=12,
                        connection_mode=p.DIRECT,
                        seed = 46,
                        use_agent_cam= use_agent_cam,
                        use_fisheye_cam= use_fisheye_cam
                        ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path= None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        super().__init__(output_dir, env_fn,
            max_steps=max_steps, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
            train_start_seed=train_start_seed, test_start_seed=test_start_seed,
            n_train=n_train, n_train_vis=n_train_vis, n_test=n_test, n_test_vis=n_test_vis,
            fps=fps, crf=crf, n_envs=n_envs, tqdm_interval_sec=tqdm_interval_sec)
