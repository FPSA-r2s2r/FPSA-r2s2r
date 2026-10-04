import pybullet as p

from sim_world.Env_Wrench import WrenchEnv
from diffusion_policy.env_runner.fpsa_image_runner import FPSAImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
import numpy as np
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.BulletBridge_util import action_rot6d_to_quat_xyzw_np, quat_xyzw_to_rot6d_np


class WrenchEngagementEnvRunner(FPSAImageRunner):
    def __init__(self, output_dir,            
                 max_steps=2000,
                 n_obs_steps=2,
                 n_action_steps=8,
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
        self.rotation_transformer = RotationTransformer(
            from_rep='quaternion',
            to_rep='rotation_6d',
        )
        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    WrenchEnv(
                        sim_steps_per_action=12,
                        connection_mode=p.DIRECT,
                        seed = 46,
                        if_FPSA = False,
                        randomize_objcolor = True,
                        randomize_image_noise=True,
                        randomize_lighting=True,
                        randomize_objpose=True,
                        randomize_distractors=True,
                        randomize_outlscene = True,
                        randomize_plane_height = True,
                        randomize_campose = True
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

    def _prepare_obs(self, obs):
        np_obs_dict = dict(obs)

        # image: uint8 NHWC -> float32 NCHW, range [0, 1]
        for key in ['agentview_image', "robot0_eye_in_hand_image"]:
            if key in np_obs_dict:
                np_obs_dict[key] = np.moveaxis(
                    np_obs_dict[key], -1, -3
                ).astype(np.float32) / 255.0

        if 'robot0_eef_pos' in np_obs_dict:
            np_obs_dict['robot0_eef_pos'] = np.asarray(
                np_obs_dict['robot0_eef_pos'], dtype=np.float32
            )

        if 'robot0_gripper_qpos' in np_obs_dict:
            np_obs_dict['robot0_gripper_qpos'] = np.asarray(
                np_obs_dict['robot0_gripper_qpos'], dtype=np.float32
            )

        # Env gives quaternion in xyzw order; policy expects rotation_6d.
        # Remove robot0_eef_quat after conversion so the policy only receives
        # keys defined in the 6D shape_meta / normalizer.
        if 'robot0_eef_quat' in np_obs_dict:
            quat_xyzw = np.asarray(np_obs_dict.pop('robot0_eef_quat'), dtype=np.float32)
            np_obs_dict['robot0_eef_rot6d'] = quat_xyzw_to_rot6d_np(quat_xyzw, 
                                                                    self.rotation_transformer
                                                                    ).astype(np.float32)
        else:
            raise KeyError(
                "Expected observation to contain 'robot0_eef_quat' "
                "or 'robot0_eef_rot6d'."
            )
        return np_obs_dict

    def _prepare_action(self, action):
        return action_rot6d_to_quat_xyzw_np(action, self.rotation_transformer)
