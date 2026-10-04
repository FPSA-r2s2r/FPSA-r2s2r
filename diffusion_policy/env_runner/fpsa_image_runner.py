"""Shared FPSA rollout lifecycle: one fresh worker group per evaluation chunk."""
import collections
import math
import pathlib

import dill
import numpy as np
import torch
import tqdm
import wandb

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper


class FPSAImageRunner(BaseImageRunner):
    image_keys = ('agentview_image', 'robot0_eye_in_hand_image')

    def __init__(self, output_dir, env_fn, *, max_steps, n_obs_steps,
                 n_action_steps, train_start_seed, test_start_seed, n_train,
                 n_train_vis, n_test, n_test_vis, fps, crf, n_envs,
                 tqdm_interval_sec):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test
        if n_envs < 1:
            raise ValueError('n_envs must be positive')

        self.env = None
        self.env_fns = [env_fn] * n_envs
        self.env_seeds = []
        self.env_prefixs = []
        self.env_render_enabled = []
        self.env_init_fn_dills = []
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        for prefix, count, n_vis, start_seed in (
            ('train/', n_train, n_train_vis, train_start_seed),
            ('test/', n_test, n_test_vis, test_start_seed),
        ):
            for i in range(count):
                seed = start_seed + i
                self.env_seeds.append(seed)
                self.env_prefixs.append(prefix)
                self.env_render_enabled.append(i < n_vis)

        self._set_rollout_step(0)

    def _set_rollout_step(self, global_step):
        """Prepare deterministic video paths for one rollout."""
        step_dir = pathlib.Path(self.output_dir).joinpath(
            'media', f'step_{global_step:08d}')
        self.env_init_fn_dills = []
        for seed, prefix, enable_render in zip(
                self.env_seeds, self.env_prefixs, self.env_render_enabled):
            split = prefix.rstrip('/').replace('/', '_')

            def init_fn(env, seed=seed, split=split,
                        enable_render=enable_render, step_dir=step_dir):
                assert isinstance(env, MultiStepWrapper)
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = step_dir.joinpath(f'{split}_seed_{seed}.mp4')
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                env.seed(seed)

            self.env_init_fn_dills.append(dill.dumps(init_fn))

    def _prepare_obs(self, obs):
        result = dict(obs)
        # uint8 NHWC -> float32 NCHW, range [0, 1].
        for key in self.image_keys:
            if key in result:
                result[key] = np.moveaxis(result[key], -1, -3).astype(np.float32) / 255.0
        for key in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
            if key in result:
                result[key] = result[key].astype(np.float32)
        return result

    def _prepare_action(self, action):
        return action

    @torch.no_grad()
    def _run_chunk(self, policy, env, init_fns, description):
        env.call_each('run_dill_function', args_list=[(fn,) for fn in init_fns])
        obs = env.reset()
        policy.reset()
        with tqdm.tqdm(total=self.max_steps, desc=description, leave=False,
                       mininterval=self.tqdm_interval_sec) as pbar:
            done = False
            while not done:
                obs_dict = dict_apply(self._prepare_obs(obs),
                    lambda x: torch.from_numpy(x).to(policy.device))
                action = self._prepare_action(policy.predict_action(obs_dict)['action'].cpu().numpy())
                obs, _, done, _ = env.step(action)
                done = np.all(done)
                pbar.update(action.shape[1] if action.ndim >= 2 else 1)
        return env.render(), env.call('get_attr', 'reward')

    def close(self):
        if self.env is not None:
            # Join old workers before allocating the next group's EGL contexts.
            self.env.close(timeout=10.0)
            self.env = None

    def run(self, policy, global_step=0):
        self._set_rollout_step(global_step)
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            init_fns = self.env_init_fn_dills[start:end]
            # Keep the original inference batch size, padding and RNG consumption.
            init_fns += [self.env_init_fn_dills[0]] * (n_envs - len(init_fns))
            self.env = AsyncVectorEnv(self.env_fns, shared_memory=False, context='spawn')
            try:
                videos, rewards = self._run_chunk(policy, self.env, init_fns,
                    f'Eval {type(self).__name__} chunk {chunk_idx + 1}/{n_chunks}')
                all_video_paths[start:end] = videos[:end - start]
                all_rewards[start:end] = rewards[:end - start]
            finally:
                self.close()

        return self._summarize_rollout(all_rewards, all_video_paths)

    def _summarize_rollout(self, all_rewards, all_video_paths):
        max_rewards = collections.defaultdict(list)
        log_data = {}
        for seed, prefix, rewards, video_path in zip(
                self.env_seeds, self.env_prefixs, all_rewards, all_video_paths):
            max_reward = float(np.max(rewards))
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
            if video_path is not None:
                log_data[prefix + f'sim_video_{seed}'] = wandb.Video(video_path, format='mp4')
        summary = []
        total_successes = 0
        total_episodes = 0
        for prefix, value in max_rewards.items():
            mean_score = float(np.mean(value))
            log_data[prefix + 'mean_score'] = mean_score
            # Checkpoint metrics also need names without '/'.
            clean_prefix = prefix.rstrip('/').replace('/', '_')
            log_data[clean_prefix + '_mean_score'] = mean_score
            # FPSA environments return is_success() as a boolean reward.
            successes = int(np.count_nonzero(np.asarray(value) > 0))
            episodes = len(value)
            log_data[prefix + 'success_rate'] = successes / episodes
            summary.append(f'{clean_prefix}: {successes}/{episodes} ({successes / episodes:.1%})')
            total_successes += successes
            total_episodes += episodes
        log_data['eval/success_count'] = total_successes
        log_data['eval/episode_count'] = total_episodes
        if total_episodes:
            success_rate = total_successes / total_episodes
            log_data['eval/success_rate'] = success_rate
            summary.append(f'overall: {total_successes}/{total_episodes} ({success_rate:.1%})')
            tqdm.tqdm.write(f'[Rollout {type(self).__name__}] ' + ' | '.join(summary))
        else:
            tqdm.tqdm.write(f'[Rollout {type(self).__name__}] No evaluation episodes.')
        return log_data
