"""CPU regression checks: python -m unittest discover -s tests -v."""
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import gym
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import default_collate
import zarr
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.dataset.PhyFisheyeImageDataset import PhyFisheyeImageDataset
from diffusion_policy.dataset.PhyAgentImageDataset import PhyAgentImageDataset
from diffusion_policy.env_runner import fpsa_image_runner
from diffusion_policy.env_runner.FPSA_Assembly_runner import AssemblyEnvRunner
from diffusion_policy.env_runner.FPSA_Gear_runner import GearEnvRunner
from diffusion_policy.env_runner.FPSA_GearHorizon_runner import GearEnvRunner as GearHorizonEnvRunner
from diffusion_policy.env_runner.FPSA_PickUp_runner import PickUpEnvRunner
from diffusion_policy.env_runner.FPSA_PickUpEnv_debug_runner import PickUpEnvDebugRunner
from diffusion_policy.env_runner.FPSA_wrench_runner import WrenchEngagementEnvRunner
from diffusion_policy.env_runner.FPSA_wrench_6d_runner import WrenchEngagementEnvRunner as Wrench6DEnvRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.FPSA_policy import FPSAImagePolicy
from diffusion_policy.workspace.FPSA_workspace import (
    epoch_batches, make_dataloader, rollout_memory,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(14, 8)

    def output_shape(self):
        return (8,)

    def forward(self, obs):
        features = [value.mean(dim=(-2, -1)) if value.ndim == 4 else value
                    for _, value in sorted(obs.items())]
        return self.linear(torch.cat(features, dim=-1))


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        'robot0_eef_pos': gym.spaces.Box(-1e9, 1e9, shape=(3,), dtype=np.float32),
        'robot0_eef_quat': gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)})
    action_space = gym.spaces.Box(-1, 1, shape=(8,), dtype=np.float32)

    current_seed = 43

    def seed(self, seed=None):
        self.current_seed = seed

    def reset(self):
        return {'robot0_eef_pos': np.array([os.getpid(), 0, 0], dtype=np.float32),
                'robot0_eef_quat': np.array([0, 0, 0, 1], dtype=np.float32)}

    def step(self, action):
        np.testing.assert_array_equal(action, [0, 0, 0, 0, 0, 0, 1, 0])
        return self.reset(), float(self.current_seed % 2), True, {}

    def slow(self):
        time.sleep(30)


class SlowCloseEnv(TinyEnv):
    def close(self):
        time.sleep(30)


def wrapped_env():
    return MultiStepWrapper(VideoRecordingWrapper(
        TinyEnv(), VideoRecorder.create_h264(fps=10)),
        n_obs_steps=2, n_action_steps=2, max_episode_steps=2)


class TinyPolicy:
    device = torch.device('cpu')

    def reset(self):
        pass

    def predict_action(self, obs):
        action = torch.zeros(len(obs['robot0_eef_pos']), 2, 8)
        action[..., 6] = 1
        return {'action': action}


class Tiny6DPolicy(TinyPolicy):
    def predict_action(self, obs):
        assert 'robot0_eef_quat' not in obs
        assert obs['robot0_eef_rot6d'].shape[-1] == 6
        action = torch.zeros(len(obs['robot0_eef_pos']), 2, 10)
        action[..., 3] = action[..., 7] = 1
        return {'action': action}


RUNNERS = (AssemblyEnvRunner, GearEnvRunner, GearHorizonEnvRunner,
           PickUpEnvRunner, WrenchEngagementEnvRunner, Wrench6DEnvRunner)


class MemoryRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.zarr_path = str(Path(cls.tmp.name) / 'tiny.zarr')
        root = zarr.open_group(cls.zarr_path, mode='w')
        data = root.create_group('data')
        rng = np.random.default_rng(7)
        for key in ('agentview_image', 'robot0_eye_in_hand_image'):
            data.array(key, rng.integers(0, 256, (42, 8, 8, 3), dtype=np.uint8))
        for key, dim in (('robot0_eef_pos', 3), ('robot0_eef_quat', 4),
                         ('robot0_gripper_qpos', 1), ('action', 2)):
            data.array(key, rng.normal(size=(42, dim)).astype(np.float32))
        root.create_group('meta').array('episode_ends', np.array([1, 3, 12, 42]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def dataset(self, n_obs_steps=None):
        return PhyFisheyeImageDataset(self.zarr_path, horizon=16,
            pad_before=15, pad_after=15, n_obs_steps=n_obs_steps, val_ratio=0.25)

    def test_all_samples_and_normalizer(self):
        full, short = self.dataset(), self.dataset(2)
        for left, right in ((full, short),
                            (full.get_validation_dataset(), short.get_validation_dataset())):
            self.assertEqual(len(left), len(right))
            np.testing.assert_array_equal(left.sampler.indices, right.sampler.indices)
            for idx in range(len(left)):
                a, b = left[idx], right[idx]
                torch.testing.assert_close(a['action'], b['action'], rtol=0, atol=0)
                for key in a['obs']:
                    torch.testing.assert_close(a['obs'][key][:2], b['obs'][key], rtol=0, atol=0)
        for key, value in full.get_normalizer().state_dict().items():
            torch.testing.assert_close(value, short.get_normalizer().state_dict()[key], rtol=0, atol=0)

    def test_agent_only_dataset_matches_full_observations(self):
        kwargs = dict(horizon=16, pad_before=15, pad_after=15, val_ratio=0.25)
        full = PhyAgentImageDataset(self.zarr_path, **kwargs)
        short = PhyAgentImageDataset(self.zarr_path, n_obs_steps=2, **kwargs)
        for left, right in ((full, short),
                            (full.get_validation_dataset(), short.get_validation_dataset())):
            self.assertEqual(len(left), len(right))
            np.testing.assert_array_equal(left.sampler.indices, right.sampler.indices)
            for idx in range(len(left)):
                a, b = left[idx], right[idx]
                torch.testing.assert_close(a['action'], b['action'], rtol=0, atol=0)
                for key in a['obs']:
                    torch.testing.assert_close(a['obs'][key][:2], b['obs'][key], rtol=0, atol=0)

    def test_loss_gradients_and_update(self):
        full, short = self.dataset(), self.dataset(2)
        policy = FPSAImagePolicy(shape_meta={'action': {'shape': [2]}},
            noise_scheduler=DDIMScheduler(num_train_timesteps=8), obs_encoder=TinyEncoder(),
            horizon=16, n_action_steps=8, n_obs_steps=2, diffusion_step_embed_dim=8,
            down_dims=(8, 16), n_groups=4)
        policy.set_normalizer(full.get_normalizer())
        other = copy.deepcopy(policy)
        batches = [default_collate([dataset[i] for i in (0, 20, 40)])
                   for dataset in (full, short)]
        losses, gradients = [], []
        for model, batch in zip((policy, other), batches):
            optimizer = torch.optim.AdamW(model.parameters())
            torch.manual_seed(123)
            loss = model.compute_loss(batch)
            loss.backward()
            losses.append(loss.detach())
            gradients.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
            optimizer.step()
        torch.testing.assert_close(*losses, rtol=0, atol=0)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(policy.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_task_config_conditioning(self):
        config_dir = str(Path(__file__).resolve().parents[1] / 'diffusion_policy/config')
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            for task in ('Assembly', 'Gear', 'GearHorizon', 'Wrench'):
                for global_cond in (True, False):
                    cfg = hydra.compose(config_name='FPSA_FishEye_workspace', overrides=[
                        f'task=FPSA_{task}', f'obs_as_global_cond={global_cond}'])
                    self.assertEqual(cfg.task.dataset.n_obs_steps, 2 if global_cond else 16)

            for global_cond in (True, False):
                cfg = hydra.compose(config_name='FPSA_AgentOnly_workspace', overrides=[
                    f'obs_as_global_cond={global_cond}'])
                self.assertEqual(cfg.task.dataset.n_obs_steps, 2 if global_cond else 16)
                self.assertGreater(cfg.dataloader.batch_size, 124)
                self.assertEqual(cfg.dataloader.prefetch_factor, 1)

    def test_checkpoint_saves_are_serialized(self):
        workspace = BaseWorkspace(OmegaConf.create({}), self.tmp.name)
        workspace.model = nn.Linear(2, 2)
        first_path = Path(self.tmp.name) / 'first.ckpt'
        second_path = Path(self.tmp.name) / 'second.ckpt'
        workspace.save_checkpoint(first_path)
        first_thread = workspace._saving_thread
        workspace.save_checkpoint(second_path)
        self.assertFalse(first_thread.is_alive())
        workspace._wait_for_saving_thread()
        self.assertTrue(first_path.is_file())
        self.assertTrue(second_path.is_file())

    def test_full_horizon_conditioning(self):
        full, configured = self.dataset(), self.dataset(16)
        policy = FPSAImagePolicy(shape_meta={'action': {'shape': [2]}},
            noise_scheduler=DDIMScheduler(num_train_timesteps=8), obs_encoder=TinyEncoder(),
            horizon=16, n_action_steps=8, n_obs_steps=2, obs_as_global_cond=False,
            diffusion_step_embed_dim=8, down_dims=(8, 16), n_groups=4)
        policy.set_normalizer(full.get_normalizer())
        losses = []
        for dataset in (full, configured):
            torch.manual_seed(123)
            losses.append(policy.compute_loss(default_collate([dataset[0], dataset[20]])))
        torch.testing.assert_close(*losses, rtol=0, atol=0)

    def check_offload(self, device):
        for use_ema in (False, True):
            model = nn.Linear(3, 2).to(device)
            optimizer = torch.optim.AdamW(model.parameters())
            inputs = torch.ones(4, 3, device=device)
            model(inputs).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            model(inputs).sum().backward()  # pending gradient accumulation
            reference, reference_optimizer = copy.deepcopy((model, optimizer))
            for actual, expected in zip(model.parameters(), reference.parameters()):
                expected.grad = actual.grad.clone()
            policy = copy.deepcopy(model) if use_ema else model
            state_devices = [[v.device for v in state.values() if torch.is_tensor(v)]
                             for state in optimizer.state.values()]
            with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                with rollout_memory(model, optimizer, policy):
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    self.assertEqual(next(policy.parameters()).device, torch.device(device))
                    self.assertTrue(all(v.device.type == 'cpu' for state in optimizer.state.values()
                                        for v in state.values() if torch.is_tensor(v)))
                    raise RuntimeError('test interruption')
            self.assertEqual(state_devices, [[v.device for v in state.values() if torch.is_tensor(v)]
                                            for state in optimizer.state.values()])
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_offload_cpu(self):
        self.check_offload('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_offload_cuda(self):
        self.check_offload('cuda:0')

    def test_loader_early_exit(self):
        # Plain integers isolate worker lifecycle from Tensor IPC.
        for workers in (0, 2):
            config = OmegaConf.create(dict(batch_size=2, num_workers=workers,
                prefetch_factor=1, persistent_workers=False))
            loader = make_dataloader(list(range(100)), config)
            loader.collate_fn = list
            with epoch_batches(loader, disable=True) as batches:
                next(iter(batches))
                processes = list(batches.iterable._workers) if workers else []
            self.assertTrue(all(not p.is_alive() for p in processes))

    def test_runner_restart_and_failure_cleanup(self):
        processes = []

        class RecordingVectorEnv(AsyncVectorEnv):
            def __init__(self, *args, **kwargs):
                if any(p.is_alive() for p in processes):
                    raise AssertionError('Previous chunk still running')
                if kwargs.get('context') != 'spawn':
                    raise AssertionError('FPSA workers must use spawn')
                super().__init__(*args, **kwargs)
                processes.extend(self.processes)

        for runner_cls in RUNNERS:
            with self.subTest(runner=runner_cls.__module__):
                processes.clear()
                runner = runner_cls(self.tmp.name, n_envs=2, n_train=1,
                    n_test=2, n_train_vis=0, n_test_vis=0, max_steps=2)
                self.assertIsNone(runner.env)
                self.assertEqual(runner.env_seeds, [43, 20043, 20044])
                runner.env_fns = [wrapped_env] * 2
                policy = Tiny6DPolicy() if runner_cls is Wrench6DEnvRunner else TinyPolicy()
                with patch.object(fpsa_image_runner, 'AsyncVectorEnv', RecordingVectorEnv):
                    result = runner.run(policy)
                    self.assertEqual(result['train_mean_score'], 1.0)
                    self.assertEqual(result['test_mean_score'], 0.5)
                    self.assertEqual(result['eval/success_count'], 2)
                    self.assertEqual(result['eval/episode_count'], 3)
                    self.assertEqual(result['eval/success_rate'], 2 / 3)
                    self.assertEqual(len(processes), 4)
                    self.assertEqual(len({p.pid for p in processes}), 4)
                    self.assertTrue(all(not p.is_alive() for p in processes))
                    with patch.object(type(policy), 'predict_action', side_effect=RuntimeError('test failure')):
                        with self.assertRaisesRegex(RuntimeError, 'test failure'):
                            runner.run(policy)
                    self.assertTrue(all(not p.is_alive() for p in processes))
                    self.assertIsNone(runner.env)
                    runner.close()

    def test_rollout_summary_and_video_format(self):
        runner = AssemblyEnvRunner(self.tmp.name, n_envs=2, n_train=1, n_test=2)
        with patch.object(fpsa_image_runner.wandb, 'Video') as video, \
                patch.object(fpsa_image_runner.tqdm.tqdm, 'write') as output:
            result = runner._summarize_rollout([[0, 1], [0, 0], [1]], ['test.mp4', None, None])
            video.assert_called_once_with('test.mp4', format='mp4')
            self.assertEqual(result['train/success_rate'], 1)
            self.assertEqual(result['test/success_rate'], 0.5)
            self.assertEqual(result['eval/success_rate'], 2 / 3)
            self.assertIn('overall: 2/3 (66.7%)', output.call_args.args[0])
        empty = AssemblyEnvRunner(self.tmp.name, n_envs=1, n_train=0, n_test=0)
        result = empty._summarize_rollout([], [])
        self.assertEqual(result['eval/episode_count'], 0)
        self.assertNotIn('eval/success_rate', result)

    def test_rollout_video_path_contains_step_split_and_seed(self):
        runner = AssemblyEnvRunner(
            self.tmp.name, n_envs=1, n_train=1, n_train_vis=1,
            n_test=1, n_test_vis=1)
        runner._set_rollout_step(1234)
        expected = [
            Path(self.tmp.name) / 'media' / 'step_00001234' / 'train_seed_43.mp4',
            Path(self.tmp.name) / 'media' / 'step_00001234' / 'test_seed_20043.mp4',
        ]
        for init_fn_dill, expected_path in zip(runner.env_init_fn_dills, expected):
            env = wrapped_env()
            try:
                fpsa_image_runner.dill.loads(init_fn_dill)(env)
                self.assertEqual(Path(env.env.file_path), expected_path)
            finally:
                env.close()

    def test_runner_observation_conversion(self):
        rng = np.random.default_rng(42)
        obs = {'agentview_image': rng.integers(0, 256, (2, 2, 8, 8, 3), dtype=np.uint8),
               'robot0_eye_in_hand_image': rng.integers(0, 256, (2, 2, 8, 8, 3), dtype=np.uint8),
               'robot0_eef_pos': rng.normal(size=(2, 2, 3)),
               'robot0_eef_quat': np.broadcast_to([0., 0., 0., 1.], (2, 2, 4)),
               'robot0_gripper_qpos': np.zeros((2, 2, 1))}
        for runner_cls in RUNNERS:
            runner = runner_cls(self.tmp.name, n_envs=1)
            result = runner._prepare_obs(obs)
            for key in ('agentview_image', 'robot0_eye_in_hand_image'):
                expected = obs[key] if runner_cls is PickUpEnvRunner and key == 'robot0_eye_in_hand_image' else (
                    np.moveaxis(obs[key], -1, -3).astype(np.float32) / 255.)
                np.testing.assert_array_equal(result[key], expected)
            if runner_cls is Wrench6DEnvRunner:
                self.assertNotIn('robot0_eef_quat', result)
                np.testing.assert_array_equal(result['robot0_eef_rot6d'],
                    np.broadcast_to([1, 0, 0, 0, 1, 0], (2, 2, 6)))

    def test_debug_runner_closes_on_return_and_error(self):
        runner = PickUpEnvDebugRunner(self.tmp.name)
        self.assertIsNone(runner.env)
        for fail in (False, True):
            env = wrapped_env()
            runner.env_fn = lambda: env
            with patch.object(env, 'close', wraps=env.close) as close:
                if fail:
                    with patch.object(TinyPolicy, 'predict_action', side_effect=RuntimeError('test failure')):
                        with self.assertRaisesRegex(RuntimeError, 'test failure'):
                            runner.run(TinyPolicy())
                else:
                    runner.run(TinyPolicy())
                close.assert_called_once()
                self.assertIsNone(runner.env)

    def test_vector_close_timeout(self):
        for factory, pending in ((SlowCloseEnv, False), (TinyEnv, True)):
            env = AsyncVectorEnv([factory], dummy_env_fn=TinyEnv,
                                 shared_memory=False, context='spawn')
            try:
                if pending:
                    env.call_async('slow')
                start = time.monotonic()
                env.close(timeout=0.1)
                self.assertLess(time.monotonic() - start, 5)
                self.assertTrue(all(not p.is_alive() for p in env.processes))
            finally:
                env.close(terminate=True)

    def test_vector_close_dead_worker(self):
        env = AsyncVectorEnv([TinyEnv], shared_memory=False, context='spawn')
        env.processes[0].terminate()
        env.processes[0].join(timeout=2)
        env.close(timeout=0.1)
        self.assertTrue(all(not p.is_alive() for p in env.processes))


if __name__ == '__main__':
    unittest.main()
