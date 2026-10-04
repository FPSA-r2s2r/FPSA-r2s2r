from typing import Dict, Union
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.BulletBridge_util import action_quat_xyzw_to_rot6d_np, quat_xyzw_to_rot6d_np
class PhyAgentImage_6D_Dataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=['action', 'agentview_image', 'robot0_eef_pos', 
        #                      'robot0_eef_quat', 'robot0_gripper_qpos'])
        
        
        # RotationTransformer('quaternion', 'rotation_6d') expects quaternion in
        # PyTorch3D convention: wxyz. Your collected data is xyzw, so every
        # quaternion is reordered before calling this transformer.
        self.rotation_transformer = RotationTransformer(
            from_rep='quaternion',
            to_rep='rotation_6d',
        )
        self.replay_buffer = ReplayBuffer.create_from_path(
                                zarr_path,
                                mode='r'
                            )
                
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # Fit the normalizer on exactly the same representation returned by
        # __getitem__: robot0_eef_rot6d and 10D rot6d action.
        action_rot6d = action_quat_xyzw_to_rot6d_np(
            self.replay_buffer['action'],
            self.rotation_transformer,
        )
        robot0_eef_rot6d = quat_xyzw_to_rot6d_np(
            self.replay_buffer['robot0_eef_quat'],
            self.rotation_transformer,
        )

        data = {
            'action': action_rot6d,
            'robot0_eef_pos': self.replay_buffer['robot0_eef_pos'],
            'robot0_eef_rot6d': robot0_eef_rot6d,
            'robot0_gripper_qpos': self.replay_buffer['robot0_gripper_qpos'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['agentview_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agentview_image = np.moveaxis(
            sample['agentview_image'], -1, 1,
        ).astype(np.float32) / 255.0

        robot0_eef_rot6d = quat_xyzw_to_rot6d_np(
            sample['robot0_eef_quat'],
            self.rotation_transformer,
        )
        action_rot6d = action_quat_xyzw_to_rot6d_np(
            sample['action'],
            self.rotation_transformer,
        )

        data = {
            'obs': {
                'agentview_image': agentview_image,
                'robot0_eef_pos': sample['robot0_eef_pos'].astype(np.float32),  # T, 3
                'robot0_eef_rot6d': robot0_eef_rot6d,                           # T, 6
                'robot0_gripper_qpos': sample['robot0_gripper_qpos'].astype(np.float32),  # T, 1
            },
            'action': action_rot6d,  # T, 10 = xyz(3) + rot6d(6) + gripper(1)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


# def test():
#     import os
#     zarr_path = os.path.expanduser('/home/iadc/PhyDomain/CupOnRackBuffer.zarr')
#     dataset = PhyDomainDataset(zarr_path, horizon=16)
#     print(f"dataset length: {len(dataset)}")
#     sample = dataset[0]
#     print("obs keys:", sample['obs'].keys())
#     sample = dataset[0]
#     print(sample['obs']['agentview_image'].shape, sample['obs']['agentview_image'].dtype,
#         sample['obs']['agentview_image'].min(), sample['obs']['agentview_image'].max())
#     print(sample['obs']['robot0_eye_in_hand_image'].shape, sample['obs']['robot0_eye_in_hand_image'].dtype,
#         sample['obs']['robot0_eye_in_hand_image'].min(), sample['obs']['robot0_eye_in_hand_image'].max())
#     print(sample['obs']['robot0_eef_pos'].shape, sample['obs']['robot0_eef_pos'].dtype)
#     print(sample['obs']['robot0_eef_quat'].shape, sample['obs']['robot0_eef_quat'].dtype)
#     print(sample['obs']['robot0_gripper_qpos'].shape, sample['obs']['robot0_gripper_qpos'].dtype)
#     print(sample['action'].shape, sample['action'].dtype)


def test():
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    zarr_path = os.path.expanduser('/home/iadc/GeoBridge/DP_data/pickup/pickup.zarr')
    dataset = PhyAgentImage_6D_Dataset(zarr_path, horizon=16)

    print(f"dataset length: {len(dataset)}")
    sample = dataset[0]

    print("obs keys:", sample['obs'].keys())

    for k in ['agentview_image']:
        x = sample['obs'][k]
        print(f"{k}: shape={x.shape}, dtype={x.dtype}, min={x.min()}, max={x.max()}")

    for k in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
        x = sample['obs'][k]
        print(f"{k}: shape={x.shape}, dtype={x.dtype}")

    print("action:", sample['action'].shape, sample['action'].dtype)

    agent = sample['obs']['agentview_image'].numpy()         # (T, C, H, W)

    # Convert back to the Matplotlib display format (T, H, W, C).
    agent_vis = np.moveaxis(agent, 1, -1)

    # Inspect the first, second, and final frames.
    frame_ids = [0, min(1, agent_vis.shape[0]-1), agent_vis.shape[0]-1]

    fig, axes = plt.subplots(len(frame_ids), 2, figsize=(8, 4 * len(frame_ids)))
    if len(frame_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, t in enumerate(frame_ids):
        axes[row, 0].imshow(np.clip(agent_vis[t], 0, 1))
        axes[row, 0].set_title(f'agentview t={t}')
        axes[row, 0].axis('off')


    plt.tight_layout()
    plt.show()

    # Inspect changes between adjacent frames.
    if agent_vis.shape[0] >= 2:
        agent_diff = np.mean(np.abs(agent_vis[1] - agent_vis[0]))
        print(f"mean abs diff agentview t1-t0: {agent_diff}")
        # Also visualize the difference image.
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(np.abs(agent_vis[1] - agent_vis[0]))
        axes[0].set_title('agentview |t1-t0|')
        axes[0].axis('off')

        plt.tight_layout()
        plt.show()
        
# test()
#python -m diffusion_policy.dataset.PhyDomainDataset.py
    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
