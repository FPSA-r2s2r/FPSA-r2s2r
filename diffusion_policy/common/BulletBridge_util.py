from typing import Dict, Union
import torch
import numpy as np

from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def xyzw_to_wxyz(q: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convert quaternion convention from xyzw to wxyz."""
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convert quaternion convention from wxyz to xyzw."""
    return q[..., [1, 2, 3, 0]]


def normalize_quat_np(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize quaternion arrays along the last dimension."""
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, eps)


def quat_xyzw_to_rot6d_np(
        quat_xyzw: np.ndarray,
        rotation_transformer: RotationTransformer,
    ) -> np.ndarray:
    """
    Convert xyzw quaternion to PyTorch3D rotation_6d.

    The collected robot data is xyzw, while diffusion_policy's RotationTransformer
    uses PyTorch3D quaternion convention, i.e. wxyz.
    """
    quat_xyzw = normalize_quat_np(quat_xyzw)
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    rot6d = rotation_transformer.forward(quat_wxyz)
    return np.asarray(rot6d, dtype=np.float32)


def rot6d_to_quat_xyzw_np(
        rot6d: np.ndarray,
        rotation_transformer: RotationTransformer,
    ) -> np.ndarray:
    """
    Convert PyTorch3D rotation_6d back to xyzw quaternion.

    This is useful for env_runner rollout, where the environment still expects
    the original xyzw quaternion action format.
    """
    rot6d = np.asarray(rot6d, dtype=np.float32)
    quat_wxyz = rotation_transformer.inverse(rot6d)
    quat_wxyz = normalize_quat_np(quat_wxyz)
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)
    return np.asarray(quat_xyzw, dtype=np.float32)


def action_quat_xyzw_to_rot6d_np(
        action: np.ndarray,
        rotation_transformer: RotationTransformer,
    ) -> np.ndarray:
    """
    Convert action from pos + quat_xyzw + gripper to pos + rot6d + gripper.

    Input shape:
        (..., 8)  = xyz(3) + quat_xyzw(4) + gripper(1)
    Output shape:
        (..., 10) = xyz(3) + rotation_6d(6) + gripper(1)

    If the action has more than one gripper/control dim after quaternion, all
    remaining dimensions are preserved after the 6D rotation.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] < 8:
        raise ValueError(
            f"Expected action dim >= 8 for xyz + quat_xyzw + gripper, "
            f"got shape {action.shape}."
        )

    pos = action[..., :3]
    quat_xyzw = action[..., 3:7]
    gripper_or_rest = action[..., 7:]
    rot6d = quat_xyzw_to_rot6d_np(quat_xyzw, rotation_transformer)
    return np.concatenate([pos, rot6d, gripper_or_rest], axis=-1).astype(np.float32)


def action_rot6d_to_quat_xyzw_np(
        action: np.ndarray,
        rotation_transformer: RotationTransformer,
    ) -> np.ndarray:
    """
    Convert action from pos + rot6d + gripper back to pos + quat_xyzw + gripper.

    Input shape:
        (..., 10) = xyz(3) + rotation_6d(6) + gripper(1)
    Output shape:
        (..., 8)  = xyz(3) + quat_xyzw(4) + gripper(1)
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] < 10:
        raise ValueError(
            f"Expected action dim >= 10 for xyz + rot6d + gripper, "
            f"got shape {action.shape}."
        )

    pos = action[..., :3]
    rot6d = action[..., 3:9]
    gripper_or_rest = action[..., 9:]
    quat_xyzw = rot6d_to_quat_xyzw_np(rot6d, rotation_transformer)
    return np.concatenate([pos, quat_xyzw, gripper_or_rest], axis=-1).astype(np.float32)
