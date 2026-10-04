import numpy as np
from scipy.spatial.transform import Rotation as R
from icecream import ic
class PoseDR:
    """SE3 Pose domain randomization"""

    def __init__(self, bullet_client=None, seed=42):
        self.bullet_client = bullet_client
        self.rng = np.random.default_rng(seed)
        self.base_pos = None
        self.base_orn = None

    def sample_pos_randomization(
        self,
        pos=None,
        x_jitter_range=None,
        y_jitter_range=None,
        z_jitter_range=None,
    ):
        if pos is None:
            raise ValueError("Base pos is not provided")
        self.base_pos = np.asarray(pos, dtype=np.float32)

        jitter = np.array([
            self.rng.uniform(-x_jitter_range, x_jitter_range) if x_jitter_range is not None else 0.0,
            self.rng.uniform(-y_jitter_range, y_jitter_range) if y_jitter_range is not None else 0.0,
            self.rng.uniform(-z_jitter_range, z_jitter_range) if z_jitter_range is not None else 0.0,
        ], dtype=np.float32)

        return self.base_pos + jitter

    def sample_SE3_randomization(
        self,
        pos=None,
        orn=None,
        x_jitter_range=None,
        y_jitter_range=None,
        z_jitter_range=None,
        x_euler_jitter_range=None,
        y_euler_jitter_range=None,
        z_euler_jitter_range=None,
        get_matrix=False,
    ):
        if pos is None or orn is None:
            raise ValueError("Base pos an orn are not provided")

        self.base_pos = np.asarray(pos, dtype=np.float32)
        self.base_orn = np.asarray(orn, dtype=np.float32)

        pos_jitter = np.array([
            self.rng.uniform(-x_jitter_range, x_jitter_range) if x_jitter_range is not None else 0.0,
            self.rng.uniform(-y_jitter_range, y_jitter_range) if y_jitter_range is not None else 0.0,
            self.rng.uniform(-z_jitter_range, z_jitter_range) if z_jitter_range is not None else 0.0,
        ], dtype=np.float32)

        euler_jitter = np.array([
            self.rng.uniform(-x_euler_jitter_range, x_euler_jitter_range) if x_euler_jitter_range is not None else 0.0,
            self.rng.uniform(-y_euler_jitter_range, y_euler_jitter_range) if y_euler_jitter_range is not None else 0.0,
            self.rng.uniform(-z_euler_jitter_range, z_euler_jitter_range) if z_euler_jitter_range is not None else 0.0,
        ], dtype=np.float32)

        posDR = self.base_pos + pos_jitter

        base_R = R.from_quat(self.base_orn)
        delta_R = R.from_euler("xyz", euler_jitter)

        # local-frame rotation perturbation
        rotDR = base_R * delta_R
        ornDR = rotDR.as_quat()
        
        if get_matrix:
            matDR = np.eye(4, dtype=np.float32)
            matDR[:3, :3] = rotDR.as_matrix().astype(np.float32)
            matDR[:3, 3] = posDR
            return matDR

        return posDR, ornDR