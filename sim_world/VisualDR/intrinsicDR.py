import numpy as np


class IntrinsicDR:
    """Camera intrinsic / FOV domain randomization.

    Works for both pinhole K and KB4/fisheye K because it only perturbs
    fx, fy, cx, cy. Distortion coefficients are intentionally left unchanged.
    """

    def __init__(self, bullet_client=None, seed=42):
        self.bullet_client = bullet_client
        self.rng = np.random.default_rng(seed)
        self.base_K = None

    def sample_intrinsic_randomization(
        self,
        K=None,
        focal_scale_range=(1.0, 1.0),
        principal_jit_px=0.0,
        width=None,
        height=None,
    ):
        if K is None:
            raise ValueError("Base intrinsic K is not provided")

        self.base_K = np.asarray(K, dtype=np.float64)
        K_dr = self.base_K.copy()

        f_min, f_max = map(float, focal_scale_range)
        f_scale = self.rng.uniform(f_min, f_max)
        K_dr[0, 0] *= f_scale
        K_dr[1, 1] *= f_scale

        principal_jit_px = float(principal_jit_px)
        if principal_jit_px > 0.0:
            K_dr[0, 2] += self.rng.uniform(-principal_jit_px, principal_jit_px)
            K_dr[1, 2] += self.rng.uniform(-principal_jit_px, principal_jit_px)

        if width is not None:
            K_dr[0, 2] = np.clip(K_dr[0, 2], 0.0, float(width) - 1.0)
        if height is not None:
            K_dr[1, 2] = np.clip(K_dr[1, 2], 0.0, float(height) - 1.0)

        return K_dr
