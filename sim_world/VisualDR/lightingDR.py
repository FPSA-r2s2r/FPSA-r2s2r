import numpy as np


class LightingDR(object):
    """Lighting domain randomization for PyBullet camera rendering."""

    def __init__(self, bullet_client, seed=42):
        self.bullet_client = bullet_client
        self.lighting_rng = np.random.default_rng(seed)
        self.light_cfg = self.default_light_cfg()

    @staticmethod
    def default_light_cfg():
        return {
            "num_lights": 1,
            "lightDirection": [0.0, -1.0, 0.8],
            "lightColor": [1.0, 1.0, 1.0],
            "lightDistance": 0.8,
            "lightAmbientCoeff": 0.55,
            "lightDiffuseCoeff": 0.75,
            "lightSpecularCoeff": 0.25,
            "shadow": 1,
        }

    def reset_to_default(self):
        self.light_cfg = self.default_light_cfg()
        return self.light_cfg

    def sample_lighting_randomization(self):
        rng = self.lighting_rng

        num_lights = int(rng.integers(1, 4))
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        elevation = rng.uniform(np.deg2rad(25.0), np.deg2rad(75.0))

        light_dir = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ], dtype=np.float32)

        self.light_cfg = {
            "num_lights": num_lights,  # metadata for now; PyBullet getCameraImage uses one light direction.
            "lightDirection": light_dir.tolist(),
            "lightColor": rng.uniform(0.45, 1.0, size=3).tolist(),
            "lightDistance": float(rng.uniform(0.8, 4.0)),
            "lightAmbientCoeff": float(rng.uniform(0.25, 0.65)),
            "lightDiffuseCoeff": float(rng.uniform(0.45, 1.0)),
            "lightSpecularCoeff": float(rng.uniform(0.05, 0.6)),
            "shadow": 1,
        }
        return self.light_cfg
