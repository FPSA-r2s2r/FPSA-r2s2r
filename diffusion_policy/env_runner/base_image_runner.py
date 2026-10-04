from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()

    def close(self):
        """Close an owned vector environment, if the runner still has one."""
        env = getattr(self, 'env', None)
        if env is not None:
            env.close()
            self.env = None
