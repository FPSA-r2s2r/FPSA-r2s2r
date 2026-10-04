"""
Usage:
python eval.py --checkpoint data/outputs/<run>/checkpoints/latest.ckpt -o data/FPSA_eval
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.workspace.PhyDomain_Unet_workspace import PhyDomainWorkspace
# from diffusion_policy.env_runner.PhyDomain_Unet_runner import PhyDomainImageRunner

from diffusion_policy.workspace.FPSA_workspace import FPSAImageWorkspace
from diffusion_policy.env_runner.FPSA_PickUpEnv_debug_runner import PickUpEnvDebugRunner

import numpy as np

def to_jsonable(x):
    if isinstance(x, wandb.sdk.data_types.video.Video):
        return x._path
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: FPSAImageWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = PickUpEnvDebugRunner(
                _seed = 30000,
                randomize_image_noise  = True,
                randomize_lighting = True,
                randomize_objpose  = True,
                randomize_distractors  = True,
                randomize_outlscene = True,
                randomize_plane_height = True,
                randomize_campose = True,
                n_obs_steps=2,
                n_action_steps=8,
                output_dir=output_dir,
                )
    runner_log = env_runner.run(policy)
    
    # # dump log to json
    # json_log = {k: to_jsonable(v) for k, v in runner_log.items()}

    # out_path = os.path.join(output_dir, 'eval_log.json')
    # with open(out_path, 'w') as f:
    #     json.dump(json_log, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
