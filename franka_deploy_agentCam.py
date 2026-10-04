#!/usr/bin/env python3
"""
Franka sim-to-real deployment of a diffusion policy trained in PyBullet.

Loads a checkpoint trained on PickUpEnv (`FPSA_AgentOnly_workspace`)
and runs closed-loop control on the real Franka:
    - obs.agentview_image  <-  Intel RealSense L515 (serial = L515_SERIALS[0])
    - obs.robot0_eef_pos   <-  Franka tip xyz (base frame)
    - obs.robot0_eef_quat  <-  Franka tip orientation (xyzw)
    - obs.robot0_gripper_qpos <- binary state (1,): 1=open, 0=closed
    - action               ->  8-dim [xyz, qx,qy,qz,qw, gripper_per_finger]

Architecture
------------
NUC (robot-side computer):
    Terminal 1:  bash launch_polymetis.sh 10.168.1.200
    Terminal 2:  python franka_server.py --gripper_robot_ip 10.168.1.200

Workstation (control-side computer, this script):
    python franka_deploy.py \
        -c GeoBridgeCheckpoints/checkpoints/latest.ckpt \
        -o data/real_deploy

Controls
--------
    S      Start one rollout (will block until pressed)
    Esc    Quit
    P      Pause / resume the action loop
    H      Stop rollout, go back to sim-home (between rollouts)

Important
---------
The L515 must be physically mounted so its view roughly matches the sim
`get_agentview_image` extrinsic (see `PickUpSim.get_agentview_image`).
Otherwise the visual domain gap will dominate.
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import queue
import threading
import time
from collections import deque

import click
import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from pynput import keyboard as pynput_keyboard
from icecream import ic
# Make `franka_standalone/{config,franka_collect}.py` importable from project root
ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / 'franka_standalone'))

from config import (  # noqa: E402
    ROBOT_IP, ROBOT_PORT, CONTROL_FREQUENCY,
    MAX_GRIPPER_WIDTH,
    JOINTS_HOME_DURATION,
    L515_SERIALS,
)
from franka_collect import FrankaClient, L515Camera  # noqa: E402

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.common.pose_trajectory_interpolator import (  # noqa: E402
    PoseTrajectoryInterpolator,
)
# Re-import to register the workspace target class for hydra
import diffusion_policy.workspace.FPSA_workspace  # noqa: F401,E402


# ======================== Sim-home joints ========================
# Matches PickUpSim.py `jointPositions[:7]`. Using these instead of
# franka_standalone EE_HOME_POSE so the policy starts from a state it has
# actually seen during training.
SIM_HOME_JOINTS = np.array([
    -0.0768761337796847,
     0.1692503838434554,
    -0.5782208367480097,
    -1.4272947420449444,
     0.0557141137601706,
     1.5859262946844102,
     0.0,
], dtype=np.float64)

# ======================== Helpers ========================

def resize_rgb_224(rgb_uint8):
    """960x540 (or any) -> 224x224 stretched, matches sim `utility.resize_rgb`."""
    if rgb_uint8.shape[0] == 224 and rgb_uint8.shape[1] == 224:
        return rgb_uint8
    return cv2.resize(rgb_uint8, (224, 224), interpolation=cv2.INTER_AREA)


# --- Binary gripper convention (matches PickUpSim) ---
#   obs.robot0_gripper_qpos : shape (1,), 1.0 = open, 0.0 = closed
#   action[7]               : shape (), 1.0 = open, 0.0 = closed
# Sim derives the obs state by thresholding the *mean per-finger* width at
# 0.8 * (open_width + closed_width) = 0.8 * 0.04 = 0.032 m. The real Franka
# reports *total* width (0..0.08), so we halve it before thresholding.
GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0
GRIPPER_STATE_THRESHOLD_WIDTH = 0.8 * (GRIPPER_OPEN_WIDTH + GRIPPER_CLOSED_WIDTH)


def gripper_width_to_state(total_width):
    """Real Franka total width (0..0.08) -> binary state (1=open, 0=closed)."""
    per_finger = float(total_width) / 2.0
    return 1.0 if per_finger >= GRIPPER_STATE_THRESHOLD_WIDTH else 0.0


def tip_pose_to_obs(tip_pose_rotvec, gripper_width):
    """Franka tip pose (xyz + rotvec) + gripper width -> sim-format obs."""
    gripper_state = gripper_width_to_state(gripper_width)
    pos = tip_pose_rotvec[:3].astype(np.float32)
    quat_xyzw = st.Rotation.from_rotvec(tip_pose_rotvec[3:]).as_quat().astype(np.float32)
    return {
        'robot0_eef_pos': pos,
        'robot0_eef_quat': quat_xyzw,
        # shape (1,) to match shape_meta robot0_gripper_qpos: [1]
        'robot0_gripper_qpos': np.array([gripper_state], dtype=np.float32),
    }


def action_to_targets(action8):
    """[xyz, qx,qy,qz,qw, gripper_state] -> ([xyz,rotvec], gripper_state).

    gripper_state is binary-ish (model outputs ~1.0 open / ~0.0 closed).
    """
    pos = action8[:3].astype(np.float64)
    quat_xyzw = action8[3:7].astype(np.float64)
    n = np.linalg.norm(quat_xyzw)
    if n < 1e-8:
        # degenerate; fall back to identity rotation
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        quat_xyzw = quat_xyzw / n
    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
    tip_pose = np.concatenate([pos, rotvec])
    return tip_pose, float(action8[7])


# ======================== Async Inference Worker ========================

class InferenceWorker(threading.Thread):
    """Background thread that runs ``policy.predict_action`` so the main
    thread can keep streaming pose targets to the Franka without gaps.

    NOTE: torch releases the GIL during the heavy GPU work, and **this worker
    never touches zerorpc**, so the gevent hub stays exclusively on the main
    thread (which is the whole point — see ``franka_interpolation_controller``
    for why the reference uses ``mp.Process`` for the controller side).
    """

    def __init__(self, policy, device):
        super().__init__(daemon=True, name='InferenceWorker')
        self.policy = policy
        self.device = device
        self.obs_q: 'queue.Queue[dict]' = queue.Queue(maxsize=1)
        self.act_q: 'queue.Queue[tuple]' = queue.Queue()  # (sched_t, actions, inf_ms)
        # NOTE: avoid the name ``_stop`` — that shadows Thread._stop().
        self._stop_event = threading.Event()

    def submit(self, obs_np: dict) -> bool:
        """Non-blocking: try to enqueue a new obs batch. Returns False if a
        previous request is still in flight (caller may choose to skip)."""
        try:
            self.obs_q.put_nowait(obs_np)
            return True
        except queue.Full:
            return False

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                obs_np = self.obs_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                obs_t = dict_apply(
                    obs_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                t_inf = time.monotonic()
                with torch.no_grad():
                    out = self.policy.predict_action(obs_t)
                actions = out['action'][0].detach().cpu().numpy()
                inf_ms = (time.monotonic() - t_inf) * 1000.0
                self.act_q.put((time.monotonic(), actions, inf_ms))
            except Exception as e:
                print(f'[InferenceWorker] error: {e!r}')


# ======================== Interpolator helpers (main thread) ========================

def _new_interp_from_robot(robot: FrankaClient):
    curr_pose = np.asarray(robot.get_tip_pose(), dtype=np.float64)
    curr_t = time.monotonic()
    return PoseTrajectoryInterpolator(
        times=np.array([curr_t]),
        poses=np.array([curr_pose]),
    ), curr_t


def _schedule_chunk(pose_interp: PoseTrajectoryInterpolator,
                    last_waypoint_time: float,
                    sched_t: float, actions: np.ndarray,
                    action_dt: float,
                    max_pos_speed: float, max_rot_speed: float):
    """Insert all actions in this chunk into the interpolator.

    Returns: (new pose_interp, updated last_waypoint_time, gripper_targets)
    """
    grip_targets = np.zeros(len(actions), dtype=np.float64)
    for i, a in enumerate(actions):
        tip_target, grip_target = action_to_targets(a)
        target_t = sched_t + (i + 1) * action_dt
        pose_interp = pose_interp.schedule_waypoint(
            pose=tip_target,
            time=target_t,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            curr_time=sched_t,
            last_waypoint_time=last_waypoint_time,
        )
        last_waypoint_time = max(last_waypoint_time, target_t)
        grip_targets[i] = grip_target
    return pose_interp, last_waypoint_time, grip_targets


# ======================== Keyboard ========================

class KeyMonitor:
    """Pynput listener for Esc / S / P / H, non-blocking."""

    def __init__(self):
        self.quit_requested = False
        self.start_requested = False
        self.stop_requested = False
        self.pause = False
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _char(self, key):
        try:
            return key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            return None

    def _on_press(self, key):
        if key == pynput_keyboard.Key.esc:
            self.quit_requested = True
            return
        c = self._char(key)
        if c == 's':
            self.start_requested = True
        elif c == 'h':
            self.stop_requested = True
        elif c == 'p':
            self.pause = not self.pause

    def stop(self):
        self._listener.stop()


# ======================== Robot helpers ========================

def goto_sim_home(robot: FrankaClient):
    """Terminate any controller, move joints to SIM_HOME_JOINTS, then restart impedance."""
    print('[Home] terminating policy, moving to sim home ...')
    try:
        robot.terminate_current_policy()
    except Exception:
        pass
    time.sleep(0.2)
    robot.move_to_joint_positions(SIM_HOME_JOINTS, JOINTS_HOME_DURATION)
    # extra settling time
    time.sleep(0.5)
    robot.start_cartesian_impedance()
    time.sleep(0.2)
    print('[Home] sim home reached, impedance running.')


def wait_for_camera(cam: L515Camera, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        c, _ = cam.get()
        if c is not None:
            return c
        time.sleep(0.05)
    raise RuntimeError('Timed out waiting for L515 first frame.')


# ======================== Inference loop ========================

def _draw_vis(color_rgb, step, max_steps, is_open, gripper_width, tip):
    """Helper: draw debug overlay onto a copy of the latest camera frame and show it."""
    vis = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(vis, f'step {step}/{max_steps}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis,
                f"grip={'OPEN' if is_open else 'CLOSE'} w={gripper_width:.3f}m",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    p = tip
    cv2.putText(vis,
                f"tip=[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]",
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imshow('Franka Deploy (L515-0)', vis)


def run_episode(policy, robot, cam, keys: KeyMonitor,
                n_obs_steps, n_action_steps,
                exec_horizon,
                frequency, max_steps,
                gripper_state_threshold,
                device,
                control_hz: float,
                max_pos_speed: float,
                max_rot_speed: float,
                print_actions: bool = False,
                episode_dir: pathlib.Path = None):
    """Run one closed-loop rollout.

    Design:
      - Main thread: tight loop @ ``control_hz``. Each tick reads the
        pose-trajectory interpolator at the current monotonic time and sends
        the result via zerorpc. Also samples obs at policy rate and issues
        gripper commands. All zerorpc calls live here so the gevent hub stays
        on this (main) thread.
      - Inference worker thread: does the slow policy.predict_action work and
        returns chunks via ``act_q``. Never touches zerorpc.
    """
    action_dt = 1.0 / frequency
    control_dt = 1.0 / control_hz
    chunk_horizon = min(exec_horizon, n_action_steps)
    chunk_duration = chunk_horizon * action_dt  # wall-clock budget per chunk
    policy.reset()

    # ---- initial gripper state ----
    robot.gripper_release()
    time.sleep(0.6)
    gs = robot.get_gripper_state() or {}
    gripper_width = float(gs.get('width', MAX_GRIPPER_WIDTH))
    is_open = gripper_width > 0.04

    # ---- prime obs buffers from current robot/camera state ----
    color = wait_for_camera(cam)
    img224 = resize_rgb_224(color)
    #cv2.imwrite('imagedebug.jpg', cv2.cvtColor(img224, cv2.COLOR_RGB2BGR))

    tip_now = np.asarray(robot.get_tip_pose(), dtype=np.float64)
    obs0 = tip_pose_to_obs(tip_now, gripper_width)
    img_buf  = deque([img224.copy()                  for _ in range(n_obs_steps)], maxlen=n_obs_steps)
    pos_buf  = deque([obs0['robot0_eef_pos'].copy()  for _ in range(n_obs_steps)], maxlen=n_obs_steps)
    quat_buf = deque([obs0['robot0_eef_quat'].copy() for _ in range(n_obs_steps)], maxlen=n_obs_steps)
    grip_buf = deque([obs0['robot0_gripper_qpos'].copy() for _ in range(n_obs_steps)],
                     maxlen=n_obs_steps)

    saved_actions, saved_states, saved_imgs = [], [], []

    # ---- interpolator anchored at current pose ----
    pose_interp, t0 = _new_interp_from_robot(robot)
    last_waypoint_time = t0

    # ---- start inference worker ----
    worker = InferenceWorker(policy, device)
    worker.start()

    # Cadence trackers (all monotonic seconds)
    next_send_t = t0
    next_obs_t  = t0 + action_dt   # already have obs at t0 in the buffer
    next_inf_t  = t0               # request first inference right away
    chunks_done = 0
    step = 0
    iter_idx = 0
    t_episode = t0

    try:
        while step < max_steps and not keys.quit_requested:
            if keys.stop_requested:
                break
            if keys.pause:
                time.sleep(0.05)
                # advance schedule so we don't try to send a flood of catch-up packets
                t_now = time.monotonic()
                next_send_t = max(next_send_t, t_now)
                continue

            t_now = time.monotonic()

            # ============== 1. Stream interpolated pose @ control_hz ==============
            if t_now >= next_send_t:
                tip_target = pose_interp(t_now)
                try:
                    robot.update_desired_ee_pose(tip_target)
                except Exception as e:
                    # polymetis controller died (e.g. joint-limit fault); try to recover
                    print(f'[Deploy] update_desired_ee_pose failed: {e!r}')
                    print('[Deploy] restarting cartesian impedance and re-anchoring ...')
                    try:
                        robot.start_cartesian_impedance()
                    except Exception as e2:
                        print(f'[Deploy] start_cartesian_impedance failed: {e2!r}; aborting.')
                        break
                    time.sleep(0.2)
                    pose_interp, t_anchor = _new_interp_from_robot(robot)
                    last_waypoint_time = t_anchor
                    next_send_t = time.monotonic()
                    # drop any pending inference result; force a fresh inference
                    try:
                        while True:
                            worker.act_q.get_nowait()
                    except queue.Empty:
                        pass
                    next_inf_t = time.monotonic()
                    continue
                next_send_t += control_dt
                # if we've fallen behind (e.g. main thread stalled), don't try to
                # catch up by hammering — just resync to "now"
                if next_send_t < t_now - control_dt:
                    next_send_t = t_now + control_dt

            # ============== 2. Sample fresh obs @ action_dt ==============
            if t_now >= next_obs_t:
                color, _ = cam.get()
                if color is not None:
                    img224 = resize_rgb_224(color)
                    img_buf.append(img224)
                tip_now = np.asarray(robot.get_tip_pose(), dtype=np.float64)
                gs2 = robot.get_gripper_state() or {}
                gripper_width = float(gs2.get('width', gripper_width))
                obs_now = tip_pose_to_obs(tip_now, gripper_width)
                pos_buf.append(obs_now['robot0_eef_pos'])
                quat_buf.append(obs_now['robot0_eef_quat'])
                grip_buf.append(obs_now['robot0_gripper_qpos'])
                # ic(obs_now['robot0_gripper_qpos'])
                next_obs_t += action_dt
                if next_obs_t < t_now - action_dt:
                    next_obs_t = t_now + action_dt

            # ============== 3. Submit next inference request ==============
            if t_now >= next_inf_t:
                agentview_image = np.stack(list(img_buf), axis=0)
                agentview_image = np.moveaxis(agentview_image, -1, -3).astype(np.float32) / 255.0
                np_obs = {
                    'agentview_image': agentview_image,
                    'robot0_eef_pos':  np.stack(list(pos_buf),  axis=0).astype(np.float32),
                    'robot0_eef_quat': np.stack(list(quat_buf), axis=0).astype(np.float32),
                    'robot0_gripper_qpos': np.stack(list(grip_buf), axis=0).astype(np.float32),
                }
                if worker.submit(np_obs):
                    next_inf_t = t_now + chunk_duration

            # ============== 4. Consume inference results ==============
            try:
                sched_t, actions_full, inf_ms = worker.act_q.get_nowait()
                actions = actions_full[:chunk_horizon]

                pose_interp, last_waypoint_time, grip_targets = _schedule_chunk(
                    pose_interp, last_waypoint_time,
                    sched_t, actions, action_dt,
                    max_pos_speed, max_rot_speed,
                )

                # gripper from the first action of the new chunk.
                # Binary convention: action >= threshold => open(release),
                # action < threshold => closed(grasp).
                grip_cmd = float(grip_targets[0])
                if grip_cmd < gripper_state_threshold and is_open:
                    robot.gripper_grasp()
                    is_open = False
                elif grip_cmd >= gripper_state_threshold and not is_open:
                    robot.gripper_release()
                    is_open = True

                # bookkeeping
                if episode_dir is not None:
                    for a in actions:
                        saved_actions.append(a.astype(np.float32))
                        saved_states.append(np.concatenate(
                            [tip_now.astype(np.float32),
                             np.array([gripper_width], dtype=np.float32)]))
                        saved_imgs.append(img224.copy())

                step += chunk_horizon
                chunks_done += 1
                print(f'  [chunk {chunks_done}] step={step}/{max_steps}  '
                      f'inf={inf_ms:.0f}ms  horizon={chunk_horizon}  '
                      f'dur={chunk_duration*1000:.0f}ms')
            except queue.Empty:
                pass

            # ============== 5. Vis (every few iterations) ==============
            if iter_idx % max(int(control_hz // 30), 1) == 0 and color is not None:
                _draw_vis(color, step, max_steps, is_open, gripper_width, tip_now)
                if cv2.waitKey(1) == 27:
                    keys.quit_requested = True

            # ============== 6. Sleep until next send ==============
            iter_idx += 1
            slack = next_send_t - time.monotonic()
            if slack > 0.0005:
                time.sleep(min(slack, 0.005))

    finally:
        worker.stop()
        worker.join(timeout=2.0)

    duration = time.monotonic() - t_episode
    print(f'[Episode] {step} steps in {duration:.1f}s '
          f'({step / max(duration, 1e-6):.1f}Hz effective, '
          f'{chunks_done} chunks)')

    # ---- save ----
    if episode_dir is not None and len(saved_actions) > 0:
        episode_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            episode_dir / 'rollout.npz',
            actions=np.array(saved_actions),
            robot_states=np.array(saved_states),
        )
        img_dir = episode_dir / 'agentview'
        img_dir.mkdir(exist_ok=True)
        for i, im in enumerate(saved_imgs):
            cv2.imwrite(str(img_dir / f'{i:05d}.jpg'),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f'[Episode] saved -> {episode_dir}')

    return step


# ======================== Main ========================

@click.command()
@click.option('-c', '--checkpoint', required=True, type=click.Path(exists=True),
              help='Path to .ckpt')
@click.option('-o', '--output_dir', default='data/real_deploy',
              type=click.Path(),
              help='Where to save rollout .npz files (set --no_save to skip)')
@click.option('-d', '--device', default='cuda:0')
@click.option('--robot_ip', default=ROBOT_IP)
@click.option('--robot_port', default=ROBOT_PORT, type=int)
@click.option('--frequency', default=160, type=int,
              help='Control loop Hz (default 160 Hz)')
@click.option('--max_steps', default=4000, type=int)
@click.option('--exec_horizon', default=8, type=int,
              help='How many of the n_action_steps predicted actions to '
                   'execute before re-inferring. 0 = use n_action_steps from '
                   'cfg (full chunk). Lower (e.g. 2-4) = more reactive but more '
                   'compute.')
@click.option('--control_hz', default=1000.0, type=float,
              help='High-frequency pose streaming rate (Hz). The streamer '
                   'samples the pose-trajectory interpolator at this rate and '
                   'sends update_desired_ee_pose to the Franka.')
@click.option('--max_pos_speed', default=0.1, type=float,
              help='Cap on translation speed between waypoints (m/s). Prevents '
                   'big policy jumps from causing joint-velocity faults.')
@click.option('--max_rot_speed', default=0.2, type=float,
              help='Cap on rotation speed between waypoints (rad/s).')
@click.option('--gripper_threshold', default=0.5, type=float,
              help='binary gripper-state threshold; action < thr => grasp '
                   '(close), action >= thr => release (open). Model outputs '
                   '~1.0=open / ~0.0=closed, so 0.5 is the natural cutoff.')
@click.option('--n_episodes', default=1, type=int)
@click.option('--no_save', is_flag=True, default=False)
@click.option('--auto_start', is_flag=True, default=False,
              help='Start first episode immediately without waiting for S')
def main(checkpoint, output_dir, device, robot_ip, robot_port,
         frequency, max_steps, exec_horizon,
         control_hz, max_pos_speed, max_rot_speed,
         gripper_threshold, n_episodes,
         no_save, auto_start):

    output_dir = pathlib.Path(output_dir).resolve()
    if not no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ----
    print('=' * 60)
    print(f'[Deploy] loading checkpoint: {checkpoint}')
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill,
                         map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device_t = torch.device(device)
    policy.to(device_t)
    policy.eval()

    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    if exec_horizon <= 0:
        exec_horizon = n_action_steps
    exec_horizon = min(exec_horizon, n_action_steps)
    print(f'[Deploy] n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}, '
          f'exec_horizon={exec_horizon}, policy_rate={frequency}Hz, '
          f'stream={control_hz}Hz, max_steps={max_steps}')
    print(f'[Deploy] speed caps: pos={max_pos_speed} m/s, rot={max_rot_speed} rad/s')

    # ---- Camera (only L515 #0) ----
    serial = L515_SERIALS[0]
    print(f'[Camera] opening L515 {serial} ...')
    cam = L515Camera(serial)

    # ---- Robot ----
    print(f'[Robot] connecting to {robot_ip}:{robot_port}')
    robot = FrankaClient(robot_ip, robot_port)

    keys = KeyMonitor()
    print('=' * 60)
    print('  S = start episode | H = abort rollout & home | Esc = quit')
    print('  P = pause / resume')
    print('=' * 60)

    try:
        for ep in range(n_episodes):
            print(f'\n[Episode {ep+1}/{n_episodes}] resetting to sim home ...')
            goto_sim_home(robot)

            if not auto_start or ep > 0:
                print('  -> Position the object, then press S to begin. Esc to quit.')
                while not keys.start_requested and not keys.quit_requested:
                    time.sleep(0.05)
                if keys.quit_requested:
                    break
                keys.start_requested = False
            keys.stop_requested = False
            print(f'[Episode {ep+1}] running ...')

            episode_dir = None
            if not no_save:
                episode_dir = output_dir / f'episode_{ep:04d}'

            run_episode(
                policy=policy,
                robot=robot,
                cam=cam,
                keys=keys,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                exec_horizon=exec_horizon,
                frequency=frequency,
                max_steps=max_steps,
                gripper_state_threshold=gripper_threshold,
                device=device_t,
                control_hz=control_hz,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                episode_dir=episode_dir,
            )

            if keys.quit_requested:
                break

    except KeyboardInterrupt:
        print('\n[Deploy] KeyboardInterrupt')
    finally:
        print('\n[Deploy] shutting down ...')
        keys.stop()
        cv2.destroyAllWindows()
        try:
            robot.terminate_current_policy()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        print('[Deploy] done.')


if __name__ == '__main__':
    main()
