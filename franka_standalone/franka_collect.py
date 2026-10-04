#!/usr/bin/env python3
"""
Franka workstation client — Keyboard teleoperation and multi-camera data collection.

Connects to franka_server.py via ZeroRPC, streams L515 RGB-D + fisheye images,
and records episodes to disk (numpy .npz per episode).

Usage:
    python franka_collect.py -o ./data
    python franka_collect.py -o ./data --robot_ip 192.168.3.2

Controls:
    W/S   Forward / Backward  (X)
    A/D   Left / Right        (Y)
    Q/E   Up / Down           (Z)
    J/L   Yaw left / right
    I/K   Pitch up / down
    U/O   Roll left / right
    Shift Hold for 3x speed
    Z     Close gripper
    X     Open gripper
    C     Start recording episode
    V     Stop recording & save episode
    B     Drop (discard) current episode
    H     Reset to home
    Esc   Quit
"""
import argparse
import os
import time
import threading
import subprocess
from pathlib import Path

import cv2
import numpy as np
import scipy.spatial.transform as st
import zerorpc
from pynput import keyboard as pynput_keyboard

from config import (
    ROBOT_IP, ROBOT_PORT, CONTROL_FREQUENCY,
    POS_SPEED, ROT_SPEED, MAX_GRIPPER_WIDTH,
    EE_HOME_POSE, HOME_MOVE_DURATION,
    FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION,
    KX_DEFAULT, KXD_DEFAULT,
    TX_FLANGE_TIP, TX_TIP_FLANGE,
    L515_SERIALS, FISHEYE_USB_ID, FISHEYE_RESOLUTION,
    FISHEYE_FOURCC,
)


# ======================== Pose Utilities ========================

def pose_to_mat(pose):
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:]).as_matrix()
    return mat

def mat_to_pose(mat):
    pos = mat[:3, 3]
    rot = st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot])


# ======================== ZeroRPC Client ========================

class FrankaClient:
    def __init__(self, ip, port):
        self._c = zerorpc.Client(heartbeat=20)
        self._c.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        return np.array(self._c.get_ee_pose())

    def get_tip_pose(self):
        flange = self.get_ee_pose()
        return mat_to_pose(pose_to_mat(flange) @ TX_FLANGE_TIP)

    def get_joint_positions(self):
        return np.array(self._c.get_joint_positions())

    def move_to_joint_positions(self, q, t):
        self._c.move_to_joint_positions(q.tolist(), float(t))

    def start_cartesian_impedance(self, Kx=None, Kxd=None):
        Kx = KX_DEFAULT if Kx is None else Kx
        Kxd = KXD_DEFAULT if Kxd is None else Kxd
        self._c.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())

    def update_desired_ee_pose(self, pose):
        flange = mat_to_pose(pose_to_mat(pose) @ TX_TIP_FLANGE)
        self._c.update_desired_ee_pose(flange.tolist())

    def terminate_current_policy(self):
        self._c.terminate_current_policy()

    def get_gripper_state(self):
        return self._c.get_gripper_state()

    def gripper_grasp(self, speed=0.2, force=40.0):
        return self._c.gripper_grasp(float(speed), float(force))

    def gripper_release(self, speed=0.2):
        return self._c.gripper_release(float(speed))

    def close(self):
        self._c.close()


# ======================== Cameras ========================

class L515Camera:
    """Thread-based Intel RealSense L515 camera reader.

    Fixed profile: color 960x540 @ 30fps, plus depth 640x480 when
    ``enable_depth=True`` (data collection); deploy uses color-only.
    """

    def __init__(self, serial: str, enable_depth: bool = True):
        import pyrealsense2 as rs
        self.serial = serial
        self._rs = rs
        self._enable_depth = enable_depth
        self._lock = threading.Lock()
        self._color = None
        self._depth = None

        self._pipeline = rs.pipeline()
        # Align is only meaningful when depth is streamed; color-only skips it.
        self._align = rs.align(rs.stream.color) if enable_depth else None

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if not cfg.can_resolve(self._pipeline):
            raise RuntimeError(
                f"[L515 {serial}] cannot open color 960x540"
                f"{' + depth 640x480' if enable_depth else ''} @ 30fps"
                f" — check USB 3.0 connection and firmware.")
        self._pipeline.start(cfg)
        mode = ('color 960x540 + depth 640x480' if enable_depth
                else 'color 960x540 (color-only)')
        print(f"[L515 {serial}] opened: {mode} @ 30fps")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                if self._enable_depth:
                    aligned = self._align.process(frames)
                    color = np.asarray(aligned.get_color_frame().get_data())
                    depth = np.asarray(aligned.get_depth_frame().get_data())
                    with self._lock:
                        self._color = color
                        self._depth = depth
                else:
                    # Color-only: no align, no depth (self._depth stays None).
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    color = np.asarray(color_frame.get_data())
                    with self._lock:
                        self._color = color
            except Exception as e:
                print(f"[L515 {self.serial}] frame error: {e}")

    def get(self):
        """Return (color_rgb_uint8, depth_uint16) or (None, None).

        depth is always None when the camera was opened with enable_depth=False.
        """
        with self._lock:
            return (self._color.copy() if self._color is not None else None,
                    self._depth.copy() if self._depth is not None else None)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._pipeline.stop()


class FisheyeCamera:
    """Thread-based USB UVC camera reader opened DIRECTLY at 640x480 YUYV.

    On this fisheye cam, MJPG @ 640x480 is a cropped/zoomed sensor ROI (narrow
    FOV), while YUYV @ 640x480 delivers the FULL fisheye FOV that matches the
    training reference. So we request the YUYV 640x480 mode DIRECTLY from the
    camera — no higher-native capture, no downscale, no resolution conversion.
    """

    def __init__(self, device, width=640, height=480, fps=30):
        self._lock = threading.Lock()
        self._color = None

        self._width = int(width)
        self._height = int(height)

        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")

        # FOURCC=YUYV must be set BEFORE the size: on this fisheye cam the YUYV
        # 640x480 mode is the FULL-FOV mode (the MJPG 640x480 mode is a cropped
        # ROI). V4L2's standard fourcc for this format is 'YUYV'.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FISHEYE_FOURCC))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        # Log the ACTUAL negotiated mode so a mismatch is visible on hardware.
        act_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        act_fourcc = self._decode_fourcc(self._cap.get(cv2.CAP_PROP_FOURCC))
        print(f"[Fisheye {device}] opened: {act_w}x{act_h} {act_fourcc} @ {fps}fps")
        if (act_w, act_h) != (self._width, self._height) or act_fourcc != FISHEYE_FOURCC:
            print(f"[Fisheye {device}][WARN] negotiated mode differs from requested "
                  f"{self._width}x{self._height} {FISHEYE_FOURCC} — FOV may not "
                  f"match the training/reference FOV (MJPG 640x480 is a cropped ROI).")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _decode_fourcc(fourcc_value):
        """Decode a CAP_PROP_FOURCC float/int into its 4-character string."""
        code = int(fourcc_value)
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue
            # VideoCapture.read() returns BGR regardless of the source YUYV, so
            # convert to RGB. No resize: the camera already delivers 640x480.
            with self._lock:
                self._color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get(self):
        """Return color_rgb_uint8 or None."""
        with self._lock:
            return self._color.copy() if self._color is not None else None

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


def find_video_device_by_usb_id(vendor_id: str, product_id: str):
    """Locate /dev/videoN for a USB camera by vendor:product."""
    try:
        out = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=5).stdout
        if not out.strip():
            return None
        current = None
        for line in out.split('\n'):
            if not line.startswith('\t'):
                current = line.strip()
            elif '/dev/video' in line and current:
                vpath = line.strip()
                try:
                    udev = subprocess.run(
                        ['udevadm', 'info', '--query=all', vpath],
                        capture_output=True, text=True, timeout=5)
                    if (udev.returncode == 0
                            and f'ID_VENDOR_ID={vendor_id}' in udev.stdout
                            and f'ID_MODEL_ID={product_id}' in udev.stdout):
                        return vpath
                except Exception:
                    pass
    except Exception as e:
        print(f"[Camera] find device error: {e}")
    return None


# ======================== Keyboard Handler ========================

class KeyboardTeleop:
    KEY_MAP = {
        'w': 'fwd', 's': 'bwd', 'a': 'left', 'd': 'right',
        'q': 'up',  'e': 'down',
        'j': 'yaw_l', 'l': 'yaw_r',
        'i': 'pit_u', 'k': 'pit_d',
        'u': 'rol_l', 'o': 'rol_r',
    }

    def __init__(self, pos_speed=0.08, rot_speed=0.3):
        self.pos_speed = pos_speed
        self.rot_speed = rot_speed
        self._states = {v: False for v in self.KEY_MAP.values()}
        self.gripper_closing = False
        self.gripper_opening = False
        self.shift_held = False
        self.quit_requested = False
        self.home_requested = False
        self.record_start = False
        self.record_stop = False
        self.record_drop = False

        self._listener = pynput_keyboard.Listener(
            on_press=self._press, on_release=self._release)
        self._listener.start()

    def _char(self, key):
        try:
            return key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            return None

    def _press(self, key):
        if key in (pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_r):
            self.shift_held = True
            return
        c = self._char(key)
        if c in self.KEY_MAP:
            self._states[self.KEY_MAP[c]] = True
        elif c == 'z':
            self.gripper_closing = True
        elif c == 'x':
            self.gripper_opening = True
        elif c == 'h':
            self.home_requested = True
        elif c == 'c':
            self.record_start = True
        elif c == 'v':
            self.record_stop = True
        elif c == 'b':
            self.record_drop = True
        if key == pynput_keyboard.Key.esc:
            self.quit_requested = True

    def _release(self, key):
        if key in (pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_r):
            self.shift_held = False
            return
        c = self._char(key)
        if c in self.KEY_MAP:
            self._states[self.KEY_MAP[c]] = False
        elif c == 'z':
            self.gripper_closing = False
        elif c == 'x':
            self.gripper_opening = False

    def get_velocity(self, dt):
        mult = 3.0 if self.shift_held else 1.0
        s, ps, rs = self._states, self.pos_speed * mult, self.rot_speed * mult
        dp = np.zeros(3)
        if s['fwd']:   dp[0] += ps * dt
        if s['bwd']:   dp[0] -= ps * dt
        if s['left']:  dp[1] += ps * dt
        if s['right']: dp[1] -= ps * dt
        if s['up']:    dp[2] += ps * dt
        if s['down']:  dp[2] -= ps * dt

        dr = np.zeros(3)
        if s['rol_l']: dr[0] -= rs * dt
        if s['rol_r']: dr[0] += rs * dt
        if s['pit_u']: dr[1] -= rs * dt
        if s['pit_d']: dr[1] += rs * dt
        if s['yaw_l']: dr[2] += rs * dt
        if s['yaw_r']: dr[2] -= rs * dt
        return dp, dr

    def stop(self):
        self._listener.stop()


# ======================== Episode Recorder ========================

class EpisodeRecorder:
    """Accumulates data in-memory; saves to .npz on finish."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ep_idx = self._next_episode_idx()
        self.reset()

    def _next_episode_idx(self):
        existing = list(self.output_dir.glob('episode_*'))
        if not existing:
            return 0
        indices = []
        for p in existing:
            try:
                indices.append(int(p.stem.split('_')[1]))
            except (ValueError, IndexError):
                pass
        return max(indices) + 1 if indices else 0

    def reset(self):
        self.timestamps = []
        self.actions = []          # (7,) pose6d + gripper
        self.robot_states = []     # (7,) pose6d + gripper_width
        self.joint_positions = []  # (7,)
        self.camera_colors = {}    # cam_name -> list of rgb frames
        self.camera_depths = {}    # cam_name -> list of depth frames

    @property
    def is_empty(self):
        return len(self.timestamps) == 0

    def add(self, timestamp, action, robot_state, joint_pos,
            camera_frames: dict):
        """
        camera_frames: {cam_name: {'color': rgb, 'depth': depth_or_None}}
        """
        self.timestamps.append(timestamp)
        self.actions.append(action.copy())
        self.robot_states.append(robot_state.copy())
        self.joint_positions.append(joint_pos.copy())
        for name, frames in camera_frames.items():
            self.camera_colors.setdefault(name, []).append(frames['color'])
            if frames.get('depth') is not None:
                self.camera_depths.setdefault(name, []).append(frames['depth'])

    def save(self):
        """Save episode as .npz + camera video directories."""
        ep_dir = self.output_dir / f'episode_{self._ep_idx:04d}'
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            ep_dir / 'robot_data.npz',
            timestamps=np.array(self.timestamps),
            actions=np.array(self.actions),
            robot_states=np.array(self.robot_states),
            joint_positions=np.array(self.joint_positions),
        )

        for cam_name, frames in self.camera_colors.items():
            cam_dir = ep_dir / cam_name
            cam_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(cam_dir / f'color_{i:05d}.jpg'), bgr)

        for cam_name, frames in self.camera_depths.items():
            cam_dir = ep_dir / cam_name
            cam_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                cam_dir / 'depth.npz',
                depth=np.array(frames))

        n = len(self.timestamps)
        print(f"[Recorder] Saved episode_{self._ep_idx:04d} ({n} frames) -> {ep_dir}")
        self._ep_idx += 1
        self.reset()

    def drop(self):
        print(f"[Recorder] Dropped episode ({len(self.timestamps)} frames)")
        self.reset()


# ======================== Main Loop ========================

def _interpolate_pose(start, end, alpha):
    """Linearly interpolate position + SLERP rotation between two poses."""
    pos = (1 - alpha) * start[:3] + alpha * end[:3]
    r0 = st.Rotation.from_rotvec(start[3:])
    r1 = st.Rotation.from_rotvec(end[3:])
    slerp = st.Slerp([0, 1], st.Rotation.concatenate([r0, r1]))
    rot = slerp(alpha).as_rotvec()
    return np.concatenate([pos, rot])


def reset_to_home(robot: FrankaClient, frequency: int = CONTROL_FREQUENCY):
    """Joint-space home -> EE home with smooth interpolation."""
    print("[Home] joints -> home ...")
    robot.terminate_current_policy()
    time.sleep(0.1)
    robot.move_to_joint_positions(FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION)
    robot.start_cartesian_impedance()

    start_pose = robot.get_tip_pose()
    dt = 1.0 / frequency
    n_steps = max(int(HOME_MOVE_DURATION * frequency), 1)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        waypoint = _interpolate_pose(start_pose, EE_HOME_POSE, alpha)
        robot.update_desired_ee_pose(waypoint)
        time.sleep(dt)
    print("[Home] Done.")


def main():
    parser = argparse.ArgumentParser(
        description='Franka keyboard teleop + multi-camera data collection')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for episodes')
    parser.add_argument('--robot_ip', default=ROBOT_IP)
    parser.add_argument('--robot_port', type=int, default=ROBOT_PORT)
    parser.add_argument('--frequency', type=int, default=CONTROL_FREQUENCY)
    parser.add_argument('--pos_speed', type=float, default=POS_SPEED)
    parser.add_argument('--rot_speed', type=float, default=ROT_SPEED)
    parser.add_argument('--no_l515', action='store_true',
                        help='Disable L515 cameras')
    parser.add_argument('--no_fisheye', action='store_true',
                        help='Disable fisheye camera')
    parser.add_argument('--no_depth', action='store_true',
                        help='Do not record depth images')
    parser.add_argument('--init_home', action='store_true', default=True)
    args = parser.parse_args()

    dt = 1.0 / args.frequency

    # ---- Cameras ----
    l515_cams = []
    if not args.no_l515:
        for serial in L515_SERIALS:
            try:
                print(f"[Camera] Starting L515 {serial} ...")
                l515_cams.append(L515Camera(serial))
            except Exception as e:
                print(f"[Camera] L515 {serial} failed: {e}")

    fisheye_cam = None
    if not args.no_fisheye and FISHEYE_USB_ID:
        vid, pid = FISHEYE_USB_ID.split(':')
        dev = find_video_device_by_usb_id(vid, pid)
        if dev:
            try:
                print(f"[Camera] Starting fisheye at {dev} ...")
                fisheye_cam = FisheyeCamera(dev, *FISHEYE_RESOLUTION)
            except Exception as e:
                print(f"[Camera] Fisheye failed: {e}")
        else:
            print(f"[Camera] Fisheye USB {FISHEYE_USB_ID} not found")

    # ---- Robot ----
    robot = FrankaClient(args.robot_ip, args.robot_port)
    teleop = KeyboardTeleop(pos_speed=args.pos_speed, rot_speed=args.rot_speed)
    recorder = EpisodeRecorder(args.output)

    print("=" * 60)
    print("  Franka Data Collector (上位机 + cameras)")
    print(f"  Server:    {args.robot_ip}:{args.robot_port}")
    print(f"  L515:      {len(l515_cams)} cameras")
    print(f"  Fisheye:   {'yes' if fisheye_cam else 'no'}")
    print(f"  Freq:      {args.frequency} Hz")
    print(f"  Output:    {args.output}")
    print("=" * 60)
    print("  C = start recording, V = stop & save, B = drop")
    print("  WASD = move, Z/X = gripper, H = home, Esc = quit")
    print("=" * 60)

    try:
        robot.start_cartesian_impedance()
        if args.init_home:
            reset_to_home(robot, args.frequency)

        target_pose = robot.get_tip_pose()
        gripper_pos = MAX_GRIPPER_WIDTH
        robot.gripper_release()
        time.sleep(1.0)  # let cameras warm up

        is_recording = False
        last_print = 0.0

        while not teleop.quit_requested:
            t0 = time.monotonic()
            ts = time.time()

            # ---- events ----
            if teleop.record_start and not is_recording:
                teleop.record_start = False
                is_recording = True
                recorder.reset()
                print("\n>>> RECORDING STARTED")
            if teleop.record_stop and is_recording:
                teleop.record_stop = False
                is_recording = False
                if not recorder.is_empty:
                    recorder.save()
                print(">>> RECORDING STOPPED")
            if teleop.record_drop:
                teleop.record_drop = False
                if is_recording:
                    is_recording = False
                    recorder.drop()
                    print(">>> EPISODE DROPPED")
            if teleop.home_requested and not is_recording:
                teleop.home_requested = False
                reset_to_home(robot, args.frequency)
                target_pose = robot.get_tip_pose()
                gripper_pos = MAX_GRIPPER_WIDTH

            # ---- teleop ----
            dpos, drot_xyz = teleop.get_velocity(dt)
            target_pose[:3] += dpos
            drot = st.Rotation.from_euler('xyz', drot_xyz)
            target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
            robot.update_desired_ee_pose(target_pose)

            # ---- gripper ----
            if teleop.gripper_closing and gripper_pos > 0.01:
                robot.gripper_grasp()
                gripper_pos = 0.0
            elif teleop.gripper_opening and gripper_pos < 0.07:
                robot.gripper_release()
                gripper_pos = MAX_GRIPPER_WIDTH

            # ---- cameras ----
            cam_frames = {}
            vis_imgs = []
            target_h = 320

            for i, cam in enumerate(l515_cams):
                color, depth = cam.get()
                if color is not None:
                    cam_frames[f'l515_{i}'] = {
                        'color': color,
                        'depth': depth if not args.no_depth else None,
                    }
                    # build vis image
                    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]
                    scale = target_h / h
                    bgr = cv2.resize(bgr, (int(w * scale), target_h))
                    cv2.putText(bgr, f'L515-{i}', (5, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    vis_imgs.append(bgr)

            if fisheye_cam is not None:
                color = fisheye_cam.get()
                if color is not None:
                    cam_frames['fisheye'] = {'color': color, 'depth': None}
                    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]
                    scale = target_h / h
                    bgr = cv2.resize(bgr, (int(w * scale), target_h))
                    cv2.putText(bgr, 'Fisheye', (5, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    vis_imgs.append(bgr)

            # ---- record ----
            if is_recording and cam_frames:
                action = np.zeros(7)
                action[:6] = target_pose
                action[6] = gripper_pos

                joints = robot.get_joint_positions()
                gripper_w = gripper_pos
                robot_state = np.zeros(7)
                robot_state[:6] = robot.get_tip_pose()
                robot_state[6] = gripper_w

                recorder.add(ts, action, robot_state, joints, cam_frames)

            # ---- visualize ----
            if vis_imgs:
                canvas = np.hstack(vis_imgs)
                rec_color = (0, 0, 255) if is_recording else (200, 200, 200)
                label = f'Ep {recorder._ep_idx}'
                if is_recording:
                    label += f' [REC {len(recorder.timestamps)} frames]'
                cv2.putText(canvas, label, (10, target_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)

                pos_txt = (f'pos=[{target_pose[0]:.3f},{target_pose[1]:.3f},'
                           f'{target_pose[2]:.3f}]  gripper={gripper_pos:.3f}m')
                cv2.putText(canvas, pos_txt, (10, target_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow('Franka Collect', canvas)
                if cv2.waitKey(1) == 27:
                    break

            # ---- terminal print (2 Hz) ----
            now = time.monotonic()
            if now - last_print > 0.5:
                p = target_pose
                rec_flag = ' [REC]' if is_recording else ''
                print(f"\rpos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]  "
                      f"gripper={gripper_pos:.3f}m{rec_flag}   ", end='', flush=True)
                last_print = now

            # ---- frequency regulation ----
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nShutting down ...")
        teleop.stop()
        cv2.destroyAllWindows()

        if is_recording and not recorder.is_empty:
            print("[Recorder] Saving in-progress episode ...")
            recorder.save()

        for cam in l515_cams:
            cam.stop()
        if fisheye_cam:
            fisheye_cam.stop()

        try:
            robot.terminate_current_policy()
        except Exception:
            pass
        robot.close()
        print("Done.")


if __name__ == '__main__':
    main()
