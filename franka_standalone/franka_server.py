#!/usr/bin/env python3
"""
Franka robot-side Step 2 — ZeroRPC server (runs on the NUC with Polymetis).

Prerequisite: polymetis must already be running (Step 1).
  Terminal 1:  bash launch_polymetis.sh          # or launch_robot.py ...
  Terminal 2:  python franka_server.py           # this script

Exposes robot arm + Panda gripper control over the network.
The upper-level client (franka_control.py / franka_collect.py) connects via ZeroRPC.

Usage:
    python franka_server.py
    python franka_server.py --robot_ip localhost --gripper_robot_ip 10.168.1.200 --port 4242
"""
import argparse
import time
import threading

import numpy as np
import scipy.spatial.transform as st
import torch
import grpc
import zerorpc
from polymetis import RobotInterface


# ======================== Gripper (libfranka) ========================

class PandaPyGripper:
    """Non-blocking Panda gripper control via panda-python libfranka bindings."""

    def __init__(self, robot_ip: str):
        from panda_py import libfranka
        self.robot_ip = robot_ip
        self._width = 0.08
        self._max_width = 0.08
        self._gripper = None
        self._lock = threading.Lock()

        print(f"[Gripper] Connecting to {robot_ip} via panda-python...")
        self._gripper = libfranka.Gripper(robot_ip)

        print("[Gripper] Performing initial homing...")
        self._gripper.homing()

        state = self._gripper.read_once()
        self._width = state.width
        self._max_width = state.max_width
        print(f"[Gripper] Ready. Width: {self._width:.4f}m, Max: {self._max_width:.4f}m")

    def get_state(self):
        try:
            with self._lock:
                state = self._gripper.read_once()
            self._width = state.width
            self._max_width = state.max_width
            return {
                'width': state.width,
                'max_width': state.max_width,
                'is_grasped': state.is_grasped,
            }
        except Exception as e:
            print(f"[Gripper] get_state error: {e}")
            return {'width': self._width, 'max_width': self._max_width, 'is_grasped': False}

    def grasp(self, speed=0.1, force=60.0):
        def _do():
            try:
                with self._lock:
                    try:
                        self._gripper.stop()
                    except Exception:
                        pass
                    self._gripper.grasp(0.0, speed, force, 0.08, 0.08)
            except Exception as e:
                print(f"[Gripper] grasp error: {e}")
        threading.Thread(target=_do, daemon=True).start()
        self._width = 0.0
        return True

    def release(self, speed=0.1):
        def _do():
            try:
                with self._lock:
                    try:
                        self._gripper.stop()
                    except Exception:
                        pass
                    self._gripper.move(self._max_width, speed)
            except Exception as e:
                print(f"[Gripper] release error: {e}")
        threading.Thread(target=_do, daemon=True).start()
        self._width = self._max_width
        return True

    def stop(self):
        with self._lock:
            try:
                self._gripper.stop()
            except Exception:
                pass

    def close(self):
        pass


# ======================== Server Interface ========================

class FrankaInterface:
    """ZeroRPC server exposing robot arm + gripper.
    Matches the API of launch_franka_interface_server.py exactly."""

    def __init__(self, robot_ip='localhost', gripper_robot_ip='172.16.0.2'):
        print(f"Connecting to robot at {robot_ip}...")
        self.robot = RobotInterface(ip_address=robot_ip)
        self._controller_running = False

        print(f"Connecting to gripper at {gripper_robot_ip} (via panda-python)...")
        try:
            self.gripper = PandaPyGripper(robot_ip=gripper_robot_ip)
            self.gripper_connected = True
            print("Gripper connected via panda-python!")
        except Exception as e:
            print(f"Warning: Failed to connect gripper: {e}")
            print("Gripper control will be disabled.")
            self.gripper = None
            self.gripper_connected = False

        self.gripper_width = 0.08
        self.gripper_max_width = 0.08

    # ========== Robot Arm Methods ==========

    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()

    def move_to_joint_positions(self, positions, time_to_go):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )

    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )
        self._controller_running = True
        print(f"[Server] Cartesian impedance started with Kx={Kx}, Kxd={Kxd}")

    def start_joint_impedance(self, Kq=None, Kqd=None):
        kwargs = {}
        if Kq is not None:
            kwargs['Kq'] = torch.Tensor(Kq)
        if Kqd is not None:
            kwargs['Kqd'] = torch.Tensor(Kqd)
        self.robot.start_joint_impedance(**kwargs)
        self._controller_running = True
        print("[Server] Joint impedance started")

    def update_desired_ee_pose(self, pose):
        if not self._controller_running:
            print("Warning: No controller running. Starting cartesian impedance with default gains.")
            Kx = torch.Tensor([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
            Kxd = torch.Tensor([37.0, 37.0, 37.0, 3.0, 3.0, 3.0])
            self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
            self._controller_running = True

        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )

    def update_desired_joint_positions(self, positions):
        if not self._controller_running:
            print("Warning: No controller running. Starting joint impedance with default gains.")
            self.robot.start_joint_impedance()
            self._controller_running = True
        self.robot.update_desired_joint_positions(torch.Tensor(positions))

    def terminate_current_policy(self):
        if not self._controller_running:
            print("Warning: No controller running, nothing to terminate.")
            return
        try:
            self.robot.terminate_current_policy()
            print("[Server] Policy terminated successfully")
        except grpc._channel._InactiveRpcError as e:
            if "no controller running" in str(e).lower():
                print("Warning: Controller already terminated.")
            else:
                raise
        finally:
            self._controller_running = False

    # ========== Panda Gripper Methods (via libfranka, non-blocking) ==========

    def get_gripper_state(self):
        if not self.gripper_connected or self.gripper is None:
            return {
                'width': self.gripper_width,
                'is_grasped': False,
                'connected': False
            }
        try:
            state = self.gripper.get_state()
            self.gripper_width = state['width']
            self.gripper_max_width = state.get('max_width', 0.08)
            return {
                'width': float(state['width']),
                'is_grasped': bool(state.get('is_grasped', False)),
                'connected': True
            }
        except Exception as e:
            print(f"Warning: Failed to get gripper state: {e}")
            return {
                'width': self.gripper_width,
                'is_grasped': False,
                'connected': True
            }

    def gripper_goto(self, width, speed=0.1, force=10.0):
        if width < 0.04:
            return self.gripper_grasp(speed=speed, force=force)
        else:
            return self.gripper_release(speed=speed)

    def gripper_grasp(self, speed=0.1, force=60.0):
        if not self.gripper_connected or self.gripper is None:
            print("Warning: Gripper not connected")
            self.gripper_width = 0.0
            return False
        self.gripper.grasp(speed=speed, force=force)
        self.gripper_width = 0.0
        return True

    def gripper_release(self, speed=0.1):
        if not self.gripper_connected or self.gripper is None:
            print("Warning: Gripper not connected")
            self.gripper_width = self.gripper_max_width
            return False
        self.gripper.release(speed=speed)
        self.gripper_width = self.gripper_max_width
        return True


# ======================== Main ========================

def connect_with_retry(robot_ip, gripper_robot_ip, max_retries=30, interval=2.0):
    """Try connecting to polymetis, retrying until it is available."""
    for attempt in range(1, max_retries + 1):
        try:
            return FrankaInterface(
                robot_ip=robot_ip,
                gripper_robot_ip=gripper_robot_ip)
        except grpc._channel._InactiveRpcError:
            print(f"[Server] Polymetis not ready (attempt {attempt}/{max_retries}), "
                  f"retrying in {interval}s ...")
            print(f"         Make sure launch_polymetis.sh is running in another terminal.")
            time.sleep(interval)
    raise RuntimeError(
        f"Cannot connect to polymetis at {robot_ip} after {max_retries} attempts.\n"
        f"Please start polymetis first:\n"
        f"  bash launch_polymetis.sh")


def main():
    parser = argparse.ArgumentParser(
        description='Franka ZeroRPC server (Step 2, runs on NUC)')
    parser.add_argument('--robot_ip', default='localhost',
                        help='Polymetis gRPC server IP (usually localhost)')
    parser.add_argument('--gripper_robot_ip', default='10.168.1.200',
                        help='Franka robot IP for gripper (libfranka direct)')
    parser.add_argument('--port', type=int, default=4242,
                        help='ZeroRPC server port')
    parser.add_argument('--no_retry', action='store_true',
                        help='Fail immediately if polymetis is not ready')
    args = parser.parse_args()

    print("=" * 50)
    print("Franka Interface Server (下位机 Step 2)")
    print("  Robot arm: via Polymetis")
    print("  Gripper:   via libfranka (non-blocking)")
    print("=" * 50)
    print(f"Polymetis server: {args.robot_ip}")
    print(f"Franka robot IP (gripper): {args.gripper_robot_ip}")
    print(f"Server port: {args.port}")
    print("=" * 50)

    if args.no_retry:
        interface = FrankaInterface(
            robot_ip=args.robot_ip,
            gripper_robot_ip=args.gripper_robot_ip)
    else:
        interface = connect_with_retry(
            robot_ip=args.robot_ip,
            gripper_robot_ip=args.gripper_robot_ip)

    s = zerorpc.Server(interface)
    s.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"\nServer running on tcp://0.0.0.0:{args.port}")
    print("Press Ctrl+C to stop.\n")

    try:
        s.run()
    finally:
        if interface.gripper_connected and interface.gripper:
            print("\nClosing gripper connection...")
            interface.gripper.close()


if __name__ == '__main__':
    main()
