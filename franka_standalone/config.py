"""
Franka standalone configuration.
Modify these values to match your hardware setup.
"""
import numpy as np

# ==================== Network ====================
ROBOT_IP = '192.168.3.2'       # NUC IP (runs franka_server.py)
ROBOT_PORT = 4242              # ZeroRPC server port

# ==================== Camera ====================
L515_SERIALS = ['f1480807', 'f1471315']   # Intel RealSense L515 serial numbers
FISHEYE_USB_ID = '32e4:9230'              # USB vendor:product ID for fisheye camera
FISHEYE_RESOLUTION = (640, 480)
FISHEYE_FOURCC = 'YUYV'        # YUYV @ 640x480 = full fisheye FOV (MJPG 640x480 is a cropped ROI)

# ==================== Robot Control ====================
CONTROL_FREQUENCY = 10         # Hz
POS_SPEED = 0.08               # m/s for keyboard control
ROT_SPEED = 0.3                # rad/s for keyboard control

# ==================== Gripper ====================
MAX_GRIPPER_WIDTH = 0.08       # Franka hand max width (m)
GRIPPER_SPEED = 0.2            # m/s
GRIPPER_FORCE = 40.0           # N

# ==================== Home Position ====================
FRANKA_HOME_JOINTS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

EE_HOME_POSE = np.array([
    0.4,    # x (m)
    0.0,   # y (m)
    0.3,    # z (m)
    np.pi,   # rx (rotvec)
    0.0,     # ry
    0.0,     # rz
])

HOME_MOVE_DURATION = 4.0       # seconds
JOINTS_HOME_DURATION = 4.0     # seconds

# ==================== Impedance Gains ====================
KX_DEFAULT = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
KXD_DEFAULT = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])

# ==================== Tip-Flange Transform ====================
# Offset from flange to tool tip (e.g. gripper finger tip)
import scipy.spatial.transform as st

_tx_flange_tip_trans = np.identity(4)
_tx_flange_tip_trans[:3, 3] = np.array([0, 0, 0.1034])

_tx_flange_rot45 = np.identity(4)
_tx_flange_rot45[:3, :3] = st.Rotation.from_euler('z', [-np.pi / 4]).as_matrix()

TX_FLANGE_TIP = _tx_flange_rot45 @ _tx_flange_tip_trans
TX_TIP_FLANGE = np.linalg.inv(TX_FLANGE_TIP)
