"""Shared PyBullet simulation infrastructure.

Task simulators own scene construction, state machines, and success criteria.
This base class owns the Franka setup, common domain randomizers, camera model,
rendering, and generic robot/gripper control.
"""

from __future__ import annotations

import cv2
from importlib.util import find_spec
import numpy as np

from sim_world.VisualDR import (
    DistractorDR,
    ImgNoiseDR,
    IntrinsicDR,
    LightingDR,
    ObjectColorDR,
    PoseDR,
    TextureDR,
)
from sim_world.src.pybullet_utility import cvK2BulletP, cvPose2BulletView
from utility import resize_rgb


PANDA_END_EFFECTOR_INDEX = 11
PANDA_NUM_DOFS = 7
IK_LOWER_LIMITS = [-7] * PANDA_NUM_DOFS
IK_UPPER_LIMITS = [7] * PANDA_NUM_DOFS
IK_JOINT_RANGES = [7] * PANDA_NUM_DOFS

DEFAULT_JOINT_POSITIONS = (
    -0.09762531509002051,
    -0.11921289903971187,
    -0.0004173343487966217,
    -2.1853378428408976,
    -0.0002942036915626864,
    2.064449508160775,
    0.687251996585265,
    0.04,
    0.04,
)


class Base_Simulation:
    """Common robot, camera, rendering, and visual-randomization layer."""

    collect_eye_in_hand_by_default = True

    def __init__(
        self,
        bullet_client,
        offset=(0.0, 0.0, 0.0),
        control_dt=1.0 / 120.0,
        seed=42,
        *,
        cid=None,
        use_egl=False,
        joint_positions=DEFAULT_JOINT_POSITIONS,
        randomize_initial_ee_pose=True,
        initial_ee_x_jit=0.04,
        initial_ee_y_jit=0.05,
        initial_ee_z_jit=0.05,
        initial_ee_eul_jit=0.12,
        eye_far=2.5,
    ):
        self.bullet_client = bullet_client
        self.cid = cid
        if use_egl:
            if cid is None:
                raise ValueError("cid is required when use_egl=True")
            else:
                from pybullet_egl_patch import load_egl
                plugin_id = load_egl(self.bullet_client, physicsClientId=cid)

        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.asarray(offset, dtype=float)
        self.control_dt = float(control_dt)
        self.t = 0.0
        self._init_domain_randomizers(seed)
        self._init_camera_model(eye_far=eye_far)
        self._init_gripper_config()
        self._load_panda(joint_positions)
        self._randomize_and_cache_home_pose(
            randomize_initial_ee_pose=randomize_initial_ee_pose,
            x_jit=initial_ee_x_jit,
            y_jit=initial_ee_y_jit,
            z_jit=initial_ee_z_jit,
            eul_jit=initial_ee_eul_jit,
        )

    def _init_domain_randomizers(self, seed):
        p = self.bullet_client
        self.lightingDR = LightingDR(p, seed=seed)
        self.agentviewImgDR = ImgNoiseDR(p, seed=seed, camera_name="agentview")
        self.eyeImgDR = ImgNoiseDR(p, seed=seed + 1000, camera_name="eye_in_hand")
        self.ImgNoiseDR = self.agentviewImgDR
        self.camposeDR = PoseDR(p, seed=seed)
        self.outsceneDR = PoseDR(p, seed=seed)
        self.collplaneDR = PoseDR(p, seed=seed)
        self.distractorDR = DistractorDR(p, seed=seed)
        self.objectColorDR = ObjectColorDR(p, seed=seed)
        self.robotTextureDR = TextureDR(p, seed=seed + 4000)
        self.skyboxTextureDR = TextureDR(p, seed=seed + 7000)
        self.fisheyeCamDR = PoseDR(p, seed=seed)
        self.initialEePoseDR = PoseDR(p, seed=seed + 2000)
        self.camIntrinsicDR = IntrinsicDR(seed=seed + 3000)

    def _init_camera_model(self, eye_far):
        self.agentview_width = 960
        self.agentview_height = 540
        self.agentview_near = 0.02
        self.agentview_far = 2.0
        self.agentview_base_extrinsic_cam = np.array(
            [
                [-0.808, 0.3283, -0.4892, 0.8758],
                [0.5837, 0.3327, -0.7407, 0.6006],
                [-0.0804, -0.884, -0.4604, 0.4729],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.agentview_base_intrinsic = np.array(
            [[691.7508, 0.0, 486.7637], [0.0, 692.2195, 273.4784], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.agentview_intrinsic = self.agentview_base_intrinsic.copy()
        self.extrinsic_cam = self.agentview_base_extrinsic_cam.copy()

        self.eye_raw_width = 640
        self.eye_raw_height = 480
        self.eye_obs_width = 224
        self.eye_obs_height = 224
        self.eye_face_size = 256
        self.eye_near = 0.005
        self.eye_far = float(eye_far)
        self.eye_base_K = np.array(
            [[242.78325327, 0.0, 316.28355305], [0.0, 243.37493553, 211.80725024], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.eye_K = self.eye_base_K.copy()
        self.eye_D = np.array(
            [-0.04749038, 0.01991722, -0.02552236, 0.0084535],
            dtype=np.float64,
        )
        self.T_eye_base_parent_cam = np.array(
            [[0, -1, 0, 0.05054945], [1, 0, 0, -0.00619893], [0, 0, 1, 0.01294445], [0, 0, 0, 1.0]],
            dtype=np.float64,
        )
        self.T_eye_parent_cam = self.T_eye_base_parent_cam.copy()
        self.eye_parent_link = 8
        self.eye_face_names = ("pos_z", "pos_x", "neg_x", "pos_y", "neg_y")
        self.eye_face_basis = self._build_eye_face_basis()
        self.eye_face_proj_90 = self.bullet_client.computeProjectionMatrixFOV(
            fov=90.0,
            aspect=1.0,
            nearVal=self.eye_near,
            farVal=self.eye_far,
        )
        self.eye_fisheye_remap = self.build_eye_fisheye_remap(
            out_width=self.eye_obs_width,
            out_height=self.eye_obs_height,
            face_size=self.eye_face_size,
        )

    def _init_gripper_config(self):
        self.GRIPPER_CLOSED = 0.0
        self.GRIPPER_OPEN = 1.0
        self.gripper_closed_width = 0.0
        self.gripper_open_width = 0.04
        self.gripper_state_threshold_width = 0.8 * (
            self.gripper_open_width + self.gripper_closed_width
        )
        self.target_gripper = self.GRIPPER_OPEN
        self.finger_target = self.gripper_open_width
        self.grasp_world_normal = np.array([0.0, 0.0, -1.0], dtype=float)
        self.grasp_world_tangent = np.array([1.0, 0.0, 0.0], dtype=float)
        self.safe_approach = 0.05
        self.safe_grasp_offset = 0.05
        self.arm_force = 200.0
        self.gripper_force = 100.0

    def _load_panda(self, joint_positions):
        p = self.bullet_client
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.panda = p.loadURDF(
            "data/franka_panda/panda_wristcam.urdf",
            self.offset,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=flags,
        )
        constraint = p.createConstraint(
            self.panda,
            9,
            self.panda,
            10,
            jointType=p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(constraint, gearRatio=-1, erp=0.1, maxForce=100)
        movable_index = 0
        for joint_index in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, joint_index, linearDamping=0, angularDamping=0)
            joint_type = p.getJointInfo(self.panda, joint_index)[2]
            if joint_type in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
                p.resetJointState(self.panda, joint_index, joint_positions[movable_index])
                movable_index += 1

    def _randomize_and_cache_home_pose(self, *, randomize_initial_ee_pose, x_jit, y_jit, z_jit, eul_jit):
        ee_pos, ee_orn = self.get_ee_pose()
        if randomize_initial_ee_pose:
            ee_pos, ee_orn = self.initialEePoseDR.sample_SE3_randomization(
                pos=ee_pos,
                orn=ee_orn,
                x_jitter_range=x_jit,
                y_jitter_range=y_jit,
                z_jitter_range=z_jit,
                x_euler_jitter_range=eul_jit,
                y_euler_jitter_range=eul_jit,
                z_euler_jitter_range=eul_jit,
            )
            self.solve_ik_and_apply(ee_pos, ee_orn, input_frame="parent_ee", reset=True)
        self.home_joint = np.asarray(self.get_current_arm_joints(), dtype=float)
        self.home_ee_pos, self.home_ee_orn = self.get_ee_pose()
        self.home_ee_pos = np.asarray(self.home_ee_pos, dtype=float)
        self.home_ee_orn = np.asarray(self.home_ee_orn, dtype=float)

    def _create_randomized_skybox(self, patterns=("checkers", "gradient", "noise", "plain"), texture_size=256):
        """Create collision-free textured walls that EGL actually renders."""
        self.skybox_body_ids = []
        center_x, center_y, base_z = self.offset
        specs = (
            ([0.01, 1.2, 1.2], [center_x + 1.2, center_y, base_z + 0.5], True),
            ([0.01, 1.2, 1.2], [center_x - 1.2, center_y, base_z + 0.5], True),
            ([1.2, 0.01, 1.2], [center_x, center_y + 1.2, base_z + 0.5], True),
            ([1.2, 0.01, 1.2], [center_x, center_y - 1.2, base_z + 0.5], False),
            ([1.2, 1.2, 0.01], [center_x, center_y, base_z - 0.5], True),
        )
        for half_extents, position, randomize_texture in specs:
            color = [1, 1, 1, 1] if randomize_texture else [0, 0, 0, 1]
            visual = self.bullet_client.createVisualShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=color,
            )
            body_id = self.bullet_client.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=position,
            )
            self.skybox_body_ids.append(body_id)
            if randomize_texture:
                self.skyboxTextureDR.sample_and_apply_texture_randomization(
                    body_id=body_id,
                    patterns=patterns,
                    texture_size=texture_size,
                    per_link=False,
                    specular_range=(0.0, 0.05),
                    alpha=1.0,
                    original_texture_prob=0.0,
                )

    def get_ee_pose(self):
        """Return the original Franka end-effector pose.

        Tool/TCP frames are deliberately task-specific. A task with an attached
        tool must override this method instead of teaching the base class about
        that tool's transform convention.
        """
        state = self.bullet_client.getLinkState(
            self.panda,
            PANDA_END_EFFECTOR_INDEX,
            computeForwardKinematics=True,
        )
        return np.asarray(state[4], dtype=float), np.asarray(state[5], dtype=float)

    def gripper_state_to_width(self, state):
        return self.gripper_open_width if float(state) >= 0.5 else self.gripper_closed_width

    def gripper_width_to_state(self, width):
        return self.GRIPPER_OPEN if float(width) >= self.gripper_state_threshold_width else self.GRIPPER_CLOSED

    def set_gripper_width(self, opening):
        self.finger_target = float(opening)
        for joint_index in (9, 10):
            self.bullet_client.setJointMotorControl2(
                self.panda,
                joint_index,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=self.finger_target,
                force=self.gripper_force,
            )

    def set_gripper_state(self, state):
        self.target_gripper = self.GRIPPER_OPEN if float(state) >= 0.5 else self.GRIPPER_CLOSED
        self.set_gripper_width(self.gripper_state_to_width(self.target_gripper))

    def set_gripper(self, command):
        command = float(command)
        if command in (self.GRIPPER_CLOSED, self.GRIPPER_OPEN):
            self.set_gripper_state(command)
        else:
            self.set_gripper_width(command)

    def solve_ik_and_apply(self, target_pos, target_orn, input_frame="ee", reset=False):
        p = self.bullet_client
        input_frame = str(input_frame).lower()
        parent_frames = {"parent_tcp", "parent_ee", "ee", "parent", "original_ee"}
        if input_frame not in parent_frames:
            raise ValueError(
                f"Base_Simulation only supports the original Franka EE frame; "
                f"got input_frame={input_frame!r}. Tool frames belong in the task subclass."
            )
        ik_pos = np.asarray(target_pos, dtype=float)
        ik_orn = np.asarray(target_orn, dtype=float)
        joints = p.calculateInverseKinematics(
            self.panda,
            PANDA_END_EFFECTOR_INDEX,
            np.asarray(ik_pos).tolist(),
            np.asarray(ik_orn).tolist(),
            IK_LOWER_LIMITS,
            IK_UPPER_LIMITS,
            IK_JOINT_RANGES,
            self.get_current_arm_joints(),
            maxNumIterations=50,
        )
        for joint_index in range(PANDA_NUM_DOFS):
            if reset:
                p.resetJointState(self.panda, joint_index, joints[joint_index])
            p.setJointMotorControl2(
                self.panda,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joints[joint_index],
                force=self.arm_force,
            )
        return np.asarray(joints[:PANDA_NUM_DOFS], dtype=float)

    def setJoint(self, joint_poses):
        for joint_index in range(PANDA_NUM_DOFS):
            self.bullet_client.setJointMotorControl2(
                self.panda,
                joint_index,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=joint_poses[joint_index],
                force=self.arm_force,
            )

    def get_current_arm_joints(self):
        return [self.bullet_client.getJointState(self.panda, i)[0] for i in range(PANDA_NUM_DOFS)]

    def get_gripper_mean_width(self):
        return float(np.mean([self.bullet_client.getJointState(self.panda, i)[0] for i in (9, 10)]))

    def collect_observation(
        self,
        use_agent_cam=True,
        direct=False,
        use_eye_in_hand=None,
    ):
        """Collect policy observations in the original Franka EE frame."""
        # Bypass task-specific get_ee_pose() overrides (for example a tool TCP).
        eef_pos, eef_quat = Base_Simulation.get_ee_pose(self)
        if use_eye_in_hand is None:
            use_eye_in_hand = self.collect_eye_in_hand_by_default
        obs = {
            "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32),
            "robot0_eef_quat": np.asarray(eef_quat, dtype=np.float32),
            "robot0_gripper_qpos": np.asarray(
                [self.gripper_width_to_state(self.get_gripper_mean_width())],
                dtype=np.float32,
            ),
        }
        if use_agent_cam:
            if direct:
                image = self.direct_get_agent_view()
            else:
                image = resize_rgb(self.get_agentview_image()[2][..., :3], out_size=224)
            obs["agentview_image"] = image.astype(np.uint8)
        if use_eye_in_hand:
            obs["robot0_eye_in_hand_image"] = self.get_eye_in_hand_image().astype(np.uint8)
        return obs

    def collect_action(self):
        """Collect [EE xyz, EE quaternion, binary gripper] for policy data."""
        return np.concatenate(
            [
                np.asarray(self.target_pos, dtype=float),
                np.asarray(self.target_orn, dtype=float),
                np.asarray([self.target_gripper], dtype=float),
            ],
            axis=0,
        ).astype(np.float32)

    def _build_eye_face_basis(self):
        return {
            "pos_z": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            "pos_x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
            "neg_x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float),
            "pos_y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
            "neg_y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
            "neg_z": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float),
        }

    def pose_to_T(self, pos, quat_xyzw):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            self.bullet_client.getMatrixFromQuaternion(quat_xyzw),
            dtype=np.float64,
        ).reshape(3, 3)
        transform[:3, 3] = np.asarray(pos, dtype=np.float64)
        return transform

    def print_panda_link_info(self):
        """Print joint/link indices for camera-calibration debugging."""
        for joint_index in range(self.bullet_client.getNumJoints(self.panda)):
            info = self.bullet_client.getJointInfo(self.panda, joint_index)
            joint_name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
            link_name = info[12].decode() if isinstance(info[12], bytes) else str(info[12])
            print(f"joint/link index {joint_index}: joint={joint_name}, link={link_name}")

    def debug_draw_axes(self, transform, length=0.08, life_time=0.1):
        """Draw a transform's XYZ axes in the PyBullet GUI."""
        origin = transform[:3, 3]
        rotation = transform[:3, :3]
        colors = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        for axis_index, color in enumerate(colors):
            self.bullet_client.addUserDebugLine(
                origin,
                origin + length * rotation[:, axis_index],
                color,
                2,
                life_time,
            )

    def get_eye_in_hand_T_world_cam(self):
        state = self.bullet_client.getLinkState(
            self.panda,
            self.eye_parent_link,
            computeForwardKinematics=True,
        )
        return self.pose_to_T(state[4], state[5]) @ self.T_eye_parent_cam

    def kb4_fisheye_rays_for_obs(self, out_width, out_height):
        fx, fy = float(self.eye_K[0, 0]), float(self.eye_K[1, 1])
        cx, cy = float(self.eye_K[0, 2]), float(self.eye_K[1, 2])
        k1, k2, k3, k4 = map(float, self.eye_D)
        u, v = np.meshgrid(np.arange(out_width, dtype=float), np.arange(out_height, dtype=float))
        u = (u + 0.5) * self.eye_raw_width / float(out_width) - 0.5
        v = (v + 0.5) * self.eye_raw_height / float(out_height) - 0.5
        mx, my = (u - cx) / fx, (v - cy) / fy
        theta_d = np.sqrt(mx * mx + my * my)
        phi = np.arctan2(my, mx)
        theta = theta_d.copy()
        for _ in range(8):
            t2 = theta * theta
            t4, t6, t8 = t2 * t2, t2 * t2 * t2, t2 * t2 * t2 * t2
            value = theta * (1 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d
            derivative = 1 + 3 * k1 * t2 + 5 * k2 * t4 + 7 * k3 * t6 + 9 * k4 * t8
            theta -= value / np.maximum(np.abs(derivative), 1e-12)
        rays = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        )
        rays[theta_d < 1e-12] = [0, 0, 1]
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1e-12)

    def build_eye_fisheye_remap(self, out_width=224, out_height=224, face_size=256):
        rays = self.kb4_fisheye_rays_for_obs(out_width, out_height)
        x, y, z = (rays[..., i] for i in range(3))
        ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
        face_index = np.full((out_height, out_width), -1, dtype=np.int32)
        lookup = {name: i for i, name in enumerate(self.eye_face_names)}
        conditions = {
            "pos_z": (az >= ax) & (az >= ay) & (z >= 0),
            "pos_x": (ax > az) & (ax >= ay) & (x >= 0),
            "neg_x": (ax > az) & (ax >= ay) & (x < 0),
            "pos_y": (ay > ax) & (ay > az) & (y >= 0),
            "neg_y": (ay > ax) & (ay > az) & (y < 0),
            "neg_z": (az >= ax) & (az >= ay) & (z < 0),
        }
        for name, condition in conditions.items():
            if name in lookup:
                face_index[condition] = lookup[name]
        remap = {}
        for name in self.eye_face_names:
            ray_face = rays @ self.eye_face_basis[name]
            rz = ray_face[..., 2]
            valid = (face_index == lookup[name]) & (rz > 1e-8)
            map_x = ((ray_face[..., 0] / np.maximum(rz, 1e-8) + 1) * 0.5 * (face_size - 1)).astype(np.float32)
            map_y = ((ray_face[..., 1] / np.maximum(rz, 1e-8) + 1) * 0.5 * (face_size - 1)).astype(np.float32)
            map_x[~valid] = 0
            map_y[~valid] = 0
            remap[name] = {"mask": valid, "map_x": map_x, "map_y": map_y}
        self.eye_invalid_pixel_count = int(np.sum(face_index < 0))
        return remap

    def get_eye_face_view_matrix(self, world_camera, face_name):
        camera_face = np.eye(4)
        camera_face[:3, :3] = self.eye_face_basis[face_name]
        return cvPose2BulletView(world_camera @ camera_face)

    def render_camera_raw(self, width, height, view_matrix, proj_matrix):
        config = self.lightingDR.light_cfg
        w, h, rgba, depth, _ = self.bullet_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL,
            flags=self.bullet_client.ER_NO_SEGMENTATION_MASK,
            lightDirection=config["lightDirection"],
            lightColor=config["lightColor"],
            lightDistance=config["lightDistance"],
            lightAmbientCoeff=config["lightAmbientCoeff"],
            lightDiffuseCoeff=config["lightDiffuseCoeff"],
            lightSpecularCoeff=config["lightSpecularCoeff"],
            shadow=config["shadow"],
        )
        return w, h, np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4).copy(), np.asarray(depth, dtype=np.float32).reshape(h, w), None

    def render_eye_cubemap_faces(self):
        world_camera = self.get_eye_in_hand_T_world_cam()
        return {
            name: self.render_camera_raw(
                self.eye_face_size,
                self.eye_face_size,
                self.get_eye_face_view_matrix(world_camera, name),
                self.eye_face_proj_90,
            )[2][..., :3]
            for name in self.eye_face_names
        }

    def render_camera(self, width, height, view_matrix, proj_matrix, segmentation=False):
        config = self.lightingDR.light_cfg
        w, h, rgba, depth, seg = self.bullet_client.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL,
            flags=0 if segmentation else self.bullet_client.ER_NO_SEGMENTATION_MASK,
            lightDirection=config["lightDirection"],
            lightColor=config["lightColor"],
            lightDistance=config["lightDistance"],
            lightAmbientCoeff=config["lightAmbientCoeff"],
            lightDiffuseCoeff=config["lightDiffuseCoeff"],
            lightSpecularCoeff=config["lightSpecularCoeff"],
            shadow=config["shadow"],
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4).copy()
        rgba[..., :3] = self.agentviewImgDR.apply_image_noise(rgba[..., :3])
        depth = np.asarray(depth, dtype=np.float32).reshape(h, w)
        seg = np.asarray(seg, dtype=np.int64).reshape(h, w) if segmentation else None
        return w, h, rgba, depth, seg

    def get_agentview_image(self, segmentation=False):
        projection = cvK2BulletP(
            self.agentview_intrinsic,
            self.agentview_width,
            self.agentview_height,
            self.agentview_near,
            self.agentview_far,
        )
        return self.render_camera(
            self.agentview_width,
            self.agentview_height,
            cvPose2BulletView(self.extrinsic_cam),
            projection,
            segmentation=segmentation,
        )

    def get_cropped_agentview_image(self, out_size=224):
        return resize_rgb(self.get_agentview_image()[2][..., :3], out_size)

    def direct_get_agent_view(self, out_size=224):
        intrinsic = self.agentview_intrinsic.copy()
        intrinsic[0, (0, 2)] *= out_size / self.agentview_width
        intrinsic[1, (1, 2)] *= out_size / self.agentview_height
        projection = cvK2BulletP(intrinsic, out_size, out_size, self.agentview_near, self.agentview_far)
        return self.render_camera(
            out_size,
            out_size,
            cvPose2BulletView(self.extrinsic_cam),
            projection,
        )[2][..., :3].astype(np.uint8)

    def get_eye_in_hand_image(self, width=None, height=None, face_size=None):
        width = self.eye_obs_width if width is None else int(width)
        height = self.eye_obs_height if height is None else int(height)
        old_face_size = self.eye_face_size
        if width == self.eye_obs_width and height == self.eye_obs_height and (face_size is None or int(face_size) == old_face_size):
            remap = self.eye_fisheye_remap
        else:
            if face_size is not None:
                self.eye_face_size = int(face_size)
            remap = self.build_eye_fisheye_remap(width, height, self.eye_face_size)
        faces = self.render_eye_cubemap_faces()
        output = np.zeros((height, width, 3), dtype=np.uint8)
        for name in self.eye_face_names:
            mapping = remap[name]
            sampled = cv2.remap(
                faces[name],
                mapping["map_x"],
                mapping["map_y"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            output[mapping["mask"]] = sampled[mapping["mask"]]
        self.eye_face_size = old_face_size
        return self.eyeImgDR.apply_image_noise(output).astype(np.uint8)

    def enable_high_quality_rendering(self):
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_SHADOWS, 1)
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_TINY_RENDERER, 0)
