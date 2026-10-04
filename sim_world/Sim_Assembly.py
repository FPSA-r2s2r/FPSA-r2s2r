import numpy as np
import math
import pybullet_data
from sim_world.VisualDR import FPSAObjectDR, PoseDR
from sim_world.src.pybullet_utility import (
    load_models,
    coacd_convex_decomposition,
    get_com,
    get_true_PositionAndOrientation,
    quat_slerp,
)

from utility import (
    load_initial_grasp_pose, 
    normalize_vector, 
    quat_from_rotation_matrix,
    ) 

from icecream import ic
from scipy.spatial.transform import Rotation as R
from sim_world.Base_Simulation import Base_Simulation

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
ul = [7] * pandaNumDofs
jr = [7] * pandaNumDofs

jointPositions = [
    -0.09762531509002051,
    -0.11921289903971187,
    -0.0004173343487966217,
    -2.1853378428408976,
    -0.0002942036915626864,
    2.064449508160775,
    0.687251996585265,
    0.04,
    0.04
]
rp = jointPositions


class AssemblySim(Base_Simulation):
    def __init__(
        self,
        bullet_client,
        cid,
        use_egl=True,
        offset=(0.0, 0.0, 0.0),
        control_dt=1.0 / 120.0,
        seed=42,
        randomize_initial_ee_pose=True,
        initial_ee_x_jit=0.04,
        initial_ee_y_jit=0.05,
        initial_ee_z_jit=0.05,
        initial_ee_eul_jit=0.12,
    ):
        super().__init__(
            bullet_client=bullet_client,
            cid=cid,
            use_egl=use_egl,
            offset=offset,
            control_dt=control_dt,
            seed=seed,
            joint_positions=jointPositions,
            randomize_initial_ee_pose=randomize_initial_ee_pose,
            initial_ee_x_jit=initial_ee_x_jit,
            initial_ee_y_jit=initial_ee_y_jit,
            initial_ee_z_jit=initial_ee_z_jit,
            initial_ee_eul_jit=initial_ee_eul_jit,
            eye_far=2.5,
        )
        self.parentPoseDR = PoseDR(self.bullet_client, seed=seed)
        self.objposeDR = self.parentPoseDR
        self.childPoseDR = PoseDR(self.bullet_client, seed=seed + 10)
        self.objectInHandPoseDR = PoseDR(self.bullet_client, seed=seed + 5000)
        self.fpsaObjectDR = FPSAObjectDR(seed=seed + 6000)
        self.gripper_force = 200.0

        self.states = [
            "home",
            "move_pregrasp",
            "open_gripper",
            "move_grasp",
            "close_gripper",
            "lift_parent",
            "move_preassembly",
            "assemble",
            "hold_assembled",
        ]
        self.state_durations = [0.01, 4.0, 0.25, 2.0, 0.75, 2.0, 4.0, 3.0, 0.5]
        self.state_idx = 0
        self.state = self.states[self.state_idx]
        self.state_t = 0.0

        # target pose, target gripper state (1=open, 0=closed)
        self.target_pos = None
        self.target_orn = None
        self.target_gripper = None
        self.done = False

        # Current state's interpolated motion segment.
        self.motion_start_pos = None
        self.motion_start_orn = None
        self.motion_target_pos = None
        self.motion_target_orn = None

        # Grasp and assembly references.
        self.last_grasp_pose = None
        self.last_grasp_orn = None
        self.parent_motion_start_pos = None
        self.parent_motion_start_orn = None
        self.parent_motion_target_pos = None
        self.parent_motion_target_orn = None
        self.initial_parent_pos = None
        self.initial_parent_orn = None
        self.initial_child_pos = None
        self.initial_child_orn = None

        # Post-grasp rigid attachment / object-in-hand DR state.
        self.parent_grasp_constraint_id = None
        self.grasp_parent_to_ee = None
        # Finger-joint position measured at the end of the physical nominal
        # close_gripper phase. After parent/gripper collisions are filtered and
        # the parent is rigidly attached, keep commanding this width instead of
        # allowing the fingers to collapse to the fully-closed target (0.0).
        self.nominal_grasp_finger_width = None
        self.hold_nominal_gripper_width_after_attach = True
        self.nominal_ee_to_parent = None
        self.randomized_ee_to_parent = None
        self.object_in_hand_delta_pos = np.zeros(3, dtype=float)
        self.object_in_hand_delta_euler = np.zeros(3, dtype=float)

        # Defaults are overwritten by make_scene().
        self.fix_parent_to_gripper = True
        self.randomize_object_in_hand_pose = True
        self.object_in_hand_pos_jit = np.array([0.002, 0.002, 0.003], dtype=float)
        self.object_in_hand_eul_jit = np.array([0.0174533, 0.0174533, 0.0349066], dtype=float)
        self.object_in_hand_debug = False

        # A fixed joint does not disable contact generation.  Once the parent
        # is rigidly attached, disable collision only against the local gripper
        # links so contact constraints do not fight the JOINT_FIXED constraint.
        # Collisions against the child, plane, environment and distractors stay on.
        self.disable_parent_gripper_collision = True
        self.parent_gripper_collision_links = (8, 9, 10, pandaEndEffectorIndex)
        self.parent_gripper_collision_disabled = False

        self.prepare_state(self.state)

 

    def get_parent_obj_pose(self):
        return tuple(
            np.asarray(v, dtype=float)
            for v in get_true_PositionAndOrientation(
                self.bullet_client, self.assembly_parent_id
            )
        )

    def get_child_obj_pose(self):
        return tuple(
            np.asarray(v, dtype=float)
            for v in get_true_PositionAndOrientation(
                self.bullet_client, self.assembly_child_id
            )
        )

    def get_parent_to_ee_transform(self, ee_pos=None, ee_orn=None):
        """Return the current parent-object-frame -> EE transform.

        With no explicit EE pose, this reads both bodies from PyBullet every
        time, so grasp slip is reflected immediately. An explicit EE pose is
        only used when building annotation-based debug/distractor waypoints.
        """
        parent_pos, parent_orn = self.get_parent_obj_pose()
        if ee_pos is None or ee_orn is None:
            ee_pos, ee_orn = self.get_ee_pose()

        inv_parent_pos, inv_parent_orn = self.bullet_client.invertTransform(
            parent_pos.tolist(), parent_orn.tolist()
        )
        rel_pos, rel_orn = self.bullet_client.multiplyTransforms(
            inv_parent_pos,
            inv_parent_orn,
            np.asarray(ee_pos, dtype=float).tolist(),
            np.asarray(ee_orn, dtype=float).tolist(),
        )
        return np.asarray(rel_pos, dtype=float), np.asarray(rel_orn, dtype=float)

    def set_parent_gripper_collision_enabled(self, enabled):
        """Enable/disable only parent-vs-local-gripper collision pairs.

        ``JOINT_FIXED`` constrains relative motion but does not suppress contact
        generation.  This helper therefore filters collision only between the
        parent object's base link and the configured Panda hand/finger/EE links.
        Parent collisions with the child, plane, environment and distractors are
        intentionally untouched.
        """
        panda_id = getattr(self, "panda", None)
        parent_id = getattr(self, "assembly_parent_id", None)
        if panda_id is None or parent_id is None:
            self.parent_gripper_collision_disabled = False
            return

        enable_flag = 1 if bool(enabled) else 0
        num_robot_joints = self.bullet_client.getNumJoints(panda_id)
        configured_links = getattr(
            self,
            "parent_gripper_collision_links",
            (8, 9, 10, pandaEndEffectorIndex),
        )

        for robot_link in configured_links:
            robot_link = int(robot_link)
            if robot_link < -1 or robot_link >= num_robot_joints:
                if getattr(self, "object_in_hand_debug", False):
                    print(
                        "Skipping invalid parent/gripper collision link:",
                        robot_link,
                    )
                continue
            self.bullet_client.setCollisionFilterPair(
                bodyUniqueIdA=panda_id,
                bodyUniqueIdB=parent_id,
                linkIndexA=robot_link,
                linkIndexB=-1,
                enableCollision=enable_flag,
            )

        self.parent_gripper_collision_disabled = not bool(enabled)

    def remove_parent_grasp_constraint(self):
        """Remove rigid attachment and restore parent/gripper collisions."""
        constraint_id = getattr(self, "parent_grasp_constraint_id", None)
        if constraint_id is not None:
            try:
                self.bullet_client.removeConstraint(constraint_id)
            except Exception:
                pass
        self.parent_grasp_constraint_id = None
        self.grasp_parent_to_ee = None
        self.nominal_grasp_finger_width = None

        # Restore only the pairs that this class may have disabled.
        try:
            self.set_parent_gripper_collision_enabled(True)
        except Exception:
            # This can happen during scene teardown after a body was already
            # removed.  The next loaded body starts with collisions enabled.
            self.parent_gripper_collision_disabled = False

    def get_ee_to_parent_transform(self, ee_pos=None, ee_orn=None):
        """Return the parent-object pose expressed in the EE link frame."""
        parent_pos, parent_orn = self.get_parent_obj_pose()
        if ee_pos is None or ee_orn is None:
            ee_pos, ee_orn = self.get_ee_pose()

        inv_ee_pos, inv_ee_orn = self.bullet_client.invertTransform(
            np.asarray(ee_pos, dtype=float).tolist(),
            np.asarray(ee_orn, dtype=float).tolist(),
        )
        rel_pos, rel_orn = self.bullet_client.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            parent_pos.tolist(),
            parent_orn.tolist(),
        )
        return np.asarray(rel_pos, dtype=float), np.asarray(rel_orn, dtype=float)

    def randomize_and_fix_parent_to_gripper(self):
        """Sample an object-in-hand SE(3) pose and rigidly attach the parent.

        The nominal grasp trajectory is executed first. At the end of
        ``close_gripper`` we measure the actual parent-object pose in the EE
        frame, add a small translation/orientation perturbation in that frame,
        teleport the parent to the sampled pose, and create a fixed constraint.

        No virtual force or physical slip model is used. The purpose is to
        generate image/proprioception-to-assembly-action mappings under varied
        object-in-hand poses.
        """
        if not self.fix_parent_to_gripper:
            self.grasp_parent_to_ee = None
            return

        self.remove_parent_grasp_constraint()

        # close_gripper has just completed with normal parent/finger collision.
        # Capture the physically achieved nominal grasp width before teleporting
        # the parent, filtering collision, or creating the rigid attachment.
        # The current simulator convention uses each finger joint position as
        # the gripper width command, so use the symmetric mean as the hold target.
        nominal_finger_qpos = np.array([
            self.bullet_client.getJointState(self.panda, 9)[0],
            self.bullet_client.getJointState(self.panda, 10)[0],
        ], dtype=float)
        self.nominal_grasp_finger_width = float(np.mean(nominal_finger_qpos))

        ee_pos, ee_orn = self.get_ee_pose()
        parent_true_pos, parent_true_orn = self.get_parent_obj_pose()
        parent_base_pos, parent_base_orn = self.bullet_client.getBasePositionAndOrientation(
            self.assembly_parent_id
        )
        parent_base_pos = np.asarray(parent_base_pos, dtype=float)
        parent_base_orn = np.asarray(parent_base_orn, dtype=float)

        # Nominal object pose in EE coordinates, based on the object/mesh frame
        # used by the assembly target calculations.
        nominal_rel_pos, nominal_rel_orn = self.get_ee_to_parent_transform(
            ee_pos=ee_pos, ee_orn=ee_orn
        )
        self.nominal_ee_to_parent = (
            nominal_rel_pos.copy(),
            nominal_rel_orn.copy(),
        )

        if self.randomize_object_in_hand_pose:
            randomized_rel_pos, randomized_rel_orn = (
                self.objectInHandPoseDR.sample_SE3_randomization(
                    pos=nominal_rel_pos,
                    orn=nominal_rel_orn,
                    x_jitter_range=self.object_in_hand_pos_jit[0],
                    y_jitter_range=self.object_in_hand_pos_jit[1],
                    z_jitter_range=self.object_in_hand_pos_jit[2],
                    x_euler_jitter_range=self.object_in_hand_eul_jit[0],
                    y_euler_jitter_range=self.object_in_hand_eul_jit[1],
                    z_euler_jitter_range=self.object_in_hand_eul_jit[2],
                )
            )
            randomized_rel_pos = np.asarray(randomized_rel_pos, dtype=float)
            randomized_rel_orn = np.asarray(randomized_rel_orn, dtype=float)
        else:
            randomized_rel_pos = nominal_rel_pos.copy()
            randomized_rel_orn = nominal_rel_orn.copy()

        delta_pos = randomized_rel_pos - nominal_rel_pos
        delta_euler = (
            R.from_quat(nominal_rel_orn).inv()
            * R.from_quat(randomized_rel_orn)
        ).as_euler("xyz")

        self.object_in_hand_delta_pos = delta_pos.copy()
        self.object_in_hand_delta_euler = delta_euler.copy()
        self.randomized_ee_to_parent = (
            randomized_rel_pos.copy(),
            randomized_rel_orn.copy(),
        )

        # Desired object/mesh-frame pose in world coordinates.
        randomized_true_pos, randomized_true_orn = self.bullet_client.multiplyTransforms(
            ee_pos.tolist(),
            ee_orn.tolist(),
            randomized_rel_pos.tolist(),
            randomized_rel_orn.tolist(),
        )

        # Preserve the fixed transform from the object's true/mesh frame to the
        # PyBullet base frame. resetBasePositionAndOrientation and the fixed
        # constraint operate on the base frame, while assembly targets use the
        # true/mesh frame returned by get_true_PositionAndOrientation().
        inv_true_pos, inv_true_orn = self.bullet_client.invertTransform(
            parent_true_pos.tolist(), parent_true_orn.tolist()
        )
        true_to_base_pos, true_to_base_orn = self.bullet_client.multiplyTransforms(
            inv_true_pos,
            inv_true_orn,
            parent_base_pos.tolist(),
            parent_base_orn.tolist(),
        )
        randomized_base_pos, randomized_base_orn = self.bullet_client.multiplyTransforms(
            randomized_true_pos,
            randomized_true_orn,
            true_to_base_pos,
            true_to_base_orn,
        )

        self.bullet_client.resetBasePositionAndOrientation(
            self.assembly_parent_id,
            randomized_base_pos,
            randomized_base_orn,
        )
        self.bullet_client.resetBaseVelocity(
            self.assembly_parent_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

        # Fixed-constraint parent frame is the EE link; child frame is the
        # PyBullet parent-object base.
        inv_ee_pos, inv_ee_orn = self.bullet_client.invertTransform(
            ee_pos.tolist(), ee_orn.tolist()
        )
        ee_to_base_pos, ee_to_base_orn = self.bullet_client.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            randomized_base_pos,
            randomized_base_orn,
        )
        # JOINT_FIXED does not automatically ignore collisions.  Disable only
        # parent-vs-gripper-neighborhood contacts before creating the rigid
        # attachment; all other parent collisions remain enabled.
        if self.disable_parent_gripper_collision:
            self.set_parent_gripper_collision_enabled(False)
        else:
            self.set_parent_gripper_collision_enabled(True)

        try:
            self.parent_grasp_constraint_id = self.bullet_client.createConstraint(
                parentBodyUniqueId=self.panda,
                parentLinkIndex=pandaEndEffectorIndex,
                childBodyUniqueId=self.assembly_parent_id,
                childLinkIndex=-1,
                jointType=self.bullet_client.JOINT_FIXED,
                jointAxis=[0.0, 0.0, 0.0],
                parentFramePosition=ee_to_base_pos,
                childFramePosition=[0.0, 0.0, 0.0],
                parentFrameOrientation=ee_to_base_orn,
                childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            )
        except Exception:
            # Do not leave collision filtering or a stale width hold active if
            # attachment creation fails.
            self.set_parent_gripper_collision_enabled(True)
            self.nominal_grasp_finger_width = None
            raise

        # Immediately replace the old fully-closed motor target with the width
        # achieved during nominal physical grasping. Subsequent calls to
        # set_gripper_state(CLOSED) will keep re-applying this same target.
        if self.hold_nominal_gripper_width_after_attach:
            self.set_gripper_width(self.nominal_grasp_finger_width)

        # Cache parent-object -> EE for all subsequent expert targets.
        cached_pos, cached_orn = self.bullet_client.invertTransform(
            randomized_rel_pos.tolist(), randomized_rel_orn.tolist()
        )
        self.grasp_parent_to_ee = (
            np.asarray(cached_pos, dtype=float),
            np.asarray(cached_orn, dtype=float),
        )

        if self.object_in_hand_debug:
            print(
                "object-in-hand DR:",
                "delta_pos_ee=", np.round(delta_pos, 6),
                "delta_euler_deg=", np.round(np.degrees(delta_euler), 3),
                "constraint_id=", self.parent_grasp_constraint_id,
                "parent_gripper_collision_disabled=",
                self.parent_gripper_collision_disabled,
                "filtered_robot_links=",
                self.parent_gripper_collision_links,
                "nominal_grasp_finger_width=",
                round(float(self.nominal_grasp_finger_width), 6),
            )

    def get_ee_pose_for_parent_pose(
        self, parent_pos, parent_orn, parent_to_ee=None
    ):
        """Convert a desired parent-object pose into an EE pose.

        Runtime assembly calls leave ``parent_to_ee`` unset and re-measure the
        transform every step. After rigid attachment this transform should be
        constant apart from tiny solver noise, matching the continuously
        measured parent-to-EE mapping used by the scripted expert.
        """
        if parent_to_ee is None:
            parent_to_ee = self.get_parent_to_ee_transform()
        rel_pos, rel_orn = parent_to_ee
        ee_pos, ee_orn = self.bullet_client.multiplyTransforms(
            np.asarray(parent_pos, dtype=float).tolist(),
            np.asarray(parent_orn, dtype=float).tolist(),
            np.asarray(rel_pos, dtype=float).tolist(),
            np.asarray(rel_orn, dtype=float).tolist(),
        )
        return np.asarray(ee_pos, dtype=float), np.asarray(ee_orn, dtype=float)

    def get_assembly_target_parent_pose(self):
        """Desired world pose of the square ring at the final assembly location.

        The parent OBJ origin is the ring center. Therefore the default target is
        exactly the child OBJ origin, with an optional child-frame offset.
        """
        child_pos, child_orn = self.get_child_obj_pose()
        target_pos, target_orn = self.bullet_client.multiplyTransforms(
            child_pos.tolist(),
            child_orn.tolist(),
            self.child_to_parent_target_pos.tolist(),
            self.child_to_parent_target_orn.tolist(),
        )
        return np.asarray(target_pos, dtype=float), np.asarray(target_orn, dtype=float)

    def get_preassembly_parent_pose(self):
        """Return a pose above the final target along the child local assembly axis."""
        target_pos, target_orn = self.get_assembly_target_parent_pose()
        _, child_orn = self.get_child_obj_pose()
        child_rot = R.from_quat(child_orn)
        axis_world = child_rot.apply(self.assembly_axis_local)
        axis_world = axis_world / np.linalg.norm(axis_world)
        pre_pos = target_pos + self.safe_assembly_approach * axis_world
        return pre_pos, target_orn

    def make_scene(
                   self,
                   env_mesh_path=None,
                   assembly_parent_path=None,
                   assembly_parent_collision_path=None,
                   assembly_child_path=None,
                   assembly_child_collision_path=None,
                   initial_grasp_path=None,
                   # FPSA assembly-parent object domain randomization.
                   # Samples visual mesh, COACD collision mesh, and grasp YAML atomically.
                   if_FPSA_tool=False,
                   fpsa_tool_aug_root=None,
                   fpsa_tool_include_base=False,
                   wrench_collision_path=None,
                   parentobj_pose_base=(0.45, -0.05, 0.05),
                   parentobj_euler_base=(0.0, 0.0, 0.0),
                   childobj_pose_base=(0.70, -0.05, 0.05),
                   childobj_euler_base=(0.0, 0.0, 0.0),
                   randomize_lighting=True,
                   randomize_skybox = True,
                   randomize_outlscene=True,
                   outlscene_xyz_jit=0.02,
                   outlscene_eul_jit=0.01,
                   randomize_plane_height=True,
                   plane_height_jit=0.008,
                   randomize_parent_objpose=True,
                   parentobj_x_jit=0.05,
                   parentobj_y_jit=0.10,
                   parentobj_z_jit=0.05,
                   parentobj_z_eul_jit=0.0,
                   randomize_child_objpose=True,
                   childobj_x_jit=0.05,
                   childobj_y_jit=0.10,
                   childobj_z_jit=0.05,
                   childobj_z_eul_jit=0.0,
                   randomize_campose=True,
                   cam_xyz_jit=0.004,
                   cam_eul_jit=0.002,
                   randomize_fisheye_cam=True,
                   fisheye_eyz_jit=0.005,
                   fisheye_eul_jit=0.002,
                   randomize_camera_intrinsic=True,
                   agentview_focal_scale_range=(0.88, 1.15),
                   agentview_principal_jit_px=18.0,
                   eye_focal_scale_range=(0.90, 1.12),
                   eye_principal_jit_px=8.0,
                   randomize_image_noise=True,
                   randomize_robot_texture=True,
                   robot_texture_patterns=("checkers", "gradient", "noise", "plain"),
                   robot_texture_size=128,
                   robot_texture_per_link=True,
                   robot_texture_specular_range=(0.02, 0.25),
                   robot_original_texture_prob=0.10,
                   skybox_texture_patterns=("checkers", "gradient", "noise", "plain"),
                   skybox_texture_size=256,
                   randomize_object_color=True,
                   object_color_mode="bounded",
                   object_color_strength=0.35,
                   object_recolor_palette=None,
                   object_recolor_target_color=None,
                   object_specular_range=(0.02, 0.5),
                   # Legacy wrench color arguments are accepted but intentionally ignored.
                   randomize_wrench_color=False,
                   wrench_color_mode="bounded",
                   wrench_color_strength=0.35,
                   randomize_distractors=True,
                   distractor_root="/mnt/storage/GoogleScannedObjects",
                   distractor_num_range=(1, 5),
                   distractor_target_size_range=(0.06, 0.16),
                   distractor_workspace=((0.25, 0.78), (-0.42, 0.42)),
                   distractor_clearance=0.04,
                   distractor_path_clearance=0.04,
                   distractor_min_target_mask_pixels=1,
                   parent_mass=0.85,
                   child_mass=0.2,
                   parent_lateral_friction=0.9,
                   child_lateral_friction=0.8,
                   child_rgba=(0.90, 0.03, 0.03, 1.0),
                   assembly_axis_local=(0.0, 0.0, 1.0),
                   child_to_parent_target_pos=(0.0, 0.0, 0.0),
                   child_to_parent_target_euler=(0.0, 0.0, 0.0),
                   safe_assembly_approach=0.10,
                   # Post-grasp object-in-hand domain randomization. Nominal
                   # pregrasp/grasp targets are never randomized.
                   fix_parent_to_gripper=True,
                   randomize_object_in_hand_pose=True,
                   object_in_hand_x_jit=0.002,
                   object_in_hand_y_jit=0.002,
                   object_in_hand_z_jit=0.003,
                   object_in_hand_roll_jit=0.0174533,
                   object_in_hand_pitch_jit=0.0174533,
                   object_in_hand_yaw_jit=0.0349066,
                   object_in_hand_debug=False,
                   # JOINT_FIXED does not disable collision.  Filter only the
                   # parent against these local Panda gripper/hand links.
                   disable_parent_gripper_collision=True,
                   parent_gripper_collision_links=(8, 9, 10, pandaEndEffectorIndex),
                   # Keep the physically achieved nominal grasp width after the
                   # parent is fixed and parent/gripper collision is filtered.
                   hold_nominal_gripper_width_after_attach=True,
                   ):
        """Build a pick-and-assemble scene with a movable parent ring and fixed child rod."""
        del wrench_collision_path
        del randomize_wrench_color, wrench_color_mode, wrench_color_strength
        self.if_FPSA_tool = bool(if_FPSA_tool)

        if assembly_parent_path is None:
            raise ValueError("assembly_parent_path must be provided")
        if assembly_child_path is None:
            raise ValueError("assembly_child_path must be provided")
        if initial_grasp_path is None:
            raise ValueError("initial_grasp_path must be provided")

        self.fpsa_sample = None
        if self.if_FPSA_tool:
            if fpsa_tool_aug_root is None:
                raise ValueError(
                    "if_FPSA_tool=True requires fpsa_tool_aug_root"
                )
            self.fpsa_sample = self.fpsaObjectDR.sample_asset(
                base_mesh_path=assembly_parent_path,
                base_collision_path=assembly_parent_collision_path,
                base_grasp_path=initial_grasp_path,
                fpsa_aug_root=fpsa_tool_aug_root,
                include_base=bool(fpsa_tool_include_base),
            )
            assembly_parent_path = self.fpsa_sample["obj_path"]
            assembly_parent_collision_path = self.fpsa_sample["collision_path"]
            initial_grasp_path = self.fpsa_sample["grasp_path"]

            # ic(assembly_parent_path, assembly_parent_collision_path)

        self.env_mesh_path = env_mesh_path
        self.assembly_parent_path = assembly_parent_path
        self.assembly_child_path = assembly_child_path
        self.initial_grasp_path = initial_grasp_path
        self.safe_assembly_approach = float(safe_assembly_approach)

        # Clear any attachment left by a previous scene build, then configure
        # the post-grasp object-in-hand randomization for this episode.
        self.remove_parent_grasp_constraint()
        self.fix_parent_to_gripper = bool(fix_parent_to_gripper)
        self.randomize_object_in_hand_pose = bool(randomize_object_in_hand_pose)
        self.disable_parent_gripper_collision = bool(
            disable_parent_gripper_collision
        )
        self.parent_gripper_collision_links = tuple(
            dict.fromkeys(int(link) for link in parent_gripper_collision_links)
        )
        self.parent_gripper_collision_disabled = False
        self.hold_nominal_gripper_width_after_attach = bool(
            hold_nominal_gripper_width_after_attach
        )
        self.nominal_grasp_finger_width = None
        self.object_in_hand_pos_jit = np.array(
            [object_in_hand_x_jit, object_in_hand_y_jit, object_in_hand_z_jit],
            dtype=float,
        )
        self.object_in_hand_eul_jit = np.array(
            [object_in_hand_roll_jit, object_in_hand_pitch_jit, object_in_hand_yaw_jit],
            dtype=float,
        )
        if np.any(self.object_in_hand_pos_jit < 0.0):
            raise ValueError("object-in-hand position jitter ranges must be non-negative")
        if np.any(self.object_in_hand_eul_jit < 0.0):
            raise ValueError("object-in-hand Euler jitter ranges must be non-negative")
        self.object_in_hand_debug = bool(object_in_hand_debug)
        self.grasp_parent_to_ee = None
        self.nominal_ee_to_parent = None
        self.randomized_ee_to_parent = None

        self.assembly_axis_local = np.asarray(assembly_axis_local, dtype=float)
        axis_norm = np.linalg.norm(self.assembly_axis_local)
        if axis_norm < 1e-8:
            raise ValueError("assembly_axis_local must be non-zero")
        self.assembly_axis_local /= axis_norm
        self.child_to_parent_target_pos = np.asarray(
            child_to_parent_target_pos, dtype=float
        )
        self.child_to_parent_target_orn = np.asarray(
            self.bullet_client.getQuaternionFromEuler(
                np.asarray(child_to_parent_target_euler, dtype=float)
            ),
            dtype=float,
        )

        parentobj_pose_base = np.asarray(parentobj_pose_base, dtype=float).copy()
        parentobj_euler_base = np.asarray(parentobj_euler_base, dtype=float).copy()
        childobj_pose_base = np.asarray(childobj_pose_base, dtype=float).copy()
        childobj_euler_base = np.asarray(childobj_euler_base, dtype=float).copy()

        self.parent_collision_path = (
            assembly_parent_collision_path
            if assembly_parent_collision_path is not None
            else coacd_convex_decomposition(self.assembly_parent_path)
        )
        self.child_collision_path = (
            assembly_child_collision_path
            if assembly_child_collision_path is not None
            else coacd_convex_decomposition(self.assembly_child_path)
        )
        self.com_parent = get_com(self.assembly_parent_path)
        self.com_child = get_com(self.assembly_child_path)
        self.initial_grasp_guess = load_initial_grasp_pose(initial_grasp_path)

        if randomize_lighting:
            self.lightingDR.sample_lighting_randomization()
        else:
            self.lightingDR.reset_to_default()

        if randomize_skybox:
            self._create_randomized_skybox(
                patterns=skybox_texture_patterns,
                texture_size=skybox_texture_size,
            )

        if randomize_robot_texture:
            self.robot_texture_cfg = self.robotTextureDR.sample_and_apply_robot_texture_randomization(
                body_id=self.panda,
                patterns=robot_texture_patterns,
                texture_size=robot_texture_size,
                per_link=robot_texture_per_link,
                specular_range=robot_texture_specular_range,
                alpha=None,
                original_texture_prob=robot_original_texture_prob,
            )
            # ic(self.robot_texture_cfg)
        else:
            self.robotTextureDR.reset(body_id=self.panda, restore_original=True)

        parent_base_orn = self.bullet_client.getQuaternionFromEuler(parentobj_euler_base)
        if randomize_parent_objpose:
            parent_pose, parent_orn = self.parentPoseDR.sample_SE3_randomization(
                pos=parentobj_pose_base,
                orn=parent_base_orn,
                x_jitter_range=parentobj_x_jit,
                y_jitter_range=parentobj_y_jit,
                z_jitter_range=parentobj_z_jit,
                z_euler_jitter_range=parentobj_z_eul_jit,
            )
        else:
            parent_pose, parent_orn = parentobj_pose_base, parent_base_orn

        child_base_orn = self.bullet_client.getQuaternionFromEuler(childobj_euler_base)
        if randomize_child_objpose:
            child_pose, child_orn = self.childPoseDR.sample_SE3_randomization(
                pos=childobj_pose_base,
                orn=child_base_orn,
                x_jitter_range=childobj_x_jit,
                y_jitter_range=childobj_y_jit,
                z_jitter_range=childobj_z_jit,
                z_euler_jitter_range=childobj_z_eul_jit,
            )
        else:
            child_pose, child_orn = childobj_pose_base, child_base_orn

        if randomize_image_noise:
            self.agentviewImgDR.sample_image_noise_randomization(
                brightness_range=(-16.0, 16.0),
                contrast_range=(0.85, 1.18),
                gamma_range=(0.85, 1.20),
                saturation_range=(0.75, 1.30),
                rgb_gain_range=(0.85, 1.15),
                hue_shift_deg_range=(-5.0, 5.0),
                color_matrix_strength_range=(0.0, 0.08),
                gray_mix_range=(0.0, 0.12),
                vignette_strength_range=(0.0, 0.10),
                gaussian_std_range=(0.0, 3.0),
                salt_pepper_prob_range=(0.0, 0.0015),
                blur_prob_range=(0.0, 0.12),
            )
            self.eyeImgDR.sample_image_noise_randomization(
                brightness_range=(-20.0, 20.0),
                contrast_range=(0.82, 1.22),
                gamma_range=(0.82, 1.25),
                saturation_range=(0.70, 1.35),
                rgb_gain_range=(0.82, 1.18),
                hue_shift_deg_range=(-6.0, 6.0),
                color_matrix_strength_range=(0.0, 0.10),
                gray_mix_range=(0.0, 0.16),
                vignette_strength_range=(0.0, 0.16),
                gaussian_std_range=(0.0, 3.5),
                salt_pepper_prob_range=(0.0, 0.002),
                blur_prob_range=(0.0, 0.16),
            )
        else:
            self.agentviewImgDR.reset()
            self.eyeImgDR.reset()

        outscene_base_pos = np.array([0.0, 0.0, -0.005], dtype=float)
        outscene_base_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if randomize_outlscene:
            env_mesh_pos, env_mesh_orn = self.outsceneDR.sample_SE3_randomization(
                pos=outscene_base_pos,
                orn=outscene_base_orn,
                x_jitter_range=outlscene_xyz_jit,
                y_jitter_range=outlscene_xyz_jit,
                z_jitter_range=outlscene_xyz_jit,
                x_euler_jitter_range=None,
                y_euler_jitter_range=None,
                z_euler_jitter_range=outlscene_eul_jit,
            )
        else:
            env_mesh_pos, env_mesh_orn = outscene_base_pos, outscene_base_orn

        if randomize_plane_height:
            ground_pose = self.collplaneDR.sample_pos_randomization(
                pos=[0.0, 0.0, env_mesh_pos[2]],
                z_jitter_range=plane_height_jit,
            )
        else:
            ground_pose = np.array([0.0, 0.0, env_mesh_pos[2]], dtype=np.float32)

        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_plane_id = self.bullet_client.loadURDF(
            "plane.urdf", basePosition=ground_pose
        )
        self.bullet_client.changeVisualShape(
            self.ground_plane_id, -1, rgbaColor=[1, 1, 1, 0]
        )

        if randomize_campose:
            self.extrinsic_cam = self.camposeDR.sample_SE3_randomization(
                pos=self.agentview_base_extrinsic_cam[:3, 3],
                orn=quat_from_rotation_matrix(self.agentview_base_extrinsic_cam[:3, :3]),
                x_jitter_range=cam_xyz_jit,
                y_jitter_range=cam_xyz_jit,
                z_jitter_range=cam_xyz_jit,
                x_euler_jitter_range=cam_eul_jit,
                y_euler_jitter_range=cam_eul_jit,
                z_euler_jitter_range=cam_eul_jit,
                get_matrix=True,
            )
        else:
            self.extrinsic_cam = self.agentview_base_extrinsic_cam.copy()

        if randomize_fisheye_cam:
            self.T_eye_parent_cam = self.fisheyeCamDR.sample_SE3_randomization(
                pos=self.T_eye_base_parent_cam[:3, 3],
                orn=quat_from_rotation_matrix(self.T_eye_base_parent_cam[:3, :3]),
                x_jitter_range=fisheye_eyz_jit,
                y_jitter_range=fisheye_eyz_jit,
                z_jitter_range=fisheye_eyz_jit,
                x_euler_jitter_range=fisheye_eul_jit,
                y_euler_jitter_range=fisheye_eul_jit,
                z_euler_jitter_range=fisheye_eul_jit,
                get_matrix=True,
            )
        else:
            self.T_eye_parent_cam = self.T_eye_base_parent_cam.copy()

        if randomize_camera_intrinsic:
            self.agentview_intrinsic = self.camIntrinsicDR.sample_intrinsic_randomization(
                self.agentview_base_intrinsic,
                focal_scale_range=agentview_focal_scale_range,
                principal_jit_px=agentview_principal_jit_px,
                width=self.agentview_width,
                height=self.agentview_height,
            )
            self.eye_K = self.camIntrinsicDR.sample_intrinsic_randomization(
                self.eye_base_K,
                focal_scale_range=eye_focal_scale_range,
                principal_jit_px=eye_principal_jit_px,
                width=self.eye_raw_width,
                height=self.eye_raw_height,
            )
        else:
            self.agentview_intrinsic = self.agentview_base_intrinsic.copy()
            self.eye_K = self.eye_base_K.copy()

        self.eye_fisheye_remap = self.build_eye_fisheye_remap(
            out_width=self.eye_obs_width,
            out_height=self.eye_obs_height,
            face_size=self.eye_face_size,
        )

        self.env_mesh = load_models(
            self.bullet_client,
            visual_mesh_file=self.env_mesh_path,
            vhacd_mesh_file=None,
            desired_mass=0.0,
            position=env_mesh_pos,
            baseOrientation=env_mesh_orn,
            visual_only=True,
        )

        self.assembly_child_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.assembly_child_path,
            vhacd_mesh_file=self.child_collision_path,
            desired_mass=float(child_mass),
            position=child_pose,
            baseOrientation=child_orn,
            center_of_mass=np.asarray(self.com_child),
            lateral_friction=float(child_lateral_friction),
            spinning_friction=0.0,
        )
        self.child_obj_id = self.assembly_child_id
        for link_idx in range(-1, self.bullet_client.getNumJoints(self.assembly_child_id)):
            try:
                self.bullet_client.changeVisualShape(
                    self.assembly_child_id,
                    link_idx,
                    rgbaColor=list(child_rgba),
                )
            except Exception:
                pass

        self.assembly_parent_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.assembly_parent_path,
            vhacd_mesh_file=self.parent_collision_path,
            desired_mass=float(parent_mass),
            position=parent_pose,
            baseOrientation=parent_orn,
            center_of_mass=np.asarray(self.com_parent),
            lateral_friction=float(parent_lateral_friction),
            spinning_friction=0.0002,
        )
        self.parent_obj_id = self.assembly_parent_id
        self.pick_up_obj_id = self.assembly_parent_id
        # A newly loaded body starts with normal collision behavior.
        self.parent_gripper_collision_disabled = False

        # Improve physical grasp stability without rigidly attaching the object.
        for finger_link in (9, 10):
            self.bullet_client.changeDynamics(
                self.panda,
                finger_link,
                lateralFriction=1.2,
                spinningFriction=0.01,
                rollingFriction=0.001,
            )

        if randomize_object_color:
            self.object_color_cfg = self.objectColorDR.sample_and_apply_object_color_randomization(
                body_id=self.assembly_parent_id,
                mode=object_color_mode,
                strength=object_color_strength,
                recolor_palette=object_recolor_palette,
                recolor_target_color=object_recolor_target_color,
                specular_range=object_specular_range,
                alpha=None,)
            
            self.object_color_cfg = self.objectColorDR.sample_and_apply_object_color_randomization(
                body_id=self.assembly_child_id,
                mode=object_color_mode,
                strength=object_color_strength,
                recolor_palette=object_recolor_palette,
                recolor_target_color=object_recolor_target_color,
                specular_range=object_specular_range,
                alpha=None,
            )
        else:
            self.objectColorDR.reset()
            self.object_color_cfg = None

        self.waite_scene_stable()

        if randomize_distractors:
            self.distractorDR.sample_and_load_distractors(
                distractor_root=distractor_root,
                num_range=distractor_num_range,
                target_size_range=distractor_target_size_range,
                workspace=distractor_workspace,
                clearance=distractor_clearance,
                path_clearance=distractor_path_clearance,
                min_target_mask_pixels=distractor_min_target_mask_pixels,
                target_body_id=self.assembly_parent_id,
                robot_body_id=self.panda,
                robot_base_offset=self.offset,
                planned_waypoints=self.get_state_machine_ee_waypoints(),
                render_agentview_fn=lambda: self.get_agentview_image(segmentation=True),
                end_effector_index=pandaEndEffectorIndex,
                ik_lower_limits=ll,
                ik_upper_limits=ul,
                ik_joint_ranges=jr,
                get_current_arm_joints_fn=self.get_current_arm_joints,
                quat_slerp_fn=quat_slerp,
                panda_num_dofs=pandaNumDofs,
                ground_z=float(ground_pose[2]),
                spawn_clearance=0.005,
                check_robot_plan=True,
                check_xy_safety=True,
                min_visible_fraction=0.55,
                debug=False,
            )
        else:
            self.distractorDR.clear_distractors()

        for _ in range(5):
            self.bullet_client.stepSimulation()

        self.initial_parent_pos, self.initial_parent_orn = self.get_parent_obj_pose()
        self.initial_child_pos, self.initial_child_orn = self.get_child_obj_pose()
        self.target_gripper = self.GRIPPER_OPEN
        self.set_gripper_state(self.target_gripper)

        # Rebuild the first state now that task assets and grasp annotation exist.
        if self.state == "home":
            self.prepare_state(self.state)

    def get_state_machine_ee_waypoints(self):
        """Approximate the complete pick-up and insertion path for distractor checks."""
        grasp_pos, grasp_orn = self.get_initial_guess_grasp()
        pregrasp_pos = grasp_pos + np.array([0.0, 0.0, self.safe_approach])
        lifted_pos = grasp_pos + np.array([0.0, 0.0, self.safe_grasp_offset + 0.12])

        pre_parent_pos, pre_parent_orn = self.get_preassembly_parent_pose()
        target_parent_pos, target_parent_orn = self.get_assembly_target_parent_pose()
        annotated_parent_to_ee = self.get_parent_to_ee_transform(
            ee_pos=grasp_pos, ee_orn=grasp_orn
        )
        pre_ee_pos, pre_ee_orn = self.get_ee_pose_for_parent_pose(
            pre_parent_pos, pre_parent_orn, annotated_parent_to_ee
        )
        target_ee_pos, target_ee_orn = self.get_ee_pose_for_parent_pose(
            target_parent_pos, target_parent_orn, annotated_parent_to_ee
        )

        return [
            (self.home_ee_pos.copy(), self.home_ee_orn.copy()),
            (pregrasp_pos, grasp_orn.copy()),
            (grasp_pos.copy(), grasp_orn.copy()),
            (lifted_pos, grasp_orn.copy()),
            (pre_ee_pos, pre_ee_orn),
            (target_ee_pos, target_ee_orn),
        ]

    def waite_scene_stable(self, waite_steps=1000, vel_threshold=0.005):
        steps = 0
        while steps < waite_steps:
            self.bullet_client.stepSimulation()
            steps += 1
            vel, ang_vel = self.bullet_client.getBaseVelocity(self.assembly_parent_id)
            speed = np.linalg.norm(vel) + np.linalg.norm(ang_vel)
            if speed < vel_threshold:
                print("Scene stabilized.")
                return True
        print("Warning: parent object did not stabilize within timeout.")
        return False

    def get_fixed_normal_grasp_orn(self, raw_grasp_orn=None):
        """Fix only the grasp positive normal direction in world frame.

        The EE local +Z axis is forced to align with self.grasp_world_normal.
        The in-plane x axis is taken from the original annotated grasp orientation,
        then canonicalized to avoid 180-degree flips.
        """
        z_axis = normalize_vector(self.grasp_world_normal)

        # Use the original grasp orientation to choose the in-plane direction.
        # This preserves some information from the manually annotated grasp,
        # instead of fully hard-coding the whole orientation.
        if raw_grasp_orn is not None:
            raw_rot = np.array(
                self.bullet_client.getMatrixFromQuaternion(raw_grasp_orn),
                dtype=float,
            ).reshape(3, 3)

            # EE local +X axis from the original grasp orientation.
            tangent = raw_rot[:, 0]
        else:
            tangent = np.asarray(self.grasp_world_tangent, dtype=float)

        # Project tangent onto the plane perpendicular to the fixed normal.
        tangent = tangent - np.dot(tangent, z_axis) * z_axis

        if np.linalg.norm(tangent) < 1e-8:
            tangent = np.asarray(self.grasp_world_tangent, dtype=float)
            tangent = tangent - np.dot(tangent, z_axis) * z_axis

        if np.linalg.norm(tangent) < 1e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(fallback, z_axis)) > 0.95:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            tangent = fallback - np.dot(fallback, z_axis) * z_axis

        x_axis = normalize_vector(tangent)

        # Canonicalize the x direction.
        # For a parallel gripper, +x and -x are often physically equivalent,
        # but they create a 180-degree quaternion/action jump.
        ref = np.asarray(self.grasp_world_tangent, dtype=float)
        ref = ref - np.dot(ref, z_axis) * z_axis

        if np.linalg.norm(ref) > 1e-8:
            ref = normalize_vector(ref)
            if np.dot(x_axis, ref) < 0.0:
                x_axis = -x_axis

        y_axis = normalize_vector(np.cross(z_axis, x_axis))
        x_axis = normalize_vector(np.cross(y_axis, z_axis))

        # Columns are EE local x/y/z axes expressed in world frame.
        rot = np.column_stack([x_axis, y_axis, z_axis])
        return quat_from_rotation_matrix(rot)


    def get_initial_guess_grasp(self):
        parent_world_pos, parent_world_orn = self.get_parent_obj_pose()
        grasp_pos, raw_grasp_orn = self.bullet_client.multiplyTransforms(
            parent_world_pos.tolist(),
            parent_world_orn.tolist(),
            self.initial_grasp_guess["t"],
            self.initial_grasp_guess["quat"],
        )
        grasp_orn = self.get_fixed_normal_grasp_orn(raw_grasp_orn)
        return np.asarray(grasp_pos, dtype=float), np.asarray(grasp_orn, dtype=float)

    def set_gripper_state(self, state):
        """High-level binary gripper command: 1=open, 0=closed.

        After the parent has been rigidly attached, the policy/action label stays
        CLOSED (0), but the simulated finger motor target is held at the actual
        width reached by the preceding nominal physical grasp. This prevents the
        fingers from snapping shut when parent/gripper collision is disabled.
        """
        self.target_gripper = (
            self.GRIPPER_OPEN if float(state) >= 0.5 else self.GRIPPER_CLOSED
        )

        hold_width = getattr(self, "nominal_grasp_finger_width", None)
        should_hold_nominal_width = (
            self.target_gripper == self.GRIPPER_CLOSED
            and getattr(self, "hold_nominal_gripper_width_after_attach", False)
            and getattr(self, "parent_grasp_constraint_id", None) is not None
            and hold_width is not None
        )

        if should_hold_nominal_width:
            self.set_gripper_width(hold_width)
        else:
            self.set_gripper_width(
                self.gripper_state_to_width(self.target_gripper)
            )


    def offset_pos_along_local_axis(self, pos, quat_xyzw, local_axis, distance):
        """Offset a world-frame position along an axis defined in the pose local frame.

        For pre-engage, use local_axis=[1, 0, 0] and distance=-0.1 to move
        10 cm along the engaging TCP pose's local -X direction, rather than
        subtracting from the global/world x coordinate.
        """
        pos = np.asarray(pos, dtype=float)
        quat_xyzw = np.asarray(quat_xyzw, dtype=float)
        quat_norm = np.linalg.norm(quat_xyzw)
        if quat_norm < 1e-12:
            raise ValueError("Invalid zero-norm quaternion for local-axis offset.")
        quat_xyzw = quat_xyzw / quat_norm

        local_axis = np.asarray(local_axis, dtype=float)
        axis_norm = np.linalg.norm(local_axis)
        if axis_norm < 1e-12:
            raise ValueError("Invalid zero-norm local axis for local-axis offset.")
        local_axis = local_axis / axis_norm

        axis_world = R.from_quat(quat_xyzw).apply(local_axis)
        axis_world = axis_world / np.linalg.norm(axis_world)
        return pos + float(distance) * axis_world

    def prepare_state(self, state):
        ee_pos, ee_orn = self.get_ee_pose()
        self.motion_start_pos = ee_pos.copy()
        self.motion_start_orn = ee_orn.copy()
        self.motion_target_pos = ee_pos.copy()
        self.motion_target_orn = ee_orn.copy()
        self.parent_motion_start_pos = None
        self.parent_motion_start_orn = None
        self.parent_motion_target_pos = None
        self.parent_motion_target_orn = None

        if self.target_gripper is None:
            self.target_gripper = self.GRIPPER_OPEN

        if state == "home":
            self.motion_target_pos = self.home_ee_pos.copy()
            self.motion_target_orn = self.home_ee_orn.copy()
            self.target_gripper = self.GRIPPER_OPEN
            self.set_gripper_state(self.target_gripper)

        elif state == "move_pregrasp":
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()
            self.motion_target_pos = grasp_pos + np.array(
                [0.0, 0.0, self.safe_approach], dtype=float
            )
            self.motion_target_orn = grasp_orn.copy()

        elif state == "open_gripper":
            self.target_gripper = self.GRIPPER_OPEN

        elif state == "move_grasp":
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()
            self.last_grasp_pose = grasp_pos.copy()
            self.last_grasp_orn = grasp_orn.copy()
            self.motion_target_pos = grasp_pos.copy()
            self.motion_target_orn = grasp_orn.copy()

        elif state == "close_gripper":
            self.target_gripper = self.GRIPPER_CLOSED

        elif state == "lift_parent":
            self.motion_target_pos = ee_pos + np.array(
                [0.0, 0.0, self.safe_grasp_offset + 0.12], dtype=float
            )
            self.motion_target_orn = ee_orn.copy()

        elif state in ("move_preassembly", "assemble"):
            self.parent_motion_start_pos, self.parent_motion_start_orn = (
                self.get_parent_obj_pose()
            )
            if state == "move_preassembly":
                target_parent_pose = self.get_preassembly_parent_pose()
            else:
                target_parent_pose = self.get_assembly_target_parent_pose()
            (
                self.parent_motion_target_pos,
                self.parent_motion_target_orn,
            ) = target_parent_pose

        elif state == "hold_assembled":
            self.motion_target_pos = ee_pos.copy()
            self.motion_target_orn = ee_orn.copy()
            self.target_gripper = self.GRIPPER_CLOSED

    def switch_to_next_state(self):
        # Execute nominal grasp first. Once close_gripper completes, perturb the
        # measured object-in-hand pose and rigidly attach the parent before lift.
        # Data collection can remain continuous; this intentionally models a
        # sampled post-grasp state rather than physical slip dynamics.
        if self.state == "close_gripper":
            self.randomize_and_fix_parent_to_gripper()

        self.state_idx += 1
        if self.state_idx >= len(self.states):
            self.done = True
            return

        self.state = self.states[self.state_idx]
        self.state_t = 0.0
        self.prepare_state(self.state)
        print("state ->", self.state)

    def step(self):
        if self.done:
            return self.target_pos, self.target_orn

        self.t += self.control_dt
        self.state_t += self.control_dt
        duration = self.state_durations[self.state_idx]
        s = min(self.state_t / max(duration, 1e-8), 1.0)

        if self.state in ["open_gripper", "close_gripper"]:
            self.set_gripper_state(self.target_gripper)
            self.target_pos, self.target_orn = self.get_ee_pose()
        elif self.state in ("move_preassembly", "assemble"):
            desired_parent_pos = (
                (1.0 - s) * self.parent_motion_start_pos
                + s * self.parent_motion_target_pos
            )
            desired_parent_orn = quat_slerp(
                self.parent_motion_start_orn,
                self.parent_motion_target_orn,
                s,
            )
            # Continuously measure parent-object -> EE. With the rigid
            # post-grasp constraint this is effectively fixed, while the expert
            # still follows the same live transform-composition path.
            self.target_pos, self.target_orn = self.get_ee_pose_for_parent_pose(
                desired_parent_pos, desired_parent_orn
            )
            self.solve_ik_and_apply(self.target_pos, self.target_orn)
            self.set_gripper_state(self.target_gripper)
        else:
            self.target_pos = (
                (1.0 - s) * self.motion_start_pos + s * self.motion_target_pos
            )
            self.target_orn = quat_slerp(
                self.motion_start_orn, self.motion_target_orn, s
            )
            self.solve_ik_and_apply(self.target_pos, self.target_orn)
            self.set_gripper_state(self.target_gripper)

        if self.state_t >= duration:
            self.switch_to_next_state()

        return self.target_pos, self.target_orn

    def _quat_axis(self, quat_xyzw, local_axis):
        quat_xyzw = np.asarray(quat_xyzw, dtype=float)
        local_axis = np.asarray(local_axis, dtype=float)
        qn = np.linalg.norm(quat_xyzw)
        an = np.linalg.norm(local_axis)
        if qn < 1e-12 or an < 1e-12:
            return None
        axis = R.from_quat(quat_xyzw / qn).apply(local_axis / an)
        norm = np.linalg.norm(axis)
        return None if norm < 1e-8 else axis / norm

    def get_assembly_alignment_metrics(self):
        """Measure ring-center/rod-center alignment in the child coordinate frame."""
        if not hasattr(self, "assembly_parent_id") or not hasattr(self, "assembly_child_id"):
            return None

        parent_pos, parent_orn = self.get_parent_obj_pose()
        target_pos, target_orn = self.get_assembly_target_parent_pose()
        _, child_orn = self.get_child_obj_pose()

        axis_world = self._quat_axis(child_orn, self.assembly_axis_local)
        parent_axis = self._quat_axis(parent_orn, self.assembly_axis_local)
        target_axis = self._quat_axis(target_orn, self.assembly_axis_local)
        if axis_world is None or parent_axis is None or target_axis is None:
            return None

        delta = parent_pos - target_pos
        axial_error = float(np.dot(delta, axis_world))
        radial_vec = delta - axial_error * axis_world
        radial_error = float(np.linalg.norm(radial_vec))

        # A ring axis is sign-symmetric, so +axis and -axis are equivalent.
        axis_error_rad = float(
            np.arccos(np.clip(abs(np.dot(parent_axis, target_axis)), -1.0, 1.0))
        )
        orientation_error_rad = float(
            np.linalg.norm(
                (R.from_quat(target_orn).inv() * R.from_quat(parent_orn)).as_rotvec()
            )
        )
        return {
            "radial_error": radial_error,
            "axial_error": axial_error,
            "axis_error_rad": axis_error_rad,
            "orientation_error_rad": orientation_error_rad,
            "parent_pos": parent_pos,
            "target_pos": target_pos,
        }

    def has_parent_child_contact(self, max_contact_distance=0.003):
        contacts = self.bullet_client.getContactPoints(
            bodyA=self.assembly_parent_id,
            bodyB=self.assembly_child_id,
        )
        return any(c[8] <= float(max_contact_distance) for c in contacts)

    def is_parent_grasped(self):
        if self.parent_grasp_constraint_id is not None:
            return True
        contacts = self.bullet_client.getContactPoints(
            bodyA=self.panda,
            bodyB=self.assembly_parent_id,
        )
        return len(contacts) > 0 and self.get_gripper_mean_width() < 0.03

    def is_success(self,
                   radial_tol=0.018,
                   axial_tol=0.025,
                   axis_tol_deg=15.0,
                   require_grasp=True,
                   require_parent_child_contact=False,
                   require_done=False,
                   return_info=False,
                   debug=False):
        """Check whether the parent ring is centered and inserted over the child rod."""
        info = {"success": False, "reason": None}

        def finish(value):
            info["success"] = bool(value)
            if debug:
                print("\n========== [AssemblySim is_success] ==========")
                for key, val in info.items():
                    print(f"{key}: {val}")
                print("==============================================\n")
            return (bool(value), info) if return_info else bool(value)

        if not hasattr(self, "assembly_parent_id"):
            info["reason"] = "missing_parent"
            return finish(False)
        if not hasattr(self, "assembly_child_id"):
            info["reason"] = "missing_child"
            return finish(False)
        if require_done and not self.done:
            info["reason"] = "state_machine_not_done"
            return finish(False)

        metrics = self.get_assembly_alignment_metrics()
        if metrics is None:
            info["reason"] = "invalid_alignment_metrics"
            return finish(False)

        info.update({
            "radial_error": metrics["radial_error"],
            "axial_error": metrics["axial_error"],
            "axis_error_deg": math.degrees(metrics["axis_error_rad"]),
            "radial_tol": float(radial_tol),
            "axial_tol": float(axial_tol),
            "axis_tol_deg": float(axis_tol_deg),
        })

        if metrics["radial_error"] > float(radial_tol):
            info["reason"] = "parent_not_centered_on_child"
            return finish(False)
        if abs(metrics["axial_error"]) > float(axial_tol):
            info["reason"] = "parent_not_at_insertion_depth"
            return finish(False)
        if metrics["axis_error_rad"] > math.radians(float(axis_tol_deg)):
            info["reason"] = "parent_child_axis_misaligned"
            return finish(False)

        if require_grasp:
            grasped = self.is_parent_grasped()
            info["parent_grasped"] = bool(grasped)
            if not grasped:
                info["reason"] = "parent_not_grasped"
                return finish(False)

        if require_parent_child_contact:
            has_contact = self.has_parent_child_contact()
            info["parent_child_contact"] = bool(has_contact)
            if not has_contact:
                info["reason"] = "no_parent_child_contact"
                return finish(False)

        info["reason"] = "success"
        return finish(True)
