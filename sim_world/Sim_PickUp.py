import numpy as np
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
    quat_from_rotation_matrix
    ) 

from sim_world.Base_Simulation import Base_Simulation

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
ul = [7] * pandaNumDofs
jr = [7] * pandaNumDofs

jointPositions = [
    -0.0768761337796847,
    0.1692503838434554,
    -0.5782208367480097,
    -1.4272947420449444,
    0.055714113760170644,
    1.5859262946844102,
    0.0,
    0.04,
    0.04,
]
rp = jointPositions


class PickUpSim(Base_Simulation):
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
        )
        self.objposeDR = PoseDR(self.bullet_client, seed=seed)
        self.fpsaObjectDR = FPSAObjectDR(seed=seed)
        self.safe_approach = 0.1
        self.states = [
            "home",
            "move_pregrasp",
            "open_gripper",
            "move_grasp",
            "close_gripper",
            "lift_object",
        ]
        self.state_durations = [0.01, 5.0, 0.25, 2.0, 0.5, 2.0]
        self.state_idx = 0
        self.state = self.states[self.state_idx]
        self.state_t = 0.0
        self.target_pos = None
        self.target_orn = None
        self.target_gripper = None
        self.done = False
        self.motion_start_pos = None
        self.motion_start_orn = None
        self.motion_target_pos = None
        self.motion_target_orn = None
        self.last_grasp_pose = None
        self.last_grasp_orn = None
        self.prepare_state(self.state)


    def make_scene(self, 
                   env_mesh_path        = None,
                   manipulated_obj_path = None,
                   initial_grasp_path = None,
                   if_FPSA = False,
                   fpsa_aug_root = "~/GeoBridge/data/objects/bracket/fpsa_aug_outputs",
                   fpsa_include_base = True,
                   obj_pose_base = [0.5, 0.0, 0.0],
                   obj_euler_base = [0.0, 0.0, 0.0],
                   randomize_lighting = True,
                   # outlier scene         
                   randomize_outlscene  = True,
                   outlscene_xyz_jit    = 0.02,
                   outlscene_eul_jit    = 0.01,
                   # plane height randomization
                   randomize_plane_height = True,
                   plane_height_jit = 0.008,
                   randomize_objpose  = True,
                   obj_x_jit    = 0.2,
                   obj_y_jit    = 0.2,
                   obj_z_eul_jit = np.pi,
                   randomize_campose = True,
                   cam_xyz_jit  = 0.004,
                   cam_eul_jit  = 0.002,
                   randomize_fisheye_cam = True,
                   fisheye_eyz_jit = 0.005,
                   fisheye_eul_jit = 0.002,
                   randomize_camera_intrinsic = True,
                   agentview_focal_scale_range = (0.88, 1.15),
                   agentview_principal_jit_px = 18.0,
                   eye_focal_scale_range = (0.90, 1.12),
                   eye_principal_jit_px = 8.0,
                   randomize_image_noise = True,
                   randomize_robot_texture = True,
                   robot_texture_patterns = ("checkers", "gradient", "noise", "plain"),
                   robot_texture_size = 128,
                   robot_texture_per_link = True,
                   robot_texture_specular_range = (0.02, 0.25),
                   robot_original_texture_prob = 0.10,
                   skybox_texture_patterns = ("checkers", "gradient", "noise", "plain"),
                   skybox_texture_size = 256,
                   # manipulated object color / material randomization
                   randomize_object_color = True,
                   object_color_mode = "bounded",  # "bounded" or "recolor"
                   object_color_strength = 0.35,
                   object_recolor_palette = None,
                   object_recolor_target_color = None,
                   object_specular_range = (0.02, 0.8),
                   randomize_distractors = True,
                   distractor_root = "/mnt/storage/GoogleScannedObjects",
                   distractor_num_range = (1, 5),
                   distractor_target_size_range = (0.06, 0.16),
                   distractor_workspace = ((0.25, 0.78), (-0.42, 0.42)),
                   distractor_clearance = 0.035,
                   distractor_path_clearance = 0.04,
                   distractor_min_target_mask_pixels = 1,
                   ):
        
        ## TODO: visual things to be randomized:
        # 1. random number and shape of distractors (no need accurate convex decomposition)
        # 2. slight disturbance of the object texture, ground texture
        # 3. objects pose and z axis orientation 
        # 4. number of lights in the scene
        # 5. position, orientation, specular characteristics of the lights
        # 6. types and amount of random noise added to the images 
        # TODO: advanced visual randomization:
        # 7. outlier scene pose randomization
        # 8. plane height
        # 9. camera pose and orn

        # always enable high quality rendering pipeline and shadows

        """ ################ Load basic scene assets ################ """
        # Avoid mutating caller-provided lists/default arguments when pose randomization is enabled.
        obj_pose_base = np.array(obj_pose_base, dtype=float).copy()
        obj_euler_base = np.array(obj_euler_base, dtype=float).copy()

        if if_FPSA:
            manipulated_obj_path, initial_grasp_path = self.fpsaObjectDR.sample(
                base_mesh_path=manipulated_obj_path,
                base_grasp_path=initial_grasp_path,
                fpsa_aug_root=fpsa_aug_root,
                include_base=fpsa_include_base,
            )
            self.fpsa_object_sample = self.fpsaObjectDR.last_sample
        else:
            self.fpsa_object_sample = None

        self.env_mesh_path = env_mesh_path
        self.pick_up_obj_path = manipulated_obj_path
        self.initial_grasp_path = initial_grasp_path
        self.convex_pick_up_obj_path = coacd_convex_decomposition(self.pick_up_obj_path)
        self.com_pick_up_obj = get_com(self.pick_up_obj_path)

        """ ################ Basic Visual Randomization ############### """
        if randomize_lighting:
            self.lightingDR.sample_lighting_randomization()
        else:
            self.lightingDR.reset_to_default()

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
        else:
            self.robotTextureDR.reset(body_id=self.panda, restore_original=True)
        
        if randomize_objpose:
            objPose, objOrn = self.objposeDR.sample_SE3_randomization(
                pos=obj_pose_base,
                orn=self.bullet_client.getQuaternionFromEuler(np.array(obj_euler_base)),
                x_jitter_range=obj_x_jit,
                y_jitter_range=obj_y_jit,
                z_euler_jitter_range=obj_z_eul_jit,
            )
        else:
            objPose = obj_pose_base
            objOrn = self.bullet_client.getQuaternionFromEuler(np.array(obj_euler_base))

        if randomize_image_noise:
            self.agentviewImgDR.sample_image_noise_randomization()
            self.eyeImgDR.sample_image_noise_randomization()
        else:
            self.agentviewImgDR.reset()
            self.eyeImgDR.reset()

        # Background scene mesh pose and invisible collision-plane height.
        #
        # The visual scene mesh gets its own pose randomization first.
        # Then the collision plane follows the randomized mesh z and receives
        # an extra height jitter:
        #
        #   mesh_z  = mesh_z_base
        #   plane_z = mesh_z_base + ground_z_jit
        outscene_base_pos = np.array([0.0, 0.0, -0.005], dtype=float)
        outscene_base_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        if randomize_outlscene:
            env_mesh_pos, env_mesh_orn = self.outsceneDR.sample_SE3_randomization(
                pos=outscene_base_pos,
                orn=outscene_base_orn,
                x_jitter_range=outlscene_xyz_jit,
                y_jitter_range=outlscene_xyz_jit,
                z_jitter_range=outlscene_xyz_jit,
                x_euler_jitter_range= None,
                y_euler_jitter_range= None,
                z_euler_jitter_range=outlscene_eul_jit,
            )
        else:
            env_mesh_pos = outscene_base_pos.copy()
            env_mesh_orn = outscene_base_orn.copy()

        # collision plane height
        if randomize_plane_height:
            ground_pose = self.collplaneDR.sample_pos_randomization(
                pos= [0.0, 0.0, env_mesh_pos[2]],
                z_jitter_range= plane_height_jit,
            )
        else:
            ground_pose = np.array([0.0, 0.0, env_mesh_pos[2]], dtype=np.float32)
        # Load ground plane but make it invisible.
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_plane_id = self.bullet_client.loadURDF(
            "plane.urdf",
            basePosition=ground_pose,
        )
        # Make the plane invisible but keep collision.
        self.bullet_client.changeVisualShape(
            self.ground_plane_id,
            -1,
            rgbaColor=[1, 1, 1, 0],  # alpha = 0
        )

        # camera pose randomization
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

        # Load background / outlier scene as visual-only.
        self.env_mesh = load_models(
            self.bullet_client,
            visual_mesh_file=self.env_mesh_path,
            vhacd_mesh_file=None,
            desired_mass=0.0,
            position=env_mesh_pos,
            baseOrientation=env_mesh_orn,
            visual_only=True,
        )

        self.pick_up_obj_id = load_models(
            self.bullet_client,
            visual_mesh_file=self.pick_up_obj_path,
            vhacd_mesh_file=self.convex_pick_up_obj_path,
            desired_mass=0.5,
            position= objPose,
            baseOrientation= objOrn,
            center_of_mass=np.array(self.com_pick_up_obj),
            lateral_friction=0.6,
        )

        # Object-level color / material domain randomization.
        # This is applied after the target object has been loaded into PyBullet.
        # It reads the current visual rgbaColor as the original/base color, then
        # applies a global tint through changeVisualShape(). For textured OBJ
        # assets, the PNG texture is not edited/replaced; the tint is applied on
        # top of the existing visual material, so texture details can remain.
        if randomize_object_color:
            self.object_color_cfg = self.objectColorDR.sample_and_apply_object_color_randomization(
                body_id=self.pick_up_obj_id,
                mode=object_color_mode,
                strength=object_color_strength,
                recolor_palette=object_recolor_palette,
                recolor_target_color=object_recolor_target_color,
                specular_range=object_specular_range,
                alpha=None,  # preserve current visual alpha
            )
        else:
            self.objectColorDR.reset()
            self.object_color_cfg = None

        self.initial_grasp_guess = load_initial_grasp_pose(initial_grasp_path)

        # Let the manipulated object settle first. The resulting AABB/grasp pose is
        # used to reject distractor placements around the target and robot path.
        self.waite_scene_stable()

        """############# Distractor Domain randomization ##############"""
        if randomize_distractors:
            self.distractorDR.sample_and_load_distractors(
                distractor_root=distractor_root,
                num_range=distractor_num_range,
                target_size_range=distractor_target_size_range,
                workspace=distractor_workspace,
                clearance=distractor_clearance,
                path_clearance=distractor_path_clearance,
                min_target_mask_pixels=distractor_min_target_mask_pixels,
                target_body_id=self.pick_up_obj_id,
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
                # Ground-relative distractor injection. The collision plane z can be randomized;
                # distractors should spawn relative to the actual current ground plane, not world z=0.
                ground_z=float(ground_pose[2]),
                spawn_clearance=0.005,
                # Keep this false by default. Full robot-plan IK rejection is very sensitive to
                # randomized scene/ground pose and can reject almost every candidate.
                check_robot_plan=False,
                check_xy_safety=True,
                min_visible_fraction=0.55,
                debug=False,
            )
        else:
            self.distractorDR.clear_distractors()

        # Step a few frames after static distractor creation so broad-phase contacts
        # are updated before collecting observations.
        for _ in range(5):
            self.bullet_client.stepSimulation()

        obj_pos, obj_orn = get_true_PositionAndOrientation(
            self.bullet_client, self.pick_up_obj_id
        )
        self.initial_obj_pos = np.array(obj_pos, dtype=float)
        self.initial_obj_orn = np.array(obj_orn, dtype=float)

    def get_state_machine_ee_waypoints(self):
        """Approximate the EE path used by the pick-up state machine."""
        grasp_pos, grasp_orn = self.get_initial_guess_grasp()

        pregrasp_pos = grasp_pos.copy()
        pregrasp_pos[2] += self.safe_approach

        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.2

        return [
            (self.home_ee_pos.copy(), self.home_ee_orn.copy()),
            (pregrasp_pos, grasp_orn.copy()),
            (grasp_pos.copy(), grasp_orn.copy()),
            (lift_pos, grasp_orn.copy()),
        ]

    def waite_scene_stable(self, waite_steps=1000, vel_threshold=0.005):       
        steps = 0 
        while steps < waite_steps:
            self.bullet_client.stepSimulation()
            steps += 1
            vel1, ang_vel1 = self.bullet_client.getBaseVelocity(self.pick_up_obj_id)

            speed1 = np.linalg.norm(vel1) + np.linalg.norm(ang_vel1)

            if speed1 < vel_threshold:
                print("Scene stabilized.")
                return True

        print("Warning: Scene did not stabilize within timeout.")
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
        mesh_world_pos, mesh_world_orn = get_true_PositionAndOrientation(
            self.bullet_client,
            self.pick_up_obj_id,
        )

        grasp_pose, raw_grasp_orn = self.bullet_client.multiplyTransforms(
            mesh_world_pos,
            mesh_world_orn,
            self.initial_grasp_guess["t"],
            self.initial_grasp_guess["quat"],
        )

        # Position follows the object-frame annotated grasp point.
        # Orientation fixes only the grasp normal, not the entire orientation.
        grasp_orn = self.get_fixed_normal_grasp_orn(raw_grasp_orn)

        return np.array(grasp_pose, dtype=float), np.array(grasp_orn, dtype=float)


    def prepare_state(self, state):
        ee_pos, ee_orn = self.get_ee_pose()

        self.motion_start_pos = ee_pos.copy()
        self.motion_start_orn = ee_orn.copy()
        self.motion_target_pos = ee_pos.copy()
        self.motion_target_orn = ee_orn.copy()

        if state == "home":
            self.motion_target_pos = self.home_ee_pos.copy()
            self.motion_target_orn = self.home_ee_orn.copy()
            self.target_gripper = self.GRIPPER_OPEN
            self.set_gripper_state(self.target_gripper)

        elif state == "move_pregrasp":
            # Simple pick-up: approach from above instead of deriving a lateral approach from the mug tree/rack.
            grasp_pos, grasp_orn = self.get_initial_guess_grasp()

            pregrasp_pos = grasp_pos.copy()
            pregrasp_pos[2] += self.safe_approach

            self.motion_target_pos = pregrasp_pos
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

        elif state == "lift_object":
            # After closing the gripper, lift along world Z to complete the simple pick-up.
            self.motion_target_pos = ee_pos.copy() + np.array([0.0, 0.0, 0.15], dtype=float)
            self.motion_target_orn = ee_orn.copy()

    def switch_to_next_state(self):
        self.state_idx += 1
        if self.state_idx >= len(self.states):
            self.done = True
            return

        self.state = self.states[self.state_idx]
        self.state_t = 0.0
        self.prepare_state(self.state)
        print("state ->", self.state)

    def step(self):
        self.t += self.control_dt
        self.state_t += self.control_dt

        duration = self.state_durations[self.state_idx]
        s = min(self.state_t / duration, 1.0)

        if self.state in ["open_gripper", "close_gripper"]:
            self.set_gripper_state(self.target_gripper)

            ee_pos, ee_orn = self.get_ee_pose()
            self.target_pos = ee_pos.copy()
            self.target_orn = ee_orn.copy()

        elif self.state in ["home", "move_pregrasp", "move_grasp", "lift_object"]:
            self.target_pos = (1.0 - s) * self.motion_start_pos + s * self.motion_target_pos
            self.target_orn = quat_slerp(self.motion_start_orn, self.motion_target_orn, s)
            self.solve_ik_and_apply(self.target_pos, self.target_orn)

        if self.state_t >= duration:
            self.switch_to_next_state()

        return self.target_pos, self.target_orn

    def is_success(self, lift_height_threshold=0.08, require_contact=True):
        if not hasattr(self, "pick_up_obj_id"):
            return False

        obj_pos, _ = get_true_PositionAndOrientation(
            self.bullet_client, self.pick_up_obj_id
        )
        obj_pos = np.array(obj_pos, dtype=float)

        if self.initial_obj_pos is None:
            lifted = obj_pos[2] > lift_height_threshold
        else:
            lifted = (obj_pos[2] - self.initial_obj_pos[2]) > lift_height_threshold

        if not require_contact:
            return bool(lifted)

        contacts = self.bullet_client.getContactPoints(
            bodyA=self.panda,
            bodyB=self.pick_up_obj_id,
        )
        grasped = len(contacts) > 0

        return bool(lifted and grasped)
