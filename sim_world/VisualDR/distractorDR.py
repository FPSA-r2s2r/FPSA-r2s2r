import os
import glob
import numpy as np


class DistractorDR(object):
    """
    Ground-relative distractor domain randomization.

    Key assumptions:
      - The collision ground plane z may be randomized.
      - Distractor spawn height is specified relative to the current ground_z.
      - Distractor injection should be stable even when the outlier scene / ground plane
        changes the target object's settled pose.

    The heavy robot-plan IK clearance check is optional and disabled by default. For
    data generation, the coarse XY checks + true target/other collision checks are
    usually enough and much more robust under scene/ground randomization.
    """

    def __init__(self, bullet_client, seed=42):
        self.bullet_client = bullet_client
        self.distractor_rng = np.random.default_rng(seed)
        self.distractor_ids = []
        self.distractor_metadata = []
        self.gso_bbox_cache = {}

    def clear_distractors(self):
        for body_id in self.distractor_ids:
            try:
                self.bullet_client.removeBody(body_id)
            except Exception:
                pass
        self.distractor_ids = []
        self.distractor_metadata = []

    def discover_gso_meshes(self, distractor_root):
        pattern = os.path.join(distractor_root, "*", "1", "meshes", "model.obj")
        meshes = sorted(glob.glob(pattern))
        if len(meshes) == 0:
            print(f"Warning: no GSO model.obj found under {distractor_root}")
        return meshes

    def read_obj_bbox(self, obj_path):
        if obj_path in self.gso_bbox_cache:
            return self.gso_bbox_cache[obj_path]

        vertices = []
        try:
            with open(obj_path, "r", errors="ignore") as f:
                for line in f:
                    if not line.startswith("v "):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
        except OSError as exc:
            print(f"Warning: cannot read OBJ {obj_path}: {exc}")
            return None

        if len(vertices) == 0:
            print(f"Warning: no vertices found in OBJ {obj_path}")
            return None

        vertices = np.asarray(vertices, dtype=np.float32)
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        dims = vmax - vmin
        if (not np.all(np.isfinite(dims))) or float(np.max(dims)) <= 1e-8:
            print(f"Warning: invalid bbox for OBJ {obj_path}")
            return None

        center = 0.5 * (vmin + vmax)
        bbox = (vmin, vmax, center, dims)
        self.gso_bbox_cache[obj_path] = bbox
        return bbox

    @staticmethod
    def point_segment_distance_xy(point_xy, a_xy, b_xy):
        point_xy = np.asarray(point_xy, dtype=float)
        a_xy = np.asarray(a_xy, dtype=float)
        b_xy = np.asarray(b_xy, dtype=float)
        ab = b_xy - a_xy
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(point_xy - a_xy))
        t = float(np.clip(np.dot(point_xy - a_xy, ab) / denom, 0.0, 1.0))
        closest = a_xy + t * ab
        return float(np.linalg.norm(point_xy - closest))

    @staticmethod
    def _bbox_to_scaled_half_extents(dims, target_size):
        max_dim = float(np.max(dims))
        if max_dim <= 1e-8:
            return None, None
        scale = float(target_size) / max_dim
        half_extents = 0.5 * np.asarray(dims, dtype=float) * scale
        half_extents = np.maximum(half_extents, np.array([0.008, 0.008, 0.008], dtype=float))
        return half_extents, scale

    def is_distractor_xy_safe(
        self,
        xy,
        half_extents,
        target_body_id,
        planned_waypoints=None,
        robot_base_offset=None,
        clearance=0.025,
        path_clearance=0.04,
        max_filter_radius_xy=0.075,
        max_target_filter_radius_xy=0.085,
        robot_base_exclusion=0.10,
        check_local_path=True,
    ):
        """Lightweight coarse XY check.

        This is intentionally not a hard physics check. It should only avoid obviously
        bad samples. True collision checks happen after the body is created.
        """
        xy = np.asarray(xy, dtype=float)
        half_extents = np.asarray(half_extents, dtype=float)

        raw_radius_xy = float(np.linalg.norm(half_extents[:2]))
        radius_xy = min(raw_radius_xy, float(max_filter_radius_xy))

        # Keep distractors away from the target, but cap the target radius so a large
        # or tilted settled AABB does not ban the whole workspace.
        try:
            aabb_min, aabb_max = self.bullet_client.getAABB(target_body_id)
            aabb_min = np.asarray(aabb_min, dtype=float)
            aabb_max = np.asarray(aabb_max, dtype=float)
            obj_center_xy = 0.5 * (aabb_min[:2] + aabb_max[:2])
            obj_radius_xy = 0.5 * float(np.linalg.norm(aabb_max[:2] - aabb_min[:2]))
            obj_radius_xy = min(obj_radius_xy, float(max_target_filter_radius_xy))
            if np.linalg.norm(xy - obj_center_xy) < obj_radius_xy + radius_xy + clearance:
                return False
        except Exception:
            pass

        # Small base exclusion only. The workspace should do most of this job.
        if robot_base_offset is not None:
            robot_base_offset = np.asarray(robot_base_offset, dtype=float)
            if np.linalg.norm(xy - robot_base_offset[:2]) < robot_base_exclusion + radius_xy:
                return False

        # Only protect local grasp/lift path. Do not protect home->pregrasp; that
        # segment creates a huge table-wide capsule and kills sampling.
        if check_local_path and planned_waypoints is not None and len(planned_waypoints) >= 3:
            start_i = 1
            for i in range(start_i, len(planned_waypoints) - 1):
                a = np.asarray(planned_waypoints[i][0][:2], dtype=float)
                b = np.asarray(planned_waypoints[i + 1][0][:2], dtype=float)
                dist = self.point_segment_distance_xy(xy, a, b)
                if dist < path_clearance + radius_xy + clearance:
                    return False

        for meta in self.distractor_metadata:
            other_xy = np.asarray(meta["xy"], dtype=float)
            other_r = min(float(meta.get("filter_radius_xy", meta["radius_xy"])), float(max_filter_radius_xy))
            if np.linalg.norm(xy - other_xy) < radius_xy + other_r + clearance:
                return False

        return True

    def create_bbox_collision_gso_body(
        self,
        obj_path,
        xy,
        yaw,
        target_size,
        ground_z=0.0,
        spawn_clearance=0.005,
    ):
        bbox = self.read_obj_bbox(obj_path)
        if bbox is None:
            return None, None

        _, _, bbox_center, dims = bbox
        half_extents, scale = self._bbox_to_scaled_half_extents(dims, target_size)
        if half_extents is None:
            return None, None

        visual_shift = (-np.asarray(bbox_center, dtype=float) * scale).tolist()

        # Ground-relative z. This is the important part for randomized ground planes.
        base_pos = [
            float(xy[0]),
            float(xy[1]),
            float(ground_z) + float(half_extents[2]) + float(spawn_clearance),
        ]
        base_orn = self.bullet_client.getQuaternionFromEuler([0.0, 0.0, float(yaw)])

        visual_shape = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_MESH,
            fileName=obj_path,
            meshScale=[scale, scale, scale],
            visualFramePosition=visual_shift,
        )
        collision_shape = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=half_extents.tolist(),
        )

        body_id = self.bullet_client.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=base_pos,
            baseOrientation=base_orn,
        )

        texture_path = os.path.join(os.path.dirname(obj_path), "texture.png")
        if os.path.exists(texture_path):
            try:
                texture_id = self.bullet_client.loadTexture(texture_path)
                self.bullet_client.changeVisualShape(body_id, -1, textureUniqueId=texture_id)
            except Exception:
                pass

        self.bullet_client.changeDynamics(
            body_id,
            -1,
            lateralFriction=0.8,
            spinningFriction=0.01,
            rollingFriction=0.01,
            restitution=0.0,
        )

        raw_radius_xy = float(np.linalg.norm(half_extents[:2]))
        meta = {
            "body_id": body_id,
            "obj_path": obj_path,
            "xy": [float(xy[0]), float(xy[1])],
            "z": float(base_pos[2]),
            "ground_z": float(ground_z),
            "yaw": float(yaw),
            "scale": float(scale),
            "target_size": float(target_size),
            "half_extents": half_extents.tolist(),
            "radius_xy": raw_radius_xy,
            "filter_radius_xy": min(raw_radius_xy, 0.075),
        }
        return body_id, meta

    def get_all_robot_joint_positions(self, robot_body_id):
        return [self.bullet_client.getJointState(robot_body_id, j)[0]
                for j in range(self.bullet_client.getNumJoints(robot_body_id))]

    def restore_all_robot_joint_positions(self, robot_body_id, q_all):
        for j, q in enumerate(q_all):
            self.bullet_client.resetJointState(robot_body_id, j, q)

    def distractor_too_close_to_robot_plan(
        self,
        body_id,
        robot_body_id,
        end_effector_index,
        planned_waypoints,
        ik_lower_limits,
        ik_upper_limits,
        ik_joint_ranges,
        get_current_arm_joints_fn,
        quat_slerp_fn,
        panda_num_dofs=7,
        clearance=0.025,
        samples_per_segment=3,
        skip_home_to_pregrasp=True,
    ):
        q_saved = self.get_all_robot_joint_positions(robot_body_id)
        try:
            start_i = 1 if skip_home_to_pregrasp else 0
            for i in range(start_i, len(planned_waypoints) - 1):
                p0, q0 = planned_waypoints[i]
                p1, q1 = planned_waypoints[i + 1]
                for s in np.linspace(0.0, 1.0, samples_per_segment):
                    pos = (1.0 - s) * p0 + s * p1
                    orn = quat_slerp_fn(q0, q1, float(s))
                    ik = self.bullet_client.calculateInverseKinematics(
                        robot_body_id,
                        end_effector_index,
                        pos.tolist(),
                        orn.tolist(),
                        ik_lower_limits,
                        ik_upper_limits,
                        ik_joint_ranges,
                        get_current_arm_joints_fn(),
                        maxNumIterations=80,
                    )
                    for j in range(panda_num_dofs):
                        self.bullet_client.resetJointState(robot_body_id, j, ik[j])

                    for finger_opening in [0.04, 0.0]:
                        if self.bullet_client.getNumJoints(robot_body_id) > 10:
                            self.bullet_client.resetJointState(robot_body_id, 9, finger_opening)
                            self.bullet_client.resetJointState(robot_body_id, 10, finger_opening)
                        self.bullet_client.performCollisionDetection()
                        pts = self.bullet_client.getClosestPoints(
                            bodyA=robot_body_id,
                            bodyB=body_id,
                            distance=clearance,
                        )
                        if len(pts) > 0:
                            return True
            return False
        finally:
            self.restore_all_robot_joint_positions(robot_body_id, q_saved)
            self.bullet_client.performCollisionDetection()

    @staticmethod
    def body_mask_pixel_count(render_agentview_fn, body_id):
        _, _, _, _, seg = render_agentview_fn()
        body_uid = np.bitwise_and(seg, (1 << 24) - 1)
        return int(np.count_nonzero(body_uid == int(body_id)))

    def loaded_distractor_is_safe(
        self,
        body_id,
        target_body_id,
        robot_body_id,
        end_effector_index,
        planned_waypoints,
        ik_lower_limits,
        ik_upper_limits,
        ik_joint_ranges,
        get_current_arm_joints_fn,
        quat_slerp_fn,
        render_agentview_fn,
        panda_num_dofs=7,
        clearance=0.05,
        min_target_mask_pixels=1,
        base_target_pixels=None,
        min_visible_fraction=0.55,
        check_robot_plan=False,
        debug=False,
    ):
        self.bullet_client.performCollisionDetection()

        pts = self.bullet_client.getClosestPoints(
            bodyA=body_id,
            bodyB=target_body_id,
            distance=clearance,
        )
        if len(pts) > 0:
            if debug:
                print("reject distractor: target closestPoints", len(pts))
            return False

        for other_id in self.distractor_ids:
            pts = self.bullet_client.getClosestPoints(
                bodyA=body_id,
                bodyB=other_id,
                distance=clearance,
            )
            if len(pts) > 0:
                if debug:
                    print("reject distractor: other closestPoints", len(pts))
                return False

        if check_robot_plan:
            if self.distractor_too_close_to_robot_plan(
                body_id=body_id,
                robot_body_id=robot_body_id,
                end_effector_index=end_effector_index,
                planned_waypoints=planned_waypoints,
                ik_lower_limits=ik_lower_limits,
                ik_upper_limits=ik_upper_limits,
                ik_joint_ranges=ik_joint_ranges,
                get_current_arm_joints_fn=get_current_arm_joints_fn,
                quat_slerp_fn=quat_slerp_fn,
                panda_num_dofs=panda_num_dofs,
                clearance=clearance,
            ):
                if debug:
                    print("reject distractor: robot plan")
                return False

        current_pixels = self.body_mask_pixel_count(render_agentview_fn, target_body_id)
        if base_target_pixels is None:
            pixel_threshold = int(max(1, min_target_mask_pixels))
        else:
            pixel_threshold = int(max(1, min_target_mask_pixels, min_visible_fraction * base_target_pixels))

        if current_pixels < pixel_threshold:
            if debug:
                print(
                    "reject distractor: target visibility",
                    "current=", current_pixels,
                    "threshold=", pixel_threshold,
                    "base=", base_target_pixels,
                )
            return False

        return True

    def sample_and_load_distractors(
        self,
        distractor_root="/mnt/storage/GoogleScannedObjects",
        num_range=(1, 5),
        target_size_range=(0.06, 0.16),
        workspace=((0.25, 0.78), (-0.42, 0.42)),
        clearance=0.05,
        path_clearance=0.04,
        min_target_mask_pixels=1,
        max_attempts_per_distractor=10,
        ground_z=0.0,
        spawn_clearance=0.005,
        target_body_id=None,
        robot_body_id=None,
        robot_base_offset=None,
        planned_waypoints=None,
        render_agentview_fn=None,
        end_effector_index=None,
        ik_lower_limits=None,
        ik_upper_limits=None,
        ik_joint_ranges=None,
        get_current_arm_joints_fn=None,
        quat_slerp_fn=None,
        panda_num_dofs=7,
        check_xy_safety=True,
        check_robot_plan=False,
        min_visible_fraction=0.55,
        debug=False,
    ):
        self.clear_distractors()

        required = {
            "target_body_id": target_body_id,
            "robot_body_id": robot_body_id,
            "robot_base_offset": robot_base_offset,
            "planned_waypoints": planned_waypoints,
            "render_agentview_fn": render_agentview_fn,
            "end_effector_index": end_effector_index,
            "ik_lower_limits": ik_lower_limits,
            "ik_upper_limits": ik_upper_limits,
            "ik_joint_ranges": ik_joint_ranges,
            "get_current_arm_joints_fn": get_current_arm_joints_fn,
            "quat_slerp_fn": quat_slerp_fn,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"Missing required distractorDR arguments: {missing}")

        meshes = self.discover_gso_meshes(distractor_root)
        if len(meshes) == 0:
            return []

        base_target_pixels = self.body_mask_pixel_count(render_agentview_fn, target_body_id)
        if debug:
            print("DistractorDR base_target_pixels:", base_target_pixels)
            print("DistractorDR ground_z:", ground_z)
            print("DistractorDR workspace:", workspace)

        # If the target is already invisible before distractors, adding distractors
        # cannot fix the scene. Return early instead of burning attempts.
        if base_target_pixels < int(max(1, min_target_mask_pixels)):
            print(
                "Warning: target is already not visible before distractor loading. "
                f"pixels={base_target_pixels}, min={min_target_mask_pixels}. Skip distractors."
            )
            return []

        rng = self.distractor_rng
        low, high = int(num_range[0]), int(num_range[1])
        if high < low:
            low, high = high, low
        num_distractors = int(rng.integers(low, high + 1))

        min_size, max_size = float(target_size_range[0]), float(target_size_range[1])
        if max_size < min_size:
            min_size, max_size = max_size, min_size

        x_bounds, y_bounds = workspace
        total_reject_stats = {}

        for k in range(num_distractors):
            accepted = False
            reject_stats = {}

            for attempt in range(max_attempts_per_distractor):
                obj_path = str(rng.choice(meshes))
                bbox = self.read_obj_bbox(obj_path)
                if bbox is None:
                    reject_stats["bad_bbox"] = reject_stats.get("bad_bbox", 0) + 1
                    continue

                _, _, _, dims = bbox
                target_size = float(rng.uniform(min_size, max_size))
                half_extents, _ = self._bbox_to_scaled_half_extents(dims, target_size)
                if half_extents is None:
                    reject_stats["bad_extents"] = reject_stats.get("bad_extents", 0) + 1
                    continue

                xy = np.array([
                    rng.uniform(float(x_bounds[0]), float(x_bounds[1])),
                    rng.uniform(float(y_bounds[0]), float(y_bounds[1])),
                ], dtype=float)
                yaw = float(rng.uniform(-np.pi, np.pi))

                if check_xy_safety and not self.is_distractor_xy_safe(
                    xy=xy,
                    half_extents=half_extents,
                    target_body_id=target_body_id,
                    planned_waypoints=planned_waypoints,
                    robot_base_offset=robot_base_offset,
                    clearance=clearance,
                    path_clearance=path_clearance,
                ):
                    reject_stats["xy"] = reject_stats.get("xy", 0) + 1
                    continue

                body_id, meta = self.create_bbox_collision_gso_body(
                    obj_path=obj_path,
                    xy=xy,
                    yaw=yaw,
                    target_size=target_size,
                    ground_z=ground_z,
                    spawn_clearance=spawn_clearance,
                )
                if body_id is None:
                    reject_stats["create_body"] = reject_stats.get("create_body", 0) + 1
                    continue

                ok = self.loaded_distractor_is_safe(
                    body_id=body_id,
                    target_body_id=target_body_id,
                    robot_body_id=robot_body_id,
                    end_effector_index=end_effector_index,
                    planned_waypoints=planned_waypoints,
                    ik_lower_limits=ik_lower_limits,
                    ik_upper_limits=ik_upper_limits,
                    ik_joint_ranges=ik_joint_ranges,
                    get_current_arm_joints_fn=get_current_arm_joints_fn,
                    quat_slerp_fn=quat_slerp_fn,
                    render_agentview_fn=render_agentview_fn,
                    panda_num_dofs=panda_num_dofs,
                    clearance=clearance,
                    min_target_mask_pixels=min_target_mask_pixels,
                    base_target_pixels=base_target_pixels,
                    min_visible_fraction=min_visible_fraction,
                    check_robot_plan=check_robot_plan,
                    debug=debug,
                )

                if ok:
                    self.distractor_ids.append(body_id)
                    self.distractor_metadata.append(meta)
                    accepted = True
                    break

                reject_stats["loaded"] = reject_stats.get("loaded", 0) + 1
                self.bullet_client.removeBody(body_id)

            for key, value in reject_stats.items():
                total_reject_stats[key] = total_reject_stats.get(key, 0) + value

            if not accepted:
                print(
                    f"Warning: failed to place distractor {k + 1}/{num_distractors} "
                    f"after {max_attempts_per_distractor} attempts. Reject stats: {reject_stats}"
                )

        print(
            f"Loaded {len(self.distractor_ids)}/{num_distractors} GSO distractors. "
            f"Total reject stats: {total_reject_stats}"
        )
        return self.distractor_ids
