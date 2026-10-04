import os

import numpy as np
import igl
import trimesh
import slippage_reshaping as sr
from pathlib import Path
import open3d as o3d
from sim_world.src.ultility import load_initial_grasp_pose, coacd_convex_decomposition

class ShapeAugmentor:
    def __init__(self, obj_path, initial_grasp_path=None):
        self.obj_path = Path(obj_path)

        self.mesh = trimesh.load(
            self.obj_path,
            process=False,
            maintain_order=True,
        )

        self.initial_grasp_guess = (
            load_initial_grasp_pose(initial_grasp_path)
            if initial_grasp_path is not None
            else None
        )

        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a single Trimesh, got {type(self.mesh)}")

        self.V_opt = None
        self.face_k1 = None
        self.face_k2 = None
        self._vertex_adj = None
        self._coacd_template_parts = None

        self._sync_arrays()
        """ 
        self.V        # Original reference mesh for grasp transfer and COACD barycentric binding
        self.V_ref    # Copy of the original mesh
        self.V_work   # Working mesh for chained reshaping; the next reshape starts here
        self.V_opt    # Most recent/final reshaping result
        """
        self.V_work = self.V.copy()

        if not self.is_manifold():
            print("Input mesh is not manifold. Running safe in-place repair...")
            raise ValueError("Input mesh is not manifold. Repair mesh firstly.")
        
        # Run COACD only once on the base mesh.
        # Later FPSA outputs reuse this decomposition topology and only update
        # the convex-part vertices by barycentric transfer from self.V -> self.V_opt.
        self.mesh_coacd = coacd_convex_decomposition(str(self.obj_path))

        # default slippare reshaping weights
        # VertexSolveParams defaults match reshaping_params.h.
        # Create or modify these parameters before each optimize_mesh call.
        self.vertex_solve_params = sr.VertexSolveParams()
        self.vertex_solve_params.normal_weight = 10.0
        self.vertex_solve_params.edge_weight = 1.0
        self.vertex_solve_params.sphericity_weight = 1.0
        self.vertex_solve_params.sphericity_k_max_const = 2.0
        self.vertex_solve_params.straightness_weight = 1e5
        self.vertex_solve_params.regularizer_weight = 1e-4
        self.vertex_solve_params.bc_weight = 1e5

        # Its nested parameters can also be tuned when handle error distribution is enabled.
        # self.vertex_solve_params.handle_error_distrib.num_rings = 4
        # self.vertex_solve_params.handle_error_distrib.off_zone_hc_weight = 1e5
        # self.vertex_solve_params.handle_error_distrib.weakened_hc_weight = 1e2
        # self.vertex_solve_params.handle_error_distrib.inner_ring_error_tol = 2.0
        # self.vertex_solve_params.handle_error_distrib.min_error_tol = np.deg2rad(10.0)

    # ============================================================
    # Mesh utilities
    # ============================================================

    def _sync_arrays(self):
        self.V = np.asarray(self.mesh.vertices, dtype=np.float64)
        self.F = np.asarray(self.mesh.faces, dtype=np.int64)

        # Any mesh topology/face update invalidates cached adjacency / KDTree.
        self._vertex_adj = None
        self._vertex_kdtree = None

        if self.F.ndim != 2 or self.F.shape[1] != 3:
            raise ValueError(f"Input mesh must be triangulated, got F.shape={self.F.shape}")

    def mesh_status(self):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.faces))

        return {
            "edge_manifold_allow_boundary": mesh_o3d.is_edge_manifold(allow_boundary_edges=True),
            "edge_manifold_no_boundary": mesh_o3d.is_edge_manifold(allow_boundary_edges=False),
            "vertex_manifold": mesh_o3d.is_vertex_manifold(),
            "watertight": mesh_o3d.is_watertight(),
            "orientable": mesh_o3d.is_orientable(),
        }

    def is_manifold(self):
        status = self.mesh_status()
        return (
            status["edge_manifold_allow_boundary"]
            and status["vertex_manifold"]
            and status["orientable"]
        )

    # ============================================================
    # Slippage-preserving reshaping
    # ============================================================

    def get_fk(self, radius=5, use_k_ring=True):
        V_in = self.V_work
        result = igl.principal_curvature(
            V_in,
            self.F,
            radius,
            use_k_ring,
        )

        _, _, PV1, PV2 = result[:4]

        PV1 = np.asarray(PV1, dtype=np.float64).reshape(-1)
        PV2 = np.asarray(PV2, dtype=np.float64).reshape(-1)

        if len(result) >= 5:
            bad_vertices = np.asarray(result[4], dtype=np.int64)
            if bad_vertices.size > 0:
                print(f"Warning: principal_curvature reported {bad_vertices.size} bad vertices")
                PV1[bad_vertices] = 0.0
                PV2[bad_vertices] = 0.0

        PV1 = np.nan_to_num(PV1, nan=0.0, posinf=0.0, neginf=0.0)
        PV2 = np.nan_to_num(PV2, nan=0.0, posinf=0.0, neginf=0.0)

        self.face_k1 = np.asarray(igl.average_onto_faces(self.F, PV1), dtype=np.float64)
        self.face_k2 = np.asarray(igl.average_onto_faces(self.F, PV2), dtype=np.float64)

        return self.face_k1, self.face_k2

    def slippage_reshape(
        self,
        constraint_ids,
        target_positions=None,
        max_iters=20,
        handle_error_distrib_enabled=False,
        input_name=None,
    ):
        constraint_ids = list(constraint_ids)

        if target_positions is None:
            raise ValueError("target_positions must be provided")

        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.shape != (len(constraint_ids), 3):
            raise ValueError(
                f"target_positions must have shape {(len(constraint_ids), 3)}, "
                f"got {target_positions.shape}"
            )

        # ------------------------------------------------------------
        # bbox_diag conversion layer
        #
        # External API convention:
        #   - target_positions is absolute target position in current mesh scale.
        #   - if called from displacement_reshape(), displacements are also absolute-scale
        #     because displacement_reshape() has already done:
        #         target = original + absolute_disp
        #
        # Internal C++ .deform convention:
        #   - displacement is normalized by bbox diagonal:
        #         normalized_disp = absolute_disp / bbox_diag
        #
        # This block makes the conversion explicit while keeping the public Python
        # interface absolute-scale.
        # ------------------------------------------------------------
        V_in = self.V_work
        constraint_vertices = V_in[constraint_ids].copy()

        bbox_min = V_in.min(axis=0)
        bbox_max = V_in.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)

        if bbox_diag <= 0.0:
            raise ValueError(f"Invalid mesh bbox_diag={bbox_diag}")

        absolute_displacements = target_positions - constraint_vertices

        # The C++ deformation side later divides the constraint displacement by
        # bbox_diag.  Therefore, to make the solver see the user-requested
        # displacement magnitude, we pass a bbox_diag-scaled target position.
        target_positions_for_solver = (
            constraint_vertices + absolute_displacements * bbox_diag
        )

        face_k1, face_k2 = self.get_fk()

        V_opt_raw = sr.optimize_mesh(
            V_in,
            self.F,
            face_k1,
            face_k2,
            constraint_ids,
            target_positions_for_solver,
            max_iters=max_iters,
            handle_error_distrib_enabled=handle_error_distrib_enabled,
            input_name=input_name or self.obj_path.stem,
            vertex_solve_params=self.vertex_solve_params,
        )

        # sr.optimize_mesh returns vertices whose displacement is still in the
        # same normalized units used internally by FPSA.  Convert the final mesh
        # back to the real object scale before exporting / COACD transfer / grasp
        # transfer:
        #     V_new = V_in + (V_opt_raw - V_in) / bbox_diag
        V_opt_raw = np.asarray(V_opt_raw, dtype=np.float64)
        if V_opt_raw.shape != V_in.shape:
            raise ValueError(
                f"optimize_mesh returned V_opt with shape {V_opt_raw.shape}, "
                f"expected {V_in.shape}"
            )

        V_new = V_in + (V_opt_raw - V_in) / bbox_diag

        self.V_opt = V_new.copy()
        self.V_work = V_new.copy()
        self.face_k1 = None
        self.face_k2 = None
        self._vertex_kdtree = None
        return self.V_opt

    def displacement_reshape(
        self,
        constraint_ids,
        displace_idxs,
        displacements,
        max_iters=20,
        handle_error_distrib_enabled=False,
        input_name=None,
        reshape_method="slippage",
    ):
        constraint_ids = list(constraint_ids)
        displace_idxs = list(displace_idxs)
        displacements = np.asarray(displacements, dtype=np.float64)

        if displacements.ndim == 1:
            if len(displace_idxs) != 1 or displacements.shape[0] != 3:
                raise ValueError(
                    "A 1D displacement must be a single 3D vector and requires exactly one displace_idx"
                )
            displacements = displacements.reshape(1, 3)

        if displacements.shape != (len(displace_idxs), 3):
            raise ValueError(
                f"displacements must have shape {(len(displace_idxs), 3)}, "
                f"got {displacements.shape}"
            )

        V_in = self.V_work
        target_positions = V_in[constraint_ids].copy()
        for vid, disp in zip(displace_idxs, displacements):
            if vid not in constraint_ids:
                raise ValueError(f"displace_idxs must be a subset of constraint_ids, got {vid}")

            row = constraint_ids.index(vid)
            target_positions[row] += disp

        method = str(reshape_method).lower()

        if method == "slippage":
            return self.slippage_reshape(
                constraint_ids=constraint_ids,
                target_positions=target_positions,
                max_iters=max_iters,
                handle_error_distrib_enabled=handle_error_distrib_enabled,
                input_name=input_name,
            )
        elif method == "arap":
            return self.ARAP_reshape(
                constraint_ids=constraint_ids,
                target_positions=target_positions,
                max_iters=max_iters,
                handle_error_distrib_enabled=handle_error_distrib_enabled,
                input_name=input_name,
            )
        else:
            raise ValueError(
                f"Unknown reshape_method: {reshape_method}. "
                "Supported: 'slippage', 'ARAP'"
            )

    def write_augment_obj(self, output_path, write_coacd=True, return_paths=False):
        if self.V_opt is None:
            raise ValueError("No optimized vertices found. Please run a reshape method first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        out_mesh = self.mesh.copy()
        out_mesh.vertices = np.asarray(self.V_opt, dtype=np.float64)
        out_mesh.export(output_path)

        coacd_output_path = None
        if write_coacd:
            coacd_output_path = self.write_deformed_coacd_obj(obj_filename=output_path)

        if return_paths:
            return str(output_path), coacd_output_path
        return None
    

    # ============================================================
    # ARAP deformation
    # ============================================================
    def ARAP_reshape(self,
                     constraint_ids,
                     target_positions=None,
                     max_iters=20,
                     handle_error_distrib_enabled=False,
                     input_name=None,
                     ):
        """constraint_ids= [339, 343, 345, 346, 846],
                                   displace_idxs= [846],
                                   displacements= np.array([0.21873721885442937,
                                                        -4.568995076824393e-09,
                                                        -8.609569257000194e-10]),
                                    max_iters= 100
        """
        if target_positions is None:
            raise ValueError("target_positions must be provided")

        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.shape != (len(constraint_ids), 3):
            raise ValueError(
                f"target_positions must have shape {(len(constraint_ids), 3)}, "
                f"got {target_positions.shape}"
            )
        
        V_in = self.V_work

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V_in)
        mesh.triangles = o3d.utility.Vector3iVector(self.F)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        # ic(constraint_ids, target_positions)
        # ic(V_in[constraint_ids])

        constraint_ids_o3d = o3d.utility.IntVector([int(i) for i in constraint_ids])
        target_positions_o3d = o3d.utility.Vector3dVector(
            np.asarray(target_positions, dtype=np.float64)
        )

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug):
            mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids_o3d,
                                                        target_positions_o3d,
                                                        max_iter=max_iters,
                                                        energy = o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,)# Smoothed

        V_new = np.asarray(mesh_prime.vertices, dtype=np.float64)
        if V_new.shape != V_in.shape:
            raise ValueError(
                f"ARAP returned V_new with shape {V_new.shape}, expected {V_in.shape}"
            )

        self.V_opt = V_new.copy()
        self.V_work = V_new.copy()
        self.face_k1 = None
        self.face_k2 = None
        self._vertex_kdtree = None
        return self.V_opt

    # ============================================================
    # Cached COACD deformation transfer
    # ============================================================

    @staticmethod
    def _load_scene_parts_as_meshes(mesh_path):
        """
        Load a COACD OBJ as separate convex parts.

        COACD is exported as a trimesh.Scene, so loading it as one forced mesh
        would lose the part boundaries.  Keeping parts separated is useful for
        pybullet / simulator collision meshes.
        """
        mesh_path = Path(mesh_path)
        loaded = trimesh.load(str(mesh_path), force="scene", process=False)

        if isinstance(loaded, trimesh.Trimesh):
            return [loaded.copy()]

        if not isinstance(loaded, trimesh.Scene):
            raise TypeError(f"Expected Trimesh or Scene from {mesh_path}, got {type(loaded)}")

        dumped = loaded.dump(concatenate=False)
        if isinstance(dumped, trimesh.Trimesh):
            dumped = [dumped]

        parts = []
        for part in dumped:
            if not isinstance(part, trimesh.Trimesh):
                continue
            if len(part.vertices) == 0 or len(part.faces) == 0:
                continue

            components = part.split(only_watertight=False)
            if len(components) == 0:
                components = [part]

            for component in components:
                if len(component.vertices) > 0 and len(component.faces) > 0:
                    parts.append(component.copy())

        if len(parts) == 0:
            raise ValueError(f"No mesh parts found in COACD file: {mesh_path}")

        return parts

    @staticmethod
    def _copy_visual_data(mesh):
        """
        Copy only concrete visual arrays from a Trimesh.

        Do not store/copy trimesh.visual.ColorVisuals itself here: after it is
        detached from its source mesh, ColorVisuals.copy() may try to read
        self.mesh.faces and crash with "NoneType has no attribute faces".
        """
        visual_data = {}
        visual = getattr(mesh, "visual", None)
        if visual is None:
            return visual_data

        # COACD parts in this file are colored through vertex_colors, but after
        # OBJ export/reload trimesh may expose usable face_colors instead.  Store
        # both when their lengths match the mesh topology.
        try:
            vertex_colors = np.asarray(visual.vertex_colors).copy()
            if vertex_colors.ndim == 2 and len(vertex_colors) == len(mesh.vertices):
                visual_data["vertex_colors"] = vertex_colors
        except Exception:
            pass

        try:
            face_colors = np.asarray(visual.face_colors).copy()
            if face_colors.ndim == 2 and len(face_colors) == len(mesh.faces):
                visual_data["face_colors"] = face_colors
        except Exception:
            pass

        return visual_data

    @staticmethod
    def _apply_visual_data(mesh, visual_data):
        """Apply visual arrays copied by _copy_visual_data() to a new mesh."""
        if not isinstance(visual_data, dict):
            return

        vertex_colors = visual_data.get("vertex_colors")
        face_colors = visual_data.get("face_colors")

        if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
            mesh.visual.vertex_colors = np.asarray(vertex_colors).copy()
        elif face_colors is not None and len(face_colors) == len(mesh.faces):
            mesh.visual.face_colors = np.asarray(face_colors).copy()

    def _build_coacd_deformation_template(self):
        """
        Precompute how each base COACD vertex attaches to the base mesh surface.

        Each convex-part vertex is projected once onto the original FPSA mesh.
        We store (face_id, barycentric coordinate).  For every later V_opt, the
        same barycentric coordinate on the same original face id gives the
        corresponding deformed COACD vertex.

        This avoids rerunning coacd.run_coacd() after every shape deformation.
        """
        if self._coacd_template_parts is not None:
            return self._coacd_template_parts

        coacd_parts = self._load_scene_parts_as_meshes(self.mesh_coacd)
        template_parts = []

        for part_idx, part in enumerate(coacd_parts):
            part_vertices = np.asarray(part.vertices, dtype=np.float64)
            part_faces = np.asarray(part.faces, dtype=np.int64)

            sqrD, face_ids, closest_points = igl.point_mesh_squared_distance(
                part_vertices,
                self.V,
                self.F,
            )

            face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
            closest_points = np.asarray(closest_points, dtype=np.float64)

            barycentric = np.zeros((len(part_vertices), 3), dtype=np.float64)
            for i, face_id in enumerate(face_ids):
                tri = self.F[int(face_id)]
                a, b, c = self.V[tri[0]], self.V[tri[1]], self.V[tri[2]]
                barycentric[i] = self._point_triangle_barycentric(
                    closest_points[i],
                    a,
                    b,
                    c,
                )

            template_parts.append({
                "name": f"coacd_part_{part_idx:04d}",
                "faces": part_faces.copy(),
                "face_ids": face_ids.copy(),
                "barycentric": barycentric,
                "visual_data": self._copy_visual_data(part),
                "metadata": dict(part.metadata) if part.metadata is not None else {},
                "mean_bind_error": float(np.sqrt(np.asarray(sqrD, dtype=np.float64)).mean()),
                "max_bind_error": float(np.sqrt(np.asarray(sqrD, dtype=np.float64)).max()),
            })

        self._coacd_template_parts = template_parts
        return self._coacd_template_parts

    def _deform_coacd_vertices_from_template(self, template_part, deformed_vertices):
        face_ids = np.asarray(template_part["face_ids"], dtype=np.int64)
        barycentric = np.asarray(template_part["barycentric"], dtype=np.float64)
        deformed_vertices = np.asarray(deformed_vertices, dtype=np.float64)

        tri_ids = self.F[face_ids]
        tri_vertices = deformed_vertices[tri_ids]

        # Sum_j barycentric[:, j] * deformed_vertices[face_vertex_j].
        return np.einsum("ni,nij->nj", barycentric, tri_vertices)

    @staticmethod
    def _write_obj_parts(output_path, parts):
        """Write each convex part as its own OBJ object group."""
        vertex_offset = 0
        colors = (np.random.RandomState(0).rand(len(parts), 3) * 255).astype(np.uint8)
        mtl_path = output_path.with_suffix(".mtl")

        with open(mtl_path, "w", encoding="utf-8") as mtl:
            for part_idx, color in enumerate(colors):
                mtl.write(f"newmtl coacd_part_{part_idx:04d}\n")
                mtl.write(f"Kd {color[0] / 255:.6f} {color[1] / 255:.6f} {color[2] / 255:.6f}\n\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# deformed COACD convex decomposition\n")
            f.write(f"mtllib {mtl_path.name}\n")

            for part_idx, part in enumerate(parts):
                name = part.get("name", f"coacd_part_{part_idx:04d}")
                vertices = np.asarray(part["vertices"], dtype=np.float64)
                faces = np.asarray(part["faces"], dtype=np.int64)

                if vertices.ndim != 2 or vertices.shape[1] != 3:
                    raise ValueError(f"Invalid vertices for {name}: {vertices.shape}")
                if faces.ndim != 2 or faces.shape[1] != 3:
                    raise ValueError(f"Invalid triangular faces for {name}: {faces.shape}")

                f.write(f"o {name}\n")
                f.write(f"usemtl coacd_part_{part_idx:04d}\n")
                for x, y, z in vertices:
                    f.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")

                for i, j, k in faces:
                    a = int(i) + vertex_offset + 1
                    b = int(j) + vertex_offset + 1
                    c = int(k) + vertex_offset + 1
                    f.write(f"f {a} {b} {c}\n")

                vertex_offset += len(vertices)

    def write_deformed_coacd_obj(self, obj_filename=None, output_path=None):
        """
        Write the COACD mesh corresponding to the current FPSA deformation.

        The OBJ exporter is intentionally manual: PyBullet separates convex
        pieces by OBJ ``o`` groups, while trimesh.Scene.export() may collapse
        the scene into one object group.
        """
        if self.V_opt is None:
            raise ValueError("No optimized vertices found. Please run a reshape method first.")

        if output_path is None:
            if obj_filename is None:
                raise ValueError(
                    "obj_filename must be provided when output_path is None, "
                    "otherwise the base mesh _coacd.obj could be overwritten."
                )
            output_path = os.path.splitext(str(obj_filename))[0] + "_coacd.obj"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parts = []
        for template_part in self._build_coacd_deformation_template():
            parts.append({
                "name": template_part["name"],
                "vertices": self._deform_coacd_vertices_from_template(
                    template_part,
                    self.V_opt,
                ),
                "faces": template_part["faces"],
            })

        self._write_obj_parts(output_path, parts)
        print(f"[COACD]: Wrote deformed cached decomposition mesh {output_path}")
        return str(output_path)

    # ============================================================
    # Geometry helpers for anchors and patches
    # ============================================================

    @staticmethod
    def _normalize(v, eps=1e-12):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < eps:
            return v
        return v / n

    @staticmethod
    def _point_triangle_barycentric(p, a, b, c, eps=1e-12):
        """
        Compute barycentric coordinates of point p w.r.t. triangle (a, b, c).
        Returns bary = [u, v, w], where p ~= u*a + v*b + w*c.
        """
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)

        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < eps:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        bary = np.array([u, v, w], dtype=np.float64)

        # closest_points from igl should already be inside/on the triangle, but this
        # avoids tiny numerical negatives.
        bary = np.clip(bary, 0.0, 1.0)
        s = bary.sum()
        if s < eps:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        return bary / s

    @staticmethod
    def _barycentric_point(V, F, face_id, bary):
        tri = F[int(face_id)]
        b0, b1, b2 = np.asarray(bary, dtype=np.float64)
        return b0 * V[tri[0]] + b1 * V[tri[1]] + b2 * V[tri[2]]

    def _build_vertex_adjacency(self):
        adj = [set() for _ in range(len(self.V))]
        for tri in self.F:
            i, j, k = map(int, tri)
            adj[i].update([j, k])
            adj[j].update([i, k])
            adj[k].update([i, j])
        return adj

    def get_k_ring_vertices(self, seed_vertex_ids, k_ring=2):
        """
        Return k-ring vertex ids around seed vertices.
        k_ring=0 returns only seed vertices.
        """
        seed_vertex_ids = [int(v) for v in seed_vertex_ids]

        if self._vertex_adj is None:
            self._vertex_adj = self._build_vertex_adjacency()

        visited = set(seed_vertex_ids)
        frontier = set(seed_vertex_ids)

        for _ in range(int(k_ring)):
            new_frontier = set()
            for v in frontier:
                new_frontier.update(self._vertex_adj[v])
            new_frontier -= visited
            visited.update(new_frontier)
            frontier = new_frontier

        return np.array(sorted(visited), dtype=np.int64)

    def get_xyz_nearest_vertices(self, point, num_vertices=32):
        """Return vertex ids nearest to point in xyz Euclidean space."""
        point = np.asarray(point, dtype=np.float64).reshape(1, 3)
        num_vertices = int(min(num_vertices, len(self.V)))

        if self._vertex_kdtree is None:
            from scipy.spatial import cKDTree
            self._vertex_kdtree = cKDTree(self.V)

        _, vertex_ids = self._vertex_kdtree.query(point, k=num_vertices)
        return np.asarray(vertex_ids, dtype=np.int64).reshape(-1)

    def closest_surface_anchor(
        self,
        point,
        k_ring=2,
        patch_method="k_ring",
        num_patch_vertices=32,
    ):
        """
        Project a 3D point to the original mesh surface and create an anchor.

        Args:
            patch_method:
                "k_ring": use topological k-ring around the closest face.
                "xyz": use KDTree xyz-nearest vertices around the input point/TCP.

        Returns:
            anchor = {
                "face_id": int,
                "barycentric": [b0, b1, b2],
                "anchor_point_old": [x, y, z],
                "patch_vertex_ids": [...],
                "source_point_old": [x, y, z],
                "k_ring": int,
                "patch_method": str,
            }
        """
        point = np.asarray(point, dtype=np.float64).reshape(1, 3)

        sqrD, face_ids, closest_points = igl.point_mesh_squared_distance(
            point,
            self.V,
            self.F,
        )

        face_id = int(face_ids[0])
        anchor_point_old = np.asarray(closest_points[0], dtype=np.float64)

        tri = self.F[face_id]
        a, b, c = self.V[tri[0]], self.V[tri[1]], self.V[tri[2]]
        bary = self._point_triangle_barycentric(anchor_point_old, a, b, c)

        if patch_method == "k_ring":
            patch_vertex_ids = self.get_k_ring_vertices(
                seed_vertex_ids=tri,
                k_ring=k_ring,
            )
        elif patch_method in ("xyz", "xyz_nearest", "kdtree"):
            patch_vertex_ids = self.get_xyz_nearest_vertices(
                point=point,
                num_vertices=num_patch_vertices,
            )
        else:
            raise ValueError(
                f"Unknown patch_method={patch_method}. "
                "Supported: 'k_ring', 'xyz'/'xyz_nearest'/'kdtree'."
            )

        return {
            "face_id": face_id,
            "barycentric": bary.tolist(),
            "anchor_point_old": anchor_point_old.tolist(),
            "patch_vertex_ids": patch_vertex_ids.tolist(),
            "source_point_old": point.reshape(3).tolist(),
            "k_ring": int(k_ring),
            "patch_method": str(patch_method),
            "num_patch_vertices": int(len(patch_vertex_ids)),
            "squared_distance_to_mesh": float(sqrD[0]),
        }

    def _patch_weights_from_anchor(self, patch_ids, anchor_point_old, sigma=None):
        """
        Distance weights for local shape matching.
        Vertices closer to the anchor receive larger weights.
        """
        patch_ids = np.asarray(patch_ids, dtype=np.int64)
        P = self.V[patch_ids]
        anchor_point_old = np.asarray(anchor_point_old, dtype=np.float64).reshape(3)

        d = np.linalg.norm(P - anchor_point_old[None, :], axis=1)

        if sigma is None:
            sigma = np.median(d) + 1e-12

        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        return w + 1e-8

    # ============================================================
    # Local shape matching / Kabsch rigid fit
    # ============================================================

    @staticmethod
    def fit_rigid_transform(P, Q, weights=None):
        """
        Fit local rigid transform by Kabsch / Procrustes.

        Mathematical convention:
            q_i ~= R @ p_i + t

        Numpy row-vector convention:
            Q_hat = P @ R.T + t

        Args:
            P: old patch vertices, shape (K, 3)
            Q: new patch vertices, shape (K, 3)
            weights: optional, shape (K,)

        Returns:
            R: shape (3, 3), in SO(3)
            t: shape (3,)
        """
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        if P.shape != Q.shape:
            raise ValueError(f"P and Q must have same shape, got {P.shape} and {Q.shape}")
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"P and Q must be Kx3, got {P.shape}")
        if P.shape[0] < 3:
            raise ValueError("Need at least 3 points to fit a stable rigid transform")

        K = P.shape[0]

        if weights is None:
            weights = np.ones(K, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if weights.shape[0] != K:
                raise ValueError(f"weights must have shape ({K},), got {weights.shape}")

        weights = weights / (weights.sum() + 1e-12)

        p_center = np.sum(P * weights[:, None], axis=0)
        q_center = np.sum(Q * weights[:, None], axis=0)

        P0 = P - p_center
        Q0 = Q - q_center

        H = P0.T @ (Q0 * weights[:, None])

        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Avoid reflection.
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = q_center - R @ p_center
        return R, t

    # ============================================================
    # SE(3) helpers
    # ============================================================

    @staticmethod
    def _project_to_SO3(R):
        """
        Project a near-rotation matrix to SO(3).
        Useful if loaded R has small numerical errors.
        """
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")

        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt

        if np.linalg.det(R_proj) < 0:
            U[:, -1] *= -1.0
            R_proj = U @ Vt

        return R_proj

    @staticmethod
    def _quat_to_R(quat, order="xyzw"):
        """
        Convert quaternion to rotation matrix.

        order:
            "xyzw": [x, y, z, w]
            "wxyz": [w, x, y, z]
        """
        q = np.asarray(quat, dtype=np.float64).reshape(4)

        if order == "xyzw":
            x, y, z, w = q
        elif order == "wxyz":
            w, x, y, z = q
        else:
            raise ValueError("quat order must be 'xyzw' or 'wxyz'")

        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-12:
            raise ValueError("Quaternion norm is too small")

        w, x, y, z = w / n, x / n, y / n, z / n

        R = np.array([
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

        return R

    @staticmethod
    def _R_to_quat(R, order="xyzw"):
        """
        Convert rotation matrix to quaternion.
        Returns xyzw by default.
        """
        R = ShapeAugmentor._project_to_SO3(R)
        tr = np.trace(R)

        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        q_xyzw = np.array([x, y, z, w], dtype=np.float64)
        q_xyzw = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-12)

        if order == "xyzw":
            return q_xyzw
        if order == "wxyz":
            return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        raise ValueError("quat order must be 'xyzw' or 'wxyz'")

    def grasp_guess_to_SE3(self, grasp_guess=None, quat_order="xyzw"):
        """
        Convert loaded grasp guess to a 4x4 SE(3).

        Supported formats:
            1. 4x4 ndarray
            2. dict["T_mesh_hand_tcp"] : 4x4
            3. dict["T"]              : 4x4
            4. dict["R"] + dict["t"]
            5. dict["quat"] + dict["t"]

        Convention:
            T_mesh_hand_tcp means hand_tcp pose expressed in mesh frame.
            T[:3, 3] is TCP position in mesh frame.
            T[:3, :3] is TCP orientation in mesh frame.
        """
        if grasp_guess is None:
            grasp_guess = self.initial_grasp_guess

        if grasp_guess is None:
            raise ValueError("No grasp guess provided and self.initial_grasp_guess is None")

        if isinstance(grasp_guess, np.ndarray):
            T = np.asarray(grasp_guess, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"grasp_guess ndarray must be 4x4, got {T.shape}")
            T = T.copy()
            T[:3, :3] = self._project_to_SO3(T[:3, :3])
            T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
            return T

        if not isinstance(grasp_guess, dict):
            raise TypeError(f"grasp_guess must be dict or 4x4 ndarray, got {type(grasp_guess)}")

        if "T_mesh_hand_tcp" in grasp_guess:
            T = np.asarray(grasp_guess["T_mesh_hand_tcp"], dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"T_mesh_hand_tcp must be 4x4, got {T.shape}")
            T = T.copy()
            T[:3, :3] = self._project_to_SO3(T[:3, :3])
            T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
            return T

        if "T" in grasp_guess:
            T = np.asarray(grasp_guess["T"], dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"T must be 4x4, got {T.shape}")
            T = T.copy()
            T[:3, :3] = self._project_to_SO3(T[:3, :3])
            T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
            return T

        if "t" not in grasp_guess:
            raise KeyError("grasp_guess must contain 't' if no 4x4 transform is provided")

        t = np.asarray(grasp_guess["t"], dtype=np.float64).reshape(3)

        if "R" in grasp_guess:
            R = np.asarray(grasp_guess["R"], dtype=np.float64)
            R = self._project_to_SO3(R)
        elif "quat" in grasp_guess:
            R = self._quat_to_R(grasp_guess["quat"], order=quat_order)
        else:
            raise KeyError("grasp_guess must contain either 'R' or 'quat' if no 4x4 transform is provided")

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    # ============================================================
    # SE(3) grasp transfer
    # ============================================================

    def make_grasp_anchor_from_SE3(
        self,
        T_grasp_old=None,
        k_ring=2,
        quat_order="xyzw",
        patch_method="k_ring",
        num_patch_vertices=32,
    ):
        """
        Build and return the surface anchor for an old SE(3) grasp pose.

        By default, the anchor is the closest surface point to the old TCP
        translation T[:3, 3]. For more accurate grasping, you can later replace
        this with real finger contact anchors.
        """
        T_grasp_old = self.grasp_guess_to_SE3(T_grasp_old, quat_order=quat_order)
        tcp_old = T_grasp_old[:3, 3]
        return self.closest_surface_anchor(
            point=tcp_old,
            k_ring=k_ring,
            patch_method=patch_method,
            num_patch_vertices=num_patch_vertices,
        )

    def transfer_grasp_SE3_by_anchor(
        self,
        T_grasp_old,
        anchor,
        use_distance_weights=True,
        quat_order="xyzw",
    ):
        """
        Transfer a full SE(3) grasp pose after mesh deformation.

        Args:
            T_grasp_old: 4x4 SE(3), or dict/format accepted by grasp_guess_to_SE3.
                Hand TCP pose in original mesh frame.

            anchor:
                Output from closest_surface_anchor(...).
                Usually built from old TCP translation:
                    anchor = make_grasp_anchor_from_SE3(T_grasp_old)

        Returns:
            T_grasp_new: 4x4 SE(3)
                Hand TCP pose in deformed mesh frame.

            debug_info: dict

        Formula:
            R_new = R_patch @ R_old
            t_new = anchor_new + R_patch @ (t_old - anchor_old)
        """
        if self.V_opt is None:
            raise ValueError("No optimized vertices found. Run a reshape method first.")

        T_grasp_old = self.grasp_guess_to_SE3(T_grasp_old, quat_order=quat_order)
        R_old = T_grasp_old[:3, :3]
        t_old = T_grasp_old[:3, 3]

        face_id = int(anchor["face_id"])
        bary = np.asarray(anchor["barycentric"], dtype=np.float64)
        patch_ids = np.asarray(anchor["patch_vertex_ids"], dtype=np.int64)

        if len(patch_ids) < 3:
            raise ValueError("Anchor patch must contain at least 3 vertices")

        P = self.V[patch_ids]
        Q = self.V_opt[patch_ids]

        anchor_point_old = self._barycentric_point(self.V, self.F, face_id, bary)
        anchor_point_new = self._barycentric_point(self.V_opt, self.F, face_id, bary)

        if use_distance_weights:
            weights = self._patch_weights_from_anchor(
                patch_ids=patch_ids,
                anchor_point_old=anchor_point_old,
            )
        else:
            weights = None

        # Local shape matching:
        #   Q ~= R_patch @ P + t_patch
        R_patch, t_patch = self.fit_rigid_transform(P, Q, weights=weights)
        R_patch = self._project_to_SO3(R_patch)

        T_grasp_new = np.eye(4, dtype=np.float64)
        T_grasp_new[:3, :3] = self._project_to_SO3(R_patch @ R_old)
        T_grasp_new[:3, 3] = anchor_point_new + R_patch @ (t_old - anchor_point_old)

        Q_pred = P @ R_patch.T + t_patch
        per_vertex_error = np.linalg.norm(Q_pred - Q, axis=1)

        debug_info = {
            "R_patch": R_patch,
            "t_patch": t_patch,
            "anchor_point_old": anchor_point_old,
            "anchor_point_new": anchor_point_new,
            "fit_error_mean": float(per_vertex_error.mean()),
            "fit_error_max": float(per_vertex_error.max()),
            "num_patch_vertices": int(len(patch_ids)),
            "det_R_patch": float(np.linalg.det(R_patch)),
        }

        return T_grasp_new, debug_info

    def transfer_grasp_SE3(
        self,
        T_grasp_old=None,
        anchor=None,
        k_ring=2,
        use_distance_weights=True,
        quat_order="xyzw",
        patch_method="k_ring",
        num_patch_vertices=32,
    ):
        """
        Public API for SE(3) grasp transfer.

        Args:
            T_grasp_old:
                If None, uses self.initial_grasp_guess.
                Otherwise can be 4x4 ndarray or supported dict.

            anchor:
                Optional precomputed anchor. If None, this function creates one
                from the old TCP position.

            patch_method:
                "k_ring": use topological k-ring around the closest face. 
                "xyz": use KDTree xyz-nearest vertices around the input point/TCP.                

        Returns:
            T_grasp_new, anchor, debug_info
        """
        T_grasp_old = self.grasp_guess_to_SE3(T_grasp_old, quat_order=quat_order)

        if anchor is None:
            anchor = self.closest_surface_anchor(
                point=T_grasp_old[:3, 3],
                k_ring=k_ring,
                patch_method=patch_method,
                num_patch_vertices=num_patch_vertices,
            )

        # ic(anchor)
        T_grasp_new, debug_info = self.transfer_grasp_SE3_by_anchor(
            T_grasp_old=T_grasp_old,
            anchor=anchor,
            use_distance_weights=use_distance_weights,
            quat_order=quat_order,
        )

        return T_grasp_new, anchor, debug_info

    def transfer_initial_grasp_guess(
        self,
        anchor=None,
        k_ring=2,
        use_distance_weights=True,
        quat_order="xyzw",
        return_format="dict",
        patch_method = "k_ring",
    ):
        """
        Convenience wrapper for self.initial_grasp_guess.
        patch_method:
            "k_ring": use topological k-ring around the closest face. 
            "xyz": use KDTree xyz-nearest vertices around the input point/TCP.  
        return_format:
            "T":    return 4x4 SE(3)
            "dict": return dict with T_mesh_hand_tcp, t, R, quat
        """
        T_new, anchor, debug_info = self.transfer_grasp_SE3(
            T_grasp_old=None,
            anchor=anchor,
            k_ring=k_ring,
            use_distance_weights=use_distance_weights,
            quat_order=quat_order,
            patch_method=patch_method,
        )

        if return_format == "T":
            return T_new, anchor, debug_info

        if return_format != "dict":
            raise ValueError("return_format must be 'T' or 'dict'")

        result = {
            "T_mesh_hand_tcp": T_new,
            "t": T_new[:3, 3].copy(),
            "R": T_new[:3, :3].copy(),
            "quat": self._R_to_quat(T_new[:3, :3], order=quat_order),
        }

        return result, anchor, debug_info

    def write_transferred_grasp(self, output_path, grasp_result):
        """
        Save transferred grasp result to .npz.

        grasp_result can be:
            - 4x4 T
            - dict returned by transfer_initial_grasp_guess(return_format="dict")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(grasp_result, dict):
            np.savez(
                output_path,
                T_mesh_hand_tcp=np.asarray(grasp_result["T_mesh_hand_tcp"], dtype=np.float64),
                t=np.asarray(grasp_result["t"], dtype=np.float64),
                R=np.asarray(grasp_result["R"], dtype=np.float64),
                quat=np.asarray(grasp_result["quat"], dtype=np.float64),
            )
        else:
            T = np.asarray(grasp_result, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"grasp_result must be dict or 4x4 T, got {T.shape}")
            np.save(output_path, T)

    # ============================================================
    # Visualization
    # ============================================================

    def _make_o3d_mesh_from_vertices(self, vertices, color=(0.75, 0.75, 0.75)):
        """
        Build an Open3D TriangleMesh from given vertices and self.F.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.shape != self.V.shape:
            raise ValueError(
                f"vertices must have shape {self.V.shape}, got {vertices.shape}"
            )

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.F.astype(np.int32))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color)
        return mesh_o3d

    @staticmethod
    def _make_pose_axis(T, axis_size=0.05):
        """
        Create an RGB coordinate frame at SE(3) pose T.

        Open3D convention:
            x-axis: red
            y-axis: green
            z-axis: blue
        """
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T must be 4x4, got {T.shape}")

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(axis_size),
            origin=np.zeros(3),
        )
        axis.transform(T)
        return axis

    @staticmethod
    def _make_sphere(center, radius, color=(1.0, 0.2, 0.2)):
        """
        Small marker sphere for anchor/contact visualization.
        """
        center = np.asarray(center, dtype=np.float64).reshape(3)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
        sphere.translate(center)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        return sphere

    def _default_vis_scale(self):
        """
        Compute a reasonable visualization scale from the deformed mesh bbox.
        """
        if self.V_opt is not None:
            V_show = np.asarray(self.V_opt, dtype=np.float64)
        else:
            V_show = np.asarray(self.V, dtype=np.float64)

        bbox_min = V_show.min(axis=0)
        bbox_max = V_show.max(axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        if diag < 1e-12:
            diag = 1.0
        return diag

    def visualize_reshaped_mesh(self):
        mesh_def = self._make_o3d_mesh_from_vertices(
            self.V_opt,
            color=(0.72, 0.72, 0.72),
        )
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

        o3d.visualization.draw_geometries(
            [mesh_def, world_frame],
            window_name="deformed mesh",
            width=1280,
            height=900,
        )

    def visualize_deformed_grasp_pose(
        self,
        T_grasp_new=None,
        anchor=None,
        debug_info=None,
        quat_order="xyzw",
        show_anchor=True,
        show_patch=True,
        show_old_grasp=False,
        T_grasp_old=None,
    ):
        """
        Visualize deformed mesh + transferred SE(3) grasp pose.

        Main use:
            T_new, anchor, debug = augmentor.transfer_initial_grasp_guess(
                k_ring=2,
                return_format="T",
            )

            augmentor.visualize_deformed_grasp_pose(
                T_grasp_new=T_new,
                anchor=anchor,
                debug_info=debug,
            )

        Visual elements:
            - deformed mesh: self.V_opt
            - RGB axis frame: transferred grasp TCP pose
            - optional red sphere: deformed surface anchor
            - optional patch vertices: small blue-ish spheres on deformed mesh
            - optional old grasp axis: original grasp pose, for comparison
        """
        if self.V_opt is None:
            raise ValueError(
                "No optimized vertices found. Run slippage_reshape() or "
                "displacement_reshape() before visualization."
            )

        scale = self._default_vis_scale()
        axis_size = 0.2 * scale
        marker_radius = 0.012 * scale

        geoms = []

        # 1. Deformed mesh.
        mesh_def = self._make_o3d_mesh_from_vertices(
            self.V_opt,
            color=(0.72, 0.72, 0.72),
        )
        geoms.append(mesh_def)

        # 2. New grasp pose axis.
        grasp_axis = self._make_pose_axis(T_grasp_new, axis_size=axis_size)
        geoms.append(grasp_axis)

        # 3. Optional anchor marker and patch vertices.
        if anchor is not None and show_anchor:
            face_id = int(anchor["face_id"])
            bary = np.asarray(anchor["barycentric"], dtype=np.float64)
            anchor_point_new = self._barycentric_point(self.V_opt, self.F, face_id, bary)
            anchor_sphere = self._make_sphere(
                anchor_point_new,
                radius=marker_radius,
                color=(1.0, 0.1, 0.1),
            )
            geoms.append(anchor_sphere)

        if anchor is not None and show_patch:
            patch_ids = np.asarray(anchor["patch_vertex_ids"], dtype=np.int64)
            patch_points = self.V_opt[patch_ids]

            # Use a point cloud for patch vertices to keep it lightweight.
            patch_pcd = o3d.geometry.PointCloud()
            patch_pcd.points = o3d.utility.Vector3dVector(patch_points)
            patch_pcd.paint_uniform_color((0.1, 0.3, 1.0))
            geoms.append(patch_pcd)

        # 4. Optional old grasp pose axis, shown in original mesh frame.
        # This is only for visual comparison. It is not transformed to V_opt.
        if show_old_grasp:
            T_grasp_old = self.grasp_guess_to_SE3(
                T_grasp_old,
                quat_order=quat_order,
            )
            old_axis = self._make_pose_axis(T_grasp_old, axis_size=0.75 * axis_size)
            geoms.append(old_axis)

        if debug_info is not None:
            print("Grasp transfer visualization debug:")
            if "fit_error_mean" in debug_info:
                print(f"  fit_error_mean: {debug_info['fit_error_mean']:.6g}")
            if "fit_error_max" in debug_info:
                print(f"  fit_error_max:  {debug_info['fit_error_max']:.6g}")
            if "num_patch_vertices" in debug_info:
                print(f"  num_patch_vertices: {debug_info['num_patch_vertices']}")

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        geoms.append(world_frame)

        o3d.visualization.draw_geometries(
            geoms,
            window_name="Deformed mesh with transferred grasp pose",
            width=1280,
            height=900,
        )
        return None
    
    def apply_transformation_to_mesh(self, T):
        """
        Apply a 4x4 SE(3) transformation to the original mesh vertices.

        Args:
            T: 4x4 SE(3) transformation matrix.
        """
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T must be 4x4, got {T.shape}")

        R = T[:3, :3]
        t = T[:3, 3]

        self.V_opt = (self.V_opt @ R.T) + t[None, :]
